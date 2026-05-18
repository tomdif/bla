"""Object-file geometry adapter for BLA System-2 readout.

Phase 18θ established that raw frozen OF-JEPA slot features
(768-dim) do not directly support goal-relative value prediction
at the 720-sample data budget. Phase 18λ tests whether a small
structured adapter can extract goal-relative geometric features
(cube_xy, eef_xyz, push_dir, etc.) from slots, then feed those
derived features to the value head.

This module implements the adapter as a supervised regressor:
`(slot_flat, goal_xy) → 10-dim geometry`. The ground-truth
geometry comes from the simulator (Phase 18β `state_features`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ObjectFileGeometryAdapter(nn.Module):
    """Maps (slot_flat, goal_xy) to 10-dim goal-relative geometry.

    Output dimensions match Phase 18β `state_features`:
      cube_xy(2) + eef_xy(2) + eef_z(1) + cube_z(1) + goal_xy(2) + push_dir(2)
      = 10
    """

    def __init__(
        self,
        slot_dim: int,
        goal_dim: int = 2,
        out_dim: int = 10,
        hidden: int = 256,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.goal_dim = int(goal_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_hidden = int(n_hidden)
        in_dim = self.slot_dim + self.goal_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, slot: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """slot: [B, slot_dim]; goal: [B, goal_dim] → [B, out_dim]."""
        if slot.ndim == 1:
            slot = slot.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        x = torch.cat([slot.float(), goal.float()], dim=-1)
        return self.net(x)


@dataclass
class AdapterStats:
    initial_train_mse: float
    final_train_mse: float
    initial_val_mse: float
    final_val_mse: float
    per_feature_val_pearson: list[float]
    per_feature_val_spearman: list[float]
    mean_val_pearson: float
    mean_val_spearman: float
    n_train: int
    n_val: int
    steps: int


def _per_feature_corrs(pred: torch.Tensor, target: torch.Tensor) -> tuple[list[float], list[float]]:
    """Per-feature Pearson + Spearman between predicted and target."""
    import numpy as np
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    pearson = []
    spearman = []
    for j in range(p.shape[1]):
        pj = p[:, j]; tj = t[:, j]
        if pj.std() < 1e-9 or tj.std() < 1e-9:
            pearson.append(float("nan"))
            spearman.append(float("nan"))
            continue
        pearson.append(float(np.corrcoef(pj, tj)[0, 1]))
        prk = np.argsort(np.argsort(pj)).astype(float)
        trk = np.argsort(np.argsort(tj)).astype(float)
        spearman.append(float(np.corrcoef(prk, trk)[0, 1]))
    return pearson, spearman


def train_adapter_supervised(
    adapter: ObjectFileGeometryAdapter,
    slots: torch.Tensor,
    goals: torch.Tensor,
    target_geo: torch.Tensor,
    *,
    val_split: float = 0.2,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 0,
) -> AdapterStats:
    """Fit adapter(slot, goal) -> target_geo with MSE."""
    if not (len(slots) == len(goals) == len(target_geo)):
        raise ValueError("slots / goals / target_geo must align")
    n = int(len(slots))
    if n == 0:
        raise ValueError("empty adapter dataset")

    device = next(adapter.parameters()).device
    slots = slots.float().to(device)
    goals = goals.float().to(device)
    target_geo = target_geo.float().to(device)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_split * n))
    val_idx = perm[:n_val].to(device)
    tr_idx = perm[n_val:].to(device)
    n_train = int(tr_idx.numel())

    def mse_on(idx):
        with torch.no_grad():
            pred = adapter(slots[idx], goals[idx])
            return float(((pred - target_geo[idx]) ** 2).mean())

    initial_tr = mse_on(tr_idx)
    initial_val = mse_on(val_idx)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr,
                              weight_decay=weight_decay)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    final_tr = initial_tr
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n_train, (min(batch_size, n_train),),
                                    generator=gen, device=device)]
        pred = adapter(slots[sel], goals[sel])
        loss = ((pred - target_geo[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
        opt.step()
        final_tr = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"  adapter step {step+1}/{steps} mse={final_tr:.5f}",
                   flush=True)

    final_tr = mse_on(tr_idx)
    final_val = mse_on(val_idx)
    with torch.no_grad():
        val_pred = adapter(slots[val_idx], goals[val_idx])
        pearson, spearman = _per_feature_corrs(val_pred, target_geo[val_idx])
    mean_pearson = float(sum(p for p in pearson if p == p) / max(1, sum(1 for p in pearson if p == p)))
    mean_spearman = float(sum(s for s in spearman if s == s) / max(1, sum(1 for s in spearman if s == s)))

    return AdapterStats(
        initial_train_mse=initial_tr, final_train_mse=final_tr,
        initial_val_mse=initial_val, final_val_mse=final_val,
        per_feature_val_pearson=pearson,
        per_feature_val_spearman=spearman,
        mean_val_pearson=mean_pearson,
        mean_val_spearman=mean_spearman,
        n_train=n_train, n_val=int(val_idx.numel()), steps=steps,
    )


def adapter_config(adapter: ObjectFileGeometryAdapter) -> dict:
    return {
        "slot_dim": adapter.slot_dim,
        "goal_dim": adapter.goal_dim,
        "out_dim": adapter.out_dim,
        "hidden": adapter.hidden,
        "n_hidden": adapter.n_hidden,
    }


def build_adapter_from_config(cfg: dict) -> ObjectFileGeometryAdapter:
    return ObjectFileGeometryAdapter(
        slot_dim=int(cfg["slot_dim"]),
        goal_dim=int(cfg.get("goal_dim", 2)),
        out_dim=int(cfg.get("out_dim", 10)),
        hidden=int(cfg["hidden"]),
        n_hidden=int(cfg.get("n_hidden", 3)),
    )


# ---------- Phase 18λ-v2: end-to-end adapter + value head ----------
class End2EndAdapterValue(nn.Module):
    """Joint adapter + value head trained on episode_imp directly.

    The 10-dim "latent" is free — not constrained to match engineered
    geometry. Loss is MSE on episode improvement; adapter and value
    head are trained jointly.
    """

    def __init__(
        self,
        slot_dim: int,
        goal_dim: int = 2,
        latent_dim: int = 10,
        action_dim: int = 7,
        plan_horizon: int = 10,
        adapter_hidden: int = 256,
        adapter_n_hidden: int = 3,
        value_hidden: int = 256,
        value_n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Import here to avoid circular at module top
        from system1_jepa.value_head import GoalProgressValueHead
        self.adapter = ObjectFileGeometryAdapter(
            slot_dim=slot_dim, goal_dim=goal_dim, out_dim=latent_dim,
            hidden=adapter_hidden, n_hidden=adapter_n_hidden,
            dropout=dropout,
        )
        self.value_head = GoalProgressValueHead(
            state_dim=latent_dim, action_dim=action_dim,
            plan_horizon=plan_horizon, goal_dim=goal_dim,
            hidden=value_hidden, n_hidden=value_n_hidden, dropout=dropout,
        )

    def forward(self, slot, goal, actions):
        latent = self.adapter(slot, goal)
        return self.value_head(latent, goal, actions)


@dataclass
class End2EndStats:
    initial_loss: float
    final_loss: float
    initial_val_loss: float
    final_val_loss: float
    n_train: int
    n_val: int
    steps: int


def train_end2end_supervised(
    model: End2EndAdapterValue,
    slots: torch.Tensor,
    goals: torch.Tensor,
    plans: torch.Tensor,
    labels: torch.Tensor,
    *,
    val_split: float = 0.2,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 0,
) -> End2EndStats:
    """Joint MSE training of adapter+value head on episode_imp."""
    n = int(len(slots))
    if n == 0:
        raise ValueError("empty end2end dataset")
    device = next(model.parameters()).device
    slots = slots.float().to(device)
    goals = goals.float().to(device)
    plans = plans.float().to(device)
    labels = labels.float().to(device)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_split * n))
    val_idx = perm[:n_val].to(device)
    tr_idx = perm[n_val:].to(device)
    n_train = int(tr_idx.numel())

    def mse_on(idx):
        with torch.no_grad():
            pred = model(slots[idx], goals[idx], plans[idx])
            return float(((pred - labels[idx]) ** 2).mean())

    initial_tr = mse_on(tr_idx)
    initial_val = mse_on(val_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    final_tr = initial_tr
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n_train, (min(batch_size, n_train),),
                                    generator=gen, device=device)]
        pred = model(slots[sel], goals[sel], plans[sel])
        loss = ((pred - labels[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        final_tr = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"  end2end step {step+1}/{steps} loss={final_tr:.5f}",
                   flush=True)

    final_tr = mse_on(tr_idx)
    final_val = mse_on(val_idx)
    return End2EndStats(
        initial_loss=initial_tr, final_loss=final_tr,
        initial_val_loss=initial_val, final_val_loss=final_val,
        n_train=n_train, n_val=int(val_idx.numel()), steps=steps,
    )


def end2end_config(model: End2EndAdapterValue) -> dict:
    return {
        "slot_dim": model.adapter.slot_dim,
        "goal_dim": model.adapter.goal_dim,
        "latent_dim": model.adapter.out_dim,
        "action_dim": model.value_head.action_dim,
        "plan_horizon": model.value_head.plan_horizon,
        "adapter_hidden": model.adapter.hidden,
        "adapter_n_hidden": model.adapter.n_hidden,
        "value_hidden": model.value_head.hidden,
        "value_n_hidden": model.value_head.n_hidden,
    }


@dataclass
class ScheduledAuxStats:
    initial_value_loss: float
    final_value_loss: float
    initial_val_value_loss: float
    final_val_value_loss: float
    initial_geo_mse: float
    final_geo_mse: float
    n_train: int
    n_val: int
    total_steps: int
    schedule_kind: str


def train_end2end_with_aux_schedule(
    model,                                  # End2EndAdapterValue
    slots: torch.Tensor,
    goals: torch.Tensor,
    plans: torch.Tensor,
    labels: torch.Tensor,
    target_geo: torch.Tensor,
    *,
    schedule_kind: str = "linear_anneal",   # "linear_anneal" or "pretrain_ft"
    steps: int = 2000,
    pretrain_steps: int = 1000,
    aux_weight_init: float = 1.0,
    aux_weight_final: float = 0.0,
    aux_residual_weight: float = 0.05,
    val_split: float = 0.2,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 0,
) -> ScheduledAuxStats:
    """Train End2EndAdapterValue with a scheduled geo-aux loss.

    Total loss at each step = λ_geo(t) * MSE(adapter_out, target_geo)
                            + λ_value(t) * MSE(model_out, labels)

    Schedules:
      linear_anneal: λ_geo linearly 1.0 → aux_weight_final over `steps`.
                     λ_value = 1 - λ_geo, but always at least 0.1 so
                     the value head also trains throughout.
      pretrain_ft:   λ_geo = 1, λ_value = 0 for first `pretrain_steps`
                     steps (adapter-only geo MSE), then λ_geo =
                     aux_residual_weight, λ_value = 1.0 for the rest
                     (joint fine-tuning with small geo regularizer).
    """
    if schedule_kind not in {"linear_anneal", "pretrain_ft"}:
        raise ValueError(f"unknown schedule_kind: {schedule_kind}")
    n = int(len(slots))
    device = next(model.parameters()).device
    slots = slots.float().to(device)
    goals = goals.float().to(device)
    plans = plans.float().to(device)
    labels = labels.float().to(device)
    target_geo = target_geo.float().to(device)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_split * n))
    val_idx = perm[:n_val].to(device)
    tr_idx = perm[n_val:].to(device)
    n_train = int(tr_idx.numel())

    def metrics_on(idx):
        with torch.no_grad():
            pred = model(slots[idx], goals[idx], plans[idx])
            value_loss = float(((pred - labels[idx]) ** 2).mean())
            adapter_out = model.adapter(slots[idx], goals[idx])
            geo_mse = float(((adapter_out - target_geo[idx]) ** 2).mean())
            return value_loss, geo_mse

    initial_val_val, initial_geo = metrics_on(val_idx)
    initial_tr_val, _ = metrics_on(tr_idx)

    def lambdas_at(step):
        if schedule_kind == "linear_anneal":
            frac = step / max(1, steps - 1)
            lam_g = aux_weight_init + (aux_weight_final - aux_weight_init) * frac
            lam_v = max(0.1, 1.0 - lam_g)
            return float(lam_g), float(lam_v)
        # pretrain_ft
        if step < pretrain_steps:
            return 1.0, 0.0
        return float(aux_residual_weight), 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    final_tr_val = initial_tr_val
    for step in range(steps):
        lam_g, lam_v = lambdas_at(step)
        sel = tr_idx[torch.randint(0, n_train, (min(batch_size, n_train),),
                                    generator=gen, device=device)]
        adapter_out = model.adapter(slots[sel], goals[sel])
        pred = model.value_head(adapter_out, goals[sel], plans[sel])
        geo_loss = ((adapter_out - target_geo[sel]) ** 2).mean()
        value_loss = ((pred - labels[sel]) ** 2).mean()
        loss = lam_g * geo_loss + lam_v * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        final_tr_val = float(value_loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"  sched step {step+1}/{steps} "
                  f"lam_g={lam_g:.2f} lam_v={lam_v:.2f} "
                  f"value={final_tr_val:.5f}", flush=True)

    final_val_val, final_geo = metrics_on(val_idx)
    final_tr_val, _ = metrics_on(tr_idx)
    return ScheduledAuxStats(
        initial_value_loss=initial_tr_val, final_value_loss=final_tr_val,
        initial_val_value_loss=initial_val_val,
        final_val_value_loss=final_val_val,
        initial_geo_mse=initial_geo, final_geo_mse=final_geo,
        n_train=n_train, n_val=int(val_idx.numel()),
        total_steps=steps, schedule_kind=schedule_kind,
    )


def build_end2end_from_config(cfg: dict) -> End2EndAdapterValue:
    return End2EndAdapterValue(
        slot_dim=int(cfg["slot_dim"]),
        goal_dim=int(cfg.get("goal_dim", 2)),
        latent_dim=int(cfg.get("latent_dim", 10)),
        action_dim=int(cfg.get("action_dim", 7)),
        plan_horizon=int(cfg.get("plan_horizon", 10)),
        adapter_hidden=int(cfg["adapter_hidden"]),
        adapter_n_hidden=int(cfg.get("adapter_n_hidden", 3)),
        value_hidden=int(cfg["value_hidden"]),
        value_n_hidden=int(cfg.get("value_n_hidden", 3)),
    )
