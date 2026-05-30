"""Goal-Conditioned Inverse Dynamics Model (GC-IDM) for OF-JEPA planning.

Amortizes test-time planning into a single forward pass, following
"Latent Geometry Beyond Search: Amortizing Planning in World Models"
(Nguyen, Xu & Huang, 2026). That paper shows that on a SIGReg-regularized
LeWM latent, a ~1.5M-param goal-conditioned inverse map `(z_t, z_g, h) -> a_t`
matches or beats CEM in 7/8 cells at 100-130x lower per-decision cost. The
enabling precondition is a smooth, action-sensitive latent geometry — which
`system1_jepa` already targets via SIGReg.

This is the zero-search endpoint of the BLA Phase-18 finding that less search
beats more search around good priors (`search_budget_zero_around_expert_demos`,
`less_search_when_score_anticorrelated`). Where `PlanProposalPolicy` proposes an
open-loop sequence (often as CEM's init mean) and `GoalProgressValueHead` SCORES
candidates, GC-IDM PROPOSES the next action directly and is replanned every step.

Two OF-JEPA-specific notes vs the paper's global-latent setting:
  * Object-file latents are identity-aligned across frames (persistent
    `id_proto`), so `z_t` and `z_g` are slot-aligned and a PER-FILE inverse map
    is well-defined — `mode="perfile"` exploits this and pools across files by
    Sinkhorn match confidence. The paper's plain MLP is `mode="flatten"`.
  * Contact-rich tasks (the BLA Robosuite suite) are exactly the regime where
    the paper reports inverse recovery WANES (Push-T). Treat this as a test with
    a predicted-mixed outcome, not a drop-in win. See docs/GC_IDM_OFJEPA.md.

Intentionally robosuite-free for unit testing (operates on tensors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_embed(values: torch.Tensor, dim: int = 64, max_period: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of a [B] scalar (remaining horizon) -> [B, dim]."""
    v = values.float().reshape(-1, 1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=v.device).float() / max(half, 1)
    )
    args = v * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=v.device)], -1)
    return emb


class _AdaLNZero(nn.Module):
    """AdaLN-Zero (Peebles & Xie 2023): h -> h*(1+gamma(c)) + beta(c), with
    gamma/beta zero-initialized so the horizon signal enters only as the loss
    demands it (matches the paper's modulation)."""

    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class GCInverseDynamics(nn.Module):
    """`(state, goal, remaining_horizon) -> next action`, single forward pass.

    mode="flatten" (paper-faithful):
        forward(state[B,Ds], goal[B,Dg], horizon[B]) -> action[B,A]
        caller flattens OF-JEPA slots (recommend `ObjectFileBatch.full_slot`
        flattened: N_files*(id_dim+state_dim)); goal is the goal frame's latent.

    mode="perfile" (OF-JEPA-native, exploits identity alignment):
        forward(state[B,N,d], goal[B,N,d], horizon[B], conf[B,N]) -> action[B,A]
        a shared MLP processes each identity-aligned file pair, then pools across
        files weighted by Sinkhorn match confidence.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: Optional[int] = None,
        action_dim: int = 7,
        hidden: int = 512,
        n_layers: int = 3,
        horizon_embed_dim: int = 64,
        dropout: float = 0.1,
        mode: str = "flatten",
        tanh_output: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim if goal_dim is not None else state_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.tanh_output = bool(tanh_output)

        in_dim = self.state_dim + self.goal_dim  # concat(z_t, z_g) per the paper
        self.in_proj = nn.Linear(in_dim, hidden)
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden),
                "mlp": nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden, hidden),
                ),
            }))
        self.blocks = nn.ModuleList(blocks)

        # horizon -> conditioning vector c (sinusoidal -> 2-layer MLP)
        self.horizon_embed_dim = int(horizon_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(horizon_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.adaln = _AdaLNZero(hidden, hidden)  # applied before the action head
        self.action_head = nn.Linear(hidden, action_dim)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = h + b["mlp"](b["norm"](h))
        return h

    def forward(self, state, goal, horizon, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "perfile":
            return self._forward_perfile(state, goal, horizon, conf)
        return self._forward_flatten(state, goal, horizon)

    def _cond(self, horizon, batch) -> torch.Tensor:
        if not torch.is_tensor(horizon):
            horizon = torch.tensor(horizon)
        horizon = horizon.reshape(-1).float().to(self.in_proj.weight.device)
        if horizon.numel() == 1 and batch > 1:
            horizon = horizon.expand(batch)
        return self.cond_mlp(sinusoidal_embed(horizon, self.horizon_embed_dim))

    def _forward_flatten(self, state, goal, horizon) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        x = torch.cat([state.float(), goal.float()], dim=-1)
        h = self._backbone(x)
        c = self._cond(horizon, x.shape[0])
        a = self.action_head(self.adaln(h, c))
        return torch.tanh(a) if self.tanh_output else a

    def _forward_perfile(self, state, goal, horizon, conf) -> torch.Tensor:
        # state/goal: [B, N, d] identity-aligned; conf: [B, N] (match confidence)
        assert state.ndim == 3 and goal.ndim == 3, "perfile expects [B, N, d]"
        B, N, _ = state.shape
        x = torch.cat([state.float(), goal.float()], dim=-1)          # [B,N,2d]
        h = self._backbone(x)                                          # [B,N,hidden]
        c = self._cond(horizon, B).unsqueeze(1).expand(B, N, self.hidden)
        h = self.adaln(h, c)                                           # [B,N,hidden]
        if conf is None:
            w = torch.full((B, N), 1.0 / N, device=h.device)
        else:
            w = torch.softmax(conf.float().to(h.device), dim=-1)      # confidence pool
        pooled = (w.unsqueeze(-1) * h).sum(dim=1)                      # [B,hidden]
        a = self.action_head(pooled)
        return torch.tanh(a) if self.tanh_output else a


@dataclass
class GCIDMStats:
    initial_loss: float
    final_loss: float
    initial_val_loss: float
    final_val_loss: float
    n_train: int
    n_val: int
    steps: int


def train_gc_idm_supervised(
    head: GCInverseDynamics,
    states: torch.Tensor,
    goals: torch.Tensor,
    horizons: torch.Tensor,
    actions: torch.Tensor,
    confs: Optional[torch.Tensor] = None,
    *,
    val_split: float = 0.2,
    steps: int = 3000,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 0,
) -> GCIDMStats:
    """Offline regression of `head(z_t, z_g, h) -> a_t` on demo tuples.

    Build tuples by sampling (t, h) along each demo trajectory with
    h ~ U[1, H_max], state=z_t, goal=z_{t+h}, action=a_t (the SAME action labels
    that trained the world model). No environment interaction.
    """
    n = len(states)
    if not (n == len(goals) == len(horizons) == len(actions)):
        raise ValueError("states/goals/horizons/actions must align")
    if n == 0:
        raise ValueError("empty GC-IDM dataset")
    device = next(head.parameters()).device
    states, goals = states.float().to(device), goals.float().to(device)
    horizons, actions = horizons.float().to(device), actions.float().to(device)
    if confs is not None:
        confs = confs.float().to(device)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_split * n))
    val_idx, tr_idx = perm[:n_val].to(device), perm[n_val:].to(device)
    n_train = int(tr_idx.numel())

    def loss_on(idx):
        with torch.no_grad():
            c = confs[idx] if confs is not None else None
            pred = head(states[idx], goals[idx], horizons[idx], c)
            return float(((pred - actions[idx]) ** 2).mean())

    initial_tr, initial_val = loss_on(tr_idx), loss_on(val_idx)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n_train, (min(batch_size, n_train),),
                                   generator=gen, device=device)]
        c = confs[sel] if confs is not None else None
        pred = head(states[sel], goals[sel], horizons[sel], c)
        loss = ((pred - actions[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip); opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  gc-idm step {step+1}/{steps} train_loss={float(loss):.5f}", flush=True)
    final_tr, final_val = loss_on(tr_idx), loss_on(val_idx)
    return GCIDMStats(initial_tr, final_tr, initial_val, final_val,
                      n_train, int(val_idx.numel()), steps)


@torch.no_grad()
def act(head: GCInverseDynamics, state, goal, remaining_horizon, conf=None) -> torch.Tensor:
    """Closed-loop control law (one step). Re-encode the observation each step,
    set remaining_horizon = T - t + 1, call this; no rollout, no search."""
    head.eval()
    a = head(state, goal, remaining_horizon, conf)
    return a.squeeze(0) if a.shape[0] == 1 else a


def gc_idm_config(head: GCInverseDynamics) -> dict:
    return {"state_dim": head.state_dim, "goal_dim": head.goal_dim,
            "action_dim": head.action_dim, "hidden": head.hidden,
            "n_layers": head.n_layers, "horizon_embed_dim": head.horizon_embed_dim,
            "mode": head.mode, "tanh_output": head.tanh_output}


def build_gc_idm_from_config(cfg: dict) -> GCInverseDynamics:
    return GCInverseDynamics(
        state_dim=int(cfg["state_dim"]), goal_dim=int(cfg["goal_dim"]),
        action_dim=int(cfg["action_dim"]), hidden=int(cfg["hidden"]),
        n_layers=int(cfg.get("n_layers", 3)),
        horizon_embed_dim=int(cfg.get("horizon_embed_dim", 64)),
        mode=cfg.get("mode", "flatten"), tanh_output=bool(cfg.get("tanh_output", True)))
