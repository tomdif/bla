"""Goal-progress / multi-step value head for BLA System-1 planning.

Phase 18γ established that the existing one-step OF-JEPA predictor
scores `(state, plan) → immediate cube motion` but cannot tell
whether a local action sequence COMPOUNDS toward the goal over
the remaining MPC trajectory. Phase 18η adds that capability as a
value head that regresses to full-episode realized improvement.

Module is intentionally robosuite-free for unit testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class GoalProgressValueHead(nn.Module):
    """Maps (state, goal, action_sequence) to predicted episode improvement.

    Input state can be either:
      - OF-JEPA slot features flattened   (n_slots * slot_dim)
      - 10-dim geometric features         (Phase 18β BC features fallback)

    The module is agnostic; caller flattens before passing in.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        plan_horizon: int = 10,
        goal_dim: int = 2,
        hidden: int = 256,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.plan_horizon = int(plan_horizon)
        self.goal_dim = int(goal_dim)
        self.hidden = int(hidden)
        self.n_hidden = int(n_hidden)
        in_dim = self.state_dim + self.goal_dim + self.plan_horizon * self.action_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """state: [B, state_dim]; goal: [B, goal_dim]; actions: [B, H, A]
        Returns: [B] predicted full-episode improvement."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"expected actions [B, H, A] got {tuple(actions.shape)}")
        if actions.shape[1] != self.plan_horizon or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} != "
                f"[B, {self.plan_horizon}, {self.action_dim}]"
            )
        flat = torch.cat([state.float(), goal.float(),
                            actions.float().flatten(start_dim=1)], dim=-1)
        return self.net(flat).squeeze(-1)


@dataclass
class ValueHeadStats:
    initial_loss: float
    final_loss: float
    initial_val_loss: float
    final_val_loss: float
    n_train: int
    n_val: int
    steps: int


def train_value_head_supervised(
    head: GoalProgressValueHead,
    states: torch.Tensor,
    goals: torch.Tensor,
    actions: torch.Tensor,
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
) -> ValueHeadStats:
    """Regress `head(state, goal, actions) -> label_episode_imp` via MSE.

    Holds out `val_split` fraction of examples at random for diagnostic
    val loss. No early stopping; reports initial/final on both splits.
    """
    if not (len(states) == len(goals) == len(actions) == len(labels)):
        raise ValueError("states/goals/actions/labels must align")
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    n = len(states)
    if n == 0:
        raise ValueError("empty value-head dataset")

    device = next(head.parameters()).device
    states = states.float().to(device)
    goals = goals.float().to(device)
    actions = actions.float().to(device)
    labels = labels.float().to(device)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_split * n))
    val_idx = perm[:n_val].to(device)
    tr_idx = perm[n_val:].to(device)
    n_train = int(tr_idx.numel())

    def loss_on(idx):
        with torch.no_grad():
            pred = head(states[idx], goals[idx], actions[idx])
            return float(((pred - labels[idx]) ** 2).mean())

    initial_tr = loss_on(tr_idx)
    initial_val = loss_on(val_idx)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    final_tr = initial_tr
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n_train, (min(batch_size, n_train),),
                                    generator=gen, device=device)]
        pred = head(states[sel], goals[sel], actions[sel])
        loss = ((pred - labels[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()
        final_tr = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"  vhead step {step+1}/{steps} train_loss={final_tr:.5f}",
                   flush=True)

    final_tr = loss_on(tr_idx)
    final_val = loss_on(val_idx)
    return ValueHeadStats(
        initial_loss=initial_tr, final_loss=final_tr,
        initial_val_loss=initial_val, final_val_loss=final_val,
        n_train=n_train, n_val=int(val_idx.numel()), steps=steps,
    )


def value_head_config(head: GoalProgressValueHead) -> dict:
    return {
        "state_dim": head.state_dim,
        "action_dim": head.action_dim,
        "plan_horizon": head.plan_horizon,
        "goal_dim": head.goal_dim,
        "hidden": head.hidden,
        "n_hidden": head.n_hidden,
    }


def build_value_head_from_config(cfg: dict) -> GoalProgressValueHead:
    return GoalProgressValueHead(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg["action_dim"]),
        plan_horizon=int(cfg["plan_horizon"]),
        goal_dim=int(cfg.get("goal_dim", 2)),
        hidden=int(cfg["hidden"]),
        n_hidden=int(cfg.get("n_hidden", 3)),
    )


def normalize_score(scores):
    """z-normalize a 1-D tensor or numpy array; safe on constant input."""
    import numpy as np
    arr = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores
    arr = arr.astype(float)
    s = arr.std()
    if s < 1e-9:
        return arr - arr.mean()
    return (arr - arr.mean()) / s


def combine_scores(predictor_scores, value_head_scores, mode: str = "sum", lam: float = 0.5):
    """Combine two per-candidate score arrays after z-normalization."""
    import numpy as np
    p = normalize_score(predictor_scores)
    v = normalize_score(value_head_scores)
    if mode == "sum":
        return lam * p + (1.0 - lam) * v
    if mode == "max":
        return np.maximum(p, v)
    if mode == "value_only":
        return v
    if mode == "predictor_only":
        return p
    raise ValueError(f"unknown combine mode {mode}")
