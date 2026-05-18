"""Learned action-sequence proposal policies for OF-JEPA planning.

This module is intentionally robosuite-free. The heavy environment loop
lives in `scripts/phase18b_policy_distill.py`; the policy and supervised
distillation utilities live here so they can be unit-tested locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanProposalPolicy(nn.Module):
    """Maps compact state features to a horizon-length action proposal.

    Input features are the Phase 16 geometric features:
    cube xy, end-effector xy/z, cube z, goal xy, push direction.
    The output is a clipped/tanh action sequence used either directly
    or as CEM's initial mean.
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden: int = 256,
        plan_horizon: int = 10,
        action_dim: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.plan_horizon = int(plan_horizon)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.plan_horizon * self.action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        flat = torch.tanh(self.net(features.float()))
        return flat.view(features.shape[0], self.plan_horizon, self.action_dim)

    @torch.no_grad()
    def propose(self, features: torch.Tensor) -> torch.Tensor:
        """Inference helper returning a single `[H, A]` plan for one state."""
        return self.forward(features)[0]


def plan_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    step_decay: float = 0.90,
) -> torch.Tensor:
    """MSE over `[B, H, A]` plans with earlier actions weighted more.

    The first few actions matter most in MPC because only `replan_every`
    actions are executed before a fresh observation. A decayed loss makes
    that deployment fact explicit while still training the full proposal.
    """

    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"expected [B, H, A], got {tuple(pred.shape)}")
    horizon = pred.shape[1]
    weights = step_decay ** torch.arange(horizon, device=pred.device, dtype=pred.dtype)
    weights = weights / weights.mean().clamp_min(1e-8)
    per_step = (pred - target).pow(2).mean(dim=-1)
    return (per_step * weights[None, :]).mean()


@dataclass
class DistillStats:
    initial_loss: float
    final_loss: float
    n_examples: int
    steps: int


def train_plan_policy_supervised(
    policy: PlanProposalPolicy,
    features: torch.Tensor,
    target_plans: torch.Tensor,
    *,
    steps: int = 2000,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    step_decay: float = 0.90,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_every: int = 0,
) -> DistillStats:
    """Fit `policy(features) -> target_plans` with weighted plan MSE."""

    if len(features) != len(target_plans):
        raise ValueError("features and target_plans must have the same first dimension")
    if target_plans.ndim != 3:
        raise ValueError(f"target_plans must be [N, H, A], got {tuple(target_plans.shape)}")

    device = next(policy.parameters()).device
    features = features.float().to(device)
    target_plans = target_plans.float().to(device)
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("empty distillation dataset")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    with torch.no_grad():
        initial = float(plan_weighted_mse(policy(features), target_plans, step_decay=step_decay))

    final = initial
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen, device=device)
        pred = policy(features[idx])
        loss = plan_weighted_mse(pred, target_plans[idx], step_decay=step_decay)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()
        final = float(loss.detach())
        if log_every and (step + 1) % log_every == 0:
            print(f"  policy step {step+1}/{steps} loss={final:.5f}", flush=True)

    with torch.no_grad():
        final = float(plan_weighted_mse(policy(features), target_plans, step_decay=step_decay))

    return DistillStats(initial_loss=initial, final_loss=final, n_examples=n, steps=steps)


def policy_config_dict(policy: PlanProposalPolicy) -> dict:
    return {
        "in_dim": policy.in_dim,
        "hidden": policy.hidden,
        "plan_horizon": policy.plan_horizon,
        "action_dim": policy.action_dim,
    }


def build_policy_from_config(config: dict) -> PlanProposalPolicy:
    return PlanProposalPolicy(
        in_dim=int(config["in_dim"]),
        hidden=int(config["hidden"]),
        plan_horizon=int(config["plan_horizon"]),
        action_dim=int(config["action_dim"]),
    )
