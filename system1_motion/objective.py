"""Loss assembly for the System-1 motion substrate (system1_motion_spec.md §3).

Factored so each term is independently togglable for the ablation matrix:
  L_pred  : action-conditioned next-state prediction (V-JEPA 2-AC)   [PRIMARY, standard]
  L_var   : per-dimension VICReg variance hinge (anti-collapse)      [Aux 4, evidence-driven]
Decode-aux (position) is computed in train.py with a stop-grad and a SEPARATE
optimizer — it never enters the substrate loss (diagnostic only, Aux 2).

Aux 3 (multi-scale temporal contrast) is CUT in spec v2: Korchinski, Favero &
Wyart (2026, arXiv:2605.27734) show a single-level latent-prediction objective is
already implicitly multi-scale on hierarchically-structured data, so the explicit
term is redundant rather than insurance. Not implemented here by design.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RunningSigma:
    """Running per-dim std of target embeddings, for normalized prediction MSE."""

    def __init__(self, momentum=0.99, eps=1e-4):
        self.m = momentum; self.eps = eps; self.var = None

    @torch.no_grad()
    def update(self, z):
        v = z.float().var(dim=0, unbiased=False)
        self.var = v if self.var is None else self.m * self.var + (1 - self.m) * v

    def sigma2(self, d_z, device):
        if self.var is None:
            return torch.ones(d_z, device=device)
        return self.var.to(device) + self.eps


def normalized_mse(pred, target, sigma2):
    """||pred - sg(target)||^2 / (d_z * sigma_z^2), per spec §3."""
    return (((pred - target.detach()) ** 2) / sigma2).mean()


def variance_hinge(z, gamma=1.0):
    """Per-dim std floor: mean_d max(0, gamma - std(z_d)). Anti-collapse (Aux 4)."""
    std = torch.sqrt(z.float().var(dim=0, unbiased=False) + 1e-6)
    return torch.clamp(gamma - std, min=0.0).mean()


def substrate_loss(z_pred, z_tgt, z_t, sigma2, *, beta_var=1.0):
    """Assemble the substrate loss (A1): prediction + variance hinge. Returns
    (total, parts-dict)."""
    L_pred = normalized_mse(z_pred, z_tgt, sigma2)
    L_var = variance_hinge(z_t)
    total = L_pred + beta_var * L_var
    return total, {"pred": float(L_pred.detach()), "var": float(L_var.detach())}
