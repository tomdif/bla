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


import math


class RunningStandardizer:
    """EMA per-dim mean/var for standardizing the flattened action window before
    E_a (so trajectory-level norm variation doesn't dominate the embedding)."""

    def __init__(self, momentum=0.99, eps=1e-5):
        self.m = momentum; self.eps = eps; self.mean = None; self.var = None

    @torch.no_grad()
    def update(self, x):
        mu = x.float().mean(0); v = x.float().var(0, unbiased=False)
        self.mean = mu if self.mean is None else self.m * self.mean + (1 - self.m) * mu
        self.var = v if self.var is None else self.m * self.var + (1 - self.m) * v

    def __call__(self, x):
        if self.mean is None:
            return x
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.eps)


def inverse_dynamics_loss(q_psi, h_t, z_future_sg, u_target_sg):
    """L_inv = MSE(q_psi(h_t, sg[z_future]), sg[E_a(a)]) / d_u. Grad -> q_psi and
    (via live h_t) f_theta; NOT E_a (target sg'd) nor the future path (sg'd)."""
    return ((q_psi(h_t, z_future_sg) - u_target_sg) ** 2).mean()


def prior_grounding_loss(prior, h_t, u_target_sg):
    """L_prior = Gaussian NLL of sg[E_a(a)] under p_rho(h_t)=N(mu,sigma). Grad ->
    p_rho and (via live h_t) f_theta; NOT E_a."""
    mu, log_sigma = prior(h_t)
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * (((u_target_sg - mu) ** 2) * inv_var + 2.0 * log_sigma + math.log(2.0 * math.pi))
    return nll.mean()


def substrate_loss(z_pred, z_tgt, z_t, sigma2, *, beta_var=1.0):
    """Assemble the substrate loss (A1): prediction + variance hinge. Returns
    (total, parts-dict)."""
    L_pred = normalized_mse(z_pred, z_tgt, sigma2)
    L_var = variance_hinge(z_t)
    total = L_pred + beta_var * L_var
    return total, {"pred": float(L_pred.detach()), "var": float(L_var.detach())}
