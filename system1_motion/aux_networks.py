"""v4 grounding networks (system1_motion_spec v4).

The A1-A3 minimal recipe converged to a low-loss, variance-healthy, position-AND
-action-free equilibrium (sanity#2: r_action=-0.001). v4 adds two grounding terms
whose absence we empirically showed produces that failure:

  E_a  (ActionEmbedder)   : action-window -> u in R^{d_u}. Trains via L_pred ONLY
                            (it feeds the dynamics T_w(z_t, u)); sg[E_a(a)] is the
                            target of both auxiliaries, so they give it no grad.
  q_psi(InverseDynamics)  : (h_t, sg[f_bar(x_future)]) -> u_hat, MSE to sg[E_a(a)].
                            Trains via L_inv only; grad flows into f_theta via h_t.
  p_rho(Prior)            : h_t -> N(mu, sigma) over u; NLL of sg[E_a(a)]. Trains
                            via L_prior only; grad into f_theta via h_t. log_sigma
                            clamped [-5,2] (VAE stability).

Gradient flow is enforced purely by stop-gradients (sg[E_a(a)], EMA-target sg) +
which term each module appears in — a single optimizer over all params is correct.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """E_a: flattened action window [S*da] -> u [d_u]. 2-layer MLP."""

    def __init__(self, in_dim: int, d_u: int = 32, hidden: int = 128):
        super().__init__()
        self.d_u = d_u
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_u))

    def forward(self, a):                       # a: [B, S*da] (standardized)
        return self.net(a)


class InverseDynamics(nn.Module):
    """q_psi: (h_t [d_z], z_future [d_z]) -> u_hat [d_u].

    `hidden` and `linear` control q_psi capacity. A high-capacity q_psi can
    MEMORIZE the action from (z_t, z_future) without the encoder representing it
    usefully, so L_inv is satisfied with no pressure on the encoder. Tightening
    the bottleneck (small hidden, or linear=True) forces the *encoder* to make
    the action easily recoverable -> the grounding pressure actually lands on it.
    Defaults (hidden=256, linear=False) preserve the canonical v4 behavior."""

    def __init__(self, d_z: int, d_u: int = 32, hidden: int = 256, linear: bool = False):
        super().__init__()
        if linear:
            self.net = nn.Linear(2 * d_z, d_u)                      # no nonlinearity, no memorization room
        else:
            self.net = nn.Sequential(nn.Linear(2 * d_z, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_u))

    def forward(self, h_t, z_future):
        return self.net(torch.cat([h_t, z_future], dim=-1))


class Prior(nn.Module):
    """p_rho: h_t [d_z] -> N(mu [d_u], sigma [d_u]); log_sigma clamped for stability."""

    def __init__(self, d_z: int, d_u: int = 32, hidden: int = 256,
                 log_sigma_min: float = -5.0, log_sigma_max: float = 2.0):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_z, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, d_u)
        self.log_sigma = nn.Linear(hidden, d_u)
        self.lo, self.hi = log_sigma_min, log_sigma_max

    def forward(self, h_t):
        z = self.backbone(h_t)
        return self.mu(z), self.log_sigma(z).clamp(self.lo, self.hi)
