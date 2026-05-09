"""SIGReg — anti-collapse regularizer ported from cser-jepa-v2.

SIGReg uses the Cramér-Wold theorem: a distribution is Gaussian iff every 1D
projection is Gaussian. Drawing K random unit-norm projections and applying
a univariate normality statistic to each catches collapse modes that
per-axis VICReg misses (oblique low-rank subspaces with healthy per-dim std).

Two variants:
  * `sigreg_epps_pulley` — 16 random directions, Epps-Pulley pairwise kernel
    statistic. Cheap, sufficient for most smoke runs.
  * `sigreg_lewm` — 1024 random directions, moment-fit error against a
    Gaussian window via fixed-knot quadrature. Stronger empirical signal;
    drove LeWorldModel to 98% on PushT with only this + prediction MSE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sigreg_epps_pulley(z: Tensor, n_directions: int = 16, eps: float = 1e-5, max_n: int = 4096) -> Tensor:
    """Epps-Pulley SIGReg with optional subsample cap.

    The kernel matrix is O(N^2 * n_directions) memory; for large batches we
    subsample down to `max_n` rows before computing. The estimator stays
    Cramer-Wold valid since the subsample is uniform.
    """

    if z.dim() == 3:
        z = z.flatten(0, 1)
    if z.dim() != 2:
        return z.new_zeros(())
    n, d = z.shape
    if n < 4:
        return z.new_zeros(())
    if n > max_n:
        idx = torch.randperm(n, device=z.device)[:max_n]
        z = z[idx]
        n = max_n

    zf = z.float()
    u = torch.randn(d, n_directions, device=zf.device, dtype=zf.dtype)
    u = F.normalize(u, p=2, dim=0)
    h = zf @ u
    h = h - h.mean(dim=0, keepdim=True)

    h_sq = h * h
    gram = torch.einsum("nm,Nm->nNm", h, h)
    d2 = (h_sq.unsqueeze(0) + h_sq.unsqueeze(1) - 2.0 * gram).clamp(min=0.0)
    k_mat = torch.exp(-(d2.clamp(max=60.0)) / 2.0)

    sum_k = k_mat.sum(dim=(0, 1)) / float(n)
    sum_l = (math.sqrt(2.0) * torch.exp(-h_sq.clamp(max=120.0) / 4.0)).sum(dim=0)
    const = float(n) / math.sqrt(3.0)

    t_dir = (sum_k - sum_l + const).clamp(min=0.0)
    return t_dir.mean().to(z.dtype)


def sigreg_lewm(z: Tensor, num_proj: int = 1024, knots: int = 17) -> Tensor:
    if z.dim() == 3:
        z = z.transpose(0, 1)
    elif z.dim() == 2:
        z = z.unsqueeze(0)

    t_dim, batch, d = z.shape
    if batch < 4:
        return z.new_zeros(())

    zf = z.float()
    t = torch.linspace(0, 3, knots, device=zf.device, dtype=zf.dtype)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=zf.device, dtype=zf.dtype)
    weights[0] = dt
    weights[-1] = dt
    window = torch.exp(-t.square() / 2.0)
    weights = weights * window

    a = torch.randn(d, num_proj, device=zf.device, dtype=zf.dtype)
    a = a.div_(a.norm(p=2, dim=0))
    proj = zf @ a

    x_t = proj.unsqueeze(-1) * t
    err = (x_t.cos().mean(dim=-3) - window).square() + x_t.sin().mean(dim=-3).square()
    statistic = (err @ weights) * float(batch)
    return statistic.mean().to(z.dtype)
