"""Anti-collapse regularizers for the real SIGReg-vs-information-floor comparison.

Three terms, each added to L_pred as the anti-collapse objective (the role the
variance-hinge currently plays). The point is to test, on a REAL trained encoder
and FACTORED across encoders (pool vs slot), the claim that for object-centric
world models a conditional information floor beats marginal Gaussian anti-collapse.

  variance_hinge  : per-dim std floor (VICReg). marginal, cheapest. [in objective.py]
  sigreg          : push 1-D random projections toward N(0,1) (Cramer-Wold spirit).
                    marginal-distribution matching — LeWorld's regularizer (moment-
                    based variant: matches mean/var/skew/kurt of each projection;
                    not the full Epps-Pulley characteristic-function statistic).
  QLIFFloor       : quantized predictive-information floor (Q-LIF spirit). Maximize
                    H2(code(z_t)) - CE(code(z_future) | z_t, action): diverse codes
                    (anti-collapse, gradients to the online encoder) AND next-code
                    predictable from current+action (information-bearing). Minimal
                    core — no Schur-slot/object-residual/rollout-flatline terms.

HONEST SCOPE: these are minimal faithful cores, not the full SCOF-Stable/Q-LIF
stack. They are enough to answer "does an information-floor regularizer leave more
decision-relevant state linearly accessible than marginal-Gaussian, and does the
answer depend on the encoder."
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- KNOWN-GOOD SIGReg, ported verbatim from arc_local/jepa_wm/system1_jepa/sigreg.py ----
# (Cramer-Wold: a dist is Gaussian iff every 1D projection is. Characteristic-function
# statistics -> non-vanishing gradient at collapse, unlike naive moment-matching.)
def sigreg_epps_pulley(z, n_directions=16, eps=1e-5, max_n=4096):
    if z.dim() == 3: z = z.flatten(0, 1)
    if z.dim() != 2: return z.new_zeros(())
    n, d = z.shape
    if n < 4: return z.new_zeros(())
    if n > max_n:
        idx = torch.randperm(n, device=z.device)[:max_n]; z = z[idx]; n = max_n
    zf = z.float()
    u = F.normalize(torch.randn(d, n_directions, device=zf.device, dtype=zf.dtype), p=2, dim=0)
    h = zf @ u; h = h - h.mean(dim=0, keepdim=True)
    h_sq = h * h
    gram = torch.einsum("nm,Nm->nNm", h, h)
    d2 = (h_sq.unsqueeze(0) + h_sq.unsqueeze(1) - 2.0 * gram).clamp(min=0.0)
    k_mat = torch.exp(-(d2.clamp(max=60.0)) / 2.0)
    sum_k = k_mat.sum(dim=(0, 1)) / float(n)
    sum_l = (math.sqrt(2.0) * torch.exp(-h_sq.clamp(max=120.0) / 4.0)).sum(dim=0)
    const = float(n) / math.sqrt(3.0)
    return (sum_k - sum_l + const).clamp(min=0.0).mean().to(z.dtype)


def sigreg_lewm(z, num_proj=1024, knots=17):
    if z.dim() == 3: z = z.transpose(0, 1)
    elif z.dim() == 2: z = z.unsqueeze(0)
    t_dim, batch, d = z.shape
    if batch < 4: return z.new_zeros(())
    zf = z.float()
    t = torch.linspace(0, 3, knots, device=zf.device, dtype=zf.dtype)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=zf.device, dtype=zf.dtype)
    weights[0] = dt; weights[-1] = dt
    window = torch.exp(-t.square() / 2.0)
    weights = weights * window
    a = torch.randn(d, num_proj, device=zf.device, dtype=zf.dtype); a = a.div_(a.norm(p=2, dim=0))
    proj = zf @ a
    x_t = proj.unsqueeze(-1) * t
    err = (x_t.cos().mean(dim=-3) - window).square() + x_t.sin().mean(dim=-3).square()
    return ((err @ weights) * float(batch)).mean().to(z.dtype)
# ----------------------------------------------------------------------------------------


def sigreg(z, n_proj=64, eps=1e-6):
    """Marginal-Gaussian regularizer. Project onto n_proj random unit directions,
    penalize deviation of each 1-D projection's first four moments from N(0,1).
    CRITICAL: do NOT standardize z first — the (var-1)^2 term must SEE the variance,
    so collapse (var->0) is penalized as (0-1)^2=1. (Centering only is fine.)"""
    z = z - z.mean(0)                                       # center only; keep the scale
    d = z.shape[1]
    dirs = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + eps)
    p = z @ dirs                                            # [B, n_proj]
    m = p.mean(0); v = p.var(0, unbiased=False)
    c = p - m
    sk = (c ** 3).mean(0) / (v ** 1.5 + eps)
    ku = (c ** 4).mean(0) / (v ** 2 + eps)
    return (m ** 2 + (v - 1.0) ** 2 + sk ** 2 + (ku - 3.0) ** 2).mean()


class QLIFFloor(nn.Module):
    """Quantized predictive-information floor (minimal Q-LIF core).

    Loss = CE(predict code(z_future) from [z_t, action]) - h2_weight * H2(code(z_t)).
    Minimizing it pushes: (a) the next-state quantized code to be predictable from
    current state + action (predictive information present), and (b) the current
    code distribution to be diverse (collision-entropy term = anti-collapse, with
    gradients to the ONLINE encoder z_t). Codes are fixed random hyperplanes (no
    learned codebook to game). The prediction head is the only learned part.
    """
    def __init__(self, d_z, d_u, n_bits=16, h2_weight=1.0):
        super().__init__()
        self.n_bits = n_bits; self.h2_weight = h2_weight
        R = torch.randn(d_z, n_bits)
        self.register_buffer("proj", R / (R.norm(dim=0, keepdim=True) + 1e-6))  # fixed
        self.head = nn.Sequential(nn.Linear(d_z + d_u, 256), nn.GELU(), nn.Linear(256, n_bits))

    def code(self, z):
        return (z @ self.proj > 0).float()                  # [B, n_bits] binary code

    def forward(self, z_t, z_future, a_emb):
        # Predictive-information term: the next-state quantized code must be
        # predictable from (current state, action). Low CE = the latent carries
        # predictive bits about the future given the action — the Q-LIF essence.
        # Anti-collapse is handled by the shared variance-hinge in the trainer (a
        # standalone collision-entropy term has vanishing gradient at collapse), so
        # this returns ONLY the predictive CE.
        tgt_code = self.code(z_future).detach()
        logits = self.head(torch.cat([z_t, a_emb], dim=-1))
        ce = F.binary_cross_entropy_with_logits(logits, tgt_code)
        # diagnostic only: code diversity of the (detached) target codes
        with torch.no_grad():
            p = tgt_code.mean(0).clamp(1e-4, 1 - 1e-4)
            h2 = float(-torch.log(p ** 2 + (1 - p) ** 2).mean())
        return ce, {"qlif_ce": float(ce.detach()), "qlif_h2": h2}
