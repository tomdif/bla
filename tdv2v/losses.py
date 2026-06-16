"""Loss terms for two-view TDV. The anti-collapse default is SIGReg (LeJEPA), per the paper's own
stated next step and the prior finding that it's more stable for prediction objectives."""
import torch, torch.nn.functional as F


def temporal_mse(pred_tokens, target_tokens):
    """ẑ_{t+1} vs stop-grad teacher z_{t+1}, over all tokens (the TDV causal-prediction loss)."""
    return F.mse_loss(pred_tokens, target_tokens.detach())


def crossview_mse(pred_tokens, target_tokens):
    """Predicted other-view tokens vs stop-grad teacher tokens (CroCo-style completion)."""
    return F.mse_loss(pred_tokens, target_tokens.detach())


def commutativity(T_A, V_t1, V_t, T_B):
    """Cycle consistency on the (time × view) grid: motion-then-view = view-then-motion.
    T = temporal latent delta, V = cross-view latent delta. Geometrically exact for a rigid scene."""
    return F.mse_loss(T_A + V_t1, V_t + T_B)


def sigreg(z, n_dirs=128, freqs=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """SIGReg anti-collapse: force random 1-D projections of the embedding batch toward N(0,1)
    (mean 0, var 1, Gaussian characteristic function). Isotropic-Gaussian features ⇒ no collapse.
    `z`: (B, D) embeddings (pool many frames into the batch for a good estimate)."""
    B, D = z.shape
    U = torch.randn(D, n_dirs, device=z.device)
    U = U / (U.norm(dim=0, keepdim=True) + 1e-8)
    P = z @ U                                   # (B, n_dirs)
    mean = P.mean(0); var = P.var(0, unbiased=False)
    loss = mean.pow(2).mean() + (var - 1.0).pow(2).mean()
    Pc = (P - mean) / (var.sqrt() + 1e-4)       # standardize, then test Gaussianity (Epps–Pulley CF)
    for t in freqs:
        c = torch.cos(t * Pc).mean(0); s = torch.sin(t * Pc).mean(0)
        tgt = math_exp(-0.5 * t * t, z.device)
        loss = loss + ((c - tgt) ** 2 + s ** 2).mean()
    return loss


def math_exp(v, device):
    return torch.exp(torch.tensor(v, device=device))


def dino(student_proj, teacher_proj, center, ts=0.1, tt=0.07):
    """Alternative anti-collapse: DINO cross-entropy between student and centered/sharpened teacher."""
    t = F.softmax((teacher_proj.detach() - center) / tt, dim=-1)
    s = F.log_softmax(student_proj / ts, dim=-1)
    return -(t * s).sum(-1).mean()


# ---- diagnostics ----
@torch.no_grad()
def effective_rank(z):
    """Participation-ratio effective rank of the embedding batch (collapse ⇒ ~1)."""
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    s = s / (s.sum() + 1e-9)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))
