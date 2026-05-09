from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=next(self.parameters()).dtype))


class TokenlessLatentBus(nn.Module):
    def __init__(
        self,
        d_jepa: int = 1024,
        d_core: int = 4096,
        hidden_dim: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.up = ProjectionMLP(d_jepa, d_core, hidden_dim)
        self.down = ProjectionMLP(d_core, d_jepa, hidden_dim)
        self.to(dtype=dtype)

    def forward_up(self, z_context: torch.Tensor) -> torch.Tensor:
        return self.up(z_context)

    def forward_down(self, z_core: torch.Tensor) -> torch.Tensor:
        return self.down(z_core)


def pool_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.mean(dim=1)
    if x.ndim == 2:
        return x
    raise ValueError(f"expected [batch, tokens, dim] or [batch, dim], got {tuple(x.shape)}")


def contrastive_infonce(
    jepa_core: torch.Tensor,
    dca_core: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric alignment objective implemented without a CE loss module."""

    a = F.normalize(pool_tokens(jepa_core).float(), dim=-1)
    b = F.normalize(pool_tokens(dca_core).float(), dim=-1)
    logits = (a @ b.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    log_prob_ab = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob_ba = logits.T - torch.logsumexp(logits.T, dim=1, keepdim=True)
    loss_ab = -log_prob_ab[torch.arange(labels.numel(), device=labels.device), labels].mean()
    loss_ba = -log_prob_ba[torch.arange(labels.numel(), device=labels.device), labels].mean()
    return 0.5 * (loss_ab + loss_ba)
