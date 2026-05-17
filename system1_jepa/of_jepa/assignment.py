"""Differentiable Sinkhorn assignment for OF-JEPA's memory-anchored matching.

The Sinkhorn iteration normalizes a [B, N, M] score matrix toward a
doubly-stochastic matrix in log-space. This is how OF-JEPA's persistent
memory cells (N files) get soft-bound to per-frame proposals (M tokens),
preserving gradient flow.
"""
from __future__ import annotations

import torch


def sinkhorn(log_scores: torch.Tensor, n_iters: int = 20,
              eps: float = 1e-9) -> torch.Tensor:
    """Differentiable Sinkhorn normalization to a doubly-stochastic matrix.

    log_scores: [B, N, M] log-domain similarity scores.
    Returns:    [B, N, M] non-negative entries summing to 1 along both
                rows and columns (approximately).
    """
    log_p = log_scores
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()
