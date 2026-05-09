from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"expected [batch, tokens, dim], got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1]).float()


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("off-diagonal extraction requires a square matrix")
    return x.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def vicreg_loss(
    z_context: torch.Tensor,
    z_target: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg regularizer over JEPA token latents.

    The target branch is detached by the caller in normal JEPA training.
    Computation is promoted to float32 for stable covariance estimates.
    """

    x = _flatten_tokens(z_context)
    y = _flatten_tokens(z_target)
    if x.shape != y.shape:
        raise ValueError(f"VICReg shape mismatch: {tuple(x.shape)} != {tuple(y.shape)}")

    invariance = F.mse_loss(x, y)

    std_x = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    std_y = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
    variance = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    denom = max(x.shape[0] - 1, 1)
    cov_x = (x.T @ x) / denom
    cov_y = (y.T @ y) / denom
    covariance = (_off_diagonal(cov_x).pow(2).sum() + _off_diagonal(cov_y).pow(2).sum()) / x.shape[1]

    return (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )


def jepa_loss(
    predicted_target: torch.Tensor,
    target_latent: torch.Tensor,
    context_latent: torch.Tensor,
    vicreg_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    target = target_latent.detach()
    prediction = F.smooth_l1_loss(predicted_target.float(), target.float())
    regularizer = vicreg_loss(context_latent, target)
    total = prediction + vicreg_weight * regularizer
    return {
        "loss": total,
        "prediction": prediction.detach(),
        "vicreg": regularizer.detach(),
    }

