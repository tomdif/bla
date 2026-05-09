from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .sigreg import sigreg_epps_pulley, sigreg_lewm


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1]).float()


def collapse_regularizer(
    z_predicted: torch.Tensor,
    z_target: torch.Tensor,
    variant: str = "epps_pulley",
) -> torch.Tensor:
    """Anti-collapse regularizer (SIGReg) applied to predicted and target features.

    `variant` is "epps_pulley" (16 random directions, cheap) or "lewm"
    (1024 directions, stronger empirical signal). Both are Cramér-Wold-correct.
    """

    p = _flatten_tokens(z_predicted)
    t = _flatten_tokens(z_target)
    if variant == "lewm":
        return sigreg_lewm(p) + sigreg_lewm(t)
    if variant == "epps_pulley":
        return sigreg_epps_pulley(p) + sigreg_epps_pulley(t)
    raise ValueError(f"unknown SIGReg variant: {variant}")


def jepa_loss(
    predicted_target: torch.Tensor,
    target_latent: torch.Tensor,
    sigreg_weight: float = 1.0,
    sigreg_variant: str = "epps_pulley",
) -> Dict[str, torch.Tensor]:
    """JEPA prediction loss + SIGReg collapse regularizer.

    predicted_target / target_latent are both [B, K_tgt, D] -- features at the
    same target patch positions.
    """

    target = target_latent.detach()
    prediction = F.smooth_l1_loss(predicted_target.float(), target.float())
    regularizer = collapse_regularizer(predicted_target, target, variant=sigreg_variant)
    total = prediction + sigreg_weight * regularizer
    return {
        "loss": total,
        "prediction": prediction.detach(),
        "sigreg": regularizer.detach(),
    }
