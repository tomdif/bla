"""Spectral/carrier priors for frame-level latent rollout.

The Phase 1E temporal predictor learns next-frame pooled JEPA latents directly.
This module adds a small, explicit carrier model over the latent history and
lets the transformer learn the residual. It is intentionally narrow: the prior
is a closed-form extrapolator over [B, T, D] latent histories, so it can be
benchmarked against the existing pipeline without changing the JEPA encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .temporal import TemporalConfig, TemporalPredictor


def carrier_prior(frame_embeds: torch.Tensor, kind: str = "affine") -> torch.Tensor:
    """Predict the next latent by extrapolating the carrier in history.

    frame_embeds: [B, T, D]

    Supported priors:
      - "last": zero-frequency/constant carrier.
      - "affine": least-squares line fit over time, evaluated at T.
      - "quadratic": least-squares quadratic fit, evaluated at T.
    """

    if frame_embeds.ndim != 3:
        raise ValueError(f"expected [B, T, D], got {tuple(frame_embeds.shape)}")
    _, steps, _ = frame_embeds.shape
    if steps < 1:
        raise ValueError("carrier prior needs at least one history frame")
    if kind == "last" or steps == 1:
        return frame_embeds[:, -1]
    if kind == "affine":
        return _affine_carrier_prior(frame_embeds)
    if kind == "quadratic":
        if steps < 3:
            return _affine_carrier_prior(frame_embeds)
        return _poly_carrier_prior(frame_embeds, degree=2)
    raise ValueError(f"unsupported carrier prior: {kind}")


def _affine_carrier_prior(frame_embeds: torch.Tensor) -> torch.Tensor:
    _, steps, _ = frame_embeds.shape
    t = torch.arange(steps, device=frame_embeds.device, dtype=frame_embeds.dtype)
    centered_t = t - t.mean()
    centered_z = frame_embeds - frame_embeds.mean(dim=1, keepdim=True)
    denom = centered_t.square().sum().clamp_min(torch.finfo(frame_embeds.dtype).eps)
    slope = (centered_z * centered_t.view(1, steps, 1)).sum(dim=1) / denom
    intercept = frame_embeds.mean(dim=1) - slope * t.mean()
    return intercept + slope * float(steps)


def _poly_carrier_prior(frame_embeds: torch.Tensor, degree: int) -> torch.Tensor:
    _, steps, _ = frame_embeds.shape
    t = torch.arange(steps, device=frame_embeds.device, dtype=frame_embeds.dtype)
    cols = [t.pow(i) for i in range(degree + 1)]
    design = torch.stack(cols, dim=1)  # [T, degree + 1]
    pred_row = torch.tensor(
        [float(steps) ** i for i in range(degree + 1)],
        device=frame_embeds.device,
        dtype=frame_embeds.dtype,
    )
    coeff = torch.linalg.pinv(design.float()).to(frame_embeds.dtype) @ frame_embeds
    return torch.einsum("p,bpd->bd", pred_row, coeff)


@dataclass
class SpectralTemporalConfig:
    temporal: TemporalConfig
    prior_kind: str = "last"
    residual_scale: float = 0.25
    direct_weight: float = 0.01


class SpectralResidualTemporalPredictor(nn.Module):
    """Temporal predictor that learns residuals around a carrier prior."""

    def __init__(self, cfg: SpectralTemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.residual = TemporalPredictor(cfg.temporal)
        self.log_residual_scale = nn.Parameter(torch.tensor(float(cfg.residual_scale)).log())

    def forward(
        self,
        frame_embeds: torch.Tensor,
        a_chunk: torch.Tensor,
        return_aux: bool = False,
    ):
        prior = carrier_prior(frame_embeds, self.cfg.prior_kind)
        residual_out = self.residual(frame_embeds, a_chunk, return_aux=return_aux)
        scale = self.log_residual_scale.exp()
        if not return_aux:
            return prior + scale * residual_out
        out = dict(residual_out)
        out["prior"] = prior
        out["z"] = prior + scale * out["z"]
        out["residual_scale"] = scale
        return out

    @torch.no_grad()
    def rollout(self, frame_embeds: torch.Tensor, action_seq: torch.Tensor) -> dict:
        if action_seq.ndim == 3:
            action_seq = action_seq.unsqueeze(2)
        horizon = action_seq.shape[1]
        max_ctx = self.residual.frame_pos_embed.size(1)
        ctx = frame_embeds
        zs, priors = [], []
        for h in range(horizon):
            out = self.forward(ctx, action_seq[:, h], return_aux=True)
            zs.append(out["z"])
            priors.append(out["prior"])
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return {
            "z": torch.stack(zs, dim=1),
            "prior": torch.stack(priors, dim=1),
        }


class SpectralBlendTemporalPredictor(nn.Module):
    """Blend a carrier prior with the learned temporal predictor.

    This is the conservative integration point for Phase 1E: the prior can be
    used immediately, while the transformer only takes over if training finds a
    consistently useful correction.
    """

    def __init__(self, cfg: SpectralTemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.direct = TemporalPredictor(cfg.temporal)
        initial = min(max(float(cfg.direct_weight), 1e-6), 1.0 - 1e-6)
        self.direct_logit = nn.Parameter(torch.tensor(initial / (1.0 - initial)).log())

    def forward(
        self,
        frame_embeds: torch.Tensor,
        a_chunk: torch.Tensor,
        return_aux: bool = False,
    ):
        prior = carrier_prior(frame_embeds, self.cfg.prior_kind)
        direct_out = self.direct(frame_embeds, a_chunk, return_aux=return_aux)
        direct_weight = self.direct_logit.sigmoid()
        if not return_aux:
            return torch.lerp(prior, direct_out, direct_weight)
        out = dict(direct_out)
        out["prior"] = prior
        out["z"] = torch.lerp(prior, out["z"], direct_weight)
        out["direct_weight"] = direct_weight
        return out

    @torch.no_grad()
    def rollout(self, frame_embeds: torch.Tensor, action_seq: torch.Tensor) -> dict:
        if action_seq.ndim == 3:
            action_seq = action_seq.unsqueeze(2)
        horizon = action_seq.shape[1]
        max_ctx = self.direct.frame_pos_embed.size(1)
        ctx = frame_embeds
        zs, priors = [], []
        for h in range(horizon):
            out = self.forward(ctx, action_seq[:, h], return_aux=True)
            zs.append(out["z"])
            priors.append(out["prior"])
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return {
            "z": torch.stack(zs, dim=1),
            "prior": torch.stack(priors, dim=1),
        }


class SpectralFeatureTemporalPredictor(nn.Module):
    """Carrier-corrected temporal predictor with residualized input features.

    Unlike output blending, this model does not average a trained predictor with
    the carrier prior. It starts exactly at the prior through a zero-initialized
    correction head, then learns action-dependent residual dynamics from:
      - raw latent history,
      - latent history centered against the next-step carrier prior,
      - first finite differences over latent history.
    """

    def __init__(self, cfg: SpectralTemporalConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.temporal.d
        self.feature_proj = nn.Linear(3 * d, d)
        self.feature_norm = nn.Identity()
        _init_feature_projection_to_raw(self.feature_proj, d)
        self.residual = TemporalPredictor(cfg.temporal)
        self.correction_head = nn.Linear(d, d)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self.log_correction_scale = nn.Parameter(torch.tensor(float(cfg.residual_scale)).log())

    def forward(
        self,
        frame_embeds: torch.Tensor,
        a_chunk: torch.Tensor,
        return_aux: bool = False,
    ):
        features, prior = self._carrier_features(frame_embeds)
        residual_out = self.residual(features, a_chunk, return_aux=return_aux)
        residual_z = residual_out["z"] if return_aux else residual_out
        correction = self.correction_head(residual_z)
        scale = self.log_correction_scale.exp()
        z = prior + scale * correction
        if not return_aux:
            return z
        out = dict(residual_out)
        out["prior"] = prior
        out["correction"] = correction
        out["correction_scale"] = scale
        out["z"] = z
        return out

    @torch.no_grad()
    def rollout(self, frame_embeds: torch.Tensor, action_seq: torch.Tensor) -> dict:
        if action_seq.ndim == 3:
            action_seq = action_seq.unsqueeze(2)
        horizon = action_seq.shape[1]
        max_ctx = self.residual.frame_pos_embed.size(1)
        ctx = frame_embeds
        zs, priors, corrections = [], [], []
        for h in range(horizon):
            out = self.forward(ctx, action_seq[:, h], return_aux=True)
            zs.append(out["z"])
            priors.append(out["prior"])
            corrections.append(out["correction"])
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return {
            "z": torch.stack(zs, dim=1),
            "prior": torch.stack(priors, dim=1),
            "correction": torch.stack(corrections, dim=1),
        }

    def _carrier_features(self, frame_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior = carrier_prior(frame_embeds, self.cfg.prior_kind)
        centered = frame_embeds - prior.unsqueeze(1)
        delta = torch.zeros_like(frame_embeds)
        delta[:, 1:] = frame_embeds[:, 1:] - frame_embeds[:, :-1]
        features = torch.cat([frame_embeds, centered, delta], dim=-1)
        return self.feature_norm(self.feature_proj(features)), prior


class SpectralAugmentedTemporalPredictor(nn.Module):
    """Temporal predictor with carrier/residual diagnostics as input features.

    The predictor still owns the full next-latent prediction. The carrier prior
    is only used to construct additional context features, avoiding the output
    bottleneck that hurts action-dependent dynamics.
    """

    def __init__(self, cfg: SpectralTemporalConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.temporal.d
        self.feature_proj = nn.Linear(3 * d, d)
        self.feature_norm = nn.Identity()
        _init_feature_projection_to_raw(self.feature_proj, d)
        self.predictor = TemporalPredictor(cfg.temporal)

    def forward(
        self,
        frame_embeds: torch.Tensor,
        a_chunk: torch.Tensor,
        return_aux: bool = False,
    ):
        features, prior = self._carrier_features(frame_embeds)
        out = self.predictor(features, a_chunk, return_aux=return_aux)
        if not return_aux:
            return out
        out = dict(out)
        out["prior"] = prior
        return out

    @torch.no_grad()
    def rollout(self, frame_embeds: torch.Tensor, action_seq: torch.Tensor) -> dict:
        if action_seq.ndim == 3:
            action_seq = action_seq.unsqueeze(2)
        horizon = action_seq.shape[1]
        max_ctx = self.predictor.frame_pos_embed.size(1)
        ctx = frame_embeds
        zs, priors = [], []
        for h in range(horizon):
            out = self.forward(ctx, action_seq[:, h], return_aux=True)
            zs.append(out["z"])
            priors.append(out["prior"])
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return {
            "z": torch.stack(zs, dim=1),
            "prior": torch.stack(priors, dim=1),
        }

    def _carrier_features(self, frame_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior = carrier_prior(frame_embeds, self.cfg.prior_kind)
        centered = frame_embeds - prior.unsqueeze(1)
        delta = torch.zeros_like(frame_embeds)
        delta[:, 1:] = frame_embeds[:, 1:] - frame_embeds[:, :-1]
        features = torch.cat([frame_embeds, centered, delta], dim=-1)
        return self.feature_norm(self.feature_proj(features)), prior


def generic_multistep_rollout_loss(
    predictor: nn.Module,
    encoder_pool_fn: Callable[[torch.Tensor], torch.Tensor],
    image_history: torch.Tensor,
    action_chunks: torch.Tensor,
    image_future: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out any predictor with a TemporalPredictor-compatible forward."""

    horizon = action_chunks.shape[1]
    if image_future.shape[1] != horizon:
        raise ValueError(
            f"image_future horizon {image_future.shape[1]} != actions horizon {horizon}"
        )

    history_emb = encoder_pool_fn(image_history)
    future_emb = encoder_pool_fn(image_future)

    ctx = history_emb
    max_ctx = _max_context(predictor)
    losses = []
    for h in range(horizon):
        z_h = predictor(ctx, action_chunks[:, h])
        target_h = future_emb[:, h].detach()
        losses.append(F.mse_loss(z_h.float(), target_h.float()))
        ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
        if ctx.size(1) > max_ctx:
            ctx = ctx[:, -max_ctx:]

    rollout_loss = torch.stack(losses).mean()
    single_step = losses[0].detach()
    return rollout_loss, single_step


@torch.no_grad()
def prior_multistep_rollout_loss(
    encoder_pool_fn: Callable[[torch.Tensor], torch.Tensor],
    image_history: torch.Tensor,
    action_chunks: torch.Tensor,
    image_future: torch.Tensor,
    prior_kind: str,
    max_context: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out a closed-form carrier prior with no learned residual."""

    horizon = action_chunks.shape[1]
    history_emb = encoder_pool_fn(image_history)
    future_emb = encoder_pool_fn(image_future)
    ctx = history_emb
    losses = []
    for h in range(horizon):
        z_h = carrier_prior(ctx, prior_kind)
        losses.append(F.mse_loss(z_h.float(), future_emb[:, h].float()))
        ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
        if ctx.size(1) > max_context:
            ctx = ctx[:, -max_context:]
    rollout_loss = torch.stack(losses).mean()
    single_step = losses[0].detach()
    return rollout_loss, single_step


def _max_context(predictor: nn.Module) -> int:
    if hasattr(predictor, "frame_pos_embed"):
        return predictor.frame_pos_embed.size(1)
    if hasattr(predictor, "residual") and hasattr(predictor.residual, "frame_pos_embed"):
        return predictor.residual.frame_pos_embed.size(1)
    if hasattr(predictor, "direct") and hasattr(predictor.direct, "frame_pos_embed"):
        return predictor.direct.frame_pos_embed.size(1)
    if hasattr(predictor, "predictor") and hasattr(predictor.predictor, "frame_pos_embed"):
        return predictor.predictor.frame_pos_embed.size(1)
    raise ValueError("predictor does not expose frame_pos_embed/max context")


def _init_feature_projection_to_raw(proj: nn.Linear, d: int) -> None:
    with torch.no_grad():
        proj.weight.zero_()
        proj.weight[:, :d].copy_(torch.eye(d, device=proj.weight.device, dtype=proj.weight.dtype))
        if proj.bias is not None:
            proj.bias.zero_()
