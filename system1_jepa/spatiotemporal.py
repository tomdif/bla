"""Unified spatiotemporal JEPA — V-JEPA-2-style.

The encoder consumes [B, T, C, H, W] and emits [B, T, N_patches, D] with
sincos-2d spatial position embeddings + a learned temporal position
embedding. The masking strategy samples (t, p) tuples; targets are
predicted at those positions given context patches from all other (t, p)
positions plus an action sequence.

This complements the per-frame patch JEPA in `model.py`: that model
trains spatial features within a single frame; this one trains
spatiotemporal features across a clip. Use this when you have video; use
the patch JEPA when you have stills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .vit import sincos_2d_position


@dataclass
class STMask:
    """Spatiotemporal mask: which (t, p) tuples are context vs target."""

    context_t: torch.Tensor  # [K_ctx] long
    context_p: torch.Tensor  # [K_ctx] long
    target_t: torch.Tensor  # [K_tgt] long
    target_p: torch.Tensor  # [K_tgt] long
    grid_h: int
    grid_w: int
    n_frames: int

    @property
    def num_context(self) -> int:
        return int(self.context_t.numel())

    @property
    def num_target(self) -> int:
        return int(self.target_t.numel())


def sample_tube_mask(
    n_frames: int,
    grid_h: int,
    grid_w: int,
    target_ratio: float = 0.5,
    device: torch.device | None = None,
) -> STMask:
    """Tube masking: pick a fraction of patch positions, mask them across all frames.

    Stronger temporal-consistency signal than independent per-frame masks.
    """

    device = device or torch.device("cpu")
    n_patch = grid_h * grid_w
    n_target_p = max(1, min(n_patch - 1, int(round(n_patch * target_ratio))))
    perm = torch.randperm(n_patch, device=device)
    target_p_set = perm[:n_target_p]
    context_p_set = perm[n_target_p:]
    t_idx = torch.arange(n_frames, device=device)
    target_t = t_idx.repeat_interleave(target_p_set.numel())
    target_p = target_p_set.repeat(n_frames)
    context_t = t_idx.repeat_interleave(context_p_set.numel())
    context_p = context_p_set.repeat(n_frames)
    return STMask(
        context_t=context_t,
        context_p=context_p,
        target_t=target_t,
        target_p=target_p,
        grid_h=grid_h,
        grid_w=grid_w,
        n_frames=n_frames,
    )


@dataclass
class SpatiotemporalConfig:
    in_channels: int = 3
    image_size: int = 16
    patch_size: int = 4
    n_frames: int = 4
    d: int = 32
    encoder_depth: int = 1
    encoder_heads: int = 4
    predictor_depth: int = 1
    predictor_heads: int = 4
    action_dim: int = 32
    target_ratio: float = 0.5
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class SpatiotemporalEncoder(nn.Module):
    """ViT applied per frame, returning [B, T, N, D] with shared weights across time."""

    def __init__(self, cfg: SpatiotemporalConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.image_size % cfg.patch_size:
            raise ValueError(f"image_size {cfg.image_size} not divisible by patch_size {cfg.patch_size}")
        self.grid = cfg.image_size // cfg.patch_size
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.d, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, cfg.n_frames, 1, cfg.d))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d,
            nhead=cfg.encoder_heads,
            dim_feedforward=int(cfg.d * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.encoder_depth)
        self.norm = nn.LayerNorm(cfg.d)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        patches = self.patch_embed(flat)
        tokens = patches.flatten(2).transpose(1, 2)
        n = tokens.shape[1]
        pos = sincos_2d_position(self.grid, self.grid, self.cfg.d, tokens.device, tokens.dtype)
        tokens = tokens + pos.unsqueeze(0)
        tokens = self.norm(self.blocks(tokens))
        tokens = tokens.reshape(b, t, n, self.cfg.d)
        tokens = tokens + self.temporal_pos_embed[:, :t]
        return tokens


class SpatiotemporalPredictor(nn.Module):
    """Predicts target (t, p) features given context (t, p) features + action chunk."""

    def __init__(self, cfg: SpatiotemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.image_size // cfg.patch_size
        self.context_proj = nn.Linear(cfg.d, cfg.d)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(cfg.n_frames, cfg.d))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d,
            nhead=cfg.predictor_heads,
            dim_feedforward=int(cfg.d * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.predictor_depth)
        self.norm = nn.LayerNorm(cfg.d)
        self.output_proj = nn.Linear(cfg.d, cfg.d)

    def _gather_st(self, tokens: torch.Tensor, t_idx: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T, N, D], t_idx/p_idx: [K] -> [B, K, D]"""

        flat_idx = t_idx * tokens.shape[2] + p_idx
        flat = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        gathered = flat[:, flat_idx]
        return gathered

    def forward(
        self,
        context_features: torch.Tensor,
        actions: torch.Tensor,
        mask: STMask,
    ) -> torch.Tensor:
        """context_features: [B, T, N, D] (encoded clip),
        actions: [B, T_action, action_dim],
        Returns predictions at target (t, p) positions: [B, K_tgt, D].
        """

        batch = context_features.shape[0]
        dim = context_features.shape[-1]
        device = context_features.device
        dtype = context_features.dtype

        ctx = self._gather_st(context_features, mask.context_t, mask.context_p)
        ctx = self.context_proj(ctx)

        spatial_pos = sincos_2d_position(mask.grid_h, mask.grid_w, dim, device, dtype)
        target_spatial = spatial_pos.index_select(0, mask.target_p)
        target_temporal = self.temporal_pos_embed.to(dtype=dtype).index_select(0, mask.target_t)
        target_tokens = self.mask_token.to(dtype=dtype)[None, None, :].expand(batch, mask.num_target, -1)
        target_tokens = target_tokens + target_spatial.unsqueeze(0) + target_temporal.unsqueeze(0)

        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        action_embed = self.action_proj(actions).mean(dim=1, keepdim=True)
        ctx = ctx + action_embed
        target_tokens = target_tokens + action_embed

        seq = torch.cat([ctx, target_tokens], dim=1)
        seq = self.blocks(seq)
        seq = self.norm(seq)
        target_out = seq[:, mask.num_context :, :]
        return self.output_proj(target_out)


class SpatiotemporalJEPA(nn.Module):
    """Encoder + EMA target encoder + predictor for clip-level JEPA."""

    def __init__(self, cfg: SpatiotemporalConfig, ema_tau: float = 0.996):
        super().__init__()
        self.cfg = cfg
        self.ema_tau = ema_tau
        self.context_encoder = SpatiotemporalEncoder(cfg)
        self.target_encoder = SpatiotemporalEncoder(cfg)
        self.predictor = SpatiotemporalPredictor(cfg)
        self._copy_context_to_target()
        self.target_encoder.requires_grad_(False)

    def _copy_context_to_target(self) -> None:
        with torch.no_grad():
            for tgt, src in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                tgt.data.copy_(src.detach().to(dtype=tgt.dtype))

    @torch.no_grad()
    def update_target_ema(self, tau: float | None = None) -> None:
        tau = self.ema_tau if tau is None else tau
        for target_param, source_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            source_fp32 = source_param.detach().to(dtype=target_param.dtype)
            target_param.mul_(tau).add_(source_fp32, alpha=1.0 - tau)

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        mask: STMask | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, STMask]:
        if mask is None:
            mask = sample_tube_mask(
                n_frames=frames.shape[1],
                grid_h=self.context_encoder.grid,
                grid_w=self.context_encoder.grid,
                target_ratio=self.cfg.target_ratio,
                device=frames.device,
            )
        context_features = self.context_encoder(frames)
        with torch.no_grad():
            target_features = self.target_encoder(frames)
            target_at_targets = self.predictor._gather_st(target_features, mask.target_t, mask.target_p)
        z_hat = self.predictor(context_features, actions, mask)
        return z_hat, target_at_targets, mask

    def training_loss(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        mask: STMask | None = None,
    ) -> dict:
        z_hat, z_target, mask = self(frames, actions, mask=mask)
        prediction = F.smooth_l1_loss(z_hat.float(), z_target.detach().float())
        return {"loss": prediction, "prediction": prediction.detach(), "mask": mask}
