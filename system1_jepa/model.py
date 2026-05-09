from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .losses import jepa_loss
from .masking import PatchMask, gather_tokens, sample_patch_mask
from .predictor import ActionConditionedPredictor
from .vit import PatchViTEncoder


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


@dataclass
class JEPAConfig:
    d_jepa: int = 1024
    in_channels: int = 3
    patch_size: int = 14
    encoder_depth: int = 32
    encoder_heads: int = 16
    encoder_mlp_ratio: float = 4.0
    predictor_depth: int = 4
    predictor_heads: int = 8
    action_dim: int = 1024
    ema_tau: float = 0.996
    sigreg_weight: float = 1.0
    sigreg_variant: str = "epps_pulley"
    target_mask_ratio: float = 0.5
    dtype: str = "bfloat16"

    @classmethod
    def tiny(cls) -> "JEPAConfig":
        return cls(
            d_jepa=32,
            patch_size=4,
            encoder_depth=1,
            encoder_heads=4,
            predictor_depth=1,
            predictor_heads=4,
            action_dim=32,
            ema_tau=0.9,
            dtype="float32",
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype)


class BLAJEPAModel(nn.Module):
    """System 1: masked context encoder, EMA full-image target encoder, predictor.

    Mixed precision: the context encoder and predictor follow `config.dtype`.
    The target encoder is always held in float32 to keep EMA updates numerically
    safe even when the trained branch runs in bf16.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.context_encoder = PatchViTEncoder(
            in_channels=config.in_channels,
            latent_dim=config.d_jepa,
            patch_size=config.patch_size,
            depth=config.encoder_depth,
            heads=config.encoder_heads,
            mlp_ratio=config.encoder_mlp_ratio,
        )
        self.target_encoder = PatchViTEncoder(
            in_channels=config.in_channels,
            latent_dim=config.d_jepa,
            patch_size=config.patch_size,
            depth=config.encoder_depth,
            heads=config.encoder_heads,
            mlp_ratio=config.encoder_mlp_ratio,
        )
        self.predictor = ActionConditionedPredictor(
            latent_dim=config.d_jepa,
            action_dim=config.action_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
        )
        self._copy_context_to_target()
        self.target_encoder.requires_grad_(False)

        self.context_encoder.to(dtype=config.torch_dtype)
        self.predictor.to(dtype=config.torch_dtype)

    def _copy_context_to_target(self) -> None:
        with torch.no_grad():
            for tgt, src in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                tgt.data.copy_(src.detach().to(dtype=tgt.dtype))
            for tgt, src in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
                tgt.data.copy_(src.detach().to(dtype=tgt.dtype))

    @torch.no_grad()
    def update_target_ema(self, tau: float | None = None) -> None:
        tau = self.config.ema_tau if tau is None else tau
        for target_param, source_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            source_fp32 = source_param.detach().to(dtype=target_param.dtype)
            target_param.mul_(tau).add_(source_fp32, alpha=1.0 - tau)

    def sample_mask(self, image: torch.Tensor) -> PatchMask:
        h, w = image.shape[-2], image.shape[-1]
        ps = self.config.patch_size
        if h % ps or w % ps:
            raise ValueError(f"image {h}x{w} is not divisible by patch_size {ps}")
        return sample_patch_mask(
            grid_h=h // ps,
            grid_w=w // ps,
            target_ratio=self.config.target_mask_ratio,
            device=image.device,
        )

    def forward(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
        mask: PatchMask | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, PatchMask]:
        mask = mask if mask is not None else self.sample_mask(image)
        ctx_dtype = next(self.context_encoder.parameters()).dtype
        tgt_dtype = next(self.target_encoder.parameters()).dtype

        z_context, grid_h, grid_w = self.context_encoder(
            image.to(dtype=ctx_dtype), keep_idx=mask.context_idx
        )
        with torch.no_grad():
            z_full_target, _, _ = self.target_encoder(image.to(dtype=tgt_dtype))
            z_target = gather_tokens(z_full_target, mask.target_idx)

        z_hat = self.predictor(
            context_tokens=z_context,
            action=action.to(dtype=ctx_dtype),
            target_positions=mask.target_idx,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        return z_hat, z_target, mask

    def training_loss(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
        mask: PatchMask | None = None,
    ) -> Dict[str, torch.Tensor]:
        z_hat, z_target, _ = self(image, action, mask=mask)
        return jepa_loss(
            predicted_target=z_hat,
            target_latent=z_target,
            sigreg_weight=self.config.sigreg_weight,
            sigreg_variant=self.config.sigreg_variant,
        )
