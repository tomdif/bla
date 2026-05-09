from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .losses import jepa_loss
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
    vicreg_weight: float = 1.0
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
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype)


class BLAJEPAModel(nn.Module):
    """System 1: context encoder, EMA target encoder, and latent predictor."""

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
        self.to(dtype=config.torch_dtype)

    def _copy_context_to_target(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

    @torch.no_grad()
    def update_target_ema(self, tau: float | None = None) -> None:
        tau = self.config.ema_tau if tau is None else tau
        for target_param, source_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target_param.mul_(tau).add_(source_param.detach(), alpha=1.0 - tau)

    def forward(
        self,
        masked_state: torch.Tensor,
        unmasked_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = next(self.parameters()).dtype
        masked_state = masked_state.to(dtype=dtype)
        unmasked_state = unmasked_state.to(dtype=dtype)
        action = action.to(dtype=dtype)

        z_context = self.context_encoder(masked_state)
        with torch.no_grad():
            z_target = self.target_encoder(unmasked_state)
        z_hat = self.predictor(z_context, action)
        return z_hat, z_target, z_context

    def training_loss(
        self,
        masked_state: torch.Tensor,
        unmasked_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hat, z_target, z_context = self(masked_state, unmasked_state, action)
        return jepa_loss(
            predicted_target=z_hat,
            target_latent=z_target,
            context_latent=z_context,
            vicreg_weight=self.config.vicreg_weight,
        )

