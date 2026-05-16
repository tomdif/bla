"""ConvNeXt-T encoder wrapper for SlotAttention.

The current toy slot pipeline uses a small 4-layer CNN that's fine for
32×32 patch scenes but underpowered for 128×128+ MOVi/CLEVRER imagery.
This wrapper:

  1. Loads ConvNeXt-T from torchvision (random init by default; pretrained
     weights are loadable but trade off domain mismatch — MOVi is
     rendered, not natural images).
  2. Freezes the first two stages and exposes the final feature map as
     [B, C, H', W'] tokens for SlotAttention.
  3. Projects channels to `slot_dim` so it drops in cleanly to the
     existing SlotAttention module (slot.py).

The output is a sequence of patch tokens [B, N_patches, slot_dim] that
SlotAttention can iterate over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ConvNeXtEncoderConfig:
    input_size: int = 128
    slot_dim: int = 128
    pretrained: bool = False
    freeze_early_stages: int = 2  # freeze the first N of the 4 stages
    pos_embed: bool = True


class ConvNeXtSlotEncoder(nn.Module):
    """ConvNeXt-T trunk → [B, N_patches, slot_dim] feature tokens."""

    def __init__(self, cfg: ConvNeXtEncoderConfig = ConvNeXtEncoderConfig()):
        super().__init__()
        self.cfg = cfg

        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = convnext_tiny(weights=weights)

        # ConvNeXt-T `features` is a sequential of 8 blocks (4 stages, each
        # preceded by a downsample/stem). Stages live at indices 1,3,5,7.
        # For an input of 128×128:
        #   after stem (4× downsample): 32×32
        #   stage 1: 32×32, C=96
        #   stage 2: 16×16, C=192
        #   stage 3: 8×8,   C=384
        #   stage 4: 4×4,   C=768
        # We use the stage-3 (8×8) output — 64 tokens, channel 384.
        self.features = model.features
        self._use_up_to = 6  # include through stage 3 (index 5 is stage 3 norm-block)
        out_channels = 384

        # Optional freeze.
        if cfg.freeze_early_stages > 0:
            # Stages live at even indices in features: 1, 3, 5, 7.
            # freeze_early_stages=2 → freeze stages 0 and 1 (indices 0..3 inclusive).
            freeze_until = 2 * cfg.freeze_early_stages
            for i, block in enumerate(self.features):
                if i < freeze_until:
                    for p in block.parameters():
                        p.requires_grad = False

        self.proj = nn.Linear(out_channels, cfg.slot_dim)

        # Spatial pos embed for slot binding.
        # 8×8 = 64 patches at 128×128 input.
        self.n_patches_h = cfg.input_size // 16  # 16× downsample at stage 3
        self.n_patches_w = cfg.input_size // 16
        n_patches = self.n_patches_h * self.n_patches_w
        if cfg.pos_embed:
            self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.slot_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.register_parameter("pos", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] → [B, n_patches, slot_dim]"""
        # Use only the first 6 children (through stage 3).
        for i, block in enumerate(self.features):
            x = block(x)
            if i >= self._use_up_to - 1:
                break
        # x: [B, C, H', W']
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H'*W', C]
        tokens = self.proj(tokens)
        if self.pos is not None:
            tokens = tokens + self.pos
        return tokens

    @property
    def n_tokens(self) -> int:
        return self.n_patches_h * self.n_patches_w
