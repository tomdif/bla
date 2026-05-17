"""ProposalEncoder — wraps ConvNeXt-T as OF-JEPA's per-frame proposal feature extractor.

Per-frame output is a grid of patch tokens, each carrying appearance +
implicit spatial position via the encoder's positional embedding. The
ObjectFileMemory's Sinkhorn matcher then binds persistent memory cells
to these tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProposalEncoder(nn.Module):
    """Per-frame proposal feature extractor.

    Output: [B, N_props, proposal_dim] — the patch-token grid where
    each token carries both appearance and approximate spatial
    location information (the spatial pos-embed in the encoder lets
    the matcher use position implicitly).
    """
    def __init__(self, input_size: int, proposal_dim: int):
        super().__init__()
        from system1_jepa.convnext_encoder import (
            ConvNeXtEncoderConfig, ConvNeXtSlotEncoder,
        )
        self.encoder = ConvNeXtSlotEncoder(ConvNeXtEncoderConfig(
            input_size=input_size, slot_dim=proposal_dim,
            pretrained=False, freeze_early_stages=0,
        ))
        self.n_tokens = self.encoder.n_tokens

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: [T, 3, H, W] → [T, n_tokens, proposal_dim]"""
        return self.encoder(video)
