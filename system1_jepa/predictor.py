from __future__ import annotations

import torch
from torch import nn

from .vit import sincos_2d_position


class ActionConditionedPredictor(nn.Module):
    """Predicts JEPA target-patch latents from context tokens + an action vector.

    The predictor consumes:
      - context tokens (with their own position embeddings carried from the encoder)
      - target position queries (a learned mask token + sincos-2d position for each target patch)
      - a global action vector (broadcast onto every input token)

    Self-attention is full (no causal mask): patches are spatial, not temporal.
    The predictor returns features at the target positions only.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 1024,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.context_token_proj = nn.Linear(latent_dim, latent_dim)
        self.mask_token = nn.Parameter(torch.zeros(latent_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=int(latent_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def _broadcast_action(self, action: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if action.ndim == 2:
            return action[:, None, :].expand(-1, num_tokens, -1)
        if action.ndim == 3 and action.shape[1] == 1:
            return action.expand(-1, num_tokens, -1)
        if action.ndim == 3 and action.shape[1] == num_tokens:
            return action
        raise ValueError(
            "action must be [B, action_dim], [B, 1, action_dim], or [B, N, action_dim]; "
            f"got {tuple(action.shape)}"
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        action: torch.Tensor,
        target_positions: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Predict target features.

        context_tokens:    [B, K_ctx, D]  -- encoder output at context patches
        action:            [B, action_dim] (or [B, 1, action_dim] / [B, N, action_dim])
        target_positions:  [K_tgt] long indices into the grid_h*grid_w patch sequence
        returns:           [B, K_tgt, D]
        """

        batch, k_ctx, dim = context_tokens.shape
        k_tgt = int(target_positions.numel())
        device, dtype = context_tokens.device, context_tokens.dtype

        target_pos_emb = sincos_2d_position(grid_h, grid_w, dim, device, dtype)
        target_pos_emb = target_pos_emb.index_select(0, target_positions)
        target_tokens = self.mask_token.to(dtype=dtype)[None, None, :].expand(batch, k_tgt, -1)
        target_tokens = target_tokens + target_pos_emb.unsqueeze(0)

        ctx = self.context_token_proj(context_tokens)
        seq = torch.cat([ctx, target_tokens], dim=1)

        action_tokens = self._broadcast_action(action, seq.shape[1])
        seq = seq + self.action_proj(action_tokens)

        seq = self.blocks(seq)
        seq = self.norm(seq)
        target_out = seq[:, k_ctx:, :]
        return self.output_proj(target_out)
