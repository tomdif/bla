from __future__ import annotations

import torch
from torch import nn


def causal_token_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class ActionConditionedPredictor(nn.Module):
    """Shallow causal transformer that predicts future JEPA latents from actions."""

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
        self.input_proj = nn.Linear(latent_dim * 2, latent_dim)
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

    def _expand_action(self, action: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if action.ndim == 2:
            return action[:, None, :].expand(-1, num_tokens, -1)
        if action.ndim == 3:
            if action.shape[1] == 1:
                return action.expand(-1, num_tokens, -1)
            if action.shape[1] == num_tokens:
                return action
        raise ValueError(
            "action must be [batch, action_dim], [batch, 1, action_dim], "
            f"or [batch, tokens, action_dim], got {tuple(action.shape)}"
        )

    def forward(self, z_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        num_tokens = z_context.shape[1]
        action_tokens = self._expand_action(action, num_tokens)
        action_latent = self.action_proj(action_tokens)
        x = self.input_proj(torch.cat([z_context, action_latent], dim=-1))
        mask = causal_token_mask(num_tokens, x.device)
        x = self.blocks(x, mask=mask)
        return self.output_proj(self.norm(x))

