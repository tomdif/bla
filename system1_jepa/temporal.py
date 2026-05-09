"""Frame-level temporal predictor + multi-step rollout loss.

The patch-level JEPA in `model.py` learns spatial features within a single
frame. This module adds a temporal head on top: each frame's patch tokens
are pooled to one D-dim vector and a small transformer learns to predict
the next-frame embedding given a chunk of actions. Multi-step rollout
slides the context window with the predictor's own outputs and supervises
each horizon, training self-consistency under recursion (Stage AH from
cser-jepa-v2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def pool_patch_tokens(z_patches: torch.Tensor) -> torch.Tensor:
    """Mean-pool patch tokens to a single per-frame embedding.

    z_patches: [B, N_patches, D] -> [B, D]
    """

    if z_patches.ndim != 3:
        raise ValueError(f"expected [B, N, D], got {tuple(z_patches.shape)}")
    return z_patches.mean(dim=1)


@dataclass
class TemporalConfig:
    d: int = 1024
    action_dim: int = 1024
    chunk_size: int = 1
    n_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_context: int = 16
    dropout: float = 0.0

    @classmethod
    def tiny(cls) -> "TemporalConfig":
        return cls(d=32, action_dim=32, chunk_size=1, n_layers=1, num_heads=4, max_context=8)


class TemporalPredictor(nn.Module):
    """Frame-level transformer: history of frame embeddings + action -> next frame.

    Includes per-step reward head (predicts K scalars over the action chunk)
    and a value head (return-to-go bootstrap). Both default to zeros so they
    contribute nothing to L_pred unless explicitly weighted in training.
    """

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Linear(cfg.action_dim * cfg.chunk_size, cfg.d)
        self.target_token = nn.Parameter(torch.zeros(1, 1, cfg.d))
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_context, cfg.d))
        self.role_embed = nn.Parameter(torch.zeros(1, 3, cfg.d))
        nn.init.trunc_normal_(self.target_token, std=0.02)
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.role_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.d * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d)

        self.reward_head = nn.Linear(cfg.d, cfg.chunk_size)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        self.value_head = nn.Linear(cfg.d, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.value_head(z).squeeze(-1)

    def forward(
        self,
        frame_embeds: torch.Tensor,
        a_chunk: torch.Tensor,
        return_aux: bool = False,
    ):
        """frame_embeds: [B, T, D], a_chunk: [B, K, action_dim] -> [B, D]

        If return_aux=True, returns a dict {z, r, v} with the predicted next-state
        embedding z, per-step reward r [B, chunk_size], and value v [B].
        """

        batch, t, _ = frame_embeds.shape
        if t > self.frame_pos_embed.size(1):
            raise ValueError(f"context {t} > max_context {self.frame_pos_embed.size(1)}")
        a_flat = a_chunk.reshape(batch, -1)
        a_tok = self.action_embed(a_flat).unsqueeze(1)
        target_tok = self.target_token.expand(batch, -1, -1)

        frames = frame_embeds + self.frame_pos_embed[:, :t] + self.role_embed[:, 0:1]
        a_tok = a_tok + self.role_embed[:, 1:2]
        target_tok = target_tok + self.role_embed[:, 2:3]

        x = torch.cat([frames, a_tok, target_tok], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        z_pred = x[:, -1]
        if not return_aux:
            return z_pred
        return {
            "z": z_pred,
            "r": self.reward_head(z_pred),
            "v": self.value(z_pred),
        }

    @torch.no_grad()
    def rollout(
        self,
        frame_embeds: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> dict:
        """Multi-step rollout for planning.

        frame_embeds: [B, T, D]
        action_seq:   [B, H, K, action_dim]   H steps of action chunks.
        Returns {z: [B, H, D], r: [B, H, K], v: [B, H]}.
        """

        if action_seq.ndim == 3:
            action_seq = action_seq.unsqueeze(2)
        batch, horizon = action_seq.shape[:2]
        max_ctx = self.frame_pos_embed.size(1)
        ctx = frame_embeds
        zs, rs, vs = [], [], []
        for h in range(horizon):
            out = self.forward(ctx, action_seq[:, h], return_aux=True)
            zs.append(out["z"])
            rs.append(out["r"])
            vs.append(out["v"])
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
        return {
            "z": torch.stack(zs, dim=1),
            "r": torch.stack(rs, dim=1),
            "v": torch.stack(vs, dim=1),
        }


def multistep_rollout_loss(
    predictor: TemporalPredictor,
    encoder_pool_fn,
    image_history: torch.Tensor,
    action_chunks: torch.Tensor,
    image_future: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out the temporal predictor self-consistently over the future horizon.

    image_history:  [B, T_ctx, C, H, W]
    action_chunks:  [B, H, K, action_dim]      (one chunk per future step)
    image_future:   [B, H, C, H, W]            (ground-truth next frames)

    Returns (mean rollout MSE, single-step MSE for diagnostic comparison).
    The encoder_pool_fn must accept [B, T, C, H, W] frames flattened over time
    and return [B, T, D] pooled embeddings.
    """

    batch, t_ctx = image_history.shape[:2]
    horizon = action_chunks.shape[1]
    if image_future.shape[1] != horizon:
        raise ValueError(
            f"image_future horizon {image_future.shape[1]} != actions horizon {horizon}"
        )

    history_emb = encoder_pool_fn(image_history)
    future_emb = encoder_pool_fn(image_future)

    ctx = history_emb
    max_ctx = predictor.frame_pos_embed.size(1)
    losses = []
    z_pred_first = None
    for h in range(horizon):
        a_h = action_chunks[:, h]
        z_h = predictor(ctx, a_h)
        if z_pred_first is None:
            z_pred_first = z_h
        target_h = future_emb[:, h].detach()
        losses.append(F.mse_loss(z_h.float(), target_h.float()))
        ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
        if ctx.size(1) > max_ctx:
            ctx = ctx[:, -max_ctx:]

    rollout_loss = torch.stack(losses).mean()
    single_step = losses[0].detach()
    return rollout_loss, single_step
