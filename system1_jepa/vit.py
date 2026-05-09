from __future__ import annotations

import torch
from torch import nn


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(dim // 2, 1)
    omega = torch.arange(half, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half - 1, 1)))
    out = positions.float()[:, None] * omega[None, :]
    emb = torch.cat([out.sin(), out.cos()], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def sincos_2d_position(
    grid_h: int,
    grid_w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.arange(grid_h, device=device)
    x = torch.arange(grid_w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    y_dim = dim // 2
    x_dim = dim - y_dim
    pos = torch.cat([_sincos_1d(yy, y_dim), _sincos_1d(xx, x_dim)], dim=1)
    return pos.to(dtype=dtype)


class PatchViTEncoder(nn.Module):
    """Minimal ViT encoder used by JEPA context and EMA target branches.

    The encoder can process either the full patch grid or a subset of patches
    selected by `keep_idx`. When `keep_idx` is provided, position embeddings
    are gathered for the corresponding patch indices so the encoder sees
    correctly-placed tokens for the masked-context view.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 1024,
        patch_size: int = 14,
        depth: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim % heads != 0:
            raise ValueError("latent_dim must be divisible by heads")
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            latent_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
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
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        patches = self.patch_embed(x)
        _, _, grid_h, grid_w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens, grid_h, grid_w

    def forward(
        self,
        x: torch.Tensor,
        keep_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        tokens, grid_h, grid_w = self.patchify(x)
        dim = tokens.shape[-1]
        pos = sincos_2d_position(grid_h, grid_w, dim, tokens.device, tokens.dtype)
        if keep_idx is not None:
            tokens = tokens.index_select(1, keep_idx)
            pos = pos.index_select(0, keep_idx)
        tokens = tokens + pos.unsqueeze(0)
        return self.norm(self.blocks(tokens)), grid_h, grid_w
