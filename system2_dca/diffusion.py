from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(dim // 2, 1)
    freq = torch.arange(half, device=t.device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * freq / max(half - 1, 1))
    args = t.float()[:, None] * freq[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


def sequence_position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)
    return sinusoidal_embedding(pos, dim).to(dtype=dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiT1DBlock(nn.Module):
    def __init__(self, d_core: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_core)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_core,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(d_core)
        self.mlp = nn.Sequential(
            nn.Linear(d_core, int(d_core * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_core * mlp_ratio), d_core),
        )
        self.cond = nn.Linear(d_core, d_core * 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, shift_m, scale_m = self.cond(cond).chunk(4, dim=-1)
        attn_in = modulate(self.norm_attn(x), shift_a, scale_a)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        mlp_in = modulate(self.norm_mlp(x), shift_m, scale_m)
        return x + self.mlp(mlp_in)


class LatentDiffusionEngine(nn.Module):
    """1D DiT that predicts the diffusion/flow velocity field over latent plans."""

    def __init__(
        self,
        d_core: int = 4096,
        depth: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_core % heads != 0:
            raise ValueError("d_core must be divisible by heads")
        self.d_core = d_core
        self.time_mlp = nn.Sequential(
            nn.Linear(d_core, d_core * 4),
            nn.SiLU(),
            nn.Linear(d_core * 4, d_core),
        )
        self.memory_proj = nn.Linear(d_core, d_core)
        self.in_proj = nn.Linear(d_core, d_core)
        self.blocks = nn.ModuleList(
            [DiT1DBlock(d_core, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_core)
        self.out_proj = nn.Linear(d_core, d_core)

    def forward(
        self,
        canvas: torch.Tensor,
        timesteps: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        canvas = canvas.to(dtype=dtype)
        memory_state = memory_state.to(dtype=dtype)
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(canvas.shape[0])
        pos = sequence_position(canvas.shape[1], self.d_core, canvas.device, dtype)
        x = self.in_proj(canvas + pos.unsqueeze(0))
        time = sinusoidal_embedding(timesteps.to(canvas.device), self.d_core).to(dtype=dtype)
        cond = self.time_mlp(time) + self.memory_proj(memory_state)
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(self.norm(x))


def diffusion_score_matching_loss(
    model: LatentDiffusionEngine,
    x0: torch.Tensor,
    memory_state: torch.Tensor,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    if timesteps is None:
        timesteps = torch.rand(x0.shape[0], device=x0.device)
    t = timesteps.reshape(-1, 1, 1).to(dtype=x0.dtype)
    x_t = (1.0 - t) * x0 + t * noise
    pred = model(x_t, timesteps, memory_state)
    return F.mse_loss(pred.float(), noise.float())

