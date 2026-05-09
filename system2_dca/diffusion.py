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
    """adaLN-Zero DiT block with self-attention, cross-attention to memory, and MLP.

    `cond` produces six modulations (shift/scale/gate for each of the self-attn
    and MLP residuals). The conditioning linear is zero-initialized so each block
    is the identity at the start of training.
    """

    def __init__(
        self,
        d_core: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_core, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_core, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm_xattn = nn.LayerNorm(d_core)
        self.xattn = nn.MultiheadAttention(
            embed_dim=d_core, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(d_core, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_core, int(d_core * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_core * mlp_ratio), d_core),
        )
        self.cond = nn.Linear(d_core, d_core * 6)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.cond(cond).chunk(6, dim=-1)
        attn_in = modulate(self.norm_attn(x), shift_a, scale_a)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_a[:, None, :] * attn_out

        xattn_in = self.norm_xattn(x)
        xattn_out, _ = self.xattn(xattn_in, memory_tokens, memory_tokens, need_weights=False)
        x = x + xattn_out

        mlp_in = modulate(self.norm_mlp(x), shift_m, scale_m)
        return x + gate_m[:, None, :] * self.mlp(mlp_in)


class LatentDiffusionEngine(nn.Module):
    """Rectified-flow velocity field over latent plans, with memory cross-attention.

    Trained to predict v* = noise - x0 under the linear interpolant
    x_t = (1-t) * x0 + t * noise. Sampling integrates dx/dt = v(x, t) from t=1
    (pure noise) to t=0 (data) with explicit Euler.
    """

    def __init__(
        self,
        d_core: int = 4096,
        depth: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        memory_tokens: int = 4,
    ):
        super().__init__()
        if d_core % heads != 0:
            raise ValueError("d_core must be divisible by heads")
        self.d_core = d_core
        self.memory_tokens = memory_tokens

        self.time_mlp = nn.Sequential(
            nn.Linear(d_core, d_core * 4),
            nn.SiLU(),
            nn.Linear(d_core * 4, d_core),
        )
        self.memory_global = nn.Linear(d_core, d_core)
        self.memory_to_tokens = nn.Linear(d_core, d_core * memory_tokens)
        self.in_proj = nn.Linear(d_core, d_core)

        self.blocks = nn.ModuleList(
            [DiT1DBlock(d_core, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_core)
        self.out_proj = nn.Linear(d_core, d_core)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _expand_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        batch = memory_state.shape[0]
        tokens = self.memory_to_tokens(memory_state)
        return tokens.view(batch, self.memory_tokens, self.d_core)

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
        time_emb = sinusoidal_embedding(timesteps.to(canvas.device), self.d_core).to(dtype=dtype)
        cond = self.time_mlp(time_emb) + self.memory_global(memory_state)
        memory_tokens = self._expand_memory(memory_state)

        for block in self.blocks:
            x = block(x, cond, memory_tokens)
        return self.out_proj(self.norm(x))

    @torch.no_grad()
    def sample(
        self,
        memory_state: torch.Tensor,
        seq_len: int,
        steps: int = 16,
        prior: torch.Tensor | None = None,
        t_start: float = 1.0,
    ) -> torch.Tensor:
        """Integrate the velocity field from t=t_start to t=0.

        Pure-Gaussian sampling: prior=None, t_start=1.0 (default).
        JEPA-prior warm start: pass `prior` of shape [B, seq_len, d_core] (e.g.
        bus.forward_up of pooled JEPA features) and a `t_start < 1.0`. The
        canvas is initialized at the linear interpolant
        x_{t_start} = (1 - t_start) * prior + t_start * noise, which carries
        JEPA's intuition into the integration so fewer steps are needed to
        reach a coherent x_0 (Pillar 3: JEPA as Diffusion Prior).
        """

        if not 0.0 < t_start <= 1.0:
            raise ValueError(f"t_start must be in (0, 1], got {t_start}")
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        batch = memory_state.shape[0]

        noise = torch.randn(batch, seq_len, self.d_core, device=device, dtype=dtype)
        if prior is None:
            x = noise if t_start == 1.0 else (1.0 - t_start) * 0.0 + t_start * noise
        else:
            if prior.shape != (batch, seq_len, self.d_core):
                raise ValueError(
                    f"prior shape {tuple(prior.shape)} != [{batch}, {seq_len}, {self.d_core}]"
                )
            prior = prior.to(device=device, dtype=dtype)
            x = (1.0 - t_start) * prior + t_start * noise

        ts = torch.linspace(t_start, 0.0, steps + 1, device=device, dtype=torch.float32)
        for i in range(steps):
            t_now = ts[i].expand(batch)
            v = self(x, t_now, memory_state)
            dt = (ts[i + 1] - ts[i]).to(dtype=dtype)
            x = x + dt * v
        return x


def diffusion_score_matching_loss(
    model: LatentDiffusionEngine,
    x0: torch.Tensor,
    memory_state: torch.Tensor,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rectified-flow training loss: predict the velocity v* = noise - x0."""

    if noise is None:
        noise = torch.randn_like(x0)
    if timesteps is None:
        timesteps = torch.rand(x0.shape[0], device=x0.device)
    t = timesteps.reshape(-1, 1, 1).to(dtype=x0.dtype)
    x_t = (1.0 - t) * x0 + t * noise
    v_target = noise - x0
    pred = model(x_t, timesteps, memory_state)
    return F.mse_loss(pred.float(), v_target.float())
