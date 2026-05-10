"""Procedural CPU — the System 2 reasoner for B.L.A.

A standard transformer decoder, intentionally compact (target 200M-1B
params). Loaded with weights trained on the synthetic logic curriculum,
not on web text — the procedural-only training is what the asymmetric-
scaling thesis hinges on.

This is the architecture that will be tested at Phase 6 against
GPT-2 1.5B parametric on math/code/proof/planning tasks.

The class is FSDP-friendly: no buffers that block sharding, no
cross-shard parameter access, single forward path that returns logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProceduralCoreConfig:
    vocab_size: int = 50_257  # GPT-2 BPE
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    mlp_ratio: int = 4
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10_000.0

    @classmethod
    def small_500m(cls) -> "ProceduralCoreConfig":
        # ~500M params: d=1280, layers=24, heads=20
        return cls(d_model=1280, n_layers=24, n_heads=20)

    @classmethod
    def micro(cls) -> "ProceduralCoreConfig":
        # ~25M params for CPU smoke
        return cls(d_model=256, n_layers=4, n_heads=4, max_seq_len=512)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def _rotary_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = torch.arange(0, half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freqs / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.einsum("s,d->sd", t, inv_freq)  # [S, D/2]
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, D] (head_dim D). cos/sin: [S, D/2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    rx1 = x1 * cos_b - x2 * sin_b
    rx2 = x1 * sin_b + x2 * cos_b
    return torch.cat([rx1, rx2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ProceduralCoreConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # PyTorch SDPA dispatches to FlashAttention on CUDA
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, S, D)
        return self.proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ProceduralCoreConfig):
        super().__init__()
        self.norm_attn = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm_mlp = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * cfg.mlp_ratio, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ProceduralCore(nn.Module):
    """Compact transformer reasoner trained on synthetic logic only.

    Public surface: forward(token_ids) -> logits [B, S, V].
    """

    def __init__(self, cfg: ProceduralCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  # tied
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard GPT-style init: scaled normal for linears, std=0.02
        baseline; residual projections scaled by 1/sqrt(2 * n_layers)."""
        std = 0.02
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=std)
        for block in self.blocks:
            for name, p in block.named_parameters():
                if p.dim() < 2:
                    continue
                if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                    # residual-out projections: scale by 1/sqrt(2L)
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.cfg.n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = token_ids.device
        seq_len = token_ids.shape[1]
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"input length {seq_len} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_embed(token_ids)
        cos, sin = _rotary_cos_sin(seq_len, self.cfg.head_dim, self.cfg.rope_base, device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        return self.head(x)

    def loss(self, token_ids: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100) -> torch.Tensor:
        """Standard causal LM loss on [B, S] tokens.

        Labels follow the HuggingFace convention: same shape as token_ids;
        prompt positions are -100. We do the causal shift inside: predict
        labels[t+1] from logits[t]. Cross-entropy in fp32 for stability.
        """
        logits = self.forward(token_ids).float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.transpose(1, 2),  # [B, V, S-1]
            shift_labels,
            ignore_index=ignore_index,
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
