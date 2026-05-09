from __future__ import annotations

import math

import torch
from torch import nn


class DeterministicLexicalDecoder(nn.Module):
    """Frozen VQ/codebook projection with argmax decoding and temperature 0."""

    def __init__(
        self,
        d_core: int = 4096,
        vocab_size: int = 128_000,
        dtype: torch.dtype = torch.bfloat16,
        normalize_codebook: bool = True,
    ):
        super().__init__()
        weight = torch.randn(vocab_size, d_core, dtype=torch.float32) / math.sqrt(d_core)
        if normalize_codebook:
            weight = torch.nn.functional.normalize(weight, dim=-1)
        self.codebook = nn.Parameter(weight.to(dtype=dtype), requires_grad=False)

    def logits(self, x0: torch.Tensor) -> torch.Tensor:
        return x0.float() @ self.codebook.float().T

    @torch.no_grad()
    def decode(self, x0: torch.Tensor) -> torch.Tensor:
        return self.logits(x0).argmax(dim=-1)

