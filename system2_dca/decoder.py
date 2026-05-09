from __future__ import annotations

import math

import torch
from torch import nn


class DeterministicLexicalDecoder(nn.Module):
    """Frozen codebook projection with argmax decoding (temperature 0).

    The codebook is held in float32 regardless of the surrounding module's
    dtype so the inner-product argmax is robust at large vocabularies. By
    default the codebook is random unit-norm Gaussian (a placeholder); use
    `from_embeddings(weight)` to load from a real tokenizer's embedding table.
    """

    def __init__(
        self,
        d_core: int = 4096,
        vocab_size: int = 128_000,
        normalize_codebook: bool = True,
    ):
        super().__init__()
        weight = torch.randn(vocab_size, d_core, dtype=torch.float32) / math.sqrt(d_core)
        if normalize_codebook:
            weight = torch.nn.functional.normalize(weight, dim=-1)
        self.codebook = nn.Parameter(weight, requires_grad=False)

    @classmethod
    def from_embeddings(
        cls,
        weight: torch.Tensor,
        normalize_codebook: bool = True,
    ) -> "DeterministicLexicalDecoder":
        if weight.ndim != 2:
            raise ValueError(f"weight must be [vocab, d_core], got {tuple(weight.shape)}")
        decoder = cls.__new__(cls)
        nn.Module.__init__(decoder)
        w = weight.to(dtype=torch.float32)
        if normalize_codebook:
            w = torch.nn.functional.normalize(w, dim=-1)
        decoder.codebook = nn.Parameter(w, requires_grad=False)
        return decoder

    def logits(self, x0: torch.Tensor) -> torch.Tensor:
        return x0.float() @ self.codebook.float().T

    @torch.no_grad()
    def decode(self, x0: torch.Tensor) -> torch.Tensor:
        return self.logits(x0).argmax(dim=-1)
