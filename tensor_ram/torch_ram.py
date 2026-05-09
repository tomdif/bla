from __future__ import annotations

import torch
from torch import nn


class DifferentiableTensorRAM(nn.Module):
    """Torch-side MIPS RAM with differentiable READ queries.

    Keys are held as a frozen buffer (no gradients flow into the index).
    Queries flow gradients through the inner-product scores and the
    softmax-weighted reconstruction, so the model can learn *what* to fetch
    even though the RAM contents are non-parametric.

    The retrieval contract is:
      query    [B, d_ram]   or [B, M, d_ram]
      returns  [B, d_ram]   or [B, M, d_ram]   (softmax-weighted top-k facts)

    For inference at scale, swap the brute-force `query @ keys.T` with a FAISS
    top-k whose indices are detached and only the selected scores carry
    gradient — same training contract, but sub-linear retrieval.
    """

    def __init__(self, d_ram: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_ram = d_ram
        self.register_buffer("keys", torch.empty(0, d_ram, dtype=dtype))

    @property
    def ntotal(self) -> int:
        return int(self.keys.shape[0])

    @torch.no_grad()
    def add(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.d_ram:
            raise ValueError(f"expected [n, {self.d_ram}], got {tuple(vectors.shape)}")
        new = vectors.detach().to(dtype=self.keys.dtype, device=self.keys.device)
        if self.keys.numel():
            self.keys = torch.cat([self.keys, new], dim=0)
        else:
            self.keys = new

    @torch.no_grad()
    def add_random(self, n: int, generator: torch.Generator | None = None) -> None:
        v = torch.randn(n, self.d_ram, generator=generator, dtype=self.keys.dtype)
        v = torch.nn.functional.normalize(v, dim=-1)
        self.add(v)

    def _topk_scores(self, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(k, self.ntotal)
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        return top_scores, top_idx

    def forward(
        self,
        query: torch.Tensor,
        k: int = 8,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft top-k retrieval. Returns (weighted_facts, top_idx).

        weighted_facts has the same shape as query.
        top_idx is detached (long), useful for logging/cache prefetch.
        """

        if self.ntotal == 0:
            raise RuntimeError("RAM is empty")
        squeeze = False
        if query.ndim == 2:
            query = query.unsqueeze(1)
            squeeze = True
        if query.ndim != 3 or query.shape[-1] != self.d_ram:
            raise ValueError(f"query must be [B, M, {self.d_ram}], got {tuple(query.shape)}")

        keys = self.keys.to(dtype=query.dtype)
        scores = torch.einsum("bmd,nd->bmn", query, keys)
        top_scores, top_idx = self._topk_scores(scores, k)
        weights = torch.softmax(top_scores / max(temperature, 1e-6), dim=-1)
        gathered = keys[top_idx]
        weighted = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
        if squeeze:
            weighted = weighted.squeeze(1)
            top_idx = top_idx.squeeze(1)
        return weighted, top_idx.detach()
