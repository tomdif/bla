"""Asynchronous RAM prefetch driven by JEPA's continuous latent stream.

While System 1 (JEPA) is running on every frame, its pooled latent is
continuously projected up to D.C.A. core space and used as a key against
the Tensor RAM. The most recent retrieved facts are cached so that when
the entropy router wakes D.C.A., the relevant facts are already in
working memory — fact-retrieval latency at wake-up is reduced to zero
(Pillar §4: Memory Synergy / Asynchronous Pre-Fetching).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from system2_dca.ram_reader import RAMReader
from tensor_ram.torch_ram import DifferentiableTensorRAM

from .bus import TokenlessLatentBus


class AsyncPrefetcher(nn.Module):
    """JEPA-driven RAM prefetch with a most-recent-facts cache."""

    def __init__(
        self,
        bus: TokenlessLatentBus,
        ram: DifferentiableTensorRAM,
        d_state: int,
        num_query_heads: int = 4,
        top_k: int = 8,
    ):
        super().__init__()
        self.bus = bus
        self.reader = RAMReader(
            d_state=d_state,
            ram=ram,
            num_query_heads=num_query_heads,
            top_k=top_k,
        )
        self._cache: Optional[torch.Tensor] = None
        self._last_idx: Optional[torch.Tensor] = None

    @torch.no_grad()
    def ping(self, z_jepa_pooled: torch.Tensor) -> torch.Tensor:
        """Project pooled JEPA latents into core space and refresh the RAM cache.

        z_jepa_pooled: [B, d_jepa] (mean-pool of JEPA patch tokens or the JEPA
        target embedding). Returns the freshly cached facts [B, num_query_heads, d_state].
        """

        if z_jepa_pooled.ndim != 2:
            raise ValueError(f"expected [B, d_jepa], got {tuple(z_jepa_pooled.shape)}")
        core_query = self.bus.forward_up(z_jepa_pooled.unsqueeze(1)).squeeze(1)
        facts, idx = self.reader(core_query)
        self._cache = facts.detach()
        self._last_idx = idx.detach()
        return self._cache

    def cached_facts(self) -> Optional[torch.Tensor]:
        return self._cache

    def cached_indices(self) -> Optional[torch.Tensor]:
        return self._last_idx
