"""RAM attention diagnostics: which entries does the RAMReader pick, and how peaked is the distribution?"""

from __future__ import annotations

from typing import Dict

import torch


def dump_ram_attention(ram_reader, query: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return a snapshot of the reader's attention weights and selected indices.

    Useful for verifying that:
      * the reader is committing (low entropy weights), not hedging
      * the reader is learning to specialize per query head
      * the RAM contents are being used (entries appear across diverse queries)
    """

    ram_reader.eval()
    with torch.no_grad():
        _, aux = ram_reader(query, return_aux=True)
    weights = aux["weights"]
    top_idx = aux["top_idx"]
    entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1)
    return {
        "weights": weights.cpu(),
        "top_idx": top_idx.cpu(),
        "entropy_per_head": entropy.cpu(),
        "mean_entropy": entropy.mean().cpu(),
        "max_possible_entropy": torch.log(torch.tensor(float(weights.shape[-1]))),
    }
