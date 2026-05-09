from __future__ import annotations

import torch
from torch import nn

from tensor_ram.torch_ram import DifferentiableTensorRAM


class RAMReader(nn.Module):
    """Generates differentiable READ queries from the working state.

    The reader projects an internal query vector into RAM-key space, performs
    soft top-k retrieval against a frozen DifferentiableTensorRAM, and projects
    the weighted result back into the working space.

    Sparsity:
      * `sparsity_weight > 0` returns an additional entropy-of-weights penalty
        in the call output, which the trainer can add to the main loss to push
        the reader toward committing to fewer entries (lower entropy = more
        committed; bound is 0 = one-hot, max = log(top_k) = uniform).
      * `hard=True` makes the forward pass return a straight-through one-hot
        retrieval at the top-1 entry. Gradient flows through the soft weights
        but the forward returns the single picked vector — the cleanest way to
        honor "physical separation of facts and logic" once you trust the
        reader's selection.
    """

    def __init__(
        self,
        d_state: int,
        ram: DifferentiableTensorRAM,
        num_query_heads: int = 4,
        top_k: int = 8,
        temperature: float = 1.0,
        sparsity_weight: float = 0.0,
        hard: bool = False,
    ):
        super().__init__()
        self.ram = ram
        self.num_query_heads = num_query_heads
        self.top_k = top_k
        self.temperature = temperature
        self.sparsity_weight = sparsity_weight
        self.hard = hard
        self.query_proj = nn.Linear(d_state, ram.d_ram * num_query_heads)
        self.value_proj = nn.Linear(ram.d_ram, d_state)
        self.norm = nn.LayerNorm(d_state)

    def _retrieve_with_diagnostics(
        self, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (weighted_facts, top_idx, weights). weights has shape [..., top_k]."""

        if self.ram.ntotal == 0:
            raise RuntimeError("RAM is empty")
        keys = self.ram.keys.to(dtype=queries.dtype)
        scores = torch.einsum("...d,nd->...n", queries, keys)
        k = min(self.top_k, self.ram.ntotal)
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        weights = torch.softmax(top_scores / max(self.temperature, 1e-6), dim=-1)

        if self.hard:
            argmax = weights.argmax(dim=-1, keepdim=True)
            hard_w = torch.zeros_like(weights).scatter_(-1, argmax, 1.0)
            weights = hard_w + (weights - weights.detach())

        gathered = keys[top_idx]
        weighted = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
        return weighted, top_idx, weights

    def forward(
        self,
        state: torch.Tensor,
        return_aux: bool = False,
    ):
        batch = state.shape[0]
        queries = self.query_proj(state).view(batch, self.num_query_heads, self.ram.d_ram)
        weighted, top_idx, weights = self._retrieve_with_diagnostics(queries)
        facts = self.norm(self.value_proj(weighted))

        if not return_aux and self.sparsity_weight == 0.0:
            return facts, top_idx.detach()

        aux = {
            "weights": weights,
            "top_idx": top_idx.detach(),
        }
        if self.sparsity_weight > 0.0:
            entropy = -(weights * (weights.clamp_min(1e-9)).log()).sum(dim=-1).mean()
            aux["sparsity_loss"] = self.sparsity_weight * entropy

        if return_aux:
            return facts, aux
        return facts, top_idx.detach()
