from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchMask:
    """Random patch mask shared across the batch.

    `context_idx` and `target_idx` are disjoint long tensors of patch indices
    into the flat [grid_h * grid_w] sequence.
    """

    context_idx: torch.Tensor
    target_idx: torch.Tensor
    grid_h: int
    grid_w: int

    @property
    def num_context(self) -> int:
        return int(self.context_idx.numel())

    @property
    def num_target(self) -> int:
        return int(self.target_idx.numel())


def sample_patch_mask(
    grid_h: int,
    grid_w: int,
    target_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> PatchMask:
    """Sample a random patch partition into context vs target indices.

    target_ratio is the fraction of patches reserved as prediction targets.
    The split is shared across the batch so all sequences have the same length.
    """

    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")
    n = grid_h * grid_w
    perm = torch.randperm(n, device=device, generator=generator)
    n_target = max(1, min(n - 1, int(round(n * target_ratio))))
    target_idx = perm[:n_target].sort().values
    context_idx = perm[n_target:].sort().values
    return PatchMask(
        context_idx=context_idx,
        target_idx=target_idx,
        grid_h=grid_h,
        grid_w=grid_w,
    )


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather a subset of tokens along the sequence axis.

    tokens: [B, N, D]
    idx:    [K] long
    returns [B, K, D]
    """

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, D], got {tuple(tokens.shape)}")
    expanded = idx.view(1, -1, 1).expand(tokens.shape[0], -1, tokens.shape[-1])
    return tokens.gather(1, expanded)
