"""Slot Attention (Locatello et al. 2020) — untyped slot binding from tokens.

Used as the bridge between dense JEPA features and a persistent commitment
ledger. Iterates K rounds of competitive cross-attention so each slot
binds to a different region of the input.

Why softmax-over-slots: standard self-attention is softmax-over-keys
(each query picks among inputs). Slot Attention softmaxes over slots so
each input is forced to *choose one slot*, which creates competition and
prevents all slots from collapsing onto the same dominant input. The
subsequent re-normalization over inputs turns the binding into an
input-side affinity that we use as the cross-attention weight.

The first prototype keeps slots untyped (no labels, no affordance heads).
The question is whether competition + iteration alone produces stable
binding under occlusion. Typed heads can be added later if the untyped
slots prove insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SlotAttentionConfig:
    n_slots: int = 16
    slot_dim: int = 64
    n_iters: int = 3
    mlp_ratio: int = 4
    eps: float = 1e-8


class SlotAttention(nn.Module):
    def __init__(self, input_dim: int, cfg: SlotAttentionConfig | None = None):
        super().__init__()
        self.cfg = cfg or SlotAttentionConfig()
        self.input_dim = input_dim
        d = self.cfg.slot_dim
        # Learned initialisation: per-slot mean and log_sigma. Each slot
        # is sampled from its own Gaussian at the start of an episode so
        # the slots are non-degenerate from step 0.
        self.slots_mu = nn.Parameter(torch.zeros(1, self.cfg.n_slots, d))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, self.cfg.n_slots, d))
        nn.init.trunc_normal_(self.slots_mu, std=0.02)
        nn.init.trunc_normal_(self.slots_log_sigma, std=0.02)

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(d)
        self.norm_pre_mlp = nn.LayerNorm(d)

        self.proj_k = nn.Linear(input_dim, d, bias=False)
        self.proj_v = nn.Linear(input_dim, d, bias=False)
        self.proj_q = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        hidden = d * self.cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def init_slots(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        """Sample fresh slots from the learned per-slot priors. Used at the
        start of an episode (or whenever caller wants a hard reset)."""
        mu = self.slots_mu.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        log_sigma = self.slots_log_sigma.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        sigma = log_sigma.exp()
        noise = torch.randn_like(mu)
        return mu + sigma * noise

    def forward(self, inputs: torch.Tensor,
                init_slots: torch.Tensor | None = None,
                return_attention: bool = False):
        """Bind `inputs` [B, N, input_dim] into `n_slots` [B, S, slot_dim].

        `init_slots` lets the caller seed the iteration from a previous
        timestep's slot state instead of fresh samples — that's how the
        slot system carries memory across frames.

        If `return_attention`, also returns the final-iteration `attn`
        tensor of shape [B, N, S] — the per-input slot competition
        distribution. `attn.sum(dim=1)` gives per-slot binding mass and
        is the source of truth for whether a slot has anything to bind
        to in the current observation.
        """
        b, _, _ = inputs.shape
        d = self.cfg.slot_dim
        inputs_n = self.norm_inputs(inputs)
        k = self.proj_k(inputs_n)
        v = self.proj_v(inputs_n)

        if init_slots is None:
            slots = self.init_slots(b, inputs.device, inputs.dtype)
        else:
            slots = init_slots

        scale = d ** -0.5
        attn = None
        for _ in range(self.cfg.n_iters):
            slots_prev = slots
            q = self.proj_q(self.norm_slots(slots))
            # Slot-side softmax → competition. Each input row gets a
            # distribution over slots that sums to 1, so the slots
            # compete for input tokens.
            attn_logits = torch.einsum("bsd,bnd->bns", q, k) * scale
            attn = attn_logits.softmax(dim=-1)
            # Re-normalise over inputs so the cross-attention weights
            # sum to 1 per slot (now reading from inputs).
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + self.cfg.eps)
            updates = torch.einsum("bns,bnd->bsd", attn_norm, v)
            # GRU-style refinement; flat over (B*S) for the cell.
            slots = self.gru(
                updates.reshape(b * self.cfg.n_slots, d),
                slots_prev.reshape(b * self.cfg.n_slots, d),
            ).reshape(b, self.cfg.n_slots, d)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
        if return_attention:
            return slots, attn
        return slots
