"""Persistent object-file memory bank — the canonical OF-JEPA v0 primitive.

This module contains the architectural break that ended the Phase 7-8
falsification arc:

  - SlotAttention's exchangeable per-episode slot init was replaced by
    `id_proto`, a learned parameter shared across all training. Each
    memory cell HAS a persistent identity (its row in `id_proto`).
  - Observation queries are matched to memory via Sinkhorn assignment
    (see `assignment.sinkhorn`), not the reverse.
  - Different update rules per role: `id_key` evolves slowly via EMA
    toward the matched proposal's id projection; `state_value` evolves
    via sparse-delta (`state + change_mask * tanh(delta)`).
  - Inter-frame LayerNorm on both halves prevents the additive
    recurrence from blowing up — Phase 7B lesson that hit OF-JEPA v0
    too on first attempt.

The locked canonical class is `ObjectFileMemory` (v0).

`ObjectFileMemoryV1` adds null-Sinkhorn + transition + visibility
heads for the Phase 9B visibility-gated experiment. **It is currently
falsified** — Phase 9B showed it regresses every identity-conditioned
metric while only improving the (gameable) anonymous Hungarian one.
Kept here for reproducibility and for the case where true streaming
data (object birth/death, deep occlusion) flips the verdict.

Memory entries: [[feedback-identity-as-address]],
[[feedback-prediction-vs-assignment]],
[[feedback-slot-persistence-layernorm]].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assignment import sinkhorn


@dataclass
class OFJEPAConfig:
    """Architectural hyperparameters for OF-JEPA v0/v1."""
    n_files: int = 12
    id_dim: int = 64
    state_dim: int = 64
    proposal_dim: int = 128       # per-token feature dim from encoder
    n_proposals: int = 64          # token grid resolution from encoder
    id_ema_alpha: float = 0.05      # slow id_key update step
    state_delta_scale: float = 0.2  # bounded sparse delta on state
    sinkhorn_iters: int = 20
    sinkhorn_temperature: float = 0.1


class ObjectFileMemory(nn.Module):
    """Canonical OF-JEPA v0 object-file memory bank.

    The id_key is a *learned parameter* shared across all episodes —
    it's the persistent "address" each memory cell uses to query the
    frame. The state_value is per-batch-episode dynamic content
    initialized to zero at episode start.

    This is the architectural break from SlotAttention: slots aren't
    initialized from a learned distribution per episode; they ARE the
    learned distribution. The model has N_files persistent memory
    cells whose identities are baked into the weights.
    """

    def __init__(self, cfg: OFJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.id_proto = nn.Parameter(torch.randn(cfg.n_files, cfg.id_dim) * 0.5)
        self.proposal_id_proj = nn.Linear(cfg.proposal_dim, cfg.id_dim)
        self.proposal_state_proj = nn.Linear(cfg.proposal_dim, cfg.state_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.proposal_dim, cfg.state_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.state_dim * 2, cfg.state_dim),
        )
        self.change_head = nn.Linear(cfg.state_dim + cfg.proposal_dim, 1)
        from system1_jepa.id_consistency import SlotPosAuxHead
        self.slot_to_pos_aux = SlotPosAuxHead(cfg.id_dim + cfg.state_dim)

    def init_episode(self, batch_size: int, device: torch.device) -> dict:
        """Returns the initial per-episode dynamic state. id_key starts
        at the persistent prototype; state_value at zero."""
        N = self.cfg.n_files
        return {
            "id_key": self.id_proto.expand(batch_size, -1, -1).to(device),
            "state_value": torch.zeros(batch_size, N, self.cfg.state_dim, device=device),
        }

    def step(self, memory: dict, proposals: torch.Tensor) -> Tuple[dict, dict]:
        """One frame's update.

        memory: dict with id_key [B, N, id_dim] and state_value [B, N, state_dim]
        proposals: [B, n_props, proposal_dim]

        Returns (new_memory, diagnostics) where diagnostics includes the
        Sinkhorn assignment matrix [B, N, n_props].
        """
        cfg = self.cfg
        id_key = memory["id_key"]
        state = memory["state_value"]

        prop_id = self.proposal_id_proj(proposals)

        id_norm = F.normalize(id_key, dim=-1, eps=1e-6)
        prop_norm = F.normalize(prop_id, dim=-1, eps=1e-6)
        sim = torch.einsum("bnd,bmd->bnm", id_norm, prop_norm)
        log_scores = sim / cfg.sinkhorn_temperature

        assignment = sinkhorn(log_scores, n_iters=cfg.sinkhorn_iters)

        matched_proposal = torch.einsum("bnm,bmd->bnd", assignment, proposals)

        # id_key: slow EMA, LayerNormed on both ends to bound the recurrence.
        matched_id = F.layer_norm(
            self.proposal_id_proj(matched_proposal),
            (cfg.id_dim,),
        )
        id_key_new = id_key + cfg.id_ema_alpha * (matched_id - id_key)
        id_key_new = F.layer_norm(id_key_new, (cfg.id_dim,))

        # state_value: sparse delta gated by change_head.
        delta_in = torch.cat([state, matched_proposal], dim=-1)
        change_mask = torch.sigmoid(self.change_head(delta_in))
        delta = cfg.state_delta_scale * torch.tanh(self.delta_head(delta_in))
        state_new = state + change_mask * delta
        state_new = F.layer_norm(state_new, (cfg.state_dim,))

        return (
            {"id_key": id_key_new, "state_value": state_new},
            {"assignment": assignment, "change_mask": change_mask},
        )


class ObjectFileMemoryV1(ObjectFileMemory):
    """[FALSIFIED — Phase 9B] Visibility-gated OF-JEPA.

    Adds three pieces beyond v0:
      1. NULL column in Sinkhorn so files can "no match" → low
         match_confidence → transition update instead of delta-from-
         proposal. Prevents the (apparent) v0 failure mode where
         occluded files bind to arbitrary visible proposals.
      2. transition_model MLP for occluded-state evolution.
      3. visibility_head logit, supervised against GT visibility.

    Phase 9B finding: under the **right** metric
    (identity-conditioned hidden MSE, not anonymous Hungarian), v0 was
    already passing. v1's added complexity regressed every meaningful
    metric while only improving the gameable anonymous one. Kept here
    for reproducibility and possible revival under true streaming /
    birth-death data regimes.

    See `docs/phases/PHASE_9B_JEPA_DECISION.md` and the memory entry
    [[feedback-identity-conditioned-metrics]].
    """

    def __init__(self, cfg: OFJEPAConfig):
        super().__init__(cfg)
        self.null_bias = nn.Parameter(torch.zeros(1))
        self.transition = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.state_dim * 2, cfg.state_dim),
        )
        self.visibility_head = nn.Linear(cfg.state_dim, 1)

    def step(self, memory: dict, proposals: torch.Tensor):
        cfg = self.cfg
        id_key = memory["id_key"]
        state = memory["state_value"]
        B, N_files, _ = id_key.shape
        _, n_props, _ = proposals.shape

        prop_id = self.proposal_id_proj(proposals)

        id_norm = F.normalize(id_key, dim=-1, eps=1e-6)
        prop_norm = F.normalize(prop_id, dim=-1, eps=1e-6)
        sim = torch.einsum("bnd,bmd->bnm", id_norm, prop_norm)
        log_scores_props = sim / cfg.sinkhorn_temperature

        null_score = self.null_bias.view(1, 1, 1).expand(B, N_files, 1)
        log_scores = torch.cat([log_scores_props, null_score], dim=-1)

        assignment = sinkhorn(log_scores, n_iters=cfg.sinkhorn_iters)

        prop_assign = assignment[..., :n_props]
        match_confidence = prop_assign.sum(dim=-1, keepdim=True)

        prop_assign_renorm = prop_assign / (match_confidence + 1e-6)
        matched_proposal = torch.einsum("bnm,bmd->bnd", prop_assign_renorm, proposals)

        matched_id = F.layer_norm(
            self.proposal_id_proj(matched_proposal),
            (cfg.id_dim,),
        )
        id_step = match_confidence * cfg.id_ema_alpha * (matched_id - id_key)
        id_key_new = F.layer_norm(id_key + id_step, (cfg.id_dim,))

        delta_in = torch.cat([state, matched_proposal], dim=-1)
        change_mask = torch.sigmoid(self.change_head(delta_in))
        delta = cfg.state_delta_scale * torch.tanh(self.delta_head(delta_in))
        matched_state = state + change_mask * delta
        transition_state = state + 0.05 * self.transition(state)

        state_new = match_confidence * matched_state + (1 - match_confidence) * transition_state
        state_new = F.layer_norm(state_new, (cfg.state_dim,))

        visibility_logit = self.visibility_head(state_new).squeeze(-1)

        return (
            {"id_key": id_key_new, "state_value": state_new},
            {
                "assignment": assignment,
                "match_confidence": match_confidence,
                "change_mask": change_mask,
                "visibility_logit": visibility_logit,
            },
        )
