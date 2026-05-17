"""Object-File JEPA (Phase 8C) — predictive object-file memory.

The Phase 7+8 falsification arc proved that identity-stable object
binding cannot be extracted from a single content-trained slot vector.
The fix is to decompose slot into:

    object_file = (id_key, state_value, [appearance, visibility, ...])

with very different update rules per component:

    id_key:       slow EMA, acts as a persistent ADDRESS
    state_value:  sparse delta, acts as dynamic CONTENT
    appearance:   EMA only when visible (v1)
    visibility:   updated every frame (v1)

And the binding mechanism is *differentiable assignment* (Sinkhorn)
between persistent memory cells and frame proposals — not random
exchangeable SlotAttention slots.

OF-JEPA v0 (this file): persistent id_keys + memory-anchored
cross-attention + Sinkhorn matching + sparse-delta state_value
updates. Fixed N_files, all active. JEPA loss on state_value +
cross-entropy on assignment vs MOVi GT.

OF-JEPA v1 (deferred to next file): adds spawn/retire lifecycle,
appearance head, visibility belief.

Memory entries motivating this design:
[[feedback-prediction-vs-assignment]], [[feedback-joint-metric-vs-single-axis]],
[[feedback-slot-persistence-layernorm]].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OFJEPAConfig:
    n_files: int = 12
    id_dim: int = 64
    state_dim: int = 64
    proposal_dim: int = 128       # per-token feature dim from encoder
    n_proposals: int = 64          # token grid resolution from encoder
    id_ema_alpha: float = 0.05      # slow id_key update step
    state_delta_scale: float = 0.2  # bounded sparse delta on state
    sinkhorn_iters: int = 20
    sinkhorn_temperature: float = 0.1


def sinkhorn(log_scores: torch.Tensor, n_iters: int = 20,
              eps: float = 1e-9) -> torch.Tensor:
    """Differentiable Sinkhorn normalization to a doubly-stochastic matrix.

    log_scores: [B, N, M] (log-domain similarity scores)
    Returns: [B, N, M] non-negative entries summing to 1 along both
    rows and columns (approximately).
    """
    log_p = log_scores
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


class ProposalEncoder(nn.Module):
    """Wraps ConvNeXt-T into a per-frame proposal feature extractor.

    Output: [B, N_props, proposal_dim] — the patch-token grid where
    each token carries both appearance and approximate spatial
    location information (the spatial pos-embed in the encoder lets
    the matcher use position implicitly).
    """
    def __init__(self, input_size: int, proposal_dim: int):
        super().__init__()
        from system1_jepa.convnext_encoder import (
            ConvNeXtEncoderConfig, ConvNeXtSlotEncoder,
        )
        self.encoder = ConvNeXtSlotEncoder(ConvNeXtEncoderConfig(
            input_size=input_size, slot_dim=proposal_dim,
            pretrained=False, freeze_early_stages=0,
        ))
        self.n_tokens = self.encoder.n_tokens

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: [T, 3, H, W] → [T, n_tokens, proposal_dim]"""
        return self.encoder(video)


class ObjectFileMemory(nn.Module):
    """Persistent object-file memory bank.

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
        # Persistent prototype id_keys: learned across all training.
        self.id_proto = nn.Parameter(torch.randn(cfg.n_files, cfg.id_dim) * 0.5)
        # Aux head: project proposal features into id-key space for matching.
        self.proposal_id_proj = nn.Linear(cfg.proposal_dim, cfg.id_dim)
        # Aux head: project proposal features into state-value space.
        self.proposal_state_proj = nn.Linear(cfg.proposal_dim, cfg.state_dim)
        # Delta head: predict state delta from (current state, matched proposal).
        self.delta_head = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.proposal_dim, cfg.state_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.state_dim * 2, cfg.state_dim),
        )
        # Change-mask head: per-file scalar gating the delta update.
        self.change_head = nn.Linear(cfg.state_dim + cfg.proposal_dim, 1)
        # Aux slot→pos for position eval probe (compatibility with existing tooling).
        from system1_jepa.id_consistency import SlotPosAuxHead
        self.slot_to_pos_aux = SlotPosAuxHead(cfg.id_dim + cfg.state_dim)

    def init_episode(self, batch_size: int, device: torch.device) -> dict:
        """Returns the initial per-episode dynamic state. id_key starts
        at the persistent prototype; state_value at zero."""
        N = self.cfg.n_files
        return {
            "id_key": self.id_proto.expand(batch_size, -1, -1).to(device),     # [B, N, id_dim]
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
        id_key = memory["id_key"]                         # [B, N, id_dim]
        state = memory["state_value"]                     # [B, N, state_dim]

        # Project proposals into id-key space.
        prop_id = self.proposal_id_proj(proposals)        # [B, n_props, id_dim]

        # Cosine-similarity matching matrix between id_keys and projected proposals.
        id_norm = F.normalize(id_key, dim=-1, eps=1e-6)
        prop_norm = F.normalize(prop_id, dim=-1, eps=1e-6)
        sim = torch.einsum("bnd,bmd->bnm", id_norm, prop_norm)  # [B, N, n_props]
        log_scores = sim / cfg.sinkhorn_temperature

        # Differentiable Sinkhorn assignment.
        assignment = sinkhorn(log_scores, n_iters=cfg.sinkhorn_iters)  # [B, N, n_props]

        # For each file, the "matched proposal" is the assignment-weighted
        # combination of proposals — soft binding.
        matched_proposal = torch.einsum("bnm,bmd->bnd", assignment, proposals)  # [B, N, proposal_dim]

        # Update id_key: slow EMA toward matched proposal's id projection.
        # LayerNorm the projection before EMA so the matched_id magnitude
        # doesn't blow up the recurrence across 24 frames × N training steps
        # (Phase 7B lesson; failed in Phase 8C v0 first attempt).
        matched_id = F.layer_norm(
            self.proposal_id_proj(matched_proposal),
            (cfg.id_dim,),
        )                                                              # [B, N, id_dim]
        id_key_new = id_key + cfg.id_ema_alpha * (matched_id - id_key)
        # Post-EMA LayerNorm to absolutely bound id_key magnitude.
        id_key_new = F.layer_norm(id_key_new, (cfg.id_dim,))

        # Update state_value: sparse delta gated by change_head.
        delta_in = torch.cat([state, matched_proposal], dim=-1)       # [B, N, state_dim+proposal_dim]
        change_mask = torch.sigmoid(self.change_head(delta_in))        # [B, N, 1]
        delta = cfg.state_delta_scale * torch.tanh(self.delta_head(delta_in))
        state_new = state + change_mask * delta
        # Also LayerNorm state_value to prevent the same recurrence blowup.
        state_new = F.layer_norm(state_new, (cfg.state_dim,))

        return (
            {"id_key": id_key_new, "state_value": state_new},
            {"assignment": assignment, "change_mask": change_mask},
        )


class ObjectFileMemoryV1(ObjectFileMemory):
    """Phase 9B: visibility-gated OF-JEPA.

    Adds three pieces beyond v0:
      1. NULL column in Sinkhorn. Each file can choose "no match this
         frame" → low match_confidence → transition update instead of
         delta-from-proposal. Prevents the v0 failure mode where
         occluded files get bound to arbitrary visible proposals and
         their state_value gets corrupted.
      2. transition_model MLP. When a file's match_confidence is low
         (entity occluded), state_value evolves via this learned
         transition rather than from the proposal.
      3. visibility_head. Per-file logit predicting whether the entity
         is currently visible. Trained against GT visibility.

    The state update is a soft blend between "delta from proposal"
    and "transition only", weighted by match_confidence.
    """

    def __init__(self, cfg: OFJEPAConfig):
        super().__init__(cfg)
        # Learnable null-column bias used by Sinkhorn so files can "no match".
        self.null_bias = nn.Parameter(torch.zeros(1))
        # Transition model for occluded-entity state evolution.
        self.transition = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.state_dim * 2, cfg.state_dim),
        )
        # Visibility belief head.
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

        # Append null column: a constant score the model can pick when no
        # proposal is good enough. The null score is learnable globally —
        # this is the "should this slot just predict forward via transition?"
        # threshold.
        null_score = self.null_bias.view(1, 1, 1).expand(B, N_files, 1)
        log_scores = torch.cat([log_scores_props, null_score], dim=-1)  # [B, N, n_props+1]

        # Sinkhorn over the augmented matrix.
        assignment = sinkhorn(log_scores, n_iters=cfg.sinkhorn_iters)  # [B, N, n_props+1]

        # Split: assignment to real proposals vs to null column.
        prop_assign = assignment[..., :n_props]                   # [B, N, n_props]
        match_confidence = prop_assign.sum(dim=-1, keepdim=True)  # [B, N, 1] in [0, 1]

        # Weighted matched proposal (only from real proposals).
        # Renormalize prop_assign per file to make it a proper distribution
        # over proposals conditional on "did match" — guards against tiny
        # match_confidence corrupting the matched_proposal direction.
        prop_assign_renorm = prop_assign / (match_confidence + 1e-6)
        matched_proposal = torch.einsum("bnm,bmd->bnd", prop_assign_renorm, proposals)

        # id_key update: slow EMA toward matched proposal's id projection,
        # GATED by match_confidence. Occluded files (low confidence) keep
        # their address frozen — they were the LAST seen identity.
        matched_id = F.layer_norm(
            self.proposal_id_proj(matched_proposal),
            (cfg.id_dim,),
        )
        id_step = match_confidence * cfg.id_ema_alpha * (matched_id - id_key)
        id_key_new = F.layer_norm(id_key + id_step, (cfg.id_dim,))

        # State update: blend delta-from-proposal (when matched) with
        # transition-only (when occluded), weighted by match_confidence.
        delta_in = torch.cat([state, matched_proposal], dim=-1)
        change_mask = torch.sigmoid(self.change_head(delta_in))
        delta = cfg.state_delta_scale * torch.tanh(self.delta_head(delta_in))
        matched_state = state + change_mask * delta
        transition_state = state + 0.05 * self.transition(state)

        state_new = match_confidence * matched_state + (1 - match_confidence) * transition_state
        state_new = F.layer_norm(state_new, (cfg.state_dim,))

        # Visibility logit (predicted from state).
        visibility_logit = self.visibility_head(state_new).squeeze(-1)  # [B, N]

        return (
            {"id_key": id_key_new, "state_value": state_new},
            {
                "assignment": assignment,
                "match_confidence": match_confidence,
                "change_mask": change_mask,
                "visibility_logit": visibility_logit,
            },
        )


class OFJEPA(nn.Module):
    """Object-File JEPA wrapper for MOVi-A training.

    Pulls per-frame proposals from the encoder, runs the object-file
    memory step-by-step across all T frames of an episode, and exposes:

        slot_states_per_frame: [T, B, N_files, id_dim + state_dim]
        assignments_per_frame: [T, B, N_files, n_proposals]

    The "full slot state" returned for compatibility with the existing
    identity probe is concat(id_key, state_value).
    """
    def __init__(self, image_size: int, cfg: OFJEPAConfig = OFJEPAConfig(),
                 version: str = "v0"):
        super().__init__()
        self.cfg = cfg
        self.version = version
        self.proposal_encoder = ProposalEncoder(image_size, cfg.proposal_dim)
        # v0 uses ObjectFileMemory; v1 adds null-Sinkhorn + transition + visibility.
        if version == "v1":
            self.memory = ObjectFileMemoryV1(cfg)
        else:
            self.memory = ObjectFileMemory(cfg)
        # Compatibility shim with existing trainer:
        self.id_dim = cfg.id_dim
        self.use_id_cons = False
        self.use_id_contrast = False
        self.mode = f"of_jepa_{version}"
        # Slot dim that the existing probe expects = id_dim + state_dim
        self.slot_dim = cfg.id_dim + cfg.state_dim
        self.n_slots = cfg.n_files
        self.slot_to_pos_aux = self.memory.slot_to_pos_aux

    def encode_video(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """video: [T, 3, H, W] → (slot_states [T, n_files, full_dim], proposals [T, n_props, prop_dim])

        For v1, also tracks per-frame visibility logits in self._last_vis_logits
        for the trainer to read out.
        """
        T = video.shape[0]
        proposals = self.proposal_encoder(video)
        memory = self.memory.init_episode(batch_size=1, device=video.device)
        slot_states = []
        vis_logits = []
        for t in range(T):
            memory, diag = self.memory.step(memory, proposals[t:t+1])
            full = torch.cat([memory["id_key"], memory["state_value"]], dim=-1)
            slot_states.append(full[0])
            if "visibility_logit" in diag:
                vis_logits.append(diag["visibility_logit"][0])
        self._last_vis_logits = torch.stack(vis_logits, dim=0) if vis_logits else None
        return torch.stack(slot_states, dim=0), proposals

    def encode_video_grad(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video)
