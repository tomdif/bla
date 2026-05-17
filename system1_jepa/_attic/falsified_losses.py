"""Falsified identity losses from the Phase 7C and Phase 8A sweeps.

These functions were active during the Phase 7B → 8A interventions on
content-based slot identity. **All falsified.**

  - `identity_consistency_loss`: Phase 7C — encoder identity consistency
    via Hungarian-matched same-entity slot pairs. Reduced id_drift but
    did not move full-slot switch rate. See
    `docs/phases/PHASE_7C_JEPA_DECISION.md`.

  - `identity_contrastive_loss`: Phase 8A — InfoNCE on the id-half of
    slot states. At λ=0.1 no effect; at λ ∈ {0.3, 1.0, 3.0} collapsed
    the id subspace, trading off content for stability. See
    `docs/phases/PHASE_8A_JEPA_DECISION.md`.

The canonical OF-JEPA v0 architecture (Phase 8C → 9B) uses NEITHER of
these. Identity stability is provided structurally by persistent
learned `id_proto` + Sinkhorn matching, not by content-side loss
terms. See `feedback-identity-as-address` and
`feedback-prediction-vs-assignment` in the memory layer.

Kept here so the falsification sweeps remain reproducible and so a
future regime change (e.g. true streaming data) can resurrect them
with full context.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from system1_jepa.id_consistency import SlotPosAuxHead, _assign_slots_to_entities


def identity_consistency_loss(
    slot_states: torch.Tensor,         # [T, S, D]
    slot_to_pos_aux: SlotPosAuxHead,
    gt_pos: torch.Tensor,              # [T, E, 2]
    gt_visible: torch.Tensor,          # [T, E] bool
    id_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[FALSIFIED — Phase 7C] Returns (consistency_loss, aux_pos_loss).

    Pushes slot[i_t][:id_dim] toward sg(slot[i_{t-1}][:id_dim]) for
    consecutive frames where the same entity was visible at both.
    Reduced id_drift; did not move full-slot switch rate.
    """
    T, S, D = slot_states.shape
    pred_pos = slot_to_pos_aux(slot_states)

    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)

    aux_terms = []
    for t in range(T):
        for s in range(S):
            e = assignments[t, s]
            if e >= 0:
                aux_terms.append(F.mse_loss(pred_pos[t, s], gt_pos[t, e]))
    aux_loss = torch.stack(aux_terms).mean() if aux_terms else \
                torch.tensor(0.0, device=slot_states.device)

    cons_terms = []
    for t in range(1, T):
        for s_t in range(S):
            e = assignments[t, s_t]
            if e < 0:
                continue
            prev_rows = np.where(assignments[t-1] == e)[0]
            if prev_rows.size == 0:
                continue
            s_prev = int(prev_rows[0])
            id_cur = slot_states[t, s_t, :id_dim]
            id_prev = slot_states[t-1, s_prev, :id_dim].detach()
            cons_terms.append(F.mse_loss(id_cur, id_prev))
    cons_loss = torch.stack(cons_terms).mean() if cons_terms else \
                 torch.tensor(0.0, device=slot_states.device)

    return cons_loss, aux_loss


def identity_contrastive_loss(
    slot_states: torch.Tensor,
    slot_to_pos_aux: SlotPosAuxHead,
    gt_pos: torch.Tensor,
    gt_visible: torch.Tensor,
    id_dim: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """[FALSIFIED — Phase 8A] InfoNCE on id-half of slot states.

    Pulls same-entity slot id_keys close across frames, pushes
    different-entity ones apart. At any λ that actually moves switch
    rate, collapses id_keys to a constant — content fails because the
    representation has no degree of freedom left for state.
    """
    T, S, D = slot_states.shape
    id_keys = slot_states[..., :id_dim]
    pred_pos = slot_to_pos_aux(slot_states)

    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)

    id_norm = F.normalize(id_keys, dim=-1, eps=1e-6)

    loss_terms = []
    for t in range(T - 1):
        for i_anchor in range(S):
            e = assignments[t, i_anchor]
            if e < 0:
                continue
            pos_rows = np.where(assignments[t + 1] == e)[0]
            if pos_rows.size == 0:
                continue
            j_pos = int(pos_rows[0])

            anchor = id_norm[t, i_anchor]
            candidates = id_norm[t + 1]
            sims = (candidates @ anchor) / temperature
            log_probs = F.log_softmax(sims, dim=0)
            loss_terms.append(-log_probs[j_pos])

    if not loss_terms:
        return torch.tensor(0.0, device=slot_states.device, requires_grad=True)
    return torch.stack(loss_terms).mean()
