"""Encoder-side identity consistency loss (Phase 7C).

Phase 7B established that the predictor-side id_dyn_split fix alone is
insufficient: the encoder still emits identity-subspace features that
drift frame-to-frame, and the JEPA loss forces the predictor to chase
that drift. The identity problem is **upstream** in the encoder.

This module supplies an auxiliary training-time loss that directly
penalizes encoder-output identity drift. Sketch:

    1. Tiny auxiliary head `slot_to_pos_aux(slot) -> [x, y]` trained
       alongside JEPA by MSE on GT positions for visible entities.
    2. Per frame, Hungarian-match `slot_to_pos_aux(slots[t])` against
       GT entity positions → assignment a_t: slot_idx → entity_id.
    3. For each pair of consecutive frames and each entity visible at
       both, find the slot bound to that entity at t-1 (i_{t-1}) and
       at t (i_t). The identity loss:

           L_id = mean ||slot_id_half(t)[i_t]
                          - stopgrad(slot_id_half(t-1)[i_{t-1}])||²

       The stop-grad anchors the *current* slot to the *previous*
       slot's identity coordinate — the encoder is pushed to make
       identity-subspace features temporally stable across frames
       for the same entity.

This is GT-supervised at training time (uses MOVi's per-entity GT
positions to drive the Hungarian match). At eval time the assignment
is recomputed from the regular identity probe — no leak.

The id_drift / dyn_drift diagnostic is also defined here so we can
verify the split is actually doing what we want.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa.identity_probe import hungarian_assign


class SlotPosAuxHead(nn.Module):
    """slot -> (x, y) auxiliary head used to drive per-frame Hungarian
    matching at training time. Tiny — single Linear."""

    def __init__(self, slot_dim: int):
        super().__init__()
        self.head = nn.Linear(slot_dim, 2)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.head(slots)


def _assign_slots_to_entities(
    pred_pos: torch.Tensor,   # [T, S, 2]
    gt_pos: torch.Tensor,     # [T, E, 2]
    gt_visible: torch.Tensor, # [T, E] bool
) -> np.ndarray:
    """Returns assignments[T, S] where assignments[t, s] = matched entity_id
    (in [0, E)) or -1 if unassigned / matched-to-invisible-entity.
    """
    T, S, _ = pred_pos.shape
    _, E, _ = gt_pos.shape
    pred_np = pred_pos.detach().cpu().numpy()
    gt_np = gt_pos.detach().cpu().numpy()
    vis_np = gt_visible.detach().cpu().numpy()

    out = -np.ones((T, S), dtype=np.int64)
    for t in range(T):
        rows, cols, _ = hungarian_assign(pred_np[t], gt_np[t])
        for r, c in zip(rows, cols):
            if vis_np[t, c]:
                out[t, r] = int(c)
    return out


def identity_consistency_loss(
    slot_states: torch.Tensor,         # [T, S, D] — encoder output, with grads
    slot_to_pos_aux: SlotPosAuxHead,
    gt_pos: torch.Tensor,              # [T, E, 2]
    gt_visible: torch.Tensor,          # [T, E] bool
    id_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (consistency_loss, aux_pos_loss).

    consistency_loss: pushes slot[i_t][:id_dim] toward sg(slot[i_{t-1}][:id_dim])
    for pairs of consecutive frames where the same entity was visible at both.

    aux_pos_loss: MSE between slot_to_pos_aux(slot_visible) and gt_pos for
    visible-entity rows (trains the aux head used for matching).
    """
    T, S, D = slot_states.shape
    pred_pos = slot_to_pos_aux(slot_states)              # [T, S, 2]

    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)  # [T, S]

    # Aux-head loss: for each (t, s) with a visible-entity assignment,
    # MSE between pred_pos and gt_pos of that entity.
    aux_terms = []
    for t in range(T):
        for s in range(S):
            e = assignments[t, s]
            if e >= 0:
                aux_terms.append(F.mse_loss(pred_pos[t, s], gt_pos[t, e]))
    aux_loss = torch.stack(aux_terms).mean() if aux_terms else \
                torch.tensor(0.0, device=slot_states.device)

    # Consistency loss: for each (t > 0, slot s_t), find the slot s_{t-1}
    # assigned to the SAME entity. If found, accumulate id-half loss.
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
    slot_states: torch.Tensor,         # [T, S, D]
    slot_to_pos_aux: SlotPosAuxHead,
    gt_pos: torch.Tensor,              # [T, E, 2]
    gt_visible: torch.Tensor,          # [T, E] bool
    id_dim: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss on the id-half of slot states.

    For each consecutive frame pair (t, t+1) and each anchor slot i at t
    that is Hungarian-matched to some entity e (also visible at t+1):
      - positive: slot j at t+1 matched to the same entity e
      - negatives: all OTHER slots at t+1 (matched to other entities or
        unassigned)

    The loss is:
        -log( exp(sim(anchor, pos)/τ) /
              sum_{k in {pos, neg_1, ..., neg_K}} exp(sim(anchor, k)/τ) )

    Cosine similarity in the id-subspace. Returns mean over anchors.

    This is the Phase 8A intervention: directly reward the encoder for
    putting same-entity slot id_keys close to each other across frames,
    and different-entity ones far apart. Operates only on the id-half so
    we don't constrain the dyn-half's freedom to encode state changes.
    """
    T, S, D = slot_states.shape
    id_keys = slot_states[..., :id_dim]                  # [T, S, id_dim]
    pred_pos = slot_to_pos_aux(slot_states)               # [T, S, 2]

    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)  # [T, S]

    # Normalize id_keys for cosine similarity.
    id_norm = F.normalize(id_keys, dim=-1, eps=1e-6)

    loss_terms = []
    for t in range(T - 1):
        for i_anchor in range(S):
            e = assignments[t, i_anchor]
            if e < 0:
                continue
            # Positive: slot at t+1 with the same entity ID.
            pos_rows = np.where(assignments[t + 1] == e)[0]
            if pos_rows.size == 0:
                continue
            j_pos = int(pos_rows[0])

            anchor = id_norm[t, i_anchor]                  # [id_dim]
            candidates = id_norm[t + 1]                     # [S, id_dim]
            sims = (candidates @ anchor) / temperature      # [S]
            # InfoNCE: log p(positive | candidates).
            log_probs = F.log_softmax(sims, dim=0)
            loss_terms.append(-log_probs[j_pos])

    if not loss_terms:
        return torch.tensor(0.0, device=slot_states.device, requires_grad=True)
    return torch.stack(loss_terms).mean()


@torch.no_grad()
def cosine_diagnostic(
    slot_states: torch.Tensor,         # [T, S, D]
    slot_to_pos_aux: SlotPosAuxHead,
    gt_pos: torch.Tensor,              # [T, E, 2]
    gt_visible: torch.Tensor,          # [T, E] bool
    id_dim: int,
) -> dict:
    """For Phase 8A diagnostic: mean cosine similarity between id_keys of
    same-entity pairs (consecutive frames) and different-entity pairs.

    A working contrastive signal should show:
        same_object_cos ↑ (close to 1)
        diff_object_cos ↓ (close to 0 or negative)

    Returns {'same_cos': float, 'diff_cos': float, 'gap': float, 'n_pairs': int}.
    """
    T, S, D = slot_states.shape
    id_keys = slot_states[..., :id_dim]
    id_norm = F.normalize(id_keys, dim=-1, eps=1e-6)
    pred_pos = slot_to_pos_aux(slot_states)
    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)

    same_sims = []
    diff_sims = []
    for t in range(T - 1):
        a_t = assignments[t]
        a_n = assignments[t + 1]
        for i in range(S):
            e = a_t[i]
            if e < 0:
                continue
            for j in range(S):
                e2 = a_n[j]
                if e2 < 0:
                    continue
                sim = float((id_norm[t, i] * id_norm[t + 1, j]).sum().item())
                if e == e2:
                    same_sims.append(sim)
                else:
                    diff_sims.append(sim)

    same_mean = float(np.mean(same_sims)) if same_sims else float("nan")
    diff_mean = float(np.mean(diff_sims)) if diff_sims else float("nan")
    return {
        "same_cos": same_mean,
        "diff_cos": diff_mean,
        "gap": same_mean - diff_mean,
        "n_same_pairs": len(same_sims),
        "n_diff_pairs": len(diff_sims),
    }


@torch.no_grad()
def drift_diagnostic(
    slot_states: torch.Tensor,         # [T, S, D]
    gt_pos: torch.Tensor,              # [T, E, 2]
    gt_visible: torch.Tensor,          # [T, E] bool
    slot_to_pos_aux: SlotPosAuxHead,
    id_dim: int,
) -> dict:
    """Per-mode diagnostic: mean ||id_half(t) - id_half(t-1)[match]|| and
    same for dyn_half. If id_drift ≈ dyn_drift the split is cosmetic.

    Returns:
        {'id_drift': float, 'dyn_drift': float, 'ratio': float}
    """
    T, S, D = slot_states.shape
    pred_pos = slot_to_pos_aux(slot_states)
    assignments = _assign_slots_to_entities(pred_pos, gt_pos, gt_visible)

    id_diffs = []
    dyn_diffs = []
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
            id_prev = slot_states[t-1, s_prev, :id_dim]
            dyn_cur = slot_states[t, s_t, id_dim:]
            dyn_prev = slot_states[t-1, s_prev, id_dim:]
            id_diffs.append((id_cur - id_prev).norm().item())
            dyn_diffs.append((dyn_cur - dyn_prev).norm().item())

    id_drift = float(np.mean(id_diffs)) if id_diffs else float("nan")
    dyn_drift = float(np.mean(dyn_diffs)) if dyn_diffs else float("nan")
    ratio = id_drift / dyn_drift if dyn_drift > 0 else float("nan")
    return {"id_drift": id_drift, "dyn_drift": dyn_drift, "ratio": ratio,
             "n_pairs": len(id_diffs)}
