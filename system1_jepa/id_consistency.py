"""Identity-binding diagnostics + the SlotPosAuxHead used by OF-JEPA eval.

This module is the **canonical** site for the small slot→position
auxiliary head and the per-frame Hungarian assignment helper, plus the
two diagnostics that survived the Phase 7-8 falsification arc:

  - `SlotPosAuxHead`        — slot → 2D position projection for matching
  - `_assign_slots_to_entities` — per-frame Hungarian helper
  - `cosine_diagnostic`     — same-entity vs different-entity id_key cosine
  - `drift_diagnostic`      — per-frame id_drift vs dyn_drift

Two training-time loss functions that were active in Phase 7C and
Phase 8A — `identity_consistency_loss` and `identity_contrastive_loss`
— have been **falsified** and moved to `system1_jepa/_attic/falsified_losses.py`.
See `docs/phases/PHASE_7C_JEPA_DECISION.md` and
`docs/phases/PHASE_8A_JEPA_DECISION.md` for the falsification arc and
`feedback-prediction-vs-assignment` in the memory layer for why
content-side identity losses can't fix what is structurally an
assignment problem.

The canonical OF-JEPA v0 (post-Phase-9B) trains with only the JEPA
temporal-smoothness loss on `state_value` + supervised position loss
via `SlotPosAuxHead`. No identity-consistency or contrastive terms.
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


# Falsified Phase 7C / 8A loss functions moved to
# `system1_jepa/_attic/falsified_losses.py`. The canonical OF-JEPA v0
# does not import them. Code that wants to reproduce the falsification
# sweeps imports from `_attic` explicitly.


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
