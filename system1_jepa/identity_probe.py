"""Identity-aware Hungarian probe for object-centric evaluation.

Position-only Hungarian matching is blind to entity identity. With moving,
occluded, multi-entity scenes, two slots can swap their matched entity
across frames and a position-only metric won't notice — yet "identity
preservation" is exactly the property we want from a slot world-model.

This module adds three primitives that make identity a first-class metric:

1. `hungarian_assign` — matches slot predictions to ground-truth entities
   using position + categorical attributes (color/shape/material). With
   unique attribute tuples the match is identity-respecting; without them
   it gracefully degrades to position-only.

2. `identity_switch_rate` — given a sequence of per-frame slot→entity_id
   assignments, returns the fraction of (slot, consecutive-frame) pairs
   where the assigned entity changed. A perfectly stable model scores 0.

3. `identity_aware_probe_eval` — fits a linear probe slot→entity-feature
   under per-frame Hungarian targets, then evaluates position MSE and
   identity-switch rate on a held-out episode split. Backbone-agnostic:
   works with toy navigate slots, ConvNeXt-CLEVRER slots, or Kubric slots.

Designed for the Phase 6.5 / 7 (CLEVRER / Kubric) decisive evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class IdentityProbeResult:
    """All six decisive-test metrics, plus diagnostics."""
    # Position MSE under identity-aware Hungarian matching.
    visible_position_mse: float
    hidden_position_mse: float
    # Identity-switch rate on visible-then-hidden frames.
    identity_switch_rate: float
    # Per-hidden-step degradation.
    per_step_position_mse: List[Optional[float]] = field(default_factory=list)
    per_step_switch_rate: List[Optional[float]] = field(default_factory=list)
    # Sample counts (sanity).
    n_visible: int = 0
    n_hidden: int = 0
    # Mean number of *distinct* entity IDs each slot is assigned to across
    # an episode. 1.0 = perfectly stable; n_entities = constantly shuffling.
    mean_slot_diversity: float = 0.0


# ---------- Hungarian matching ----------------------------------------------

def _scipy_assign(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Try scipy.optimize.linear_sum_assignment, fall back to greedy."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        S, E = cost.shape
        rows, cols = [], []
        used = set()
        for r in range(S):
            order = sorted(range(E), key=lambda c: cost[r, c])
            for c in order:
                if c not in used:
                    rows.append(r); cols.append(c); used.add(c); break
        return np.asarray(rows), np.asarray(cols)


def hungarian_assign(
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    pred_attr: Optional[np.ndarray] = None,
    true_attr: Optional[np.ndarray] = None,
    attr_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Hungarian-match slot predictions to GT entities by position+attribute cost.

    pred_pos:  [S, 2]   predicted slot positions
    true_pos:  [E, 2]   GT entity positions
    pred_attr: [S, A]   predicted slot attribute embedding (optional)
    true_attr: [E, A]   GT entity attribute embedding (optional)

    Returns (rows, cols, matched_cost) where rows index pred slots and cols
    index true entities. Handles S != E by padding the cost matrix.
    """
    S = pred_pos.shape[0]
    E = true_pos.shape[0]
    pos_cost = ((pred_pos[:, None, :] - true_pos[None, :, :]) ** 2).sum(axis=-1)
    if pred_attr is not None and true_attr is not None:
        attr_cost = ((pred_attr[:, None, :] - true_attr[None, :, :]) ** 2).sum(axis=-1)
        cost = pos_cost + attr_weight * attr_cost
    else:
        cost = pos_cost

    if S == E:
        rows, cols = _scipy_assign(cost)
    else:
        big = max(S, E)
        padded = np.full((big, big), 1e9, dtype=np.float64)
        padded[:S, :E] = cost
        rows, cols = _scipy_assign(padded)
        keep = (rows < S) & (cols < E)
        rows, cols = rows[keep], cols[keep]

    matched = float(cost[rows, cols].sum())
    return rows, cols, matched


# ---------- Identity-switch metric ------------------------------------------

def identity_switch_rate(assignments_per_frame: List[np.ndarray]) -> float:
    """Fraction of (slot, consecutive-frame) pairs where the matched entity ID changed.

    assignments_per_frame[t]: [S] int array. assignments[t][s] = entity_id
    matched to slot s at frame t, or -1 if slot s was unassigned.

    Excludes pairs where either frame's assignment is -1.
    """
    if len(assignments_per_frame) < 2:
        return 0.0
    n_pairs = 0
    n_switch = 0
    for t in range(1, len(assignments_per_frame)):
        prev = assignments_per_frame[t - 1]
        curr = assignments_per_frame[t]
        for s in range(len(curr)):
            if prev[s] >= 0 and curr[s] >= 0:
                n_pairs += 1
                if prev[s] != curr[s]:
                    n_switch += 1
    return n_switch / max(n_pairs, 1)


def slot_diversity(assignments_per_frame: List[np.ndarray]) -> float:
    """Mean number of distinct entity IDs each slot was matched to across the sequence.

    1.0 = each slot bound to one entity throughout (ideal).
    """
    if not assignments_per_frame:
        return 0.0
    S = len(assignments_per_frame[0])
    per_slot = []
    for s in range(S):
        ids = set()
        for a in assignments_per_frame:
            if a[s] >= 0:
                ids.add(int(a[s]))
        if ids:
            per_slot.append(len(ids))
    return float(np.mean(per_slot)) if per_slot else 0.0


# ---------- Linear probe with Hungarian targets ------------------------------

class _SlotToEntityProbe(nn.Module):
    """Per-slot linear: slot_state -> (pos_2 + attr_A) prediction."""
    def __init__(self, slot_dim: int, attr_dim: int):
        super().__init__()
        self.head = nn.Linear(slot_dim, 2 + attr_dim)
        self.attr_dim = attr_dim

    def forward(self, slot_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # slot_states: [..., slot_dim]
        out = self.head(slot_states)
        return out[..., :2], out[..., 2:]


def _hungarian_targets(
    pred_pos: torch.Tensor,   # [B, S, 2]
    pred_attr: torch.Tensor,  # [B, S, A]
    true_pos: torch.Tensor,   # [B, E, 2]
    true_attr: torch.Tensor,  # [B, E, A]
    true_visible: torch.Tensor,  # [B, E]
    attr_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample Hungarian: return (target_pos[B,S,2], target_attr[B,S,A], valid_mask[B,S])

    target_*[b, s] is the GT entity matched to slot s in sample b.
    valid_mask[b, s] is True for slots that got an assignment and the
    matched entity was visible.
    """
    B, S, _ = pred_pos.shape
    _, E, A = pred_attr.shape if pred_attr.ndim == 3 else (B, true_attr.shape[1], true_attr.shape[2])
    A = true_attr.shape[-1]

    pred_pos_np = pred_pos.detach().cpu().numpy()
    pred_attr_np = pred_attr.detach().cpu().numpy()
    true_pos_np = true_pos.detach().cpu().numpy()
    true_attr_np = true_attr.detach().cpu().numpy()
    true_visible_np = true_visible.detach().cpu().numpy()

    tgt_pos = np.zeros((B, S, 2), dtype=np.float32)
    tgt_attr = np.zeros((B, S, A), dtype=np.float32)
    valid = np.zeros((B, S), dtype=bool)

    for b in range(B):
        rows, cols, _ = hungarian_assign(
            pred_pos_np[b], true_pos_np[b],
            pred_attr_np[b], true_attr_np[b],
            attr_weight=attr_weight,
        )
        for r, c in zip(rows, cols):
            tgt_pos[b, r] = true_pos_np[b, c]
            tgt_attr[b, r] = true_attr_np[b, c]
            valid[b, r] = bool(true_visible_np[b, c])

    device = pred_pos.device
    return (
        torch.from_numpy(tgt_pos).to(device),
        torch.from_numpy(tgt_attr).to(device),
        torch.from_numpy(valid).to(device),
    )


@dataclass
class ProbeFitConfig:
    lr: float = 5e-3
    epochs: int = 200
    weight_decay: float = 1e-3
    batch_size: int = 256
    attr_weight: float = 1.0


def fit_identity_probe(
    train_slot_states: torch.Tensor,   # [N, S, D]
    train_gt_pos: torch.Tensor,        # [N, E, 2]
    train_gt_attr: torch.Tensor,       # [N, E, A]
    train_gt_visible: torch.Tensor,    # [N, E] bool
    cfg: ProbeFitConfig = ProbeFitConfig(),
) -> _SlotToEntityProbe:
    """Train slot→(pos, attr) linear probe with per-sample Hungarian targets.

    Only slots whose matched entity is *visible* contribute loss for that sample
    (we don't ask the probe to predict the position of an occluded entity from
    the visible-frame training set — the slot may have learned to hold it but
    that's what we measure at eval, not what we train on).
    """
    device = train_slot_states.device
    N, S, D = train_slot_states.shape
    _, E, A = train_gt_attr.shape

    probe = _SlotToEntityProbe(slot_dim=D, attr_dim=A).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bs = min(cfg.batch_size, N)
    for _ in range(cfg.epochs):
        idx = torch.randperm(N, device=device)[:bs]
        s_b = train_slot_states[idx]            # [B, S, D]
        gtp = train_gt_pos[idx]                  # [B, E, 2]
        gta = train_gt_attr[idx]                 # [B, E, A]
        gtv = train_gt_visible[idx]              # [B, E]

        with torch.no_grad():
            pp, pa = probe(s_b)
            tgt_pos, tgt_attr, valid = _hungarian_targets(
                pp, pa, gtp, gta, gtv, attr_weight=cfg.attr_weight,
            )

        pp, pa = probe(s_b)
        if valid.any():
            pos_loss = F.mse_loss(pp[valid], tgt_pos[valid])
            attr_loss = F.mse_loss(pa[valid], tgt_attr[valid])
            loss = pos_loss + cfg.attr_weight * attr_loss
        else:
            loss = F.mse_loss(pp, torch.zeros_like(pp))  # degenerate batch

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return probe


# ---------- Full eval --------------------------------------------------------

def identity_conditioned_position_eval(
    states: torch.Tensor,       # [N, S, D]
    gt_pos: torch.Tensor,        # [N, E, 2]
    gt_visible: torch.Tensor,    # [N, E] bool
    ep_ids: torch.Tensor,        # [N] long
    frame_idx: torch.Tensor,     # [N] long
    cfg: "ProbeFitConfig",
    subspace_dims: Optional[slice] = None,
) -> Dict[str, float]:
    """Phase 9B headline metric: identity-conditioned position MSE.

    For each file s, find its **modal entity** across the episode (the
    entity it gets Hungarian-matched to in the most frames). Then for
    each (file, frame) pair, compute MSE between the probe's prediction
    for that file and the GT position of *that file's modal entity*,
    regardless of which entity might be a better local match.

    This separates two things that the standard probe conflates:
    - "anonymous Hungarian rematching" — relabel files frame-by-frame to
      minimize MSE. Slot_delta wins this trivially on occluded frames by
      matching the occluded entity to whatever slot happens to predict
      a nearby position.
    - "persistent object tracking" — does file s STAY bound to its
      entity across time, even when that entity becomes occluded?

    The latter is what OF-JEPA actually tries to do. This function
    measures it.

    Returns:
      {
        'identity_visible_mse': mean MSE for (file, frame) pairs where
          the file's modal entity is visible at frame,
        'identity_hidden_mse': same but for occluded frames,
        'identity_hidden_visible_ratio': hidden/visible,
        'n_id_visible': sample count,
        'n_id_hidden': sample count,
      }
    """
    if subspace_dims is not None:
        states = states[..., subspace_dims]
    device = states.device

    unique_eps = ep_ids.unique()
    n_eps = unique_eps.numel()
    train_eps = unique_eps[: int(n_eps * 0.8)]
    train_mask = torch.isin(ep_ids, train_eps)
    test_mask = ~train_mask

    visible_any = gt_visible.any(dim=-1)
    train_idx = (train_mask & visible_any).nonzero(as_tuple=True)[0]
    if train_idx.numel() < 16:
        return {"identity_visible_mse": float("nan"),
                 "identity_hidden_mse": float("nan"),
                 "identity_hidden_visible_ratio": float("nan"),
                 "n_id_visible": 0, "n_id_hidden": 0}

    # Train a position-only probe (no attr).
    in_dim = states.size(-1)
    probe = nn.Linear(in_dim, 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Pre-compute training pairs: for each train (frame, file), find the
    # closest visible GT entity (by position) and use that as target.
    train_pairs_x, train_pairs_y = [], []
    with torch.no_grad():
        pre_pred = probe(states[train_idx]).detach()  # noop init, just for shape
    # Sample-based: at each step, for the train sub-batch, compute Hungarian
    # match and use it as fresh targets. (Same as identity_aware_probe.)
    for _ in range(cfg.epochs):
        idx = train_idx[torch.randperm(train_idx.numel(), device=device)[:cfg.batch_size]]
        s_b = states[idx]               # [B, S, D']
        gtp = gt_pos[idx]                # [B, E, 2]
        gtv = gt_visible[idx]            # [B, E]
        pp = probe(s_b)                  # [B, S, 2]
        # Per-sample Hungarian targets restricted to visible entities.
        targets = torch.zeros_like(pp)
        valid = torch.zeros(pp.shape[:-1], dtype=torch.bool, device=device)
        pp_np = pp.detach().cpu().numpy()
        gtp_np = gtp.cpu().numpy()
        gtv_np = gtv.cpu().numpy()
        for b in range(pp.size(0)):
            vis_e = np.where(gtv_np[b])[0]
            if vis_e.size == 0:
                continue
            rows, cols, _ = hungarian_assign(pp_np[b], gtp_np[b][vis_e])
            for r, c in zip(rows, cols):
                targets[b, r] = gtp[b, vis_e[c]]
                valid[b, r] = True
        if not valid.any():
            continue
        loss = F.mse_loss(pp[valid], targets[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # ---- Eval: per-episode, compute modal entity per file, then per-frame
    # identity-conditioned position MSE.
    probe.eval()
    test_eps = unique_eps[int(n_eps * 0.8):]
    visible_mses = []
    hidden_mses = []

    with torch.no_grad():
        for ep in test_eps.tolist():
            ep_rows = (ep_ids == ep).nonzero(as_tuple=True)[0]
            if ep_rows.numel() == 0:
                continue
            order = torch.argsort(frame_idx[ep_rows])
            ep_rows = ep_rows[order]
            T = ep_rows.numel()
            S = states.size(1)
            E = gt_pos.size(1)

            # Per-frame Hungarian assignment file→entity.
            assignments = -np.ones((T, S), dtype=np.int64)
            for t_idx, r in enumerate(ep_rows.tolist()):
                pp = probe(states[r:r+1]).cpu().numpy()[0]  # [S, 2]
                tp = gt_pos[r].cpu().numpy()                  # [E, 2]
                tv = gt_visible[r].cpu().numpy()
                vis_e = np.where(tv)[0]
                if vis_e.size == 0:
                    continue
                rows, cols, _ = hungarian_assign(pp, tp[vis_e])
                for rr, cc in zip(rows, cols):
                    assignments[t_idx, rr] = int(vis_e[cc])

            # For each file, find modal entity assignment.
            modal_entity = -np.ones(S, dtype=np.int64)
            for f in range(S):
                ents = assignments[:, f][assignments[:, f] >= 0]
                if ents.size > 0:
                    vals, counts = np.unique(ents, return_counts=True)
                    modal_entity[f] = int(vals[np.argmax(counts)])

            # Per-(file, frame), compute MSE against modal entity's GT.
            for t_idx, r in enumerate(ep_rows.tolist()):
                pp = probe(states[r:r+1]).cpu().numpy()[0]
                for f in range(S):
                    e = modal_entity[f]
                    if e < 0:
                        continue
                    err = float(((pp[f] - gt_pos[r, e].cpu().numpy()) ** 2).sum())
                    if gt_visible[r, e].item():
                        visible_mses.append(err)
                    else:
                        hidden_mses.append(err)

    vis_mean = float(np.mean(visible_mses)) if visible_mses else float("nan")
    hid_mean = float(np.mean(hidden_mses)) if hidden_mses else float("nan")
    ratio = hid_mean / vis_mean if vis_mean > 0 else float("nan")
    return {
        "identity_visible_mse": vis_mean,
        "identity_hidden_mse": hid_mean,
        "identity_hidden_visible_ratio": ratio,
        "n_id_visible": len(visible_mses),
        "n_id_hidden": len(hidden_mses),
    }


def identity_aware_probe_eval(
    states: torch.Tensor,       # [N, S, D]
    gt_pos: torch.Tensor,        # [N, E, 2]
    gt_attr: torch.Tensor,       # [N, E, A]
    gt_visible: torch.Tensor,    # [N, E] bool
    gt_entity_ids: torch.Tensor, # [N, E] long  — persistent across N (within episode)
    ep_ids: torch.Tensor,        # [N] long
    frame_idx: torch.Tensor,     # [N] long — sortable frame index within episode
    hidden_step: torch.Tensor,   # [N] long — 0 = visible, k = k frames after last visible
    J: int,
    cfg: ProbeFitConfig = ProbeFitConfig(),
    subspace_dims: Optional[slice] = None,  # Phase 7B/C: restrict probe input to id or dyn subspace
) -> IdentityProbeResult:
    """Train probe on train-episode visible frames, eval on test-episode frames.

    Identity-switch rate is computed PER EPISODE on the test split: we sort
    by `frame_idx` within each test episode, run the probe + Hungarian per
    frame, and count slot-assignment changes between consecutive frames.

    If `subspace_dims` is provided, the probe input is restricted to that
    slot-feature subspace (e.g. `slice(0, id_dim)` to compute the
    id-subspace switch rate from the Phase 7C feedback loop).
    """
    if subspace_dims is not None:
        states = states[..., subspace_dims]
    device = states.device

    unique_eps = ep_ids.unique()
    n_eps = unique_eps.numel()
    train_eps = unique_eps[: int(n_eps * 0.8)]
    test_eps = unique_eps[int(n_eps * 0.8):]
    train_mask = torch.isin(ep_ids, train_eps)
    test_mask = torch.isin(ep_ids, test_eps)

    # Training split: visible frames from train episodes.
    visible_any = gt_visible.any(dim=-1)
    train_idx = (train_mask & visible_any).nonzero(as_tuple=True)[0]

    probe = fit_identity_probe(
        states[train_idx], gt_pos[train_idx],
        gt_attr[train_idx], gt_visible[train_idx],
        cfg=cfg,
    )

    # Eval: walk test episodes frame-by-frame.
    probe.eval()
    visible_mses: List[float] = []
    hidden_mses: List[float] = []
    per_step_pos: Dict[int, List[float]] = {}
    per_step_switch: Dict[int, List[float]] = {}
    all_switch_rates: List[float] = []
    all_diversity: List[float] = []
    n_vis = 0
    n_hid = 0

    with torch.no_grad():
        for ep in test_eps.tolist():
            ep_mask = (ep_ids == ep)
            ep_rows = ep_mask.nonzero(as_tuple=True)[0]
            if ep_rows.numel() == 0:
                continue
            order = torch.argsort(frame_idx[ep_rows])
            ep_rows = ep_rows[order]

            assignments: List[np.ndarray] = []
            for r in ep_rows.tolist():
                pp, pa = probe(states[r:r+1])
                pp_np = pp[0].cpu().numpy()
                pa_np = pa[0].cpu().numpy()
                tp_np = gt_pos[r].cpu().numpy()
                ta_np = gt_attr[r].cpu().numpy()
                tv_np = gt_visible[r].cpu().numpy()
                tid_np = gt_entity_ids[r].cpu().numpy()

                rows, cols, _ = hungarian_assign(
                    pp_np, tp_np, pa_np, ta_np, attr_weight=cfg.attr_weight,
                )

                S = pp_np.shape[0]
                assign = -np.ones(S, dtype=np.int64)
                for rs, cs in zip(rows, cols):
                    assign[rs] = int(tid_np[cs])
                assignments.append(assign)

                # Position MSE on visible entities in this frame.
                visible_cols = [c for c in cols if tv_np[c]]
                if visible_cols:
                    matched_rows = [r_ for r_, c in zip(rows, cols) if tv_np[c]]
                    mse_v = float(((pp_np[matched_rows] - tp_np[visible_cols]) ** 2).sum(-1).mean())
                    visible_mses.append(mse_v)
                    n_vis += 1
                # Hidden MSE on occluded entities matched.
                hidden_cols = [c for c in cols if not tv_np[c]]
                if hidden_cols:
                    matched_rows = [r_ for r_, c in zip(rows, cols) if not tv_np[c]]
                    mse_h = float(((pp_np[matched_rows] - tp_np[hidden_cols]) ** 2).sum(-1).mean())
                    hidden_mses.append(mse_h)
                    n_hid += 1
                    h_step = int(hidden_step[r].item())
                    per_step_pos.setdefault(h_step, []).append(mse_h)

            sr = identity_switch_rate(assignments)
            all_switch_rates.append(sr)
            all_diversity.append(slot_diversity(assignments))

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    # Per-step switch rate: for each hidden depth, mean of (switch count / pair count)
    # contributions limited to consecutive frames where the second is at that depth.
    # Simpler proxy: re-derive from assignments above. For first-pass we report
    # the episode-level switch rate; the per-step breakdown is left for the
    # full CLEVRER/Kubric run where N_frames per episode is large.

    per_step_pos_sorted = [
        _mean(per_step_pos[k]) if k in per_step_pos else None
        for k in range(max([0] + list(per_step_pos.keys())) + 1)
    ]

    return IdentityProbeResult(
        visible_position_mse=_mean(visible_mses),
        hidden_position_mse=_mean(hidden_mses),
        identity_switch_rate=_mean(all_switch_rates),
        per_step_position_mse=per_step_pos_sorted,
        per_step_switch_rate=[],  # filled in once we have per-frame J alignment
        n_visible=n_vis,
        n_hidden=n_hid,
        mean_slot_diversity=_mean(all_diversity),
    )
