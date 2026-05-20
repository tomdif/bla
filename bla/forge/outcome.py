"""BF-1.2 — outcome scorers + episode→demo bridge.

The EpisodeRecord schema (BF-0.4) has an `outcome` dict that the
EpisodeLogger.set_outcome(...) populates by hand. In production we want
scorers that COMPUTE outcomes from the per-step trajectory data already
stored in the record (decoded_positions, safety_events) — no human in
the loop.

This module provides:

  - cube_xy_displacement_score(ep, slot_id=0, target_m=0.10) → float in [0,1]
    Normalized max XY displacement of a decoded slot. 1.0 means the
    object moved ≥ target_m in the XY plane during the episode.

  - no_safety_events_score(ep) → float (0.0 or 1.0)
    Sanity scorer: 0.0 if any safety_event was logged, else 1.0.
    Compose with the displacement scorer via `combine_scores`.

  - combine_scores(*fns, weights=None, mode="product") → scorer
    Combine multiple scorers element-wise. Default mode='product' means
    ANY zero score kills the combined score (so a halted episode with
    big displacement still scores 0).

  - compute_outcome(ep, *, score_fn, metric_name, success_threshold)
    Returns the dict shape EpisodeLogger.set_outcome expects (apply
    AFTER finalize() with logger.outcome = ... or before, using the
    returned dict).

  - episode_to_demo(ep, *, demo_id, score_fn=None) → DemoRecord
    Converts a teleoperated EpisodeRecord into a DemoRecord ready for
    DemoBank.add(). Uses score_fn for achieved_outcome (default: the
    episode's own outcome.improvement). The action stream is exactly
    the recorded gantry_actions; initial_state is taken from the
    earliest decoded_positions snapshot.

Hardware-arrival path: same scorers; no swap needed (they're pure
functions of the EpisodeRecord, which is hardware-agnostic).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from bla.forge.episode import DemoRecord, EpisodeRecord


# ---------- primitive scorers ----------
def cube_xy_displacement_score(
    ep: EpisodeRecord,
    *,
    slot_id: int = 0,
    target_m: float = 0.10,
) -> float:
    """Max XY distance the decoded slot moved during the episode,
    normalized by `target_m` and clipped to [0, 1].

    A score of 1.0 means the slot's max-from-start displacement
    reached or exceeded `target_m` (e.g. a successful 10cm push).
    NaN positions (unbound or occluded slots) are skipped.
    """
    if target_m <= 0:
        raise ValueError(f"target_m must be > 0, got {target_m}")
    if ep.decoded_positions.size == 0:
        return 0.0
    if not (0 <= slot_id < ep.decoded_positions.shape[1]):
        raise ValueError(
            f"slot_id={slot_id} out of range for n_slots="
            f"{ep.decoded_positions.shape[1]}")

    pos = ep.decoded_positions[:, slot_id, :]   # [T, 2]
    valid = ~np.isnan(pos[:, 0])
    if not valid.any():
        return 0.0
    valid_pos = pos[valid]
    deltas = valid_pos - valid_pos[0]
    max_disp = float(np.max(np.linalg.norm(deltas, axis=1)))
    return float(min(max_disp / target_m, 1.0))


def no_safety_events_score(ep: EpisodeRecord) -> float:
    """1.0 if zero safety events, else 0.0.

    A coarse "did the episode complete without intervention" gate.
    Useful as a multiplicative factor: a perfect 10cm push that
    triggered an e-stop should NOT score 1.0.
    """
    return 1.0 if len(ep.safety_events) == 0 else 0.0


def episode_length_fraction_score(
    ep: EpisodeRecord, *, expected_steps: int,
) -> float:
    """Fraction of expected_steps that the episode actually ran.

    A halted episode that ran 4 out of 12 expected steps scores 4/12.
    Capped at 1.0 (an over-long episode is treated as a complete one).
    """
    if expected_steps <= 0:
        raise ValueError(f"expected_steps must be > 0, got {expected_steps}")
    actual = int(ep.frames.shape[0])
    return float(min(actual / expected_steps, 1.0))


# ---------- combinator ----------
def combine_scores(
    *fns: Callable[[EpisodeRecord], float],
    weights: Optional[list[float]] = None,
    mode: str = "product",
) -> Callable[[EpisodeRecord], float]:
    """Combine multiple scorers into one.

    mode='product' (default): elementwise multiplicative. Any 0 → 0.
                              Use this when each factor is a hard gate
                              (e.g. displacement * safety_clean).
    mode='weighted_mean':     Σ w_i * s_i / Σ w_i.
                              weights must be provided (positive floats).

    All scorers must return values in [0, 1]; the combined value is
    clipped to [0, 1] defensively.
    """
    if not fns:
        raise ValueError("combine_scores needs at least one scorer")
    if mode == "product":
        if weights is not None:
            raise ValueError("weights are ignored when mode='product'")

        def combined(ep: EpisodeRecord) -> float:
            v = 1.0
            for fn in fns:
                v *= float(fn(ep))
            return float(max(0.0, min(1.0, v)))
        return combined

    if mode == "weighted_mean":
        if weights is None:
            raise ValueError("weights required when mode='weighted_mean'")
        if len(weights) != len(fns):
            raise ValueError(
                f"weights ({len(weights)}) must match scorers ({len(fns)})")
        if any(w <= 0 for w in weights):
            raise ValueError("weights must be strictly positive")
        denom = float(sum(weights))

        def combined(ep: EpisodeRecord) -> float:
            num = sum(w * float(fn(ep))
                       for w, fn in zip(weights, fns))
            return float(max(0.0, min(1.0, num / denom)))
        return combined

    raise ValueError(f"unknown mode: {mode!r}")


# ---------- outcome dict builder ----------
def compute_outcome(
    ep: EpisodeRecord,
    *,
    score_fn: Callable[[EpisodeRecord], float],
    metric_name: str,
    success_threshold: float = 0.5,
    notes: str = "",
) -> dict:
    """Compute an EpisodeLogger-shape outcome dict from an EpisodeRecord.

    Returns:
        {"success": bool, "improvement": float, "metric_name": str,
         "notes": str}

    Suitable to assign to `EpisodeLogger.outcome` (or to a record's
    outcome field directly) and re-save.
    """
    if not (0.0 <= success_threshold <= 1.0):
        raise ValueError(
            f"success_threshold must be in [0,1], got {success_threshold}")
    s = float(score_fn(ep))
    s = max(0.0, min(1.0, s))   # defensive clip
    return {
        "success": s >= success_threshold,
        "improvement": s,
        "metric_name": str(metric_name),
        "notes": str(notes),
    }


# ---------- episode → demo bridge ----------
def episode_to_demo(
    ep: EpisodeRecord,
    *,
    demo_id: int,
    score_fn: Optional[Callable[[EpisodeRecord], float]] = None,
    collector_notes: str = "",
) -> DemoRecord:
    """Convert a teleoperated EpisodeRecord into a DemoRecord.

    The DemoRecord's:
      - actions = ep.gantry_actions
      - initial_state["fiducials"] is reconstructed from the earliest
        decoded_positions step (using slot_id as the fiducial key,
        since we no longer have the original fiducial_id mapping —
        callers who need true id preservation should keep it in
        ep.retrieved_demo or pass extra metadata).
      - achieved_outcome is score_fn(ep) if provided, else
        ep.outcome["improvement"] (fallback to 0.0 if missing).

    This is the conversion the operator-side workflow uses: record a
    teleop episode → review → if it's a good demo, episode_to_demo()
    it and DemoBank.add() it.
    """
    if ep.gantry_actions.size == 0:
        raise ValueError(
            "Cannot convert empty episode (no gantry_actions) to demo")

    # Build initial_state["fiducials"] from the first decoded step.
    # Slot index → string key (so the JSON is human-readable). We use
    # the slot index as the fiducial id stand-in; the bank's key_fn
    # has to know about this convention.
    fid_world: dict[str, list[float]] = {}
    first = ep.decoded_positions[0]   # [n_slots, 2]
    for s in range(first.shape[0]):
        if not np.isnan(first[s, 0]):
            fid_world[str(s)] = [float(first[s, 0]),
                                  float(first[s, 1]), 0.0]
    initial_state = {"fiducials": fid_world}

    if score_fn is not None:
        achieved = float(score_fn(ep))
    else:
        improvement = ep.outcome.get("improvement", 0.0)
        achieved = float(improvement)
    achieved = max(0.0, min(1.0, achieved))   # clip

    return DemoRecord(
        demo_id=int(demo_id),
        task=ep.task,
        initial_state=initial_state,
        actions=ep.gantry_actions.astype(np.float32),
        achieved_outcome=achieved,
        collector_notes=str(collector_notes),
    )
