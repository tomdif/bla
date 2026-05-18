# Phase 18κ — Cross-task transfer (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18μ established that the supervised geometry adapter recipe
(slot → adapter → value head → planner) is empirically a peer to
the simulator-true engineered-geo recipe at the planner level on
the Stack cube-displacement task. The remaining open question:

> Does the supervised adapter/value/action stack survive outside the
> push task it was trained on?

If yes, the BLA architecture is genuinely **task-agnostic** at the
System-2 readout layer — slot features carry enough information for
the adapter to extract task-relevant geometry on related tasks
without per-task engineering. If no, the recipe is task-specific
and only the engineered-geo recipe is portable.

This is the decisive architectural test of the Phase 18η/λ/μ arc.

## The question

> Does `combined_sum_supervised` transfer to robosuite Lift /
> PickPlace tasks without retraining the OF-JEPA dynamics predictor
> or the supervised adapter on the new task, OR does it require
> task-specific data?

## Setup — three transfer regimes (lightweight → heavy)

In increasing order of training cost / adaptation:

### Regime 1 — Zero-shot transfer (cheapest, hardest)

Use the Phase 18μ recipe trained on Stack cube-displacement,
**unchanged**, on Lift task with a comparable goal (e.g. cube
displacement in 2D, ignoring the "Lift" success criterion). The
adapter and value head have never seen Lift data.

### Regime 2 — Same-task variant (Stack with perturbations)

Same task family but with shifted env distribution: heavier cubes,
varied friction, varied start positions outside the training
distribution. Tests whether the supervised adapter's slot→geo
extraction is robust to env-shift.

### Regime 3 — Task fine-tune (lightest possible adaptation)

Re-train ONLY the value head on a small batch of new-task rollouts
(100-200 episodes), keeping OF-JEPA + adapter frozen. Tests whether
the adapter's geometric extraction transfers and only the value
function needs task-specific tuning.

## Pre-committed gates

### Headline gate (Regime 1 or Regime 3, choose one for primary)

```
G1. On the new task: combined_sum_supervised.improvement >= 0.50 ×
    combined_sum_supervised.improvement on Stack (Phase 18μ mean 0.251)
       (rough "doesn't collapse to floor" gate; new-task improvement
        of ~0.125 would be a meaningful transfer signal)
```

### Diagnostic gates

```
G2. Adapter geo-recovery Spearman on new-task rollouts >= 0.30
       (if the adapter can't extract geometry from new-task slots,
        the entire approach fails)

G3. Combined_sum_supervised >= combined_sum_geo on new task
       (since engineered-geo features are TASK-SPECIFIC and the
        adapter recipe is task-agnostic, the adapter recipe should
        ALSO match engineered-geo when both are evaluated; in
        Regime 1 this matters because engineered_geo doesn't even
        have the right features)

G4. combined_sum_supervised >= phase17_locked on new task
       (planner remains above baseline)
```

### Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3 + G4** (any regime) | The BLA System-1 → System-2 → planner architecture transfers cross-task. Lock Recipe B as the primary BLA recipe and deprecate engineered geo. |
| G1 + G2 + G3 + G4 in Regime 3 only | The adapter transfers but the value head is task-specific. Lock Recipe B with a per-task value-head retrain (small data). |
| G2 + G3 + G4 (G1 fails) | Adapter+geo-extraction transfer, but the planner-level result on new task is small. Investigate whether the new task is harder for ALL recipes or specific to adapter. |
| 0/4 in Regime 1 | Adapter+value stack is task-specific. Engineered-geo recipe remains the only validated locked recipe; BLA-native recipe needs per-task training to transfer. |

## Which regime to run first

**Regime 2 (perturbed Stack)** is the cleanest first test because:
- Re-uses existing Stack infrastructure (no new task plumbing).
- Tests env-distribution-shift robustness directly.
- If it fails, that tells us the adapter is brittle even within-task,
  which is informative before attempting cross-task.
- If it passes, Regime 1 (Lift/PickPlace zero-shot) is the natural
  next test.

Default plan:
1. Run **Regime 2** (perturbed Stack) first — 3 seeds × parallel.
2. If Regime 2 passes, run **Regime 3** (Lift with value-head
   fine-tune).
3. If Regime 3 passes, run **Regime 1** (Lift zero-shot) as the
   strong-pass test.

## What this phase is NOT

- Not a new architectural variant (no v3 adapter, no end-to-end
  retry).
- Not a new System-1 retraining (OF-JEPA frozen).
- Not multi-task training (one task at a time).

## Implementation sketch

Regime 2 (perturbed Stack):
- Modify `scripts/robosuite_collect_rollouts.py` to accept
  perturbation parameters (cube_mass, friction, start_pos_range).
- Or just augment goal-distance distribution: train on r ∈ [0.05,
  0.08]; eval on r ∈ [0.08, 0.12].
- Reuse `phase18l2_end2end.py` with the new env params.

Regime 1 (Lift zero-shot):
- New script: `scripts/phase18k_cross_task.py` — replaces
  `build_env` with `rs.make("Lift", ...)`, evaluates the locked
  recipe.
- Goal becomes "cube z >= threshold" or equivalent. Need to define
  a continuous goal target compatible with the value head's input
  format (10-dim goal-relative geometry).

Regime 3 (value-head fine-tune):
- Collect ~100 Lift rollouts using `scripted_prior_lift` (TBD).
- Re-train ONLY the value head on these rollouts.
- Eval on Lift.

## Decision before implementation

Three sub-questions to lock before implementing:

1. **Which regime first?** Default: Regime 2.
2. **Which new task for Regime 1/3?** Default: Lift (simpler than
   PickPlace; has clear continuous goal).
3. **What "perturbation" for Regime 2?** Default: goal-distance
   range [0.10, 0.15] (OOD vs training [0.05, 0.08]).

These are the cheapest decisive tests; if the supervised adapter
holds up under goal-distance shift, the cross-task case is well-
motivated.

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18K_CROSS_TASK_DECISION.md`
- Artifacts: `artifacts/phase18k_{regime2, regime3, regime1}/...`
