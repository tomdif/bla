# Phase 18θ — Slot-feature value head (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18η-multi (4/4 gates) locked `combined_sum` as the new BLA
System-1 planning recipe: a small MLP value head over **10-dim hand-
engineered geometric features** plus the OF-JEPA one-step predictor.

The BLA System-1 thesis is that *slot / object-file features* are
the right state representation for planning. Phase 18θ tests whether
swapping the value head's input from 10-dim geometry → OF-JEPA's
slot features yields a richer episode-level signal — i.e., whether
the BLA architecture's learned state actually contains
planner-valuable information beyond the hand-engineered geometric
abstractions.

## The question

> Does OF-JEPA's learned object-file state contain planner-relevant
> value information beyond hand-engineered geometry?

## Setup

Same as Phase 18η-multi, but with three value-head variants per seed:

| Head | Input | Dim |
|---|---|---|
| **geo** (reference, Phase 18η) | 10-dim BC features | 10 |
| **slot** (primary 18θ) | OF-JEPA slot features (n_slots × slot_dim) flattened | 768 |
| **geo+slot** (richer) | concat(geo, slot) | 778 |

All three use the same goal_xy (2) and action_plan (10×7=70) inputs.
All three trained for 2000 steps on the same 720 rollout samples,
labeled with full-episode realized improvement.

Eval modes (single seed first; 6 modes × 30 eps):

```
gt_closed_loop                  (oracle reference)
phase17_locked                  (predictor only — baseline)
combined_sum_geo                (Phase 18η recipe — within-seed reference)
combined_sum_slot               (slot value head + predictor)
combined_sum_geoslot            (geo+slot value head + predictor)
naive_cem                       (floor)
```

## Pre-committed gates

```
G1. combined_sum_slot.improvement >= phase17_locked.improvement + 0.02
       (slot value head + predictor beats the no-value-head baseline
        by at least the same margin Phase 18η's geo recipe did)

G2. combined_sum_geoslot.improvement >= combined_sum_geo.improvement
       (the geo+slot concat head adds value beyond geo-only — i.e.,
        slot features carry information not already in geo)

G3. slot value head held-out Spearman >= 0.20
       (positive ranking signal — Phase 18η geo head got 0.319)

G4. slot value head top-decile actual / bot-decile actual >= 2.0
       (monotonic at the extremes, as Phase 18η geo head had 0.476/0.112=4.3x)
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3 + G4** | OF-JEPA slot features are directly planner-valuable AND add information beyond engineered geometry. The architecture becomes more BLA-native: object-file state → dynamics → value → planning, all from the same learned representation. |
| G1 + G3 + G4, G2 fail | Slot features are planner-valuable but don't add beyond geometry — possibly because geometry already encodes everything in this domain. Use slot-only head for tasks where engineered features are unavailable. |
| G1 only (G3/G4 fail) | Slot-feature head plans well but its per-replan ranking is poor; the planner-level win comes from CEM noise. Investigate. |
| 0/4 | OF-JEPA slot state is useful for dynamics but value prediction still needs explicit geometric features. The planner depends on hand-engineered geometric abstractions; that's the System-2/readout layer's job. Still a clarifying result. |

## What this phase is NOT

- Not a re-training of OF-JEPA (frozen Phase 17 encoder)
- Not a new candidate distribution (locked recipe: scripted prior +
  light CEM, unchanged)
- Not yet multi-seed (Phase 18θ-multi if 18θ passes G1)
- Not yet cross-task (Phase 18κ)

## Implementation sketch

New script: `scripts/phase18t_slot_value_head.py`.

Modifications from `phase18h_value_head.py`:
1. **Collection**: at each replan boundary, also encode the agentview
   image with OF-JEPA → flatten slot features → save `slot_features`
   in the rollout cache alongside existing `features` (geo).
2. **Training**: train 3 value heads (geo, slot, geo+slot) on the
   same cache, save as `value_head_geo.pt`, `value_head_slot.pt`,
   `value_head_geoslot.pt`.
3. **Eval**: for each combined_sum variant, build a score function
   that selects the appropriate state representation per the head
   variant.

Reuse `GoalProgressValueHead` directly — its `state_dim` parameter
is already configurable.

Pod runbook (single seed first):

```bash
python3 scripts/phase18t_slot_value_head.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 300 --train-steps 2000 \
    --n-eval-episodes 30 \
    --out /workspace/phase18t \
    --seed 0
```

Estimated cost:
- Collection: ~12 min (same as 18η)
- Training: 3 heads × 2 min = ~6 min
- Eval: 6 modes × 30 ep × ~15s = ~45 min
- Total single-seed: ~65 min

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18T_SLOT_VALUE_HEAD_DECISION.md`
- Artifacts: `artifacts/phase18t/{summary.json, decile_diagnostics.json,
  per_episode_*.jsonl}`

## Sibling memory

- `[[bla-locked-planning-recipe]]` — current `combined_sum_geo` recipe;
  18θ proposes the upgrade
- `[[bla-next-architectural-lever]]` — Phase 18θ was the next-priority
  lever after 18η-multi locked
- `[[value-head-complementary-to-dynamics]]` — the lesson 18θ stress-
  tests with a richer state representation
- `[[rank-vs-candidate-quality-orthogonal]]` — the framework that
  motivates the value head architecture; 18θ tests whether richer
  state improves rank quality specifically
