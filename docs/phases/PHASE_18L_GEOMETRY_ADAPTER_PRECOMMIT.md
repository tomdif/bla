# Phase 18λ — Object-file geometry adapter (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18θ established that raw frozen OF-JEPA slot features
(768-dim, flattened) do not directly support goal-relative value
prediction at the 720-sample data budget (Spearman 0.113 vs geo's
0.362; top/bot 1.18× vs geo's 3.7×).

The reframe (locked in
`[[feedback_frozen_slots_not_enough_for_value]]`) is: the
information *is* in the slots — OF-JEPA uses it for action-
conditioned dynamics prediction in Phase 17/18d — but a small MLP
trained on episode-improvement regression cannot extract the
goal-relative geometric projection in one shot. The right
architectural lever is a **structured System-2 adapter** that
projects slot state → goal-relative geometric features, then feeds
those derived features to the value head.

Phase 18λ tests whether such an adapter exists, and whether it
recovers (or matches) the hand-engineered geometry's planner value.

## The question (two-part)

> 1. Can a small MLP recover the 10-dim goal-relative geometric
>    features (cube_xy, eef_xyz, cube_z, push_dir) from
>    `(OF-JEPA slot state, goal_xy)`?
> 2. If so, does plugging the adapter's output into the Phase 18η-
>    style value head yield a planner that beats `phase17_locked`
>    and matches `combined_sum_geo`?

## Architecture

```
ObjectFileGeometryAdapter:
  input:  flattened slot features (n_slots * slot_dim = 768)
        + goal_xy (2)
  → 2-3 hidden 256-dim MLP
  → output: 10-dim derived geometry (matches Phase 18β `state_features`
            output: cube_xy, eef_xyz, cube_z, goal_xy, push_dir)

value_head_adapter:
  input: adapter(slot, goal) [10] + goal [2] + plan [10×7=70]
  → same GoalProgressValueHead architecture as Phase 18η/θ
```

The adapter is trained **supervised**: for each
(slot, goal, true_geo) triple in the Phase 18θ cache, regress
adapter(slot, goal) → true_geo with MSE loss. This isolates whether
slots contain the geometric information; the value head is the same
proven design from Phase 18η.

Phase 18θ already collected the necessary cache
(`/workspace/phase18t_main/rollout_cache.npz`: 900 samples with both
geo and slot features), so collection is reused. Per-phase cost:
~5 min adapter training + ~5 min value-head training + ~45 min eval.

## Setup

- Adapter training: 720/180 train/val split (same deterministic
  split as Phase 18η/θ for direct comparison).
- Adapter loss: per-feature MSE, optionally weighted by feature
  variance (defer; start with uniform MSE).
- Value head: train fresh on `(adapter(slot, goal), goal, plan) →
  episode_imp` for 2000 steps, identical hyperparameters to
  Phase 18η/θ.
- Eval modes (single seed first):
    - gt_closed_loop                (oracle)
    - phase17_locked                (predictor only)
    - combined_sum_geo              (Phase 18η reference)
    - combined_sum_adapter          (NEW — adapter-derived geo + predictor)
    - naive_cem                     (floor)

## Pre-committed gates

```
G1. adapter held-out Spearman over predicted_geo vs true_geo per
    feature dimension >= 0.50 mean across 10 features
       (the adapter actually recovers geometry from slots)

G2. value_head_adapter held-out Spearman on episode_imp >= 0.25
       (adapter-derived geometry supports value prediction with
        signal at least geo's level minus tolerance; 18η geo was 0.32)

G3. value_head_adapter top-decile / bot-decile actual >= 2.0×
       (monotonicity at extremes, as Phase 18η geo head was 3.7×)

G4. combined_sum_adapter improvement >= phase17_locked.improvement
       + 0.02 (matches Phase 18η G1)

Stretch (not gated):
G5. combined_sum_adapter improvement >= 0.90 × combined_sum_geo
       (within 10% of hand-engineered-geo recipe — would mean BLA
        is no longer dependent on simulator-true geometric features)
```

## Verdict matrix

| Result | Interpretation |
|---|---|
| **G1 + G2 + G3 + G4 pass; G5 within 10%** | Slot features contain the goal-relative geometry; a small structured adapter extracts it; the resulting planner is competitive with hand-engineered geometry. **BLA is no longer dependent on simulator-true features.** The architecture becomes genuinely System-1 (frozen slots) → System-2 (geometry adapter) → planner. |
| G1 + G2 + G3 pass; G4 fail | Adapter extracts geometry and value head signal is good, but planning fails. Investigate (probably env-variance like Phase 18θ; multi-seed before lock). |
| G1 pass; G2/G3 fail | Adapter recovers geometry but value head can't use it. Suggests adapter feature space differs from geo space in a way that breaks downstream training. Try: train adapter+value-head jointly end-to-end. |
| G1 fail | Slots do NOT contain the goal-relative geometry, even with structured supervised extraction. The BLA architecture genuinely needs goal-conditioned features at the encoder level — re-train OF-JEPA with goal awareness in System-1. |

## What this phase is NOT

- Not a re-training of OF-JEPA (encoder frozen)
- Not a new dynamics predictor (Phase 17 frozen)
- Not a new candidate distribution (scripted prior + light CEM
  unchanged)
- Not yet multi-seed (18λ-multi if 18λ passes G1+G4)
- Not yet cross-task (Phase 18κ)

## Why supervised-adapter first

We could train the adapter end-to-end with value-head MSE loss
(no explicit geometry supervision). That's more BLA-native. But it
puts two unknowns in series: "can slots produce useful features
under this loss?" + "is the value-head training stable?"

Supervised adapter answers the first cleanly: we have ground-truth
geo from the simulator. If slots can't predict geo, end-to-end won't
either (it's a strictly harder problem). If slots CAN predict geo,
we know the value head is the bottleneck (and we can iterate on it).
Cheaper, more diagnostic, same end-state-architecturally.

## Implementation sketch

New module: `system1_jepa/geometry_adapter.py` with
`ObjectFileGeometryAdapter` + `train_adapter_supervised`.

New script: `scripts/phase18l_geometry_adapter.py` reads the 18θ
rollout cache, trains the adapter, trains the value head on
adapter-derived geo, evaluates 5 modes.

Pod runbook:

```bash
python3 scripts/phase18l_geometry_adapter.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-cache /workspace/phase18t_main/rollout_cache.npz \
    --adapter-train-steps 2000 \
    --vh-train-steps 2000 \
    --n-eval-episodes 30 \
    --out /workspace/phase18l_main \
    --seed 0
```

Estimated cost (cache reused):
- Adapter training: ~3 min
- Value head training: ~3 min
- Eval: 5 modes × 30 ep × ~15s = ~38 min
- Total: ~45 min single-seed

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18L_GEOMETRY_ADAPTER_DECISION.md`
- Artifacts: `artifacts/phase18l/{summary.json,
  adapter_diagnostic.json, per_episode_*.jsonl}`

## Sibling memory

- `[[bla-locked-planning-recipe]]` — combined_sum_geo recipe;
  18λ tries to replace geo with adapter-derived geo
- `[[frozen-slots-not-enough-for-value]]` — Phase 18θ lesson;
  18λ is the constructive follow-up
- `[[bla-next-architectural-lever]]` — updated: 18λ is the new
  next-priority architectural test
- `[[value-head-complementary-to-dynamics]]` — Phase 18η; same
  value head architecture used here, just different input
