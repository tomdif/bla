# Phase 18κ Regime 3 — Lift task fine-tune (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before implementation.**

## Why this phase exists

Phase 18κ R2 tested OOD goal-distance shift within the same task.
Phase 18ν tested whether scheduled aux losses unify the recipe
family. The remaining decisive transfer test is **task shift**:
does the BLA System-1 → System-2 → planner stack survive a
different task that requires different state features?

**Lift** is the right next task because:
- robosuite already provides it; minimal new infra.
- It forces **vertical geometry** (cube_z, eef_z) which the
  supervised adapter was weakest on (Spearman 0.04 - 0.14 in
  Phase 18λ).
- It changes the task structure (lift, not push), giving a
  clean cross-task probe.

This is Phase 18κ Regime 3 from the original 18κ precommit
(`e7e1381`): task fine-tune. Each learned head is re-trained on
fresh Lift data; the OF-JEPA encoder (Phase 17) stays frozen.

## Design decisions (locked)

| Question | Choice |
|---|---|
| Lift goal representation | **2-dim `goal_xy = cube_init_xy`** + improvement = `max(0, cube_z_end - cube_z_start) / target_lift_height`. Reuses the existing value-head architecture (state_dim 10 / geo + goal 2 + plan 70). Minimal code change. |
| Lift scripted prior | **3-stage scripted_lift**: reach (EEF over cube) → grasp (close gripper) → lift (+z). ~40 LOC; same pattern as scripted_push. |
| Fine-tune scope | **All 4 learned heads** (geo, supervised, end2end, pretrain+ft) re-trained on fresh Lift cache. Geo VH is also re-trained because Lift episode_imp differs from Push improvement. |
| Implementation timing | **Precommit now; implement in steps**: env plumbing → scripted prior → collection → fine-tune → eval, each step testable. |

## Episode improvement on Lift

```
episode_imp_lift = max(0.0, cube_z_end - cube_z_start) /
                    max(target_lift_height, 1e-9)
```

with `target_lift_height = 0.06m` (matching robosuite's default
success threshold for Lift; cube needs to be lifted ~6cm).

## Engineered geometric features (unchanged 10-dim)

`state_features(obs, goal_xy)`:
- cube_xy (2): current cube xy
- eef_xy (2): current eef xy
- eef_z (1): current eef z       ← Phase 18λ adapter weak here
- cube_z (1): current cube z     ← Phase 18λ adapter weak here
- goal_xy (2): target xy (= cube_init_xy for Lift)
- push_dir (2): direction from eef to cube  (no goal-direction
                                              meaning for Lift,
                                              but kept for arch
                                              compatibility)

The Phase 18λ adapter's z-feature recovery was Spearman 0.04 - 0.14.
On Lift, where z matters, the adapter HAS to learn these features
or it can't predict episode_imp_lift (which is ENTIRELY a z-gain
quantity). This phase is the decisive test of whether z-features
are recoverable from frozen slots given task-relevant data.

## Eval setup

- 3 seeds × 5 recipes (locked, geo, supervised, end2end,
  pretrain+ft) × Lift task × 30 episodes
- Fresh Lift rollouts per seed: ~200 episodes (smaller than Stack
  data since fine-tune doesn't need as much)
- Per-recipe fine-tune budget: 1000 steps (half of Stack training)

## Pre-committed gates

Per the user's locked gate spec:

```
G1. best learned recipe.improvement >= phase17_locked.improvement + 0.02
       (the best of {geo, sup, e2e, pft} beats locked baseline)

G2. pretrain_ft.improvement >= supervised.improvement - 0.02
    AND pretrain_ft.improvement >= end2end.improvement - 0.02
       (pretrain+ft is within 0.02 of both supervised and end2end,
        i.e. it's a robust middle recipe under task shift too)

G3. at least one learned-adapter recipe (sup, e2e, pft) reaches
       >= 0.90 × geo's mean improvement
       (the BLA-native recipe family doesn't collapse vs engineered
        geometry under task shift)

G4. Adapter cube_z + eef_z Spearman on Lift held-out >= 0.30 mean
    (averaged across the two z-features) for the supervised adapter
       (the height features Phase 18λ identified as weak ARE
        recoverable when the task forces them)
```

## Verdict matrix

| Pass | Verdict |
|---|---|
| **G1 + G2 + G3 + G4** | The BLA recipe family transfers to Lift via fine-tune. Pretrain+ft is genuinely the robust general-purpose adapter (lock as Recipe D). |
| **G1 + G3 + G4** (G2 marginal) | Recipe family transfers; geo and one adapter recipe are the best. Pretrain+ft is OK but not strictly the winner under task shift. |
| **G1 + G4** (G3 fail) | Geo (simulator features) is required for Lift; adapter recipes don't generalize across tasks even with fine-tune. |
| **G4 only** | The slot representation contains z-features but value prediction on Lift fails. Investigate (probably more data or different value-head architecture). |
| **0/4** | BLA architecture doesn't transfer cross-task even with fine-tune. The deployment-conditional recipe story is task-specific. |

## Implementation plan

Each step testable independently before the next:

```
1. Env plumbing  (scripts/phase18k_r3_lift.py — new file)
   - build_env_lift() that wraps rs.make("Lift", ...) with same
     image/horizon/etc. as build_env Stack
   - sample_lift_goal(obs, ep_id) returns (cube_init_xy, target_lift_height)
   - state_features_lift(obs, goal_xy) returns 10-dim engineered geo

2. Scripted prior  (scripts/phase18k_r3_lift.py)
   - scripted_lift_action(env, obs, ep_state, phase) — 3-stage
     finite-state machine: reach → grasp → lift
   - rollout_scripted_lift_prior — env-clone rollout helper

3. Collection  (scripts/phase18k_r3_collect.py)
   - Run 200 Lift episodes with scripted_lift + light CEM at fresh
     OF-JEPA model checkpoint
   - Save rollout_cache_lift.npz with same schema as Phase 18θ
     (geo_features, slot_features, goals, plans, labels) but on
     Lift task

4. Fine-tune  (scripts/phase18k_r3_finetune.py)
   - Load Phase 17 OF-JEPA model (frozen)
   - For each of 4 head types (geo, sup, e2e, pft):
     - Load Stack checkpoint as initialization
     - Fine-tune 1000 steps on Lift cache, same lr as initial training
     - Save Lift checkpoint
   - Compute decile diagnostic per head on Lift held-out

5. Eval  (scripts/phase18k_r3_eval.py)
   - For each recipe (5 modes including locked + naive): run 30 Lift
     eval episodes
   - Per-recipe improvement, success rate (cube_z >= target after
     full episode)
   - Cross-seed aggregate + gates evaluation
```

Pod budget (per seed):
- Collection: 200 ep × ~3s = ~10 min
- Fine-tune: 4 heads × 2 min = ~8 min
- Eval: 5 modes × 30 ep × ~15s = ~38 min
- Total per seed: ~56 min; parallel 3 seeds = ~60 min wall

## What this phase is NOT

- Not zero-shot transfer (Regime 1) — fine-tune is allowed.
- Not multi-task training (only Lift fine-tune per recipe).
- Not architectural redesign — same 10-dim engineered geo,
  same adapter / value-head architectures.

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18K_REGIME3_DECISION.md` (after
  results).
- Per-seed artifacts: `artifacts/phase18k_r3_seed{0,1,2}/...`
- Aggregate: `artifacts/phase18k_r3_multi/aggregate.json`

## Sibling memory

- `[[bla-locked-planning-recipe]]` — locked deployment-conditional
  family; 18κ R3 tests whether the family transfers to task shift
- `[[aux-loss-distribution-dependent]]` — 18κ R2 finding;
  R3 tests whether the same in-dist vs OOD ordering flip applies
  to task-shift
- `[[engineered-aux-loss-useful-inductive-bias]]` — 18λ-v2; tests
  whether the geo-MSE bias is useful on Lift's vertical geometry
  (where the prior adapter was weakest)
