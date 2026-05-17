# Phase 15b (Planning, recalibrated gates) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — recalibrated after Phase 15's empirical oracle ceiling.**
Supersedes `PHASE_15_PLANNING_PRECOMMIT.md`.

## Why this exists

Phase 15's original oracle threshold (improvement ≥ 0.20) was empirically
too high for the 60-frame push setup. A 30-episode oracle check
established that even the hand-coded closed-loop oracle averages:

```
30-ep oracle:  improvement ≈ 0.146   dir_score ≈ 0.152   contact ≈ 0.70
```

The task is valid enough to test planning (positive dir_score, high
contact rate, real cube motion), but the absolute ceiling is lower
than I'd guessed. The disciplined response is **not** to silently
lower the gate, but to open a new explicitly recalibrated phase.
Phase 15b restates the planning test with thresholds tied to the
observed oracle ceiling, while preserving the comparative gates
against no-action and random baselines.

## The question (unchanged)

> Does the action-conditioned OF-JEPA predictor support closed-loop
> action selection under MPC/CEM?

## Setup (unchanged from 15)

| | Value |
|---|---|
| Env | robosuite Stack |
| Episodes | 30 per (mode, K) |
| Total actions executed | 15 (at stride 4 = 60 env frames ≈ 3 sec) |
| Plan horizon | 10 actions |
| Replan cadence | 5 actions (3 replans per episode) |
| CEM iterations | 3 |
| Main K | 128; sweep 64, 128, 256 |
| Elite fraction | 20% |
| Modes | gt_closed_loop, cem_action, cem_noaction, random |

Both checkpoints (action-conditioned + no-action OF-JEPA) are reloaded
from Phase 15's saved files — no retraining needed.

## Revised oracle sanity gate

```
oracle improvement >= 0.10    (empirical floor of oracle distribution)
oracle dir_score   >  0       (oracle pushes in roughly the right direction)
oracle contact    >= 0.60     (EE actually touches the cube on most episodes)
```

All three must pass before the model-comparison eval runs.

## Revised planning gates

```
G1. improvement(cem_action) - improvement(cem_noaction) >= 0.10
       (action conditioning closes ≥10pp more of the goal gap than no-action MPC)

G2. improvement(cem_action) >= 0.10
       (matches oracle ceiling; tests "are predictions planner-grade at all")

G3. improvement(cem_action) / improvement(cem_noaction) >= 1.5
       (at matched candidate budget K=128, action conditioning gives ≥50% relative gain)
```

G3 was reformulated from Phase 15's "smaller-K matches main-K"
because no-action MPC may never reach action-MPC's level, making the
original formulation unstable. The matched-K ratio is cleaner.

## Diagnostics (logged, not gated)

```
dir_score          — directional correctness of the push
contact_rate       — does EE actually engage the cube
overshoot_rate     — did cube go past goal
mean_displacement  — magnitude of cube motion
success_rate       — final dist <= 4cm (oracle ceiling ~17%; not gated)
pred-actual_corr   — predictor calibration on the candidates it scored
```

## Joint verdict matrix

| Pass | Verdict |
|---|---------|
| 3/3 | **OF-JEPA action model supports MPC planning.** Push to Phase 16. |
| 2/3 | Partial planner usefulness. Inspect failed gate. |
| 1/3 | Mostly not planner-grade under closed-loop execution. |
| 0/3 | Action predictions do not transfer to planning. |

## What this does NOT establish

- **Absolute success rate.** Oracle ceiling is ~17% success at 4cm
  threshold; we're testing *relative* planning quality, not task
  completion.
- **Multi-seed stability.** Single seed (matches Phase 14.5/6).
- **Beyond robosuite Stack.** Same environment, same goal type.

## Reproducibility

```bash
# Models already trained in Phase 15 (checkpoints saved). Just rerun
# with revised thresholds — script reloads model_action.pt + model_noaction.pt.
python3 scripts/phase15_planning.py \
  --train-cache /workspace/robosuite_local/stack_scripted \
  --seed 0 --max-steps 1500 --jepa-stride 4 \
  --total-actions 15 --replan-every 5 \
  --plan-horizon 10 --cem-iters 3 --main-K 128 --candidate-counts 64,128,256 \
  --modes gt_closed_loop,cem_action,cem_noaction,random \
  --n-episodes 30 --oracle-sanity-n 30 \
  --oracle-min-improvement 0.10 --oracle-min-contact 0.60 \
  --g2-threshold 0.10 --g3-ratio 1.5 \
  --goal-dist-min 0.05 --goal-dist-max 0.08 --success-threshold 0.04 \
  --out /workspace/phase15b
```
