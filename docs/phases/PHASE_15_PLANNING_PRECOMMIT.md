# Phase 15 (Planning / CEM-on-Predictor) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

Phase 14 closed with the claim that OF-JEPA learns generalizable
action effects. Phase 15 asks the planner-relevant question:

> Can the action-conditioned OF-JEPA actually choose better actions,
> not just rank them offline?

## The question

CEM (cross-entropy method) uses a model's predicted futures to search
for action sequences that achieve a goal. If OF-JEPA's predictions are
planner-grade, action-conditioned CEM should outperform:
- a no-action predictor (Phase 14.5/14.6 baseline)
- a random shooting baseline (no planning at all)
and approach an oracle skyline.

## Task

**Cube-displacement goal.** Each episode:
1. Reset the env, observe initial cubeA position.
2. Sample a goal `p_goal = cubeA_init + Δ` where `Δ` is a random 2D
   displacement on the table plane, magnitude `0.10 ± 0.02 m`.
3. Plan + execute. Score by final Euclidean distance from cubeA to
   p_goal.

Goal magnitude was chosen so that a single v3 scripted push (which
moves cubes 5-6cm on average) is *insufficient*; a deliberate plan is
needed.

## Setup

| | Value |
|---|---|
| Env | robosuite Stack (same as Phase 14) |
| Action dim | 7 (Panda OSC) |
| Episodes | 50 per (mode, K) |
| Planning horizon | H = 5 stride-boundaries = 20 env frames ≈ 1 sec |
| JEPA stride k | 4 (matches training) |
| CEM iterations | 4 |
| Candidate count K | 128 (main); 64, 128, 256 for efficiency curve |
| Elite fraction | 20% |
| Initial action prior | N(0, 0.5), clipped to [-1, 1] |
| Models | re-trained from v3 (seed 0, 1500 steps; identical to 14.5/14.6) |

## Modes

| Mode | Planning predictor | Notes |
|---|---|---|
| **cem_action** | action-conditioned OF-JEPA | the test |
| **cem_noaction** | no-action OF-JEPA | mechanically degenerate — predictor doesn't differentiate by action, so CEM elite-selection is ≈ random. Reported for completeness. |
| **random** | none (single sample from prior) | tests whether planning matters at all |
| **gt_scripted** | scripted v3-style push aimed at goal | oracle skyline / "can the task even be solved" |

## Pre-committed gates — REVISED 2026-05-17

Original gates (success rate + mean final distance) had two empirically
discovered issues during oracle calibration:

1. Open-loop position goal + fixed horizon means GT-scripted oracle
   tops out at ~20% success rate (overshoots when given more time).
2. Mean final distance is gameable by a "don't move" baseline: random
   policy scored lower mean_dist (0.064) than oracle (0.085) because
   non-moving cubes park at goal_magnitude while pushed cubes can
   overshoot.

Revised primary metric: **improvement** = max(0, (start_dist - end_dist) / start_dist).
Range [0, 1]. No-movement → 0. Closing 20% of goal gap → 0.20. Perfect
push → 1.0. Overshoot is penalized through end_dist.

```
G1. improvement(cem_action) - improvement(cem_noaction) >= 0.10
       (action conditioning closes ≥10pp more of the goal gap than no-action CEM)

G2. improvement(cem_action) >= 0.20
       (action conditioning closes ≥20% of the goal gap on average)

G3. cem_action achieves G1-level improvement with K' <= 64
       (action-conditioned needs ≤50% of K=128 candidates)
```

### Diagnostics (logged, not gated)
```
dir_score    = cos_sim(cube_displacement, goal_direction) × 1[|displacement| > 2cm]
               — does the planner push in roughly the right direction?

success rate = 1[final_dist <= 5cm]
               — kept as a secondary indicator; not gated because the
                 open-loop oracle ceiling is only ~20% at this goal size

mean_dist, median_dist, mean_displacement, pred-actual correlation
```

If gates fail, this disentangles "predictor isn't planner-grade"
(low corr) from "search is the bottleneck" (high corr, gates still fail).

## Joint verdict matrix

| Pass count | Verdict |
|---|---------|
| **3/3** | **OF-JEPA action model supports planning.** Push to Phase 16 (closed-loop, longer-horizon, or task-completion). |
| 2/3 | **Partial planner usefulness.** Identify which gate failed; either tune CEM hyperparameters or the predictor head. |
| 1/3 | **Mostly not planner-grade.** Predictor's action→effect is too coarse for choice; investigate (a) whether trained-policy head helps, (b) whether the encoder slot→entity mismatch hurts goal-position assignment. |
| 0/3 | **Predictions don't transfer to planning.** Phase 14.6's offline ranking generalization didn't translate to closed-loop choice. Stronger negative result; redesign needed. |

## Why CEM, not a trained policy head

A trained policy head would confound the experiment: did the policy
improve because the model's predictions are planner-grade, or because
the policy's own training escaped a bad prediction? CEM directly asks
the world-model question:

> Are the model's predicted futures good enough to choose better
> actions?

Policy training is Phase 16. Phase 15 keeps the test pure.

## What this does NOT establish

1. **No real-task success.** Cube-displacement is simpler than
   stacking. Phase 15+1 would test the full Stack task.
2. **Single seed for training.** Plan-time stochasticity is averaged
   over 50 episodes; training stochasticity is not. Multi-seed
   retrain optional follow-up.
3. **CEM is one search method.** Other planners (MPPI, gradient-based
   shooting) might extract more from the same model. Not tested.
4. **Goal-from-initial-frame only.** We sample the goal from the
   initial cubeA position; we don't test multi-step goal reasoning
   ("first push cubeA into cubeB").

## Reproducibility — pod runbook

```bash
# (1) Retrain v3 models with checkpoint save (~60 min). Also runs
#     the planning eval in the same script.
python3 scripts/phase15_planning.py \
  --train-cache /workspace/robosuite_local/stack_scripted \
  --seed 0 --max-steps 1500 --jepa-stride 4 \
  --n-episodes 50 --plan-horizon 5 --cem-iters 4 \
  --candidate-counts 64,128,256 --main-K 128 \
  --modes cem_action,cem_noaction,random,gt_scripted \
  --out /workspace/phase15_planning
```

Artifacts: `artifacts/phase15_planning/{summary.json, per_episode_*.jsonl}`
