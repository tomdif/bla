# Phase 16 (Policy-prior CEM / BC-warm-started MPC) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 15b proved OF-JEPA's action-conditioned predictor is calibrated
(pred-actual corr=0.53) but naïve Gaussian-around-zero CEM doesn't
find contact-rich action sequences (contact 23-27% vs oracle 73%),
so the predictor's ranking ability is wasted on a candidate set that
can't move the cube regardless of which is "best."

Phase 16's question:

> If we give OF-JEPA's predictor a **contact-aware action prior** to
> refine — rather than discovering contact from scratch — does
> closed-loop planning become viable?

This isolates the missing layer (action proposal) and tests whether
"world model + proposal policy + refinement" works as a model-based
control recipe.

## The methods to compare

```
1. naïve CEM                — N(μ=0, σ=0.5) prior (Phase 15b baseline)
2. scripted-contact CEM     — v3-style approach-then-push prior + Gaussian noise
3. BC-prior CEM             — small MLP trained on scripted v3 actions
                              (BC = behavioral cloning), then CEM refines
4. oracle skyline           — closed-loop adaptive scripted policy (unchanged)
```

Same OF-JEPA predictor everywhere. Same MPC infrastructure (replan
every 5 actions, plan horizon 10, 3 CEM iters, K=128 main). Different
proposal distribution.

## Pre-committed gates

```
G1. contact rate (BC-prior CEM) >= 0.50
       (action prior actually engages the cube on most episodes)

G2. improvement (BC-prior CEM) >= 0.10
       (matches oracle floor; tests "with the right prior, OF-JEPA plans")

G3. improvement (BC-prior CEM) - improvement (naïve CEM) >= 0.05
       (the prior change is what unlocked it, not noise)
```

### Diagnostics (logged, not gated)

```
predicted-vs-actual correlation
G3'. scripted-contact CEM improvement - naïve CEM improvement
     (does scripted prior alone help, or do we need BC?)
improvement gap to oracle (BC-prior vs gt_closed_loop)
overshoot rate, dir_score (sanity)
```

## Joint verdict matrix

| Pass | Verdict |
|---|---------|
| **3/3** | **OF-JEPA is planner-grade given a competent action prior.** Strong evidence for "world model + policy" architecture. Push to Phase 17 (longer horizon, more complex tasks). |
| 2/3 | Partial: the prior helps but doesn't fully unlock planning. Likely G1 (contact) passes but G2 (absolute improvement) doesn't reach oracle floor. Diagnose. |
| 1/3 | Prior helps but not enough. Likely needs a better-trained BC policy, MPPI instead of CEM, or more candidates. |
| 0/3 | The Phase 15b conclusion doesn't generalize. Either prior alone isn't enough OR the predictor isn't actually planner-grade under closed-loop conditions. |

## Why BC + CEM, not just BC

BC alone tests the action prior but doesn't use OF-JEPA at all.
BC + CEM lets the prior do "warm start" but lets OF-JEPA's predictor
refine action choice via score-based elite selection. This is the
canonical "model + policy + search" recipe in model-based RL.

If BC alone matches BC+CEM, the predictor isn't adding value (Phase
15b conclusion stands). If BC+CEM clearly beats BC alone on
improvement, that proves OF-JEPA's planner-grade contribution is
real but needed the right prior to surface.

## Implementation sketch

### Scripted-contact prior (no training)

At plan-start, generate a base trajectory from the v3 collector's
scripted_push policy aimed at the current goal. Add Gaussian noise.
CEM iterates around this base.

### BC prior (one new training run)

Tiny MLP: `state → action`. Inputs: encoded slot states from OF-JEPA's
encoder, decoded cubeA position via aux head, current goal_xy. Output:
7-dim action. Trained on (state, action) pairs from v3 rollouts.
~10 min train.

For planning: roll out BC policy for full horizon to get a base
trajectory. Add CEM noise. Iterate.

### Candidate sampling
```
μ_t  = base_trajectory_t                  (instead of 0)
σ_t  = 0.3                                (smaller than 0.5 — refining around an informative prior)
cands = μ + σ * randn(K, H, A); clipped
```

## What this does NOT establish

- **Cross-task transfer.** Same env, same goal type, same training data.
- **Trained-policy-end-to-end.** This is BC for the prior + CEM for
  refinement. End-to-end policy training is Phase 17 or later.
- **Multi-seed.** Single seed. Phase 15b oracle showed substantial
  reset-variance; multi-seed confirmation is standard hygiene.

## Reproducibility — pod runbook

```bash
# Existing scripted_push collector already on pod (Phase 14).
# Train BC head (~10 min):
python3 scripts/phase16_bc_train.py \
  --train-cache /workspace/robosuite_local/stack_scripted \
  --out /workspace/phase16/bc_policy.pt --seed 0 --max-steps 600

# MPC with prior-aware CEM (reloads model_action.pt + bc_policy.pt):
python3 scripts/phase16_policy_prior_mpc.py \
  --train-cache /workspace/robosuite_local/stack_scripted \
  --bc-policy /workspace/phase16/bc_policy.pt \
  --model-action /workspace/phase15_mpc/model_action.pt \
  --seed 0 --jepa-stride 4 \
  --total-actions 15 --replan-every 5 --plan-horizon 10 \
  --cem-iters 3 --main-K 128 \
  --modes naive_cem,scripted_prior_cem,bc_prior_cem,gt_closed_loop \
  --n-episodes 30 --oracle-sanity-n 30 \
  --out /workspace/phase16
```

Artifacts: `artifacts/phase16/{summary.json, per_episode_*.jsonl, bc_policy.pt}`
