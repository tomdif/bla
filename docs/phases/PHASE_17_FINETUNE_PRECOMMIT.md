# Phase 17 (Focused-contact predictor fine-tuning) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 16 confirmed Phase 15b's prior-bottleneck diagnosis (scripted
goal-directed prior gets imp=0.143 ≈ 80% of oracle), but uncovered a
new failure mode at the predictor layer:

```
predictor correlation (pred-vs-actual):
  Phase 15b naïve_cem (broad scripted candidates):     +0.528
  Phase 16  scripted_prior_cem (focused contact):      −0.387
```

The action-conditioned OF-JEPA was trained on v3's broad scripted
distribution. When CEM samples cluster around a goal-directed push
prior, candidates are OOD relative to training and the predictor
ranks them backwards. Result: CEM-refined improvement (0.143) is
*worse* than prior alone (0.180).

Phase 17 tests whether **retraining the predictor on focused-contact,
goal-directed action distributions** restores its ranking signal in
that regime — letting OF-JEPA + CEM finally improve on a competent
prior.

## The question

> If we include focused-contact + goal-directed actions in the
> predictor's training distribution, does its ranking become
> reliable (corr > 0) on those candidates, and does CEM-refinement
> improve on the scripted prior?

## Setup

**Data**:
- v3 broad-distribution (200 eps, existing `stack_scripted` cache)
- goal-directed focused-contact (200 fresh eps via
  `closed_loop_gt_step` with random goals + perturbations)

**Training**:
- Fresh action-conditioned OF-JEPA, 1500 steps, identical hyper-
  parameters to Phase 15
- 50/50 mix of v3 + goal-directed at each batch
- Saves `model_action_finetuned.pt`

**Evaluation**: re-run Phase 16's exact MPC eval with the new model
(scripted_prior_cem mode, oracle skyline, naïve baseline).

## Pre-committed gates

```
G1. predictor correlation > 0.30 on scripted-prior CEM candidates
       (the calibration regime that was broken in Phase 16)

G2. CEM refinement improves scripted prior by >= 5 percentage points
       improvement(scripted_prior_cem, finetuned model) -
       improvement(gt_closed_loop) >= 0.05
       i.e., 0.143 → 0.18+ (which would be ≥ oracle)
       OR
       improvement(scripted_prior_cem, finetuned) >= 0.18

G3. improvement(scripted_prior_cem, finetuned) >= 0.16
       (approaching oracle 0.18; lower bound to count as "near-skyline")
```

A strong result:
```
scripted_prior (oracle):              0.180
scripted_prior + finetuned CEM:       >= 0.18 (matches or exceeds)
v3-trained CEM (Phase 16 baseline):   0.143
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **3/3** | **OF-JEPA-as-planner is data-distribution-fixable.** The world-model + CEM-refinement recipe works when the predictor sees the deployment-time action distribution during training. Push to Phase 18 (longer-horizon, cross-task generalization). |
| 2/3 | Predictor calibration improves but CEM-refinement still doesn't beat the prior. Diagnose: predictor better but search still wastes its ranking. |
| 1/3 | The calibration improves on isolated candidates but doesn't lift downstream planning. Phase 16's deeper finding stands: world model is offline-only. |
| 0/3 | Even with in-distribution training, CEM-on-OF-JEPA doesn't beat a competent prior. Strong negative for closed-loop OF-JEPA usage. |

## Diagnostics (logged, not gated)

```
contact rate              — should remain >= 0.90 (prior-driven)
direction score           — should improve over Phase 16's 0.063
overshoot rate
mean displacement
predicted-vs-actual scatter plot (saved as artifact)
training-data composition: 50/50 mix vs all-focused vs all-v3
```

## Why fine-tune via mixture, not pure retrain on focused data

Pure retrain risks catastrophic forgetting of broad-distribution
generalization (Phase 14.6's OOD robustness). 50/50 mix preserves
the v3 distribution coverage while adding focused-contact regime.
If mix doesn't work, pure focused-retrain is the obvious follow-up.

## Reproducibility — pod runbook

```bash
# 1. Collect 200 goal-directed-push rollouts (~10 min)
python3 scripts/robosuite_collect_rollouts.py \
  --task Stack --n-episodes 200 --horizon 80 \
  --policy goal_directed_push \
  --out /workspace/robosuite_local/stack_goal_directed

# Smoke test: cube_a_disp_mean should be >= 0.06m
# (closed_loop_gt_step is more cube-effective than v3)

# 2. Train mixed-data model + re-run Phase 16 eval (~90 min)
python3 scripts/phase17_finetune.py \
  --train-caches /workspace/robosuite_local/stack_scripted,\
/workspace/robosuite_local/stack_goal_directed \
  --train-mix 0.5,0.5 \
  --max-steps 1500 --jepa-stride 4 --seed 0 \
  --model-out /workspace/phase17/model_action_finetuned.pt

# 3. Re-run Phase 16 eval with the new model
python3 scripts/phase16_policy_prior_mpc.py \
  --model-action /workspace/phase17/model_action_finetuned.pt \
  --seed 0 --jepa-stride 4 \
  --total-actions 15 --replan-every 5 --plan-horizon 10 \
  --cem-iters 3 --main-K 128 \
  --modes gt_closed_loop,naive_cem,scripted_prior_cem \
  --n-episodes 30 --oracle-sanity-n 30 \
  --oracle-min-improvement 0.10 --oracle-min-contact 0.60 \
  --out /workspace/phase17_eval
```

Artifacts: `artifacts/phase17/{summary.json, per_episode_*.jsonl}`,
`model_action_finetuned.pt`
