# Phase 18B — Learned Proposal Policy Distillation Precommit

**Date:** 2026-05-18.
**Status:** precommitted before run.

## Goal

Phase 18D locked the current strongest recipe:

```
OF-JEPA predictor + scripted goal-conditioned prior + CEM refinement
```

This phase asks whether the scripted prior can be replaced by a learned
proposal policy distilled from the CEM-refined teacher. The world model
stays fixed. The only thing being tested is the source of CEM's initial
action-sequence mean.

## Method

1. Load the Phase 17 mixed-data OF-JEPA action-conditioned checkpoint.
2. Collect teacher examples from `scripted_prior_cem`:
   - At each MPC replan boundary, record `state_features(obs, goal)`.
   - Run the existing scripted-prior+CEM teacher.
   - Store the refined full action plan `[plan_horizon, action_dim]`.
3. Train `PlanProposalPolicy`:
   - Input: 10-dim geometric state+goal features.
   - Output: full horizon action sequence.
   - Loss: weighted plan MSE, with earlier actions weighted more because
     MPC executes only `replan_every` actions before replanning.
4. Evaluate:
   - `gt_closed_loop`
   - `scripted_prior_cem` (heavy teacher baseline)
   - `learned_policy_only`
   - `learned_policy_cem` (policy prior + light CEM)
   - `scripted_prior_light_cem` (diagnostic compute match)
   - `naive_cem`

Script:

```bash
python3 scripts/phase18b_policy_distill.py \
  --model-action /workspace/phase17/model_action_finetuned.pt \
  --teacher-episodes 120 \
  --policy-train-steps 2500 \
  --policy-K 32 --policy-cem-iters 1 \
  --n-episodes 30 \
  --out /workspace/phase18b
```

## Gates

Let `T = scripted_prior_cem`, `P = learned_policy_only`, and
`L = learned_policy_cem`.

**G1: policy-only keeps most of the teacher.**

```
P.improvement >= 0.75 * T.improvement
```

This tests whether the learned policy has actually absorbed the prior,
not merely produced contact-ish noise.

**G2: policy + light CEM matches heavy scripted teacher.**

```
L.improvement >= T.improvement - 0.02
```

A 2pp tolerance is allowed because seed/env variance at 30 episodes is
visible, but Phase 18D showed the planner-oracle gap itself can be
measured to about 1pp over three seeds.

**G3: policy + light CEM is materially cheaper.**

```
L.mean_candidates / T.mean_candidates <= 0.35
```

Default teacher compute is `3 replans * 3 iters * 128 = 1152`
candidate scores per episode. Default learned-light compute is
`3 replans * 1 iter * 32 = 96`, about 8.3% of teacher compute.

## Interpretation Matrix

| Result | Interpretation |
|---|---|
| G1+G2+G3 pass | Learned prior replaces scripted prior; proceed to larger learned-policy loop. |
| G1 fail, G2 pass | Policy is not good alone, but provides enough mean for light CEM; distill more or add recurrent state. |
| G1 pass, G2 fail | Policy learned teacher average but OF-JEPA cannot refine it cheaply; increase light-CEM budget or improve calibration. |
| G3 fail only | Behavior works but compute economy claim fails; tune `policy_K`, `policy_cem_iters`, or sigma. |
| G1+G2 fail | Keep scripted prior; policy data/capacity is insufficient. |

## Non-goals

- No OF-JEPA architecture change.
- No new perception checkpoint unless Phase 18B fails in a way that
  points back to predictor calibration.
- No claim of cross-task transfer; still robosuite Stack.

## Files

- `system1_jepa/planning_policy.py`
- `scripts/phase18b_policy_distill.py`
- `tests/test_planning_policy.py`
