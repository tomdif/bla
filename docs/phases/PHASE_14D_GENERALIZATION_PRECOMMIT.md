# Phase 14.6 (Action-Conditioning Generalization) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

The follow-up to Phase 14.5's strong pass. 14.5 proved action
conditioning works *when training data is informative*. The open
question 14.6 answers: did the model learn **action effects**, or
just **the specific scripted-policy distribution**?

## The question

> Train on scripted v3. Eval on perturbed-policy rollouts that share
> the underlying physics (same robosuite Stack task, same robot, same
> action space) but differ in surface action statistics (controller
> gain, noise level, horizon length).
>
> Does the action-conditioned model's ranking and position-prediction
> advantage transfer?

If yes: the model has learned the action→effect map, generalizable.
If no: the model has memorized the script distribution, fragile.

## Setup

| | Value |
|---|---|
| Train | scripted v3 cache (200 episodes, gain=10, σ=0.20, horizon=80) |
| Eval sets | 3 perturbed caches × 50 episodes each |
| Training | identical to Phase 14.5 (1500 steps, k=4 stride, lr=1e-4, seed 0) |
| Models | of_jepa_v0 (baseline) + of_jepa_v0_action |

### Perturbed eval sets

**A. Strength shift.** OSC gain ~ uniform[4, 16] per episode (v3 was
fixed at 10). Tests action-magnitude generalization.

**B. Noise shift.** Action noise σ ~ uniform[0.05, 0.45] per episode
(v3 was fixed at 0.20). Tests robustness to action stochasticity.

**C. Horizon shift.** 120 frames per episode (v3 was 80). Tests
longer-trajectory generalization.

(The user's original list also mentioned angle and cube-init jitter.
Those are already randomized in v3 — uniform sweep angle, robosuite
random reset — so they're in-distribution, not OOD. Dropped.)

## Pre-committed gates (per perturbation)

```
G1. top1_hit_rate(+action) ≥ 0.25      # ≥ 2× chance (0.125)
G2. pos_mse(+action) / pos_mse(base) ≤ 0.95   # action helps position decode
```

Secondary diagnostic (logged but not gated):
```
state_mse(+action) / state_mse(base)   # Phase 14.5 baseline was 0.956
```

## Joint verdict matrix

| Pass count | Verdict |
|---|---------|
| 3/3 | **Clean generalization.** Action conditioning learned EFFECTS, not script statistics. Locks the Phase 14.5 claim — push to Phase 15. |
| 2/3 | **Partial generalization.** Identify the broken axis; understand why; consider data augmentation OR a targeted architecture adjustment for that axis. |
| 1/3 | **Mostly script-bound.** Action conditioning has narrow generalization. Requires harder data distribution (Can/PickPlace) or active-exploration data before trusting beyond v3-like distributions. |
| 0/3 | **Overfit to script.** Phase 14.5's pass was distribution-specific; the architectural claim doesn't transfer. Stronger negative than 14.4 (which only said data-bound on random); this would say bounded-by-script-too. |

## Threshold rationale

- **G1 = 0.25** (vs Phase 14.5's 0.343): allows some degradation but
  requires meaningful discriminative signal. 0.25 = 2× chance, which
  is what the user explicitly proposed. Below 0.20 we'd be in noise
  territory.
- **G2 = 0.95** (same as Phase 14.5): identical threshold to enable
  apples-to-apples comparison. If +action stops helping pos_mse under
  perturbation, the effect was distribution-specific.
- **state_mse not gated.** Phase 14.5 hit −4.4% (action slightly
  better); on OOD perturbation, +5% is plausible. Don't gate on a
  secondary signal that the user explicitly downweighted.

## What this does NOT test

- Active behavior. We're still measuring static prediction quality.
- Larger distribution shifts (cross-task, cross-robot). 14.6 stays
  inside the same robosuite Stack env.
- Multi-seed stability. Single seed; if 14.6 marginally fails on one
  axis, 3-seed retry might tighten the call.

## Reproducibility — pod runbook

```bash
# 1. Collect three 50-episode perturbed eval sets (~15 min total).
for KIND in strength noise horizon; do
  python3 scripts/robosuite_collect_rollouts.py \
    --task Stack --n-episodes 50 --policy scripted_push \
    $(case $KIND in
       strength) echo "--gain-range 4,16 --horizon 80" ;;
       noise)    echo "--gain-range 10,10 --noise-range 0.05,0.45 --horizon 80" ;;
       horizon)  echo "--gain-range 10,10 --noise-range 0.20,0.20 --horizon 120" ;;
      esac) \
    --out /workspace/robosuite_local/stack_perturb_$KIND
done

# 2. Train once on v3, eval on three perturbed sets (~60 min).
python3 scripts/phase14_generalization.py \
  --train-cache /workspace/robosuite_local/stack_scripted \
  --eval-caches /workspace/robosuite_local/stack_perturb_strength,\
/workspace/robosuite_local/stack_perturb_noise,\
/workspace/robosuite_local/stack_perturb_horizon \
  --eval-labels strength,noise,horizon \
  --modes of_jepa_v0,of_jepa_v0_action \
  --seed 0 --max-steps 1500 --jepa-stride 4 --k-candidates 8 \
  --out /workspace/phase14d_generalization
```
