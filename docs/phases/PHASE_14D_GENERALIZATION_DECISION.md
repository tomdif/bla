# Phase 14.6 (Action-Conditioning Generalization) — Decision document

**Date:** 2026-05-17.
**Status:** ✅✅✅ **3/3 CLEAN GENERALIZATION.** All three pre-committed
gates pass on every perturbation axis. Locks Phase 14.5's strong-pass
claim: the action-conditioned OF-JEPA learned **action effects**, not
the specific scripted-policy distribution.

> **Headline:** Phase 14.6 shows clean OOD generalization of
> planner-facing action effects. The action-conditioned model remains
> above 2× chance on action ranking and improves future position
> prediction under strength, noise, and horizon shifts. Latent state
> matching degrades OOD, confirming that action conditioning should
> be evaluated through planner-facing predictions rather than
> perceptual self-loop state MSE.

> Phase 14.6 trained on scripted v3 and evaluated on three perturbed
> held-out eval sets (strength jitter, noise jitter, longer horizon).
> Pre-committed gates and verdict matrix were locked in
> `PHASE_14D_GENERALIZATION_PRECOMMIT.md` before training.

## Setup

| | Train | Strength eval | Noise eval | Horizon eval |
|---|---|---|---|---|
| Episodes | 200 (v3) | 50 | 50 | 50 |
| OSC gain | 10 (fixed) | U[4, 16] | 10 (fixed) | 10 (fixed) |
| Action noise σ | 0.20 (fixed) | 0.20 | U[0.05, 0.45] | 0.20 |
| Horizon | 80 | 80 | 80 | **120** |
| Mean cube disp | a=0.056 b=0.062 | a=0.062 b=0.058 | a=0.057 b=0.074 | a=0.060 b=0.087 |

Training: 1500 steps × 2 modes, identical to Phase 14.5. Single seed.

## Headline numbers

```
                 |  in-dist (14.5)  |  strength  |  noise   |  horizon
─────────────────┼──────────────────┼────────────┼──────────┼──────────
top1_hit_rate    |  0.343           |  0.312     |  0.336   |  0.322
pos_mse ratio    |  0.895           |  0.877     |  0.864   |  0.903
state_mse ratio  |  0.956           |  1.558     |  1.721   |  1.483
```

(`ratio` = +action / baseline; ≤ 1.0 means action helps.)

Three pre-committed gates and verdicts:

| Perturbation | G1 top1 ≥ 0.25 | G2 pos_ratio ≤ 0.95 | Verdict |
|---|---|---|---------|
| **strength** | 0.312 ✅ | 0.877 ✅ | **PASS** |
| **noise** | 0.336 ✅ | 0.864 ✅ | **PASS** |
| **horizon** | 0.322 ✅ | 0.903 ✅ | **PASS** |

**Joint verdict per pre-committed matrix: 3/3 clean.**

## What this establishes

1. **The action-conditioning advantage transfers across OOD action
   distributions.** Strength jitter U[4, 16] is a 4× spread around
   the trained gain of 10 — the predictor still ranks the actual
   action at 2.5× chance. Noise σ jitter U[0.05, 0.45] is a 9× spread
   — same result.

2. **The advantage transfers across longer horizons.** 120 frames
   vs 80 trained — cube reaches positions further from any state seen
   in training, yet planner-facing metrics survive.

3. **In-distribution metrics are only mild upper bounds on OOD
   performance.** top1 dropped 1-9% across perturbations; pos_ratio
   was actually *better* than in-dist on 2/3 perturbations (closer to
   0.5 means more relative gain). This suggests the action→effect
   structure the model learned is a real coarse-grained invariant of
   the dynamics, not a fitted artifact.

## What the secondary state_mse diagnostic tells us

Action-conditioned **state matching** degrades 1.48-1.72× under OOD,
vs in-distribution 0.956× (action slightly better than base). This was
explicitly downweighted in the precommit ("secondary diagnostic, not
gated") but is worth interpreting:

> The action-conditioned predictor's *latent* state map is more
> brittle to action-distribution shift than the unconditioned one.
> The unconditioned predictor essentially does autoregressive state
> extrapolation (smooth dynamics); under OOD action the smoothness
> still applies. The conditioned predictor pays attention to action;
> if action distribution shifts, its predictions drift further from
> the encoder-state target.

**Important:** this brittleness does NOT propagate to the gated metrics
because:
- `pos_mse` is decoded position (Phase 14.3 finding), which uses the
  predicted state-delta + the aux head. The aux head appears to
  partially undo state-prediction drift when decoding to position.
- `top1` is comparative across candidate actions, not absolute. The
  predictor can drift in absolute terms while still being
  discriminative: the actual action's prediction is still closer to
  GT than alternatives, even if all candidates are shifted.

This is the Phase 14.3/14.5 split-result pattern at finer granularity:
**action conditioning is robustly load-bearing for action-relevant
readouts (position, ranking); it's brittle for self-loop state
matching.** The Phase 14.4 architectural take ("action belongs in
planner readouts, not perception") is partially vindicated *for the
OOD setting* even though Phase 14.5 falsified it for in-distribution.

The honest architectural read:

> Action conditioning is universally useful for downstream planner
> readouts. For perceptual self-loop state matching, it's beneficial
> in-distribution and brittle out-of-distribution. The cleanest
> production design is the split-head pattern (one action-conditioned
> planner head + one unconditioned perception head) — recommended in
> 14.4, softened in 14.5, **reaffirmed in 14.6 with finer
> resolution**.

## What this does NOT establish

1. **No active behavior test.** The model can rank candidate actions;
   it has not been used to *select* actions by a policy or planner.
   That belongs in Phase 15 (behavioral transfer) or System-2 plan
   evaluation.
2. **Single seed.** All gates pass with margin, so seed sensitivity
   is unlikely to flip the verdict, but 3-seed confirmation is
   standard hygiene for the writeup.
3. **One task family.** All perturbations stay inside robosuite Stack.
   Cross-task transfer (e.g. Can/PickPlace) is a separate Phase.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 JEPA | ✅ | slot_delta spatial memory |
| 7-8A | ❌×5 | content-side identity fixes falsified |
| 8C/8D | ✅ | OF-JEPA v0: identity-as-address |
| 9 | ✅ | MOVi-D identity transfer (corrected metric) |
| 10 | ✅ | refactor + first-class metrics |
| 12 | ✅ | relations add 5-9% on single-slot readouts |
| 13.3 | ✅ | OF-JEPA transfers to CLEVRER |
| 13.4 | ⚠ | relations redundant when readout has pairs |
| 14.3 | ⚠ | action conditioning split on RANDOM data |
| 14.4 | ❌ | action ranking at chance on RANDOM data — wrongly architecture-blamed |
| 14.5 | ✅ | action conditioning IS load-bearing with informative data |
| **14.6** | **✅✅✅** | **action conditioning GENERALIZES across action distribution shifts** |

## Architectural take (full Phase 14 sweep)

> *Action conditioning is data-informativeness-bound for in-distribution
> use, distribution-generalizable for planner-facing readouts, and
> brittle for self-loop perception under OOD action shift. The
> production design is split-head: action-conditioned planner readouts
> + unconditioned perception loop. Object files remain the load-bearing
> primitive throughout.*

Phase 14 closes.

## Reproducibility

Data: collected in Phase 14.5 (`stack_scripted`) + three perturbed sets
via patched collector:

```bash
# Three perturbed eval sets (~15 min total)
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

# Train once on v3, eval on three perturbed sets (~70 min)
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

Artifacts: `artifacts/phase14d_generalization/{seed0_*.json,summary.json}`
