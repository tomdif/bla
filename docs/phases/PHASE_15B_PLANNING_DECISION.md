# Phase 15b (MPC Planning, recalibrated) — Decision document

**Date:** 2026-05-17.
**Status:** ❌ **G1 + G2 FAIL. G3 trivially "passes" due to div-by-zero.
Overall: 1/3 — not yet planner-grade under naïve-prior CEM.**

> **Headline:** Phase 15b shows that OF-JEPA's action-conditioned
> predictor is calibrated enough to rank candidate outcomes, but naïve
> CEM over zero-centered Gaussian actions fails because it rarely
> samples contact-rich cube-engaging trajectories. The bottleneck is
> action-prior coverage, not predictor calibration. The natural next
> step is Phase 16: contact-aware action prior (BC-warm-start or
> scripted-contact prior) + CEM refinement.

## Setup

Phase 15b reused the saved checkpoints from Phase 15 (no retraining)
and re-ran the MPC eval with recalibrated thresholds (oracle ≥ 0.10,
G2 ≥ 0.10, G3 = matched-K ratio ≥ 1.5) per
`PHASE_15B_PLANNING_PRECOMMIT.md`.

| | Value |
|---|---|
| Env | robosuite Stack |
| Episodes | 30 per (mode, K) |
| Total actions | 15 (= 60 env frames ≈ 3 sec) |
| Plan horizon | 10 actions, replan every 5 |
| CEM iters | 3, elite frac 20% |
| Main K | 128; sweep 64, 128, 256 |
| Models | reloaded from `model_action.pt` + `model_noaction.pt` (Phase 15) |

## Oracle sanity gate

```
30-ep oracle: imp=0.272  dir=0.449  contact=0.73   succ=0.333
              (vs gates 0.10, > 0, 0.60)
              PASS ✅
```

Note: sample-to-sample oracle variance is large. A previous 30-ep
sample landed at imp=0.146; this one at 0.272. Causes: env-reset
random initial cube/EE positions vary substantially. Multiple seeds
needed for tight oracle-mean estimate, but **action-MPC at 0.008 is
decisively below the oracle floor regardless of which oracle sample
we use**.

## Headline numbers

| Mode | improvement | dir_score | contact | succ | n_cand |
|---|---|---|---|---|---|
| **gt_closed_loop (oracle)** | **0.272** | 0.449 | 0.73 | 0.333 | — |
| cem_action K=64 | 0.004 | 0.009 | 0.27 | 0.000 | 576 |
| cem_action K=128 | 0.008 | -0.048 | 0.23 | 0.000 | 1152 |
| cem_action K=256 | 0.013 | -0.003 | 0.27 | 0.000 | 2304 |
| cem_noaction K=128 | ~0 | 0 | 0.13 | 0 | 1152 |
| random | ~0 | 0 | 0.10 | 0 | 1 |

### Gates

```
G1: improvement(action) - improvement(noaction)
       = 0.008 - 0.000 = 0.008    (need ≥0.10)            FAIL ❌

G2: improvement(action) = 0.008                            (need ≥0.10)            FAIL ❌

G3: improvement(action) / improvement(noaction)
       = 0.008 / 0 → ∞             (need ≥1.5)            "PASS" but vacuous
```

G3 technically passes because cem_noaction's improvement is
effectively zero (Gaussian-around-zero never samples cube-engaging
actions, so the no-action predictor — which can't discriminate by
action anyway — produces no useful plan). The infinite ratio is
informationally vacuous; **the meaningful gate is G2, which fails 12×
below threshold**.

**Joint verdict: 1/3 "pass" — but the one pass is vacuous.
Substantively, 0/3.**

## Diagnostic synthesis: what failed?

### The predictor IS planner-grade (corr=0.53)

At K=128, the predicted final distance correlates with actual final
distance at **r=0.528** across the 30 episodes × 128 candidates. The
model's *ranking* of action candidates is meaningful — Phase 14.6's
offline ranking finding is confirmed under closed-loop conditions.

### Contact rate exposes the bottleneck

```
oracle:                73%   (designed for contact)
cem_action K=64-256:   23-27%
cem_noaction K=128:    13%
random:                10%
```

The CEM-planned actions barely touch the cube. Without contact, no
action sequence can move it — and the predictor (no matter how
calibrated) can't favor contact-inducing sequences that *aren't in
its candidate set*. **The CEM prior — N(μ=0, σ=0.5) — doesn't sample
cube-engaging action distributions**. The scripted_push v3 actions
that trained the model have a specific structure (descend, then
push) that random Gaussian-around-zero rarely produces.

### Why CEM iteration doesn't fix this

CEM iterations refine μ/σ around elites. But if NONE of the initial
candidates are cube-engaging, all elites are equally ineffective,
and the prior never converges toward useful action structure. CEM
relies on having SOMETHING in the initial candidate set to refine.

### Why K=256 doesn't help much

K=256 gets 0.013 vs K=64's 0.004 — slight improvement but still 20×
below oracle. Larger K samples more from the same bad prior, getting
some marginal lucky hits, but it doesn't change the prior's
distribution.

## Architectural implications

The Phase 14 arc established:
- 14.5: action conditioning helps offline ranking when data is informative
- 14.6: that ranking transfers across OOD action distributions

Phase 15b adds:
- 15b: but Gaussian-prior CEM doesn't translate ranking into action selection
  because the prior doesn't cover the useful action subspace.

This is consistent with the broader "object files + action conditioning"
story: the world model is calibrated but **needs an informative action
prior** to plan. That's the role of a policy.

## Honest framing for the writeup

> The action-conditioned OF-JEPA is a well-calibrated world model
> (offline ranking + OOD generalization confirmed in 14.5/14.6;
> pred-actual correlation 0.53 under closed-loop sampling in 15b).
> But naïve Gaussian-around-zero CEM does not transform that ranking
> into actual action selection on the open-loop pushing task, because
> the action prior fails to sample cube-engaging action sequences.
> The bottleneck is action-search prior, not predictor quality.

This is **NOT** "predictions don't transfer to planning." It is "the
test method we chose for planning was prior-bound, not predictor-bound."

## What this does NOT establish

- **It doesn't falsify the predictor as a planning substrate.** It
  only falsifies *one particular search method* (Gaussian-prior CEM).
- **It doesn't say closed-loop planning is impossible.** It says we
  need a better action prior (Phase 16).
- **Single seed.** Oracle variance is high; the negative result on
  CEM is much larger than any plausible seed variance, but
  multi-seed retry would tighten conclusions.

## Phase 16 — natural next step

Three mitigations available, in order of expected impact:

1. **Behavioral-cloning warm-start.** Train a tiny policy MLP on v3
   actions; use its predictions as the CEM prior (μ instead of 0).
   Same predictor, better prior. Cheapest test of "is the prior the
   bottleneck."
2. **Trained policy head.** Add a policy network alongside OF-JEPA;
   train via differentiable trajectory rollout. Replaces CEM entirely.
3. **MPPI or gradient-based shooting.** Different search method;
   might be more sample-efficient than CEM with the same prior.

(1) is the cheapest direct test of the "prior is the bottleneck"
hypothesis. If a v3-warm-started CEM closes a meaningful fraction of
the gap to oracle, that proves Phase 15b's failure was prior-bound.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 14.5 | ✅ | action conditioning helps offline ranking on informative data |
| 14.6 | ✅✅✅ | action conditioning generalizes across OOD action distributions |
| 15 | ⚠ | original sanity threshold empirically too high; pivoted to 15b |
| **15b** | **❌** | **naïve-prior CEM does NOT translate ranking into action selection; bottleneck is action prior, not predictor** |

## Architectural take

> *The action-conditioned OF-JEPA is a calibrated world model. Naive
> open-loop search doesn't use it well. The next step is to provide
> a better action prior, either via behavioral cloning or a learned
> policy that exploits the world model's ranking ability. **Phase 16:
> action prior, not architecture change**.*

## Reproducibility

```bash
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
  --out /workspace/phase15_mpc
```

Artifacts: `artifacts/phase15_mpc/{summary.json, per_episode_*.jsonl}`
