# Phase 16 (Policy-prior MPC) — Decision document

**Date:** 2026-05-17.
**Status:** ⚠ **1/3 strict gates. Two simultaneously-true stories:**
> **A (BC-specific):** BC warm-start solves *contact coverage* but
> not *goal-directed pushing*. Contact 77%, improvement 0.03. BC is
> a contact prior, not a push-to-goal prior. Next fix = goal-
> conditioned push primitive or BC explicitly trained for push.
>
> **B (architecture):** A goal-conditioned push primitive already
> exists as `scripted_prior_cem` (imp=0.143 ≈ 80% of oracle). But
> CEM refinement makes it *worse* than the prior alone (0.143 <
> 0.180), and the predictor's correlation goes from +0.53 (15b
> naïve) to −0.39 (16 scripted-prior). The OF-JEPA predictor is
> calibrated on v3's broad scripted action distribution and goes
> mis-calibrated on focused-contact distributions a competent
> prior produces.

> **Headline:** Action prior is the bottleneck for closed-loop planning,
> as Phase 15b predicted. With a scripted-contact prior, contact rate
> jumps from 20% → 97% and improvement from ~0 → 0.143 (≈80% of
> oracle). But CEM refinement on top of the prior makes things slightly
> worse, and the predictor's ranking correlation goes from +0.53
> (Phase 15b naïve) to −0.39 (Phase 16 scripted-prior). The world
> model's calibration doesn't transfer to focused-contact distributions.

## Setup

Same OF-JEPA predictor (reloaded `model_action.pt` from Phase 15), four
search methods, oracle as skyline. Pre-committed gates in
`PHASE_16_POLICY_PRIOR_PRECOMMIT.md`.

| | Value |
|---|---|
| Env | robosuite Stack |
| Goal | cube_init + random Δ, |Δ|∈[0.05, 0.08] |
| Episodes | 30 per mode |
| Total actions | 15 (3 replans × 5 actions) |
| Plan horizon | 10, CEM iters 3 |
| K | 128 main |
| Prior σ | 0.2 (vs naïve σ=0.5) |
| BC | inline-trained MLP, 40 demos × 15 actions, 1200 steps, ~5 min |

## Headline numbers

| Mode | improvement | dir_score | contact | succ | mean_disp | pred-actual corr |
|---|---|---|---|---|---|---|
| **gt_closed_loop** (oracle skyline) | **0.180** | 0.106 | 0.667 | **0.233** | 0.024m | — |
| **scripted_prior_cem** | **0.143** | 0.063 | **0.967** | 0.133 | 0.052m | **−0.387** |
| bc_prior_cem | 0.031 | 0.023 | 0.767 | 0.000 | 0.013m | +0.041 |
| bc_only (no CEM) | 0.002 | -0.018 | 0.667 | 0.000 | 0.003m | — |
| naive_cem (15b baseline) | 0.000 | -0.015 | 0.200 | 0.000 | 0.004m | -0.090 |
| random | 0.000 | 0.000 | 0.233 | 0.000 | 0.000m | — |

### Pre-committed gates

```
G1. contact_rate(bc_prior) = 0.767     (>=0.50)  PASS ✅
G2. improvement(bc_prior)  = 0.031     (>=0.10)  FAIL ❌  (3× below threshold)
G3. gap_over_naive         = +0.031    (>=0.05)  FAIL ❌
```

Strict verdict: **1/3 marginal.**

## Three findings (two surprising)

### Finding 1 (expected): Phase 15b's prior-bottleneck diagnosis is correct.

`scripted_prior_cem` (CEM seeded with closed-loop scripted policy) hits:
- Contact rate **0.967** vs naïve 0.200 (5× jump)
- Improvement **0.143** vs naïve ~0 (and **80% of oracle's 0.180**)
- mean_displacement **0.052m** vs naïve 0.004m (13× jump)

Action prior coverage was the bottleneck. Once the prior actively
samples cube-engaging trajectories, all planner-relevant metrics
improve dramatically. **Phase 15b's diagnostic claim is vindicated.**

### Finding 2 (surprising): CEM-refinement slightly HURTS the prior.

`scripted_prior_cem` improvement (0.143) is *below* `gt_closed_loop`
oracle (0.180), the very policy the prior is sampled from. CEM noise
(σ=0.2) around the scripted base diversifies actions in ways that:
- Increase contact rate (0.67 oracle → 0.97 scripted_prior_cem)
- Decrease directional precision (dir_score 0.106 → 0.063)
- Decrease improvement (0.180 → 0.143)

CEM is making more contacts but less goal-directed pushes. The model
isn't biasing elites toward the truly-better candidates.

### Finding 3 (the deep one): predictor ranking goes NEGATIVE on
focused-contact distributions.

Pred-actual correlation:
- Phase 15b naïve_cem: **+0.528** (the predictor ranks broad-Gaussian candidates well)
- Phase 16 naïve_cem reproduction: −0.090 (essentially zero on this seed)
- Phase 16 **scripted_prior_cem: −0.387** (predictor anti-correlated with actual outcome)
- Phase 16 bc_prior_cem: +0.041 (essentially zero)

**The predictor's ranking signal is regime-dependent.** It was trained
on v3 scripted_push actions — a broad distribution of approach + push
with σ=0.20 noise. When CEM candidates cluster around a focused
contact-engaging prior, the model sees out-of-distribution candidates
and its scoring becomes unreliable or actively wrong.

This is consistent with Phase 14.4's observation that action ranking
fails under unfamiliar distributions, and Phase 14.6's caveat that
state_mse degraded under OOD shifts (0.96 → 1.5×). What 14.6 didn't
test was whether the PRACTICAL DOWNSTREAM USAGE (CEM scoring) breaks
under OOD action distributions. Phase 16 shows it does.

## Implication for the BLA architecture

The Phase 14 → 15 → 16 arc tells a coherent story:

- Phase 14: action-conditioned OF-JEPA learns to rank actions in
  offline tests on its training distribution. ✅
- Phase 15: in closed-loop planning, naïve CEM doesn't sample
  cube-engaging actions, so the predictor's ranking has nothing
  useful to refine. ❌ (diagnosed as prior-bound)
- Phase 16: with a competent prior, CEM samples great candidates,
  but the predictor mis-ranks them because they're OOD relative
  to the training distribution. ⚠

So the world model is calibrated *on its training distribution* but
not *on the distribution it would naturally see if combined with a
competent policy*. **OF-JEPA isn't planner-grade in the strict sense
of "useful for closed-loop refinement on top of a good prior."**

### What would fix this

The model needs to see focused-contact action distributions DURING
TRAINING, not just at test time. Options:
1. **Retrain OF-JEPA on closed-loop oracle rollouts** (or a mixture
   of broad scripted v3 + narrow oracle), then redo Phase 16.
2. **Iterative train-on-deployed**: collect rollouts from current
   policy + CEM, retrain, repeat. Standard model-based RL recipe.
3. **Auxiliary contact-action distribution loss** during training:
   explicitly hold out a small fraction of "focused" actions and
   include them in the training mix.

### What this DOESN'T break

- Phase 14.5/14.6 findings: action conditioning is data-bound, not
  architecture-bound, and offline ranking generalizes across OOD
  *action statistics* (gain, noise, horizon). Those tests were on
  the broad v3 distribution. They don't claim anything about
  focused-contact distributions.

## Honest verdict

**1/3 strict gates pass, but the gates were the wrong question.**

The interesting question turned out to be: *does the predictor
generalize to action distributions a planner would naturally
produce?* And the answer here is **not yet** — at least without
retraining the predictor on closed-loop or oracle-style data.

The cleanest publishable framing:
> *OF-JEPA + offline ranking generalizes across action statistics
> (Phase 14.6). OF-JEPA + closed-loop planning requires the predictor
> to be trained on focused-contact distributions; v3-trained
> OF-JEPA's ranking signal becomes unreliable when CEM samples
> from a competent action prior.*

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 14.5 | ✅ | action conditioning helps offline ranking with informative data |
| 14.6 | ✅✅✅ | OOD generalization across action statistics |
| 15 | ⚠ | naïve sanity threshold too high; pivoted to 15b |
| 15b | ❌ | naïve CEM doesn't translate ranking into action selection; prior-bound diagnosis |
| **16** | **1/3 + deep insight** | **prior IS bottleneck (15b vindicated); BUT predictor ranking goes negative on focused-contact dist; OF-JEPA not yet planner-grade in closed-loop** |

## Bottleneck has moved one layer downstream

The user's diagnosis after seeing the results:
```
Phase 15b bottleneck:  action prior rarely contacts cube
Phase 16 bottleneck:   contact happens, but not goal-directed pushes
```

The key diagnostic now: **direction_score conditional on contact**.
If contact is high but conditional direction score is near zero, the
prior is a contact prior, not a push-to-goal prior.

Looking at our data:
- bc_prior_cem: contact 0.767, dir_score 0.023 → mostly side-swipes, not goal-directed pushes ✓
- scripted_prior_cem: contact 0.967, dir_score 0.063 → still mostly side-swipes despite goal-aware base prior
- gt_closed_loop: contact 0.667, dir_score 0.106 → genuinely goal-directed (but only 67% contact)

Even the scripted prior, which uses closed_loop_gt_step (which IS
goal-directed at every step), is getting only dir_score 0.063 after
CEM noise smears the action distribution. So:

> CEM noise on top of a goal-directed prior degrades directionality
> faster than the predictor can recover it via re-ranking.

This is consistent with the predictor's negative correlation: the
model can't tell which CEM-noise variant is most goal-directed,
because all candidates are within a narrow band of the prior's
action distribution — a regime the predictor wasn't trained for.

## Next-phase options

**Option A — Phase 16b: retrain predictor on focused-contact data.**
Collect 200 closed-loop oracle rollouts; mix with v3 (50/50) or replace;
re-train OF-JEPA; redo Phase 16. ~2 hours wall time. Tests whether
the calibration failure is fixable with data.

**Option B — Phase 17: end-to-end trained policy.** Drop CEM; train a
policy network with OF-JEPA-as-critic via differentiable trajectory
rollout. Skips the calibration question by training policy and
world-model jointly. More infrastructure, but addresses the deeper
"world model + policy" recipe properly.

**Option C — Accept Phase 16 as the honest negative.** Pause closed-loop
work; reframe OF-JEPA's contributions as offline (identity binding,
action ranking, OOD generalization) rather than online planning.
Cleanest writeup of work-to-date.

## Reproducibility

```bash
python3 scripts/phase16_policy_prior_mpc.py \
  --model-action /workspace/phase15_mpc/model_action.pt \
  --seed 0 --jepa-stride 4 \
  --total-actions 15 --replan-every 5 --plan-horizon 10 \
  --cem-iters 3 --main-K 128 \
  --modes gt_closed_loop,naive_cem,scripted_prior_cem,bc_prior_cem,bc_only,random \
  --n-episodes 30 --oracle-sanity-n 30 \
  --oracle-min-improvement 0.10 --oracle-min-contact 0.60 \
  --bc-episodes 40 --bc-train-steps 1200 \
  --out /workspace/phase16
```

Artifacts: `artifacts/phase16/{summary.json, per_episode_*.jsonl, bc_policy.pt}`
