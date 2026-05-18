# Phase 18γ — Predictor calibration audit (Decision document)

**Date:** 2026-05-18.
**Status:** ❌🌟 **1/3 precommit gates pass — but the audit reframed the
whole question.** Predictor ranking quality and candidate quality are
**orthogonal**; the "light = trust region, heavy = exploit" thesis is
falsified in a specific way.

> **Headline:** The predictor's ranking ability and the candidate
> distribution's outcome quality are independent dimensions. Tight
> noise balls around the scripted prior (D1/D2) produce the **best
> realized outcomes** but the **worst ranking signal** — the predictor
> goes mildly anti-correlated. Broader scripted CEM (D3) and policy
> + light CEM (D5) produce **positive rank signal** but lower
> candidate quality. Phase 18β's per-episode `corr = -0.52` on
> policy+CEM was *not* a per-replan calibration failure — it was a
> cross-replan policy-drift artifact. The locked planning recipe
> (`scripted_prior + light CEM`) wins on **prior quality**, not on
> predictor ranking.

## Setup

Phase 17 model + Phase 18β policy. For each of 6 candidate
distributions, sample M=128 candidate plans per state at N=40 states,
score all with the predictor, then env-clone-execute the predictor's
top-8 + bottom-8 + 8 random per state (24 ground-truth executions
per state per distribution). Build a fresh env per (state,
distribution) pair to avoid the `env.sim.set_state` degradation the
pilot caught (`real_std=0` at ~128 cycles).

Pilot (M=64, N=20, K=6, exec_horizon=5) caught two bugs:
1. `env.sim.set_state` degrades after ~128 cycles → fixed via fresh
   env per state.
2. `exec_horizon=5` was too short (~1s physics) → bumped to 10 to
   match plan_horizon and Phase 18β's per-episode physics budget.

## Headline numbers (M=128, N=40, K=8, exec_horizon=10)

```
  Dist       top_vs_bot_gap   topK   botK   mean_R   breadth   contact   Spearman
  D1 scripted+tiny             -0.017  0.113  0.129    0.118    0.018     0.77    -0.035
  D2 scripted+light            -0.028  0.131  0.159    0.145    0.109     0.76    -0.049
  D3 scripted+heavy            +0.017  0.108  0.091    0.101    0.162     0.71    +0.024
  D4 policy+tiny               +0.003  0.053  0.050    0.052    0.020     0.77    -0.021
  D5 policy+light              +0.015  0.111  0.096    0.102    0.115     0.81    +0.021
  D6 naive Gaussian            +0.004  0.005  0.000    0.003    0.411     0.18    +0.027
```

## Gate verdicts (vs precommit)

```
G1. D2 (scripted_prior + light CEM) top_vs_bot_gap > +0.02
       −0.028                          FAIL   (predictor anti-correlated
                                                on the supposed trust manifold!)

G2. D3 gap < D2 gap by ≥ 0.02
       (+0.017) − (−0.028) = +0.045    FAIL  (sign reversed — D3 BETTER than D2)

G3. D2 topK realized > D3 topK realized
       0.131 > 0.108                   PASS  (light CEM topK ~21% > heavy)

Verdict: 1/3.  But the *direction* of the failure is informative.
```

## The reframe — two orthogonal axes

User reframed during the run (2026-05-18). The decisive artifact is
not "is predictor trustworthy here yes/no" but a 2-axis table:

```
                    Negative-rank        Positive-rank
                    (predictor misranks)  (predictor ranks)
                    
High candidate-q    D1, D2                D3, D5
(mean_R > 0.10)     0.118, 0.145          0.101, 0.102
                    breadth 0.018, 0.109  breadth 0.162, 0.115
                    
Mid candidate-q     —                     D4 (~neutral)
                                          0.052, breadth 0.020
                    
Low candidate-q     —                     D6
(mean_R < 0.02)                           0.003, breadth 0.411
```

The pattern that emerges:

| Observation | Mechanism |
|---|---|
| D1/D2 → high outcomes, bad ranking | Tight neighborhoods around a good prior produce uniformly-good outcomes; predictor has no variance to rank. Its small biases produce systematic ANTI-correlation because the noise dominates the signal. |
| D3 → middling outcomes, positive rank | Heavy CEM with σ=0.2 over 3 iters creates the broader, more variable candidate distribution the predictor was trained on (Phase 17 mixed-data: scripted + goal_directed + CEM-refined). The predictor recognizes "this is training-like" and ranks the variation positively. |
| D5 → middling outcomes, positive rank | Same predictor-recognition story applies to policy + light CEM. **Per-replan ranking IS positive** even on the policy distribution; Phase 18β's -0.52 corr was a per-episode artifact of cross-replan policy-drift. |
| D4 → policy-only, weak signal | Mid quality (no CEM breadth), neutral ranking — neither bias enough to misrank nor variance enough to rank well. |
| D6 → naive, weak positive rank but ~zero candidate quality | Predictor can distinguish "less bad" from "more bad" but all candidates are too bad to matter. |

## Reconciliation with Phase 18β

Phase 18β reported per-EPISODE pred_actual_corr (one corr per episode,
across the whole MPC trajectory):

```
naive_cem               +0.285
scripted_prior_light    −0.191
scripted_prior_heavy    −0.288
learned_policy_cem      −0.520
```

Phase 18γ reports per-replan-boundary top_vs_bot_gap (per single
state, full plan_horizon execution from a fresh env):

```
D6 naive                +0.004
D2 scripted+light       −0.028
D3 scripted+heavy       +0.017
D5 policy+light         +0.015
```

These are different metrics measuring different things:

| Phase 18β per-episode | Phase 18γ per-replan |
|---|---|
| Aggregates over ~3 replans within an MPC episode | Single replan boundary, fresh state |
| Captures cross-replan drift (policy can locally pick good actions but compound bad direction across replans) | Captures local rank quality only |
| naive_cem positive → predictor distinguishes good-from-random across the episode | naive_cem ~zero → predictor distinguishes within naive distribution, but absolute candidate quality is ~0 |
| heavy_CEM negative → CEM elites are pushed into anti-correlated region across multiple replans | heavy_CEM positive → at a single replan, broad scripted distribution is rankable |
| policy_CEM strongly negative → policy's drift compounds catastrophically | policy_CEM positive → at a single replan, policy candidates rank fine |

**18β and 18γ are not contradictory; they measure orthogonal failure
modes.** Phase 18β's heavy < light story holds because:
- The **prior** is doing the planning work; not the predictor.
- Light CEM stays close to the prior; heavy CEM drifts.
- Heavy CEM's per-replan ranking gain (which 18γ confirms) does NOT
  compensate for the cross-replan drift from prior-distance.

## The locked recipe still holds — but for a clearer reason

Phase 18β locked: `scripted_prior + LIGHT CEM` as the default.

Phase 18γ tells us **why** this won:
1. The scripted prior is the dominant good-candidate source.
2. Light CEM keeps candidates inside the prior basin → high
   candidate quality.
3. The predictor's per-replan ranking on light-CEM distribution is
   weakly *negative* (D2 -0.028) — so even removing the predictor
   shouldn't hurt much.
4. Heavy CEM gets better per-replan ranking but drifts away from the
   prior basin → mean candidate quality drops (0.145 → 0.101).

The lesson is sharper than the original "trust region" framing:
**The predictor adds little planning value in the regime where it
matters.** The planner's positive result is a property of the prior +
breadth control, not predictor scoring.

## Implications for next phases

### Phase 18δ — Trust-region CEM (pre-locked, now reframed)

Original framing: "constrain CEM updates to stay inside calibrated
action manifold."

Revised framing in light of 18γ:
- The action manifold where predictor calibration is positive (D3, D5)
  is BROADER than the manifold where candidate quality is high (D1,
  D2).
- A trust-region CEM that constrains via *prior distance* keeps
  candidate quality high but limits ranking gain.
- A trust-region CEM that constrains via *predictor confidence* might
  expand search just enough to use the predictor's positive ranking,
  but risks leaving the prior basin.
- **The right knob is probably mean_realized of the elite set**, not
  the predictor's score or prior distance. Use the predictor to rank
  among candidates that have *passed* a realized-quality screen
  (perhaps via short rollout proxy).

### Phase 18ε / 18ζ — Predictor refinement (deprioritized further)

18γ shows the predictor's failure mode is most acute exactly where
the planner least needs it (tight neighborhoods around strong priors).
Retraining the predictor on those exact tight neighborhoods may not
help — the candidates there are too similar to rank meaningfully.

Better: train the predictor to rank by **outcome-prediction quality**
(MSE on cube_end_xy) instead of by direct improvement score, so its
calibration generalizes from training-data candidate distributions to
deployment-time tight neighborhoods.

### Phase 18η (new) — Cross-replan drift instrumentation

Phase 18β's per-episode corr of -0.52 was driven by cross-replan
drift, not per-replan miscalibration. Worth instrumenting: at each
replan boundary, log the policy's *direction* relative to the goal,
and the *change in direction* from the previous replan. Hypothesis:
the policy drifts because it has no memory of "where we were
heading" across replans.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 14.5/14.6 | ✅✅✅ | offline ranking + OOD generalization |
| 15b | ❌ | naïve CEM fails (prior-bound) |
| 16 | 1/3 + diagnosis | BC fixes contact; predictor anti-corr on focused-contact |
| 17 | ✅ | mixed-data; planner beats oracle (seed 0) |
| 18δ (multi-seed) | ✅✅✅ | Phase 17 robust; planner +0.005 over oracle |
| 18β | ❌ + ⭐ | distillation falsified; light CEM > heavy CEM (8.3% compute) |
| **18γ** | **❌ + 🌟** | **predictor rank ≠ candidate quality; orthogonal axes. Phase 18β's -0.52 corr was per-episode drift, not per-replan miscalibration. Locked recipe holds for clearer reason.** |

## Reproducibility

```bash
python3 scripts/phase18g_calibration_audit.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --policy-ckpt /workspace/phase18b/plan_policy.pt \
    --M 128 --N 40 --gt-k-per-extreme 8 --exec-horizon 10 \
    --out /workspace/phase18g_main --seed 0
```

Artifacts: `/workspace/phase18g_main/{summary.json,
per_state*.jsonl, log.txt}` on pod; `artifacts/phase18g/` in repo.

## What this phase establishes
- Phase 18β's `corr = -0.52` is a per-episode aggregate effect,
  not a per-replan calibration verdict.
- The Phase 17/18δ "planner ≥ oracle" win is driven by **prior
  quality**, not by predictor calibration.
- Distribution **breadth** alone is not enough for rankability — the
  candidate distribution must also resemble training distribution
  (D3 ≈ Phase 17 mixed-data shape; D1/D2 do not).
- "Trust region" needs reframing: it's the region where candidate
  *quality* is preserved, not where the predictor *thinks* it
  knows best.
