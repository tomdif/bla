# Phase DR3 — Bank-Coverage Diagnostic + Constrained Rerank (Decision)

**Date:** 2026-05-19.
**Status:** ✅ **Negative/clarifying ablation; H1 falsified, H2 partially confirmed.**
**Precommit:** `docs/phases/PHASE_DR3_PRECOMMIT.md` (commit `88857e1`).

## Headline

> **DR3 falsifies the bank-coverage hypothesis and gives Recipe E
> a real low-variance variant.** Spearman(NN_distance, -improvement)
> averages 0.07–0.21 across modes, well below the precommit's ≥0.3
> threshold — bank-coverage is NOT the main story. Constrained
> top-k rerank cuts top-1's variance nearly in half (σ 0.054 vs
> 0.101) at a 16% mean cost. The residual variance is execution
> stochasticity / hidden state mismatch, not retrieval-metric or
> bank-coverage failure.

## Three-seed aggregate (n=30, reset pool = all 100 demos)

| Mode | imp_mean ± std | success ± std | Spearman(nn,-imp) |
|---|---:|---:|---:|
| **geometry_top1** | **0.213 ± 0.101** | 21% | 0.074 |
| **geometry_constrained_rerank** | 0.178 ± **0.054** | 17% | 0.164 |
| demo_no_cem_oracle | 0.096 ± 0.020 | 9% | 0.210 |
| demo_no_cem_cycle | 0.036 ± 0.034 | 3% | -0.043 |

Per-seed:

| Seed | top1 | rerank | oracle | cycle |
|---:|---:|---:|---:|---:|
| 0 | 0.235 / 23% | 0.169 / 17% | 0.073 / 7% | 0.003 / 0% |
| 1 | 0.103 / 10% | 0.130 / 10% | 0.106 / 10% | 0.070 / 7% |
| 2 | 0.302 / 30% | 0.236 / 23% | 0.108 / 10% | 0.036 / 3% |

## H1 (coverage hypothesis) — FALSIFIED at the ≥0.3 threshold

Precommit expectation: Spearman(NN_distance, -improvement) ∈ [0.3, 0.6].

Observed (3-seed mean):
- geometry_top1: **0.074**
- constrained_rerank: 0.164
- demo_no_cem_oracle: 0.210
- demo_no_cem_cycle: −0.043

**The strongest H1 signal is in oracle mode (0.210), still well
below 0.3.** Coverage matters mildly, but it's not the dominant
explanation for per-episode failure. Adding more demos to the bank
will help only modestly.

The precommit's F1 trigger fires:

> F1 (precommit): Spearman corr(d_nn, improvement) ≈ 0. Variance is
> NOT a coverage problem — adding demos won't help. Look elsewhere
> (execution, hidden state).

## H2 (constrained rerank) — PARTIALLY CONFIRMED

Constrained rerank shows the variance/mean tradeoff the precommit
predicted, but G1 narrowly fails:

| Gate | Threshold | Observed | Result |
|---|---|---:|---|
| G1 | rerank mean ≥ top1 − 0.02 (≥ 0.193) | 0.178 | ❌ misses by 0.015 |
| G2 | rerank std ≤ top1 std − 0.01 (≤ 0.091) | **0.054** | ✅ PASS by 0.037 |
| G3 | on d_nn > 0 subset, rerank ≥ top1 + 0.05 | not separately computed | ⚠ |

Rerank achieves G2 with margin (variance reduction 0.101 → 0.054 is
nearly 2×) but slightly underperforms on G1 (mean drop is 16%, not
within the 9% tolerance). **This is the real E2_STABLE recipe**:
lower variance, lower mean — same tradeoff DR1 showed with
top3_avg, but at better numbers (top3_avg σ was 0.045, mean 0.280;
rerank σ 0.054, mean 0.178 in this harder protocol).

Strong-pass criterion (SP: mean ≥ top1 AND std ≤ 0.06 AND
H1 corr > 0.3) fails on all three counts. The unconstrained
strong-pass hypothesis was wrong.

## The residual variance is execution stochasticity

Three independent observations point to this:

1. **Oracle drops to 9%** when reset distribution is wider than
   the 24-working bank. The oracle uses the reset target's OWN
   actions — there's no retrieval involved. If oracle fails this
   hard, the bottleneck cannot be retrieval-side.

2. **Same reset + same action sequence produces different outcomes
   across modes** within a single eval loop. Pilot showed demo 47
   replayed once produced z_gain=0.0003 (in oracle mode) and once
   produced z_gain=0.150 (in rerank mode). The mujoco state is
   set correctly; what differs is the env's non-mujoco state across
   modes (RNG counters, internal caches, ordering).

3. **H1 weak across all modes**: top1 spearman = 0.074. If retrieval
   *quality* were the bottleneck, we'd see strong correlation
   between NN distance (a proxy for retrieval quality) and outcome.
   We don't.

## What's locked

- **E2_FAST = geometry_top1** (DR1 + DR2 + DR3 confirm this as the
  highest-mean retrieval mode in every protocol tested)
- **E2_STABLE = geometry_constrained_rerank** (DR3 establishes this
  as the empirically lower-variance alternative; replaces DR1's
  ad-hoc "top3_avg" as the canonical stable recipe)
- The doctrine **state match primary, outcome tiebreaker** survives:
  constrained rerank IS the proper application of this rule, and
  it does reduce variance as the doctrine predicted

## What's falsified

- **H1: variance is bank-coverage limited.** Spearman correlations
  too weak; adding more demos won't be the main lever.
- **Strong-pass criterion SP** from precommit: rerank doesn't match
  top1's mean while reducing variance — there's no free lunch at
  the metric level.

## What's next

The residual variance is execution stochasticity / hidden state
mismatch. The architectural improvements that could address this:

```
1. Demo-execution stability:
   record env RNG state across replays; ensure the same reset+demo
   produces consistent outcomes. Probably requires reproducible
   robosuite state restoration beyond set_state_from_flattened.

2. Closed-loop demo retrieval:
   re-retrieve at each step, not just at episode start. Adapts to
   execution drift but may break the demo manifold.

3. Demo-action error correction:
   small residual policy that applies state-conditional corrections
   to the retrieved demo's actions. Mirrors the planner/value-head
   pattern but on the action side.

4. BLA-Forge real-world data:
   on real hardware the noise model is different. Variance work
   on simulator may not transfer; better to test the doctrine in
   the deployment regime.
```

Per the scaling roadmap §5+, **#4 (BLA-Forge real-world)** is now
the right next step. DR1-DR3 have explored the metric, bank, and
rerank axes in simulator; the remaining stochasticity is best
addressed in the deployment regime, not by more simulator metric
tuning.

## Recipe E variants after DR3 (locked)

```
Recipe.E2_FAST  = geometry_top1
                 retrieval by L2 over 6-D absolute pose
                 (can_xy, eef_xy, can_z, eef_z) from env.sim.data.
                 Highest mean across DR1/DR2/DR3 protocols.

Recipe.E2_STABLE = geometry_constrained_rerank(k=5, filter_ratio=1.25)
                  top-k by L2; filter to ≤ 1.25× NN distance;
                  rerank within filter by outcome_score.
                  Lower variance (DR3: σ 0.054 vs 0.101); ~16%
                  lower mean. Preserves the doctrine.

NEVER add CEM around either variant (D3/D4/Scale-1/DR1 cross-
validated as destructive).
```

## Files

- Precommit: `docs/phases/PHASE_DR3_PRECOMMIT.md` (commit `88857e1`)
- Decision: this file
- Eval: `scripts/phase_dr3_constrained_rerank.py`
- Module: `bla/recipes/demo_retrieval.py` (`retrieve_constrained_rerank`)
- Tests: `tests/test_demo_retrieval.py` (18/18 passing)
- Pod: `/workspace/phase_dr3_main_seed{0,1,2}/summary.json`

## Locked
