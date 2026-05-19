# Phase DR2 — Retrieval-Quality Ablation (Decision)

**Date:** 2026-05-19.
**Status:** ✅ **Negative / clarifying ablation — Recipe E retrieval doctrine sharpened.**
**Parents:**
- `docs/phases/PHASE_DR1_DECISION.md` (commit `98d3659`)
- `docs/phases/PHASE_DR2_PRECOMMIT.md` (commit `9b8a79a`)

## Headline

> **DR2 does not improve on `geometry_top1`, but it clarifies the
> retrieval doctrine. In demo-prior regimes, state-matched
> nearest-neighbor retrieval is the default. Raw slot embeddings
> are not a retrieval interface, and unconstrained outcome
> reranking breaks the state-match requirement.**

The locked refinement of Recipe E:

> **Retrieve by state match first. Outcome/value can only be a
> tie-breaker inside a tight state-match neighborhood, never the
> primary reranker.**

## Three-seed aggregate (n=30 per seed per mode)

| Mode | seed 0 | seed 1 | seed 2 | mean ± std | success ± std |
|---|---:|---:|---:|---:|---:|
| **geometry_top1** (DR1 baseline) | 0.337 | 0.302 | 0.468 | **0.369 ± 0.088** | 37% ± 8.8pp |
| goal_relative_top1 | 0.302 | 0.268 | 0.435 | 0.335 ± 0.088 | 33% ± 8.8pp |
| slot_state_top1 | 0.036 | 0.036 | 0.036 | **0.036 ± 0.000** | 3% ± 0pp |
| geometry_top3_avg | 0.329 | 0.239 | 0.273 | 0.280 ± **0.045** | 26% ± 5.1pp |
| geometry_topk_outcome_rerank | 0.202 | 0.136 | 0.236 | **0.191 ± 0.051** | 19% ± 5.1pp |
| demo_no_cem_oracle | 0.320 | 0.268 | 0.404 | 0.331 ± 0.069 | 32% ± 6.9pp |
| phase17_locked | 0.083 | 0.120 | 0.092 | 0.098 ± 0.019 | 7% ± 0pp |
| demo_no_cem_cycle | 0.003 | 0.003 | 0.036 | 0.014 ± 0.019 | 1% ± 1.9pp |
| naive_cem | 0.003 | 0.003 | 0.003 | 0.003 ± 0.000 | 0% ± 0pp (floor) |

## Gate evaluation

```
G1: best DR2 mode mean ≥ geometry_top1_mean(DR1) − 0.03  (≥ 0.316)
G2: best DR2 mode std ≤ 0.08
G3: best DR2 mode success ≥ phase17_locked + 10pp        (≥ 20%)
G4: best DR2 mode beats geometry_top3_avg on mean        (≥ 0.281)
```

Best DR2 mode: `geometry_top1` (mean 0.369, std 0.088).

| Gate | Threshold | Observed | Result |
|---|---|---:|---|
| G1 | ≥ 0.316 | 0.369 | ✅ PASS |
| G2 | ≤ 0.08 | **0.088** | ⚠ near-miss (off by 0.008) |
| G3 | ≥ 0.167 succ | 0.367 | ✅ PASS |
| G4 | ≥ 0.281 mean | 0.369 | ✅ PASS |

**Three of four gates pass; G2 misses by 0.008**. Strong-pass SP1
(mean ≥ 0.346 AND std ≤ 0.08) — mean clears (0.369 ≥ 0.346), std
narrowly fails (0.088 > 0.08).

## DR1 variance was overestimated

DR1 reported `geometry_top1` at mean 0.346 / σ 0.135. DR2 reports
**0.369 / 0.088**. Same mode, same protocol, different RNG
ordering. The DR2 numbers are equally valid as a variance estimate —
the "high variance" that DR2 was set up to reduce was largely a
small-sample artifact.

The actual variance gap to top3_avg (0.045) is much smaller than
DR1 suggested (0.088 vs 0.045 vs the DR1-reported 0.135 vs 0.052).
A factor-of-2 instead of factor-of-3.

## What DR2 falsified

```
1. raw OF-JEPA slot-state retrieval
   slot_state_top1 → 0.036 / 3% across 3 seeds; std ZERO meaning
   it never retrieves successfully. 768-D L2 is uninformative as
   a retrieval key. Phase 18θ lesson extends to retrieval.

2. goal-relative as a drop-in improvement
   goal_relative_top1 (0.335) underperformed geometry_top1 (0.369).
   Absolute+local geometry was already sufficient.

3. unconstrained outcome reranking
   geometry_topk_outcome_rerank (0.191) was 48% WORSE than top1
   (0.369). Reranking by outcome moves away from state match,
   picks globally-strong-but-mismatched demos, which fail to grasp.
```

The "top3_avg as variance reducer" hypothesis is partially
confirmed: top3_avg σ=0.045 vs top1 σ=0.088 — meaningfully lower
variance, but lower mean (0.280 vs 0.369). The tradeoff is real.
This recovers DR1's E2_FAST / E2_STABLE pair.

## The sharpened doctrine

Locked refinement to `feedback_state_match_primary_outcome_tiebreaker`:

```
Recipe E retrieval default:
  geometry_top1 by L2 over [can_x, can_y, eef_x, eef_y, can_z, eef_z]
  (from env.sim.data, not env._get_observations — see robosuite-obs
  gotcha)

When outcome/value is to be incorporated:
  RIGHT: top-k by state distance, FILTER to within 1.25× of NN distance,
         then pick argmax(outcome_score) within filter.
  WRONG: top-k by state distance, then argmax(outcome_score)
         (DR2 falsified — picks globally-strong-but-mismatched demos).
```

## Why DR2 is useful even as a negative result

DR2 eliminates three plausible-but-wrong scaling paths that the
team would otherwise have been tempted to invest in:

1. **"Just use slot embeddings as retrieval keys"** — broken; needs
   a structured readout (adapter) just like value prediction (18θ→18λ).
2. **"Add task-relative geometry to the key"** — no measurable
   help on PickPlaceCan; absolute geometry already captures the
   needed match.
3. **"Rerank by recorded demo quality"** — actively harmful when
   the rerank can override state match.

Each of these would have looked reasonable from outside the
empirical loop. The pilot+main combination falsifies them in
~30 minutes of compute.

## What's still open for DR3

```
1. Constrained top-k rerank
   filter top-k by state distance band, then rerank within filter.
   This is the corrected version of outcome rerank — state match
   stays primary, outcome is a tiebreaker only when distances tie.

2. Demo-bank coverage expansion
   Variance may be a coverage problem, not a metric problem.
   DR3 diagnostic:
     correlate per-episode improvement with nearest-neighbor distance.
   If bad outcomes correlate with large NN distance → expand bank.
   If bad outcomes happen even at close NN distance → execution
   stochasticity / hidden state mismatch is the bottleneck.

3. Learned adapter-based retrieval key
   slot → low-D engineered-style features via supervised adapter
   (mirror of Phase 18λ for value). Only worth doing AFTER #1+#2
   establish whether the metric is even the bottleneck.
```

## Recipe E variants after DR2

```
E2_FAST    = geometry_top1
             retrieval over a state-matched bank.
             DEFAULT for contact-sensitive demo-prior regimes.
             DR1 main: 0.346 ± 0.135 / 35% ± 14pp on PickPlaceCan.
             DR2 confirms: best of the metric-only ablations.

E2_STABLE  = UNRESOLVED at the metric level.
             top3_avg (DR1) and constrained_rerank (DR3) are
             candidates; neither is locked. Best current proxy:
             use top1 and accept its variance until bank-coverage
             work (DR3) lands.
```

## Decision

**Lock DR2 as a negative/clarifying ablation.** Recipe E2_FAST
default = `geometry_top1`. E2_STABLE remains unresolved at the
metric level and is deferred to DR3 (bank coverage diagnostic +
constrained rerank).

Next priority order (post-DR2):

```
1. DR3 — demo-bank coverage diagnostic + constrained top-k rerank
2. BLA-Forge real-world testbed
3. P4 falsification probe (optional, low-stakes)
4. v2 stateful encode_step (deferred; v1 rolling K=5 is default)
5. Model capacity scaling (still last)
```

## Files

- Precommit: `docs/phases/PHASE_DR2_PRECOMMIT.md` (commit `9b8a79a`)
- Decision: this file
- Eval: `scripts/phase_dr2_retrieval_ablation.py`
- Module additions: `bla/recipes/demo_retrieval.py` (`outcome_score`
  field + `retrieve_rerank_by_outcome` method)
- Tests: `tests/test_demo_retrieval.py` (14/14 passing)
- Pod: `/workspace/phase_dr2_main_seed{0,1,2}/summary.json`

## Locked

