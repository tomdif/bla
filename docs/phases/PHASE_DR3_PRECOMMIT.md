# Phase DR3 — Bank-Coverage + Constrained Rerank (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Gates locked before run.**
**Parents:**
- `docs/phases/PHASE_DR2_DECISION.md` (commit `1a8f9f0`)
- `feedback_state_match_primary_outcome_tiebreaker`

## Purpose

DR2 closed with `geometry_top1` as the best retrieval mode but σ
just above the 0.08 target (0.088). DR3 tests two hypotheses:

> **H1 (coverage):** the remaining variance is bank-coverage limited.
> Episodes that fail correlate with large nearest-neighbor distance;
> episodes that succeed have small NN distance.

> **H2 (constrained rerank):** when NN distance is positive (env
> init not in the bank), filtering top-k to within 1.25× of the
> nearest-neighbor distance and reranking by outcome inside the
> filter improves over pure top-1 without re-introducing
> state-mismatch.

## Why we need a different protocol

DR2's reset target was always from the 24-demo working bank, so
NN distance was 0 every episode. Constrained rerank degenerates to
top1 under that protocol. **DR3 widens reset targets to all 100
extracted demos** (24 working + 76 non-working). For non-working
reset targets, retrieval over the working bank picks the nearest
working demo — and NN distance > 0.

This is closer to a deployment scenario where env init may not
exactly match any bank entry.

## Protocol

```
Task     : robosuite PickPlaceCan
Demo bank: 24 working demos (DR1/DR2 screened)
Reset    : RANDOM from full 100 demos (seed-controlled, expected
           ~24 working + ~76 non-working per seed of 30 eps)
Init     : state-matched reset to chosen reset target
Modes    : geometry_top1, geometry_constrained_rerank, demo_no_cem_oracle,
           demo_no_cem_cycle
Logging  : per-episode NN distance + chose_demo_id + improvement
Seeds    : 0, 1, 2 (parallel)
Episodes : 30 per seed per mode (5 pilot)
```

Mode definitions:

```
geometry_top1               — top-1 NN from 24-working bank
geometry_constrained_rerank — top-5 NN; filter ≤ 1.25× NN distance;
                              rerank within filter by outcome_score
demo_no_cem_oracle          — use the reset target's OWN actions
                              (fails for the 76% non-working subset
                              — useful as a floor / sanity check)
demo_no_cem_cycle           — D3 baseline; fixed 5-demo cycle
```

## Pre-committed gates

```
H1 coverage diagnostic (no gate; descriptive output expected):
  Per-episode (geometry_top1):
    log NN distance d_nn and improvement i.
    Report Spearman correlation: corr(d_nn, -i).
    EXPECT positive correlation if bank-coverage matters.

H2 constrained rerank gates:
  G1: constrained_rerank mean ≥ geometry_top1 − 0.02
      (rerank shouldn't significantly hurt)
  G2: constrained_rerank std ≤ geometry_top1 std − 0.01
      (rerank should reduce variance)
  G3: on the subset where d_nn > 0 (non-working reset targets),
      constrained_rerank mean ≥ geometry_top1 mean + 0.05
      (rerank pays off where it can actually trigger)
```

Strong-pass:

```
SP: constrained_rerank mean ≥ geometry_top1 AND std ≤ 0.06
    AND H1 corr(d_nn, -i) > 0.3.
```

## Pre-committed predictions

```
geometry_top1 (full 100-demo reset pool):
  Expected mean 0.20-0.30 (lower than DR2's 0.369 because 76% of
  reset targets are non-working; even nearest working demo may
  not save them all). σ harder to predict; likely 0.10-0.15.

constrained_rerank:
  Expected mean ≈ geometry_top1 within ±0.03. On the d_nn > 0
  subset, slight improvement (+0.02-0.05) if outcome scores carry
  signal.

demo_no_cem_oracle (full 100):
  Expected mean ≈ 0.24 × (typical working z_gain) ≈ 0.04-0.08.
  Effectively a floor for the 76% non-working subset.

demo_no_cem_cycle:
  Expected mean ~0.01 (D3 broken baseline).

H1 (NN distance vs improvement):
  EXPECT Spearman corr(d_nn, -improvement) in [0.3, 0.6] —
  i.e. closer-NN-distance → higher improvement.
```

User's two hypotheses, in order:

> If bad outcomes correlate with large NN distance → fix is **more demos**.
> If bad outcomes happen even at close NN distance → fix is **execution stochasticity / hidden state mismatch**.

DR3 directly answers this.

## Falsification scenarios

```
F1 (coverage uncorrelated):
  Spearman corr(d_nn, improvement) ≈ 0. Variance is NOT a coverage
  problem — adding demos won't help. Look elsewhere (execution,
  hidden state).

F2 (rerank hurts):
  constrained_rerank mean < geometry_top1 mean − 0.02 even on the
  filter-active subset. Either the outcome scores are useless or
  the filter ratio (1.25×) is wrong. Tune or abandon.

F3 (rerank helps but increases variance):
  mean improves but std goes UP. The rerank is randomly trading
  bad NN choices for good outcome choices but adding noise. Not
  useful for E2_STABLE.

F4 (both modes equally bad):
  No mode achieves mean > 0.10. Reset-from-100-demos is too hard
  a regime; DR3 should re-scope to a milder protocol.
```

## Locked
