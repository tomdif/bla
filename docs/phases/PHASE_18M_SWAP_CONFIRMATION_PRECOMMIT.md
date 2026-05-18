# Phase 18μ — Locked-recipe swap confirmation (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before aggregation.**

## Why this phase exists

Phase 18λ-v2 established that `combined_sum_supervised` (slot →
supervised geometry adapter → value head) is **empirically ≈
engineered geometry at the planner level**:

- Phase 18λ-v2 (3 seeds): supervised 0.295 mean vs geo 0.260 mean
  (+13%; supervised wins on 3/3 seeds).
- Phase 18λ-multi (3 seeds): supervised 0.207 mean vs geo 0.242 mean
  (-14%; supervised wins on 1/3 seeds).
- **Combined 6 seeds**: supervised 0.251 = geo 0.251 *exactly*.

If supervised matches engineered geo across enough seeds, the locked
planning recipe should swap from
`combined_sum_geo` → `combined_sum_supervised`,
eliminating dependence on simulator-true geometric features and
validating the BLA System-1 → System-2 architecture as a peer to
hand-engineered geometry.

## The question

> Across 5+ seeds, does `combined_sum_supervised` match
> `combined_sum_geo` under identical RNG / eval batches, justifying
> a swap of the locked planning recipe?

## Setup

Use the **6 already-collected seeds** from Phase 18λ-multi (seeds
0/1/2) and Phase 18λ-v2 (seeds 0/1/2). Both phases ran
`combined_sum_geo` and `combined_sum_supervised` (= `combined_sum_
adapter` in 18λ scripts) within the SAME script run, sharing RNG
trajectory and eval episodes per seed — exactly the "identical RNG
conditions" requirement.

This gate evaluation is performed AFTER this precommit doc is
committed, against pre-existing data, with no new training or eval.

## Pre-committed gates

```
G1. mean(combined_sum_supervised) >= 0.95 × mean(combined_sum_geo)
       (supervised is within 5% of geo at the planner level mean)

G2. Per-seed: supervised beats or matches geo (supervised >= geo - 0.02)
    on at least 3/5 seeds   (using 5+ seeds; 3/5 = 60% threshold,
                              equivalent to "more than half")

G3. mean(combined_sum_supervised) >= mean(phase17_locked) + 0.02
       OR at minimum: mean(supervised) >= mean(locked) - 0.02
       (does not regress meaningfully below the locked baseline)

G4. mean adapter geo-recovery Spearman across seeds >= 0.45
       (the adapter actually extracts geometric structure from
        slots, reproducible)
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3 (strong) + G4** | Swap locked recipe to `combined_sum_supervised`. BLA System-1 → System-2 → planner is the new default. |
| G1 + G2 + G3 (acceptable: ≥ locked - 0.02) + G4 | "Acceptable swap": supervised within 5% of geo and does not regress meaningfully below locked. Swap with caveat (deferred until cross-task transfer confirms). |
| G1 only (G2/G3/G4 fail) | Supervised tracks geo on average but per-seed performance is too variable; keep engineered geo as locked recipe. |
| 0/4 | Supervised is not yet competitive; further v3 work needed (wider latent, larger data, multi-task auxiliary). |

## What this phase is NOT

- Not a new training run.
- Not a re-collection.
- Not cross-task transfer (Phase 18κ).
- Not a new architectural variant (Phase 18λ-v3 wider latent etc.).

## What aggregating already-existing data lets us claim

The 6-seed view is statistically richer than the precommitted 5-seed
target. We have:
- 6 independent seeds (3 from 18λ-multi + 3 from 18λ-v2)
- Per-seed identical-RNG comparisons (geo and supervised in same
  script run sharing eval episodes)
- Pre-committed gates evaluated AFTER precommit-doc commit (this
  doc commits before the aggregation)

This is the right time to lock or reject the swap.

## Reproducibility

Source data:
- `artifacts/phase18l_multi/{summary_seed1.json, summary_seed2.json,
  aggregate.json}` (seeds 0, 1, 2 of 18λ-multi)
- `artifacts/phase18l/summary.json` (seed 0 of 18λ; reused into
  18λ-multi seed 0 aggregate)
- `artifacts/phase18l2_multi/{summary_seed0.json, summary_seed1.json,
  summary_seed2.json, aggregate.json}` (seeds 0, 1, 2 of 18λ-v2)

Aggregation script: `scripts/phase18mu_aggregate.py` (to be written
before evaluation).

Decision doc: `docs/phases/PHASE_18M_SWAP_CONFIRMATION_DECISION.md`
(after gates evaluated).

## Sibling memory

- `[[bla-locked-planning-recipe]]` — current `combined_sum_geo`
  recipe; 18μ tests whether to swap
- `[[value-relevant-subspace-recoverable-from-slots]]` — 18λ-multi
- `[[engineered-aux-loss-useful-inductive-bias]]` — 18λ-v2 supports
  why supervised wins over end2end
