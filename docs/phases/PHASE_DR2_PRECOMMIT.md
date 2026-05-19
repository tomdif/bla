# Phase DR2 — Retrieval-Quality Ablation (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Gates locked before run.**
**Parents:**
- `docs/phases/PHASE_DR1_DECISION.md` (commit `5b00a7f` + `98d3659`)
- `docs/BLA_SCALING_ROADMAP.md` §4

## Purpose

DR1 strong-passed all 4 gates but `demo_retrieval_top1` had high
variance (σ_imp = 0.135). `demo_retrieval_top3_avg` had lower
variance (σ = 0.052) but also lower mean (0.280 vs 0.346). DR2
asks: **can we reduce top-1's variance without sacrificing mean?**

> Ideal DR2 winner: mean ≥ 0.33, σ ≤ 0.08, success ≥ DR1 top1 or
> top3_avg.

## Falsifiable claim

```
At least one DR2 mode (goal-relative key, slot-state key, or
top-k outcome reranking over geometry) achieves mean improvement
≥ geometry_top1 − 0.03 AND std ≤ 0.08 on PickPlaceCan 3-seed × n=30.
```

If no DR2 mode clears that, then the high variance of top-1 is
inherent to the demo bank itself (limited coverage / brittle demos),
not the retrieval metric. That would be a useful negative finding
and would direct DR3+ toward bank expansion.

## Protocol

```
Task     : robosuite PickPlaceCan
Demo bank: 24 working demos (DR1 screen result, ids fixed)
Init     : state-matched reset to RANDOM demo from bank
           (same protocol as DR1 — seed-controlled choice)
Seeds    : 0, 1, 2 (parallel on GPUs 0/1/2)
Episodes : 30 per seed per mode (5 for pilot)
Sigma    : per-dim, σ_motion=0.12, σ_gripper=0 (only used by phase17_locked)
```

## Modes (minimum set per user's call)

```
geometry_top1                — DR1 baseline (absolute can+eef pose)
goal_relative_top1           — relative key: (eef−can, can_height, eef_height)
geometry_top3_avg            — DR1 stable baseline
geometry_topk_outcome_rerank — top-5 by geometry, pick highest outcome score
slot_state_top1              — OF-JEPA encoded slot state as the key
```

Additional reference modes carried over from DR1:

```
demo_no_cem_oracle           — ceiling reference (uses reset target directly)
demo_no_cem_cycle            — broken-cycling baseline
phase17_locked               — CEM around geometry_top1
naive_cem                    — floor
```

## Key construction details

### geometry_top1 (DR1 baseline; 6-D absolute)

```
key = [can_x, can_y, eef_x, eef_y, can_z, eef_z]
```

### goal_relative_top1 (5-D relative)

```
key = [eef_x − can_x, eef_y − can_y,
       can_x − table_center_x, can_y − table_center_y,
       can_z, eef_z]
```

PickPlaceCan's nominal table center is (0.0, 0.0). The goal is
implicit (lift the can ≥ target_z_gain).

### slot_state_top1 (slot_dim * n_slots = 768-D)

```
key = encode_frame(model, agentview_image).flatten()
```

At bank-build time, encode each demo's init frame. At query time,
encode the current env frame.

### geometry_topk_outcome_rerank

```
top_k = geometry_retriever.retrieve(query, k=5)
best = max(top_k, key=lambda d: d.outcome_score)
```

`outcome_score` = z_gain achieved by replaying the demo on its own
state-matched init (recorded at bank-build time). Higher = stronger
demo. Reranking top-5 by outcome should prefer demos that lift
consistently over weakly-lifting neighbors.

## Pre-committed gates

```
G1: best DR2 mode mean ≥ geometry_top1_mean − 0.03      (≥ 0.316)
G2: best DR2 mode std ≤ 0.08
G3: best DR2 mode success ≥ phase17_locked + 10pp       (≥ 20%)
G4: best DR2 mode beats geometry_top3_avg on mean       (≥ 0.281)
```

Strong-pass:

```
SP1: mean ≥ 0.346  AND  std ≤ 0.08
SP2: best DR2 mode is one of {goal_relative, outcome_rerank}
     (i.e. cheap ablations win over slot_state/learned)
```

Acceptable pass:

```
mean ≥ 0.32  AND  std ≤ 0.08
```

## Pre-committed predictions

```
geometry_top1            (DR1 baseline)         expected: mean ~0.346, σ ~0.135
goal_relative_top1                              expected: mean ~0.30-0.40, σ ~0.06-0.10
                          (most likely winner per user's call)
geometry_top3_avg        (DR1 stable baseline)  expected: mean ~0.280, σ ~0.052
geometry_topk_outcome_rerank                    expected: mean ~0.34-0.40, σ ~0.05-0.08
                          (other likely winner)
slot_state_top1                                 expected: mean ~0.25-0.40, σ wide
                          (could go either way — 768-D key may overfit)
```

User's prediction (verbatim):
> Most likely winner: goal_relative_topk_rerank OR
> geometry_topk_outcome_rerank. Why: top1 variance likely comes
> from nearest-neighbor mismatch. Top-k reranking gives the system
> a way to avoid one bad neighbor without reintroducing action-
> space CEM.

## Falsification scenarios

```
F1: No DR2 mode achieves mean ≥ 0.316 AND std ≤ 0.08 →
    The variance of top-1 is inherent to the bank, not the metric.
    DR3 should expand the demo bank or relax the demo-screening cutoff.

F2: slot_state_top1 strongly beats all others →
    Surprising; suggests OF-JEPA slot features carry retrieval
    signal beyond engineered geometry. Would shift the retrieval
    story toward learned representations.

F3: outcome_rerank actively HURTS vs geometry_top1 →
    The outcome score is noisy or anti-correlated with deployment
    success. Need to revisit the screening metric.

F4: Best DR2 mode = geometry_top3_avg (no new mode wins) →
    DR1's existing stable variant is the ceiling without richer
    features. DR3 should explore other improvements (bank size,
    learned embeddings).
```

## What this phase does NOT do

- Does not introduce CEM. Recipe E remains demo-replay only.
- Does not change runtime (rolling-window K=5 is the deployment
  runtime; this eval uses the existing batched runtime for
  apples-to-apples with DR1).
- Does not learn a retrieval embedding (deferred to DR3 if needed).
- Does not cross-task validate (PickPlaceCan only; the doctrine
  doesn't need more task validations).

## Locked
