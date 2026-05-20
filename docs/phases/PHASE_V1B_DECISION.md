# Phase V1b — V-JEPA 2 Clip-Summary as DemoRetriever Key (Decision)

**Date:** 2026-05-20.
**Status:** ⚠️ **Slight loss in simulator; V-JEPA does not replace privileged geometry.**
**Parent:** `docs/phases/PHASE_V1_G0_DECISION.md` (V1-G0 finding that redirected V-JEPA from encoder swap to retrieval key).
**Compute:** ~35 minutes on B200 (V-JEPA bank build + 3 seeds × 6 modes × n=30 + cache reuse).

## Headline

> **V-JEPA clip embeddings do not pay off as a simulator
> retrieval-key replacement for privileged geometry, but they
> remain promising for BLA-Forge where true geometry is
> unavailable or noisy.**

## Three-seed aggregate (n=30 per seed, full-100 reset pool — DR3 protocol)

| Mode | seeds 0/1/2 | imp_mean ± std | succ_mean | match_rate |
|---|---|---:|---:|---:|
| demo_no_cem_oracle | 0.072 / 0.129 / 0.074 | 0.092 ± 0.033 | 6.7% | 100% |
| demo_no_cem_cycle | 0.003 / 0.069 / 0.036 | 0.036 ± 0.033 | 3.3% | 1% |
| **geometry_top1** | 0.202 / 0.169 / 0.202 | **0.191 ± 0.019** | **18.9%** | **26.7%** |
| **vjepa_top1** | 0.169 / 0.103 / 0.069 | **0.114 ± 0.051** | **11.1%** | **13.3%** |
| vjepa_top3_avg | 0.069 / 0.003 / 0.069 | 0.047 ± 0.038 | 4.4% | 0% |
| geometry_top3_avg | 0.071 / 0.236 / 0.189 | 0.165 ± 0.085 | 15.6% | 0% |

**Δ(geometry_top1 − vjepa_top1):  +0.078 imp / +7.8pp success / +13.3pp match.**

## The decisive single-number diagnostic

Across all 90 episodes (3 seeds × n=30), **24 reset targets were in
the 24-demo working bank**. For these episodes, the "right" demo
trivially exists in the bank — the question is whether each
retriever finds it.

```
Geometry correctly retrieves in-bank:  24/24 = 100.0%
V-JEPA correctly retrieves in-bank:    12/24 = 50.0%
```

V-JEPA misses **half** the easy retrievals. The 6-D mujoco pose key
finds the matched demo every single time it's in the bank.
V-JEPA's 1024-D clip embedding does not — it routes to a
visually-similar-but-state-mismatched demo half the time.

This is not a "V-JEPA is bad" finding. V1-G0 already showed clip
features are extremely stable (cosine ≥0.99 across overlapping
windows). The failure is at the **interface** level: V-JEPA's
clip embedding encodes scene-level visual semantics
(appearance, camera, gripper visual pose, background) but not
the specific physical state variables needed to choose a
state-matched demo.

## Per-episode picks (seed 0) confirm the attractor pattern

V-JEPA repeatedly picks demo **10** (9 of 30 episodes = 30% of
retrievals). Geometry's picks are spread across many demos
(47, 82, 46, 23, 30, ...). V-JEPA has a "visual prototype"
attractor — likely a demo with average/representative scene
appearance that minimizes mean distance to many query embeddings.

## Interpretation

```
geometry_top1:
  task-state key — 6-D MuJoCo pose
  privileged access to exactly what matters in sim:
    object pose, EEF pose, goal-relative geometry
  best for sim retrieval

V-JEPA clip key:
  visual/context key — 1024-D mean-pooled patch tokens
  encodes appearance, camera, gripper visual pose,
  background/lighting, global visual layout
  but UNDER-specifies physical state
```

The result fits the recurring BLA lesson:

> Foundation-model features need the right interface; raw/high-level
> embeddings are not automatically task-state keys.

Same pattern as:
- `raw-slot-state-not-for-retrieval` (DR2)
- `vjepa2-position-bound-tokens` (V1-G0)
- `frozen-slots-not-enough-for-value` (Phase 18θ)

## What V-JEPA still might be good for

V-JEPA loses in simulator because geometry retrieval has
**privileged access** to the underlying physical state via
`env.sim.data`. That privilege evaporates in BLA-Forge real-world:

```
Real-world deployment regime:
  Camera and proprioception give NOISY estimated geometry.
  V-JEPA clip embeddings are extremely stable (V1-G0: ≥0.99 cos).

The interesting test then becomes:
  estimated geometry (camera + OF-JEPA adapter) alone
  vs
  estimated geometry + V-JEPA clip features

NOT V-JEPA alone vs perfect privileged geometry.
```

## V-JEPA role after V1b

```
NOT the default simulator retrieval key.

Candidate real-world fallback when privileged geometry is unavailable.
Visual context feature for diagnostics and run-time inspection.
Possible hybrid key with camera-estimated geometry for BLA-Forge.
```

## What this phase falsifies

```
Falsified:
  V-JEPA clip-summary as a drop-in replacement for the 6-D
  geometry retrieval key in simulator deployment.

Falsified diagnostic mechanism:
  V-JEPA's failure is NOT instability of the features
  (V1-G0 proved cos ≥0.99 across windows). It IS the encoding
  of scene appearance over physical state.

Variance pattern (also a finding):
  geometry_top1 σ_imp = 0.019  (very stable across seeds)
  vjepa_top1    σ_imp = 0.051  (~2.7× higher variance)
  vjepa_top3_avg σ_imp = 0.038 (no consistent improvement)
  geometry_top3_avg σ_imp = 0.085 (worst variance; same DR2/DR3
                                    instability of top-k averaging)
```

## Runtime cost

V-JEPA model load: ~10s (V-JEPA-2 ViT-L bf16 → ~1.2GB VRAM).
V-JEPA encode per query: ~0.5s for a 4-frame 256×256 clip.
Bank build (24 demos × 4-frame encode): ~30s one-time, cached.

Geometry key: zero compute beyond `env.sim.data.get_body_xpos()`,
which is essentially instant.

Net cost: V-JEPA adds ~15 seconds per 30-episode mode vs
geometry's effectively zero. Not a deal-breaker, but irrelevant
since V-JEPA loses on quality too.

## What this phase does NOT yet test

```
1. Hybrid retrieval key
   score = geometry_distance + λ * vjepa_distance
   Combines V-JEPA's visual context with geometry's
   state-precision. Worth testing if there's downstream
   compute commitment.

2. Camera-estimated geometry vs V-JEPA in a no-privileged-data
   regime
   The "real" V-JEPA test for BLA-Forge: when sim.data
   doesn't exist, can V-JEPA stand in?

3. Constrained rerank (DR3 protocol) with V-JEPA as the
   outcome-bias secondary key
   top-k by geometry distance → filter by ≤ 1.25× NN dist →
   rerank within filter by V-JEPA clip cosine. State-match
   stays primary; V-JEPA is the tiebreaker.

These remain candidate V1c+ probes if real-world work needs them.
```

## Decision

**Lock V1b as a negative-but-clarifying ablation.** V-JEPA is
removed from the "primary retrieval key" candidate set for
simulator deployment. V-JEPA retains a candidate slot for:

```
- BLA-Forge real-world fallback (geometry-free retrieval)
- hybrid keys with noisy estimated geometry
- diagnostic/context feature for run-time inspection
```

Tasks updated:
```
#178 V1b                 completed (this decision)
#179 V1a Cosmos full     pending  (still the encoder-swap path)
#180 NEW — V-JEPA hybrid retrieval key (deferred until BLA-Forge or
                                          downstream need arises)
```

## Files

- Precommit (lightweight, in-spec): `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md` §3
- Script: `scripts/phase_v1b_vjepa_retrieval.py`
- Pod results: `/root/bla/runs/phase_v1b_main_seed{0,1,2}/summary.json`
- Local copies: `/tmp/v1b_seed{0,1,2}.json`
- Parent: `docs/phases/PHASE_V1_G0_DECISION.md`

## Locked
