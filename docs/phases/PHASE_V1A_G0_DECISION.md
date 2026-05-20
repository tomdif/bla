# Phase V1a-G0 — Cosmos-Tokenizer Token-Stability Probe (Decision)

**Date:** 2026-05-20.
**Status:** ✅ **STRONG PASS — Cosmos-Tokenizer CV4x8x8 is a valid Layer-1 encoder swap candidate.**
**Compute:** ~30 seconds on B200 (model load + 4 probe Ks).
**Parent:** `docs/phases/PHASE_V1_G0_DECISION.md` (V-JEPA 2 G0 failure).

## Headline

> **V1a-G0 passes decisively. Cosmos-Tokenizer CV4x8x8 is stable
> under rolling-window inference and is greenlit for full
> encoder-swap testing. V-JEPA 2 is redirected to clip-summary
> retrieval-key use because its per-token features are
> position-bound under rolling windows.**

Per-token cos 0.997-0.999 at K∈{9,13,17,21}, 98-99% of spatial
tokens clear the 0.95 gate. The prediction from V-JEPA 2's
failure ("VAE without RoPE shouldn't have position drift") holds.

## Per-K results (CV4x8x8, offset=4 frames = 1 latent position shift)

| K (frames) | T_lat | Per-token mean cos | Per-token frac ≥0.95 | Pooled cos | Clip-summary cos |
|---:|---:|---:|---:|---:|---:|
| 9 | 3 | **0.997** | 98.6% | 0.999 | 1.000 |
| 13 | 4 | **0.998** | 99.1% | 0.999 | 1.000 |
| 17 | 5 | **0.999** | 99.3% | 1.000 | 1.000 |
| 21 | 6 | **0.999** | 99.5% | 1.000 | 1.000 |

Determinism check (same window twice): **0.99999** (effectively 1.0;
small bfloat16 noise).

Min cosine across spatial tokens: ~0.62, but the p10 is 0.999.
A tiny fraction (~1%) of spatial tokens at the latent-grid boundary
have lower similarity — likely edge effects from the causal
convolution. The bulk of the latent volume is rock-solid.

## Comparison to V-JEPA 2 G0 (V1-G0 finding)

```
                            V-JEPA 2 ViT-L      Cosmos CV4x8x8
per-token mean cos          0.46-0.52           0.997-0.999
per-token frac ≥0.95        0.4-0.8%            98.6-99.5%
pooled-temporal mean cos    0.52-0.88           0.999
clip-summary cos            0.99+               1.000
determinism                 1.000               0.99999
strict G0 pass              ❌ all K            ✅ all K
```

The asymmetric result reflects the two architectures:

```
V-JEPA 2:    transformer with 3D RoPE on temporal axis
             → identical content at different positions → different tokens
             (by design; positional information is part of the representation)

Cosmos CV4x8x8: causal CNN/VAE encoder, no RoPE
             → translation-equivariant in time within causal constraints
             → identical content at different temporal positions → ~identical tokens
```

This confirms the prediction from
`feedback_vjepa2_position_bound_tokens` (memory, 2026-05-20).

## Implications for V1

```
V1a (Cosmos encoder swap):  GREENLIT
  Next: full M1-M5 swap test. Compare against current Phase-14
  OF-JEPA visual encoder on:
    M1. rolling-window K=5 (note: Cosmos natural stride is K=9
        for 1 latent position; will need to pick the right K)
    M2. identity-conditioned id_h_mse
    M3. demo retrieval success on PickPlaceCan DR1 protocol
    M4. recipe_router input stability
    M5. perturbation robustness

V1b (V-JEPA 2 clip-summary as DemoRetriever key):  unchanged
  Orthogonal to V1a; still high-priority for DR3 σ-gap.
```

## Cosmos-specific caveats for the full V1a swap

```
1. Input resolution: Cosmos CV4x8x8 wants 256×256; our current
   Phase-14 OF-JEPA encoder is trained on 128×128. The
   M1-M5 comparison should hold the input pipeline equal
   (re-render robosuite at 256×256 for both encoders).

2. Temporal compression: 4× temporal. If we want "per-frame"
   features at 4× the model's natural granularity, we need to
   either (a) upsample latent temporally, or (b) accept 4×
   stride. BLA's rolling-window K=5 was at 1× stride; this
   becomes K=20 frames per 5 latent positions.

3. VAE features are pixel-reconstruction-oriented, NOT
   contrastive/object-centric. The G0 test passed on rolling
   stability, but V-JEPA 2's R2 risk applies here too:
   identity-under-occlusion is untested. M2 (id_h_mse on a
   stress test with occluders) is the critical V1a metric.

4. The 16-channel latent at 32×32 = 16,384 features per latent
   frame. That's 16× richer than OF-JEPA's current 6 slots × 128
   dims = 768. Layer 2 (OF-JEPA slot attention) needs to consume
   this — either via a learned projection or by replacing the
   ViT-style patch encoder with the Cosmos latent as input.
```

## Compute summary

```
Pod:        B200 (180 GB VRAM)
Model size: ~ 100MB (encoder.jit alone)
VRAM peak:  0.65 GB
Runtime:    < 30 seconds total (4 K values + determinism check)
```

## CRITICAL CAVEAT (do not lose sight of)

Cosmos passing G0 means:
> the features are temporally stable under rolling-window overlap.

It does NOT yet mean:
> the features preserve object identity, occlusion state, or
> contact-relevant geometry.

That is what V1a full encoder-swap must test. The G0 gate proved
**interface compatibility**, not **representational quality**.
M2 (id_h_mse under occlusion) is the decisive V1a metric.

This mirrors the "interface matters" lesson from prior phases:

```
Raw token interface:
  can fail even when the foundation model is high quality.
Aggregated/adapted interface:
  can be the right use.
```

Cosmos passes the *raw token* interface test. V-JEPA 2 only
passes the *aggregated/clip-summary* interface test. They are
now slotted into the integration plan accordingly.

## Decision

**V1a is greenlit.** Next concrete step: write the V1a precommit
covering M1-M5 with the right Cosmos-specific protocol (256×256
input, K=9 or K=13 natural latent-grid alignment, occlusion stress
on M2). After precommit, build the actual swap pipeline.

V1b (V-JEPA 2 clip-summary key) remains higher-priority than V1a
for tackling DR3's σ gap — but V1a is now a real second track,
not just a placeholder.

## Files

- Script: `scripts/phase_v1a_g0_cosmos.py`
- Pod result: `/root/bla/runs/phase_v1a_g0_cosmos/summary.json`
- Parent: `docs/phases/PHASE_V1_G0_DECISION.md`
- Foundation memo: `docs/phases/PHASE_V0_FEASIBILITY_MEMO.md`

## Locked
