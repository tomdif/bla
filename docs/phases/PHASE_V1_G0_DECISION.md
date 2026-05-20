# Phase V1-G0 — V-JEPA 2 Token-Stability Probe (Decision)

**Date:** 2026-05-20.
**Status:** ⚠️ **Strict G0 fails; clarifying finding redirects V1 scope.**
**Compute:** ~3 minutes on B200 (model load + 4 probe Ks + diagnostics).
**Parent:** `docs/phases/PHASE_V0_FEASIBILITY_MEMO.md`.

## Headline

> **V-JEPA 2's per-token features are position-bound (3D RoPE);
> direct replacement of OF-JEPA's per-frame encoder is not viable.
> But V-JEPA 2's clip-summary features (mean-pooled) are
> extremely stable (cos ≥ 0.99), making V-JEPA 2 a strong
> candidate for the DemoRetriever key layer instead of the
> per-frame slot encoder.**

The strict G0 gate (per-token cosine ≥ 0.95 on overlapping windows)
fails for an architectural reason — RoPE positional encoding makes
the same source content at different temporal positions produce
different tokens. This isn't a quality issue; it's an integration
mismatch with OF-JEPA's per-frame identity-binding API.

## Per-K results (3-second probe)

| K | Per-token mean cos | Per-token frac ≥0.95 | Pooled-temporal mean cos | Clip-summary cos |
|---:|---:|---:|---:|---:|
| 4 | 0.52 | 0.8% | 0.52 | **0.992** |
| 5 | 0.52 | 0.8% | 0.52 | **0.992** |
| 6 | 0.46 | 0.6% | 0.80 | **0.996** |
| 8 | 0.48 | 0.4% | 0.88 | **0.997** |

Sanity check: re-encoding the same window twice → cosine **1.000**
(deterministic). The measurement is sound.

## What the three metrics together tell us

```
PER-TOKEN cosine (raw last_hidden_state at aligned tubelet positions):
  0.46-0.52 across all K.
  Position bias dominates the embedding — same source frames at
  different temporal positions ARE different tokens by design (3D RoPE).

POOLED-TEMPORAL cosine (mean across tubelets, per spatial patch):
  Grows monotonically with K: 0.52 (K=4) → 0.88 (K=8).
  As more tubelets get averaged, position bias cancels out. The
  CONTENT direction emerges from the noise.

CLIP-SUMMARY cosine (single 1024-D vector per window via mean over all tokens):
  ≥ 0.992 across all K.
  At the clip-summary level, V-JEPA 2's features are extremely
  stable. THIS IS THE RETRIEVAL-USEFUL SIGNAL.
```

## What this changes about V1 plan

The original V1 (`BLA_VISUAL_FOUNDATION_INTEGRATION.md` §3) framed
the swap as Layer-1-encoder replacement holding Layer 2 fixed. G0
falsifies that framing for V-JEPA 2: the tokens aren't compatible
with OF-JEPA's per-frame slot attention.

**Updated V1 scope** (locked here):

```
Drop:  V-JEPA 2 ViT-L as per-frame OF-JEPA encoder replacement
       (modes B and D from V0's mode list).

Add:   V-JEPA 2 ViT-L clip-summary features as DemoRetriever key
       (Layer 3 input, NOT Layer 2 encoder).
       Test against the current geometry_top1 6-D key on
       PickPlaceCan DR1/DR2/DR3 protocol.
```

This re-frames V-JEPA 2's role from "Layer 1 perception" to "Layer 3
retrieval key feature." That's a smaller, cleaner experiment that
respects what V-JEPA 2 actually produces (stable clip-level
representations) rather than what OF-JEPA needs (stable per-frame
position-resilient tokens).

## Why this is a useful finding, not a setback

```
1. We avoided a wrong full-V1 swap.
   Without G0, we'd have spent compute training an OF-JEPA encoder
   to consume V-JEPA 2 tokens and discovered the position bias by
   way of degraded id_h_mse on contact-sensitive tracking. G0
   caught it in 3 minutes.

2. We learned that V-JEPA 2's strength is clip-level retrieval.
   This is a real signal — clip_summary cos ≥ 0.99 means a
   1024-D V-JEPA 2 vector encodes "what this short clip looks
   like" with high fidelity. That's exactly what DemoRetriever
   needs.

3. The doctrine "state match primary, outcome tiebreaker" is
   still the right rule. The question DR2/DR3 left open was
   whether better retrieval features could close the variance
   gap. V-JEPA 2 clip features are now a candidate answer.
```

## Cosmos-Tokenizer status

Still untested at G0. Cosmos-Tokenizer is a VAE encoder (not a
RoPE-positional video transformer), so the position-bias mechanism
that hit V-JEPA 2 may not apply. If we proceed with V1, Cosmos
should get its own G0 — same probe, fresh measurement.

Prediction: Cosmos-Tokenizer per-token stability is likely better
than V-JEPA 2's (VAEs are typically position-invariant), but its
features are pixel-reconstruction-oriented and may not preserve
identity under occlusion. Different failure mode.

## Compute summary

```
Pod:           B200 (180 GB VRAM)
Model size:    300M (V-JEPA 2 ViT-L/16 fpc64-256)
VRAM:          0.65 GB allocated (BF16 weights + activations for K=8)
Time:          ~3 minutes total (load + 4 Ks + determinism check)
Cost basis:    Cheap. Reusable for Cosmos probe.
```

## Decision

**V1 scope updated.** Drop V-JEPA 2 from the encoder-swap track.
Add V-JEPA 2 to the retrieval-key track as Phase **V1b**.

Updated next moves:

```
V1a (was V1) — encoder swap test, NOW Cosmos-Tokenizer only:
                run G0 on Cosmos-Tokenizer CV4x8x8;
                if it passes, run the full M1-M5 swap test;
                if it fails, drop the encoder-swap track entirely.

V1b — NEW — V-JEPA 2 clip-summary as DemoRetriever key:
                replace 6-D geometric key with 1024-D V-JEPA 2
                clip-summary; compare to geometry_top1 / E2_FAST
                on PickPlaceCan DR1 protocol. Pre-commit gates.
```

V1b is the higher-priority next move because:
- It directly addresses DR3's open question (close the σ gap)
- It uses V-JEPA 2 in its proven-stable mode (clip-summary)
- It costs less than V1a (no per-frame encoder retraining)

## Files

- Script: `scripts/phase_v1_g0_token_stability.py`
- Pod result: `/root/bla/runs/phase_v1_g0/summary.json`
- Parent V0 memo: `docs/phases/PHASE_V0_FEASIBILITY_MEMO.md`
- Spec: `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md` (§3 needs update)

## Locked
