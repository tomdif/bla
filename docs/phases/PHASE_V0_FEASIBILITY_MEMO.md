# Phase V0 — Visual Foundation Feasibility Memo

**Date:** 2026-05-20.
**Status:** ✅ Complete — no compute committed; ~3 hours of research.
**Source spec:** `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md` §8.

## TL;DR — go / no-go per layer

```
Cosmos-Tokenizer    ✅ GO    primary V1 candidate (with V-JEPA 2)
V-JEPA 2            ✅ GO    primary V1 candidate (lead candidate)
Cosmos-Predict2     ⏸ HOLD  too heavy for encoder swap; possible V2 data gen
SANA-WM             ❌ SKIP  wrong fit — camera-only conditioning, no robotics
                              demos, CC-BY-NC-SA weight license
```

## V1 launch readiness

Both V-JEPA 2 and Cosmos-Tokenizer pass V0 cleanly:

```
Public:        both ungated, on HuggingFace, native PyTorch loaders
Licenses:      V-JEPA 2 = MIT; Cosmos = NVIDIA Open Model License
                (both permissive enough for research; both could even
                 ship into BLA-Forge commercial work if we needed it)
Compute:       both fit comfortably on B200/180GB
Feature path:  both expose encoder-only inference
                V-JEPA 2: last_hidden_state [B, N_tokens, D]
                          (ViT-L: D=1024; ViT-H: D=1280; ViT-g: D=1408)
                Cosmos-Tokenizer: latent grid [B, 16, T_lat, H_lat, W_lat]
                                  (continuous-video CV4x8x8 default)
Robotics:      V-JEPA 2 has 1B "AC" variant with Franka pick-and-place
                65–80% zero-shot on Droid.
                Cosmos-Predict2 has a 14B variant fine-tuned on DROID
                (sample_GR00T_dreams variant).
```

## Per-question answers (from spec §8)

### Q1. Are foundation feature interfaces public?

**Yes for V-JEPA 2 + Cosmos-Tokenizer. No-go for SANA-WM.**

V-JEPA 2: `AutoModel.from_pretrained(...)`, ungated, `skip_predictor=True` returns per-patch tokens directly.

Cosmos-Tokenizer: encoder is independently loadable as TorchScript `.jit` from the HF cards (`nvidia/Cosmos-{0.1,1.0}-Tokenizer-CV8x8x8` etc.).

SANA-WM: project page + paper public; code in `NVlabs/Sana` subtree; weights gated by CC-BY-NC-SA 4.0 (non-commercial only).

### Q2. Feature dimensionality and frame rate

| Model | Input | Output token shape | D |
|---|---|---|---|
| V-JEPA 2 ViT-L/16 fpc64-256 | 64 frames × 256×256 | `[T/2, H/16, W/16]` = 32×16×16 = 8192 tokens | 1024 |
| V-JEPA 2 ViT-H/16 fpc64-256 | same | same shape | 1280 |
| V-JEPA 2 ViT-g/16 fpc64-384 | 64 frames × 384×384 | 32×24×24 = 18,432 tokens | 1408 |
| V-JEPA 2 ViT-L fpc16 (SSv2 ft) | 16 frames × 256×256 | 8×16×16 = 2048 tokens | 1024 |
| Cosmos-Tokenizer CV4x8x8 | 9 frames × 512×512 | latent `[16, 3, 64, 64]` (16-ch, 3 temporal, 64×64 spatial) | 16 |
| Cosmos-Tokenizer CV8x8x8 | longer clips | 6-ch latent | 6 |

V-JEPA 2 uses **tubelet=2** by default — frame stride 1 doesn't produce equivalent tokens. Rolling-window K=5 should stride by 2 frames or accept slight phase drift.

### Q3. V-JEPA 2 short-clip identity stability

**Open empirical question — must be probed in V1.** V-JEPA 2's 3D RoPE gives shifted-window tokens the right positional encoding; tubelet=2 forces even-frame strides for clean alignment. No published streaming-stability metric exists, so V1 needs a probe:

```
Probe: encode same video at offset 0, 2, 4 frames.
Metric: cosine similarity of overlapping patch tokens.
Gate: ≥ 0.95 cosine on overlap window for V1 to proceed.
```

This becomes a V1 precommit.

### Q4. Can Cosmos generate robosuite-like rollouts?

**Yes, indirectly.** Cosmos integrates with Isaac Sim / Isaac Lab and the GR00T-Mimic blueprint. A Predict2-14B variant is fine-tuned on DROID; Predict2-2B has an action-conditioned variant. No robosuite-specific demos but the manipulation domain is covered.

Practical use for V2 perturbation suite: Cosmos-Predict2-2B-Video2World can generate stylistic variations (lighting, texture, distractor objects) from a starting frame — useful for the V2 stress test.

### Q5. Does SANA-WM help BLA?

**No.** SANA-WM is image+camera-trajectory → video for novel-view synthesis along a metric-scale path. It cannot ingest a robot demo trajectory; the camera is the only controllable variable. For BLA's V2 perturbation suite (lighting, distractor, occlusion variations of robotics scenes), domain randomization in Robosuite/Isaac is strictly better — free perturbation axes with ground-truth identity preserved.

Plus: CC-BY-NC-SA 4.0 weights block commercial use, and "search-budget-zero around expert demos" doctrine de-prioritizes minute-scale generative imagination anyway.

## V1 plan, updated

Encoder swap test compares:

```
A — current OF-JEPA visual encoder (Phase 14-trained ConvNeXt-T)        baseline
B — V-JEPA 2 ViT-L/16 fpc64-256                                          primary
C — Cosmos-Tokenizer CV4x8x8 (encoder only, BF16)                        secondary
D — V-JEPA 2 ViT-L fpc16 (SSv2 finetune)                                 secondary
    (16-frame native; natural fit for K=5 rolling window if probe Q3 passes)
```

Drop the (D) "SANA-WM data augmentation" option from the original spec
§3 mode set. SANA-WM is skipped entirely.

## V1 precommit (new gate)

Add to the V1 gate set:

```
V1-pre G0 (token-stability probe):
  For each candidate encoder, compute cosine similarity of patch tokens
  on overlapping windows offset by 2 frames.
  Gate: ≥ 0.95 cosine on the overlap region for ≥ 80% of test frames.
  Below this, the encoder is not stable enough for OF-JEPA rolling-window
  identity binding — falsifies encoder swap before measuring downstream.
```

This is cheaper than running full M1–M5 and rules out unstable backbones early.

## V2 plan, updated

V2 perturbation suite stays as planned in `BLA_VISUAL_FOUNDATION_INTEGRATION.md`
§4, but the perturbation source changes:

```
Old V2 plan:    Cosmos-Predict2 or SANA-WM video generation
New V2 plan:    Robosuite domain randomization (lighting, textures,
                 distractors, camera angles) — Cosmos-Predict2 only
                 considered later if domain randomization is insufficient.
```

Domain randomization preserves ground-truth identity through perturbations,
which the foundation-model-generated path does not.

## Open risks

```
R1. V-JEPA 2 tubelet=2 stride may force K=5 rolling window to be
    K=4 or K=6 (even strides) for clean alignment. Verify in V1-pre.

R2. Cosmos-Tokenizer is pixel-reconstruction-oriented (VAE), NOT
    contrastive / object-centric. Its features may not preserve
    identity binding under occlusion. Probe G0 catches this.

R3. V-JEPA 2 ViT-L (300M) is 4-6× the parameter count of our current
    Phase-14 encoder. Layer-1 swap will dominate inference compute.
    Decision: accept (B200 has headroom); the V1 gate is about
    QUALITY, not speed.

R4. Neither paper reports identity-conditioned tracking metrics
    (id_h_mse-style) on contact-sensitive manipulation. V1 is the
    first time this gets measured.
```

## Decision

✅ **Proceed to V1 (task #174).** V-JEPA 2 ViT-L is the lead
candidate, Cosmos-Tokenizer is the secondary. Add a V1-pre G0
token-stability probe before running the full M1–M5 swap.

Skip SANA-WM entirely. Drop Cosmos-Predict2 from V1 (it's a V2
candidate at most).

## Files

- This memo: `docs/phases/PHASE_V0_FEASIBILITY_MEMO.md`
- Parent: `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md`
- Update needed: §3 mode list (remove SANA-WM, add V-JEPA 2 fpc16
  variant); §4 (replace generative perturbation with domain rand);
  §3 gates (add G0 token-stability precommit).

## Locked
