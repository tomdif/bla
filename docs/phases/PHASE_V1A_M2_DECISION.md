# Phase V1a-M2 — Cosmos Identity-Preservation Probe (Decision)

**Date:** 2026-05-20.
**Status:** ❌ **V1a falsified. Cosmos-Tokenizer features fail M2 even though they passed G0.**
**Parents:**
- `docs/phases/PHASE_V1A_G0_DECISION.md` (Cosmos passed G0; greenlit V1a)
- `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md` §3

## Headline

> **Cosmos-Tokenizer CV4x8x8 features at high-variance (active)
> latent cells are CHAOTIC under real object motion — adjacent
> frames flip from cosine 0.99 to cosine −0.76. The encoder
> reconstructs local pixel content at each cell, not object
> identity. V1a is dead at M2; the full encoder swap is not
> worth building.**

## V1a was greenlit by G0 — what changed at M2?

G0 (V1a-G0, 2026-05-20) measured temporal stability of Cosmos
features under overlapping rolling windows on a STATIC scene
(robosuite Lift, random small motions, no actual cube
displacement). Per-token cosine was 0.997-0.999, clip-summary
1.000. The verdict: "PROCEED — Cosmos features are stable under
rolling-window inference."

M2 v1 (first attempt today) tried to test identity preservation
under motion, but both rollouts had ZERO object displacement
(scripted Lift policy failed to push the cube; first 17 frames
of a Can demo are pre-grasp approach). Cosines reported there
(0.93 / 0.41) reflected static-scene features, not real motion.

M2 v2 (this decision) replayed a FULL 135-frame Can demo:
- Can displacement: **0.87 m** lateral, **0.27 m** vertical
- Lift event begins around frame 72
- Real motion confirmed

## Three-seed-style diagnostic (one rollout, 35 latent frames)

```
High-variance latent cells (top-K = 16; cells where the encoder's
representation actually changes — these are where objects/EEF
are visible / moving):
  consecutive-frame cosine mean:    0.574
  consecutive-frame cosine median:  0.818
  consecutive-frame cosine MIN:    -0.762
  start-vs-end cosine mean:         0.765

Background latent cells (bottom-K = 16; empty regions of the scene):
  consecutive-frame cosine mean:    0.99986
  consecutive-frame cosine median:  0.99992
  start-vs-end cosine mean:         0.99866
```

Background cells are perfectly stable (as expected — empty table
doesn't change). High-variance cells are **chaotic** — adjacent
frames at the same spatial cell can have features that point in
opposite directions in 1024-D space (cosine −0.76).

## Why this happens

Cosmos-Tokenizer CV4x8x8 is a causal VAE trained for video
reconstruction. Its latent at each cell encodes "what is at this
8×8 input pixel region at this time" — a local appearance
representation. When the can enters a cell, the cell's feature
becomes "can-appearance." When the can exits and the EEF enters,
the cell's feature becomes "EEF-appearance." These are very
different features (hence the cosine inversion).

The encoder has no incentive to maintain identity across cells
during training. Identity binding is downstream OF-JEPA's job,
NOT something Cosmos provides for free.

## What G0 missed

G0 measured: "if I encode the same source content at shifted
positions, do tokens agree?" Cosmos passed because its VAE is
translation-equivariant in time within causal-padding constraints
— shifting a static scene by N frames just shifts the latent
grid by N positions, content is preserved.

M2 measured: "if the SCENE CONTENT changes (object moves through
a cell), does the cell's feature represent the moving object
identity or the cell's current pixel content?" Cosmos failed
because its VAE encodes current content, not persistent identity.

Two stability axes:
```
Temporal stability under content change → ❌ chaotic (M2)
Temporal stability under position shift → ✅ excellent (G0)
```

G0 tested only the second; we needed both.

## What this means for V1a

The V1a plan was:
```
Layer 1: Cosmos-Tokenizer (frozen, encoder)
Layer 1.5: Learned projection 16-ch → slot_dim per cell
Layer 2: OF-JEPA slot attention on top of frozen Cosmos features
```

With chaotic per-cell features, Layer 2's slot attention would
have to do ALL the identity tracking from scratch, with no
useful structure from Cosmos. That's essentially "train OF-JEPA
from scratch but with extra preprocessing steps" — strictly
worse than the current Phase-14 OF-JEPA encoder, which is
trained end-to-end and has its own learned per-pixel encoding.

## Decision

**Lock V1a as falsified at M2.** Do NOT build the full
Cosmos-Tokenizer → projection → OF-JEPA pipeline. Cosmos passed
G0 (interface stability) but fails M2 (representational fitness
for identity tracking).

## What this confirms about the broader doctrine

```
Pattern (4 phases, same shape, now with one more instance):

  Phase 18θ:   raw OF-JEPA slots → value head: fail; need adapter
  Phase DR2:   raw OF-JEPA slots → retrieval key: fail
  Phase V1-G0: V-JEPA per-token → OF-JEPA encoder: fail (RoPE)
  Phase V1b:   V-JEPA clip-summary → retrieval key: lose to geometry
  Phase V1a-M2: Cosmos latent cells → OF-JEPA encoder: fail (chaotic)
```

The repeating doctrine, sharpened:

> Foundation visual encoders trained for reconstruction or
> self-supervised prediction don't have object-centric
> representations at the per-cell or per-token level. To get a
> useful drop-in encoder for OF-JEPA-style identity binding,
> you'd need a foundation that was specifically trained with
> object-centric losses (which is what the OF-JEPA arc itself
> produces).

Said differently: object-centricity must be IN the encoder's
training objective; it can't be retrofitted by adding a slot
layer on top.

## What G0 still tells us — and a refined V0 protocol

G0 by itself is NOT sufficient as an encoder-swap gate. The
refined precommit for any future foundation-model layer-1 swap
needs both:

```
G0a — temporal stability (per-token cosine across rolling windows
      of STATIC scenes): the original G0 test. ~3 min compute.
G0b — content-change stability (per-cell cosine across frames
      where real OBJECT MOTION happens): M2 v2's test. ~3 min
      compute on a known-motion rollout.

Pass requires BOTH. G0a alone is misleading.
```

This is now the locked V0 protocol for any future Layer-1 swap
candidate (e.g., V-JEPA 3, future Cosmos variants, etc.).

## What about V-JEPA?

V-JEPA 2 already failed G0a (per-token cos 0.46-0.52). It would
likely also fail G0b for the same reason (RoPE-bound tokens
change when scene content changes too). But this is moot —
V-JEPA was already redirected from Layer 1 to Layer 3 (clip-
summary retrieval) by V1-G0, and V1b confirmed it loses there
too against privileged geometry.

V-JEPA's clip-summary role for BLA-Forge real-world deployment
(where privileged geometry isn't available) remains the only
open V-JEPA path.

## What this means for the visual-foundation roadmap

```
Layer-1 encoder swap (replacing Phase-14 OF-JEPA encoder):
  V-JEPA 2     ❌ V1-G0 failed
  Cosmos       ❌ V1a-M2 failed (this decision)

Conclusion: no current foundation encoder is a drop-in OF-JEPA
            Layer-1 replacement. The visual-foundation
            integration was the wrong abstraction.

What still has a path forward:
  V-JEPA 2 clip-summary as BLA-Forge retrieval fallback
    (when privileged geometry is unavailable)
  Cosmos-Tokenizer as data-augmentation pipeline
    (synthetic perturbations for OF-JEPA training)
  Both are AUXILIARY uses, not core encoder swaps.

What's dead:
  V1a Cosmos encoder swap (this phase)
  V1b V-JEPA retrieval key (lose to privileged geometry)
  V0's original "encoder swap or data aug" framing — too narrow
```

## Updated next-step priority

```
Highest priority now: BLA-Forge real-world testbed.
  The visual-foundation track has produced 3 negative findings
  on 3 candidates. Continuing the swap track is unlikely to
  produce different results.

  The remaining live path for foundation models in BLA is
  BLA-Forge: noise-aware retrieval where clip-summary features
  may complement noisy estimated geometry. Test there.

Deferred: Cosmos-as-data-augmenter (V2 perturbation suite).
  Still in spec but lower priority than real-world.
```

## Files

- Script v1 (broken protocol): `scripts/phase_v1a_m2_identity_probe.py`
- Script v2 (this decision): `scripts/phase_v1a_m2_v2_motion_aware.py`
- Pod result: `/root/bla/runs/phase_v1a_m2_v2/summary.json`
- Parent G0: `docs/phases/PHASE_V1A_G0_DECISION.md`

## Locked
