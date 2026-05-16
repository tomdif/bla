# Phase 7 (Kubric/MOVi) — Decision document (v1, first pass)

**Date:** 2026-05-16.
**Status:** ⚠️ **PARTIAL — slot_delta dominates positional accuracy by 20-40×; slot_dense_update dominates identity-switch rate. No single mode is best on all six metrics.**

> First decisive Phase 7 test on rendered video (MOVi-A, 256×256→128×128
> resized, 24 frames per episode, ~6.5 instances per scene, full per-instance
> 3D-pose / visibility / appearance ground truth from the Kubric simulator).
> 3 seeds × 4 modes × 3000 training steps × 40-episode held-out probe set.

## Headline numbers (mean ± std across 3 seeds)

| Mode | vis_pos_mse | hidden_pos_mse | switch_rate ↓ | diversity ↓ | h/v ratio |
|---|---|---|---|---|---|
| **slot_delta** | **1.0e-5 ± 5e-6** | **1.6e-5 ± 4e-6** | 0.705 ± 0.063 | 5.34 ± 0.18 | 1.69 ± 0.37 |
| slot_dense_update | 4.4e-4 ± 1.3e-4 | 3.7e-4 ± 1.1e-4 | **0.376 ± 0.068** | **4.45 ± 0.44** | 0.85 ± 0.07 |
| dense_jepa | 2.1e-4 ± 1.0e-5 | 2.3e-4 ± 5.6e-5 | 0.467 ± 0.048 | 4.79 ± 0.33 | 1.10 ± 0.32 |
| copy | 1.9e-3 ± 5e-4 | 1.9e-3 ± 4e-4 | 0.155 ± 0.013 | 3.00 ± 0.13 | 1.02 ± 0.10 |

Chance switch rate at n_slots=12 is **0.917**; perfect stability is **0**.

## The copy-baseline lesson — switch_rate alone is gameable

The **copy** baseline (no training, random-init ConvNeXt + SlotAttention,
identity predictor) achieves the **lowest** switch rate (0.155) and
**lowest** slot diversity (3.0). Why? With no training, the slot states
barely change across frames — they're essentially constant — so the
Hungarian assignment per frame returns nearly the same mapping every
time. **A useless constant-slot model wins on switch rate.**

This forces the result to be read **jointly** (mse AND switch), not on
either axis alone:

- **copy** wins switch_rate but loses position MSE by 200×.
- **slot_delta** wins position MSE by 20-40× but has the worst trained-
  model switch rate.
- **slot_dense_update** sits between them — middling position MSE, best
  trained-model switch rate.
- **dense_jepa** is the middle child on both.

## The slot_delta vs slot_dense_update tradeoff

These two modes share the **same** ConvNeXt + SlotAttention encoder.
They differ only in the predictor's update rule:

- `slot_delta`: `next = slot + change_mask · tanh(delta)` — sparse,
  identity-preserving by default.
- `slot_dense_update`: `next = raw_delta` — full replacement each step.

The JEPA loss backpropagates from the predictor to the encoder. What
each mode teaches the encoder is different:

- **slot_delta** rewards an encoder that produces slot states whose
  *spatial structure* is precise per-frame, because the predictor
  mostly preserves slots from t to t+1 — so the encoder must encode
  position information directly into slot identity. → **positional
  accuracy wins**.
- **slot_dense_update** rewards an encoder whose slot states are
  *temporally smooth* across frames, because the predictor has to
  fully reconstruct slots[t+1] from slots[t]. The encoder + slot
  attention must produce stable, predictable patterns. → **identity
  stability wins**.

This is a real result: the slot mechanism is a **tradeoff**, not a
single-axis win. The user-stated v0 thesis "slot_delta dominates" is
**falsified at this scale and config** for the joint (position, identity)
objective on MOVi-A.

## Methodology notes (worth committing to memory)

1. **First sweep diverged on 2/3 seeds at LR=3e-4 with SIGReg=0.01.**
   slot_delta's additive recurrence (`slot += delta`), combined with
   slot persistence across 24 frames, caused exponential parameter
   growth on seeds 0 and 2 (loss went 92k → 3M → 19M → 71M → 196M →
   456M across 1500 steps). Seed 1 was stable. Same code, same config.
   **Root cause:** LayerNorm-free slot persistence + SIGReg's
   inter-cluster repulsion creating an unbounded direction in slot
   space. **Fix:** LayerNorm the slot states between frames + drop
   SIGReg + halve LR. Sweep v2 was stable on all 3 seeds.

   This is a [[calibrate-before-hash]]-flavored lesson — never commit
   to thresholds on a brand-new training recipe until the seed-stability
   of the recipe itself has been measured.

2. **JEPA loss on MOVi-A is suspiciously easy.** Final losses landed
   at 0.0001-0.0008 across all trained modes. MOVi-A's 24-frame clips
   show very small frame-to-frame entity motion under gravity, and
   LayerNormed adjacent slot states are L2-close. The encoder gets
   away with weak dynamics modeling because next-frame prediction is
   nearly identity. Future Phase 7+ work should either (a) use larger
   inter-frame strides (predict t+4 instead of t+1) or (b) extend
   training and look at *long-horizon* prediction.

3. **Probe overfits with only 32 eval episodes.** Position MSE values
   in the 1e-5 to 1e-3 range are below the noise floor of what the
   probe can robustly estimate from 32 episodes × 24 frames. The
   ordering across modes is stable across seeds (slot_delta < dense_jepa
   < slot_dense_update on position MSE; reversed on switch rate), but
   absolute magnitudes shouldn't be over-interpreted at this data
   scale. Phase 7.5 (CLEVRER) or a larger Phase 7.1 (MOVi train shards)
   will improve confidence.

## What this run does and doesn't establish

**Establishes:**
- The full Phase 7 pipeline works end-to-end on rendered video:
  TFDS streaming + .npz cache + ConvNeXt-T encoder + persistent
  SlotAttention + JEPA training + identity-aware Hungarian probe with
  position-and-attribute matching + identity-switch metric.
- The identity-aware Hungarian probe is a real signal: it
  discriminates between modes consistently across 3 seeds (slot_delta
  is always tightest on position MSE; copy is always lowest on
  switch rate; slot_dense_update is always best trained-model switch).
- A degenerate "constant slot" baseline gets switch_rate ≈ 0.15 — so
  any future reporting that hails "low switch rate" without a
  position-MSE accompaniment is **wrong**.

**Does NOT establish:**
- That slot_delta dominates as a world-model architecture. It doesn't.
- That the slot mechanism beats the dense baseline on every metric.
  It doesn't.
- A strong external-benchmark number — single-dataset, small data
  regime, no published baseline comparison.
- That the JEPA objective is sufficient on its own — the trivially-low
  training loss suggests the objective is too easy at this clip length
  and inter-frame stride.

## Decision

**Phase 7 v1: partial pass.** The slot mechanism *survives* the move to
rendered video. The slot encoder produces meaningfully-different
representations than the dense baseline (~20× position MSE gap). But
the v0 mental picture — "sparse-delta is the load-bearing mechanism" —
is at best one half of the story. **slot_dense_update wins on the
identity-stability axis.**

For the published narrative, the right framing is:

> "Slot-update architecture is a design choice that *trades off* spatial
> accuracy against identity stability. slot_delta optimizes spatial
> accuracy; slot_dense_update optimizes identity stability. Which is
> right depends on what you want the world model to do."

For Phase 7.1+ (next steps, in priority order):
- **Larger inter-frame stride** (predict t+4 not t+1) to make the JEPA
  loss non-trivial.
- **More training data** (pull MOVi-A train shards or run on MOVi-C
  which has real backgrounds).
- **CLEVRER external benchmark** (Phase 7.5, tasks 73-78 in the queue
  — still gated on Google Drive scene-annotation download).
- **Architectural exploration**: hybrid slot_delta + identity loss
  (push the encoder to produce slots that are BOTH spatially-tight AND
  temporally-stable).

## Reproducibility

Code committed at `dfb1415` (scaffolding) and `188647f` (training
orchestrator). Artifacts at `artifacts/phase7_run2/seed_{0,1,2}/`.

Run command:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_a_local/validation \
    --modes slot_delta,slot_dense_update,dense_jepa,copy \
    --seeds $seed --epochs 10 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --out /workspace/phase7_run2/seed_$seed > seed_$seed.log 2>&1 &
done
```

## Updated claim stack

| Phase | Status | Headline |
|---|---|---|
| 2A-B | ✅ | slot mechanism + ablation (within-episode) |
| 3 | ✅ | stress matrix passes |
| 4A | ✅ | pixel noise survives |
| 4B | ✅ | appearance randomization (+5-12% via Hungarian) |
| 5A | ✅ | within-episode capacity curve |
| 5B | ✅ | dynamic-64 = fixed-64 at parity, -60-75% updates |
| 5C | ✅ | Hungarian methodology fix |
| 5D | ✅ | Phase-4B restated under Hungarian |
| 5E | ✅ | slot-count sweet spot 8-24 |
| 6 (light) | ✅ | sub-pixel rendering preserves +5-12% margin |
| **7 (MOVi-A v1)** | ⚠️ | **slot mechanism survives rendered video, but slot_delta vs slot_dense_update is a TRADEOFF, not dominance** |

This honest stack is the correct one to take into BLA Phase 6
integration. Adjust the BLA architecture-level claims accordingly:
the slot mechanism gives a *family* of tradeoffs, and the right pick
depends on the downstream task.
