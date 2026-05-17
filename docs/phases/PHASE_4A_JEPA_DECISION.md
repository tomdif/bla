# Phase 4A (JEPA Track) — Decision document

**Date:** 2026-05-16.
**Status:** ✅ **PASSED — perceptual-noise stress survived.**

> The slot-delta mechanism survived a transition from the controlled
> state-channel observations of Phase 3 to a noisier image-perception
> channel (Gaussian σ=0.1 on pixels). All 54/54 gates pass with the
> same controls as Phase 3, tightest margin **14.1×**. Worst
> slot_delta cell across the entire perceptual-noise matrix:
> **4.47 hidden MSE**, comfortably under the pre-committed 2× Phase-3
> ceiling of 7.0.

## Setup

**Hardware:** RunPod 8× RTX PRO 6000 Blackwell, torch 2.8 + CUDA 12.8 (same pod as Phase 3).

**What's new vs Phase 3:** `perceptual_noise = 0.1` — Gaussian noise
σ=0.1 applied per-pixel after rendering, sampled from the env's RNG
for reproducibility. Noise is re-masked through the partial-
observability circle so masked-out regions stay exactly zero (an
honest occluder doesn't leak signal). All other stress flags
(moving_distractors, partial_observability) carry over from Phase 3.

**Matrix:**

| Knob | Values |
|---|---|
| seeds | 0, 1, 2, 3, 4 |
| modes | slot_delta, slot_dense_update, dense_jepa_flatten, copy |
| n_targets | 3, 5, 8 |
| n_distractors | 2, 5, 10 |
| K | 5 |
| J | 20, 40, 80 (dropped J=10; less informative under noise) |
| moving_distractors | true |
| partial_observability | true (obs_radius 8) |
| perceptual_noise | **0.1** |
| training steps | 3000 |

Total: 5 seeds × 4 modes × 3 targets × 3 distractors × 1 K = **180 sub-runs**, eval at 3 J values = **540 probe rows**. ~14 min wall.

## Headline numbers

Hidden-frame MSE averaged across all seeds / targets / distractors:

| Mode | J=20 | J=40 | J=80 |
|---|---|---|---|
| **slot_delta** | **3.28** | **3.22** | **3.40** |
| slot_dense_update | 67.95 | 64.99 | 67.55 |
| dense_jepa_flatten | 68.85 | 65.60 | 68.46 |
| copy | 68.72 | 65.70 | 68.55 |

## Phase-4A gates (pre-committed)

1. **`slot_delta` hidden MSE at J=40 ≤ 0.75 × best non-slot-delta baseline.**
   slot_delta J=40 = 3.22; best control J=40 ≈ 65 → 0.75 × 65 = 48.7. **PASS by 15× margin.** ✓
2. **`slot_delta` worst-cell hidden MSE ≤ 2 × Phase 3 worst-cell (= 7.0).**
   Phase 4A worst cell: **4.47**. **PASS.** ✓

Plus the standard per-cell gates from Phase 3 schema, all 54/54 pass.

## What changed under noise (Phase 3 → Phase 4A)

| Slice | Phase 3 | Phase 4A | Δ |
|---|---|---|---|
| slot_delta worst cell | 3.50 | **4.47** | +28% |
| slot_delta mean (J=20/40/80) | 2.68 | **3.30** | +23% |
| tightest gate margin | 20.6× | **14.1×** | -32% |

Adding pixel noise pushed everything up slightly but did **not** flip
the qualitative picture. The bounded-forgetting plateau still holds:
slot_delta hidden MSE is essentially flat across J=20→40→80 (3.28 →
3.22 → 3.40). The mechanism continues to *not* compound error through
occlusion.

## Per-axis behaviour under noise

| Axis | Phase 4A slot_delta hidden MSE |
|---|---|
| Worst cell | nt=5, nd=2, J=80: 4.47 ± 1.77 |
| Best cell | nt=3, nd=10, J=80: 2.52 ± 1.41 |
| Tightest gate | nt=5, nd=2, J=40 vs slot_dense_update: 14.1× margin |

The cluster of harder cells is at **`n_targets=5, n_distractors=2`**
— the configuration with the most targets and the fewest distractors,
which is a slightly counterintuitive pattern. Under noise, fewer
distractors may actually give the slot system *less* gradient signal
to learn entity separation, since distractors function as additional
training signal for the binding mechanism. This is a hypothesis;
needs more data to confirm.

## Interpretation

Most defensible reading:

> Sparse-delta slot memory transfers from clean state-channel
> observations to noisier image-perception observations without
> qualitative degradation. The mechanism preserves entity state under
> moving distractors, partial observability, AND pixel-level
> perceptual noise simultaneously, across 8× longer occlusion than
> the training window.

Carry-forward caveat:

> σ=0.1 Gaussian pixel noise is one fairly mild perceptual-realism
> step. Further-realism tests should include color randomization
> (breaking the channel-as-label shortcut), proper rendered sprites
> with shading, and eventually real video frames. Phase 4B/4C.

## Reproducibility

Artifacts at `artifacts/phase4a_run1/`:

```
manifest.json       full config; records perceptual_noise=0.1 +
                    phase4A_rendered_obs=true
raw_all.jsonl       540 per-(sub-run × J) rows
aggregate_all.csv   180 cells with mean ± stderr ± 95% CI on 5 seeds
gates_all.json      54 gate decisions
```

Run command:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_phase3.py \
    --seeds $seed --targets 3,5,8 --distractors 2,5,10 \
    --K 5 --J 20,40,80 --J-train 10 \
    --modes slot_delta,slot_dense_update,dense_jepa_flatten,copy \
    --moving-distractors --partial-observability --obs-radius 8 \
    --perceptual-noise 0.1 \
    --steps 3000 --probe-episodes 16 --probe-epochs 200 \
    --mask-bias-init 0.0 \
    --out /workspace/phase4a_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```

## Decision

**Phase 4A passes.** Four independent claims now hold:

1. *Phase 2A*: slot mechanism validates on a small static env.
2. *Phase 2B*: slot_delta beats fair patch-level dense JEPA.
3. *Phase 3*: that advantage survives every stress dimension we tested.
4. *Phase 4A*: that advantage survives perceptual noise on the
   observation channel, in addition to the Phase-3 stress dimensions.

The "synthetic and probe-based" caveat narrows: it now refers
specifically to (a) rendered observations are still toy
(patches-on-canvas, no real visual structure), and (b) representation
quality, not behaviour.

## What's next

Two tracks open up, both more meaningful now that perception-channel
robustness is established:

- **Phase 4B — perception realism, stronger.** Randomize entity colors
  (break channel-as-label shortcut) + use small textured sprites. The
  question: does the slot system learn to bind entities by *trajectory
  + interaction* rather than just by channel?
- **Phase 3b — behavioural transfer.** Replay-buffer BC or DAGGER on
  the frozen Phase-4A slot representation. Tests whether the
  representational win translates to action under perceptual noise.

Phase 4B is the higher-information next test for the architecture
itself; Phase 3b is the higher-information next test for whether the
architecture is *useful*. Either is a clean next step.
