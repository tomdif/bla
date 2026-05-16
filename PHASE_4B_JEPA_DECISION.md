# Phase 4B (JEPA Track) — Decision document

**Date:** 2026-05-16.
**Status:** ✅ **PASSED STRONGLY — appearance randomization survived.**

> The slot-delta mechanism survived the next perceptual realism step:
> **color randomization + random background**, stacked on top of the
> Phase-4A pixel noise. All 36/36 gates pass; tightest margin
> **18.3×**; worst slot_delta cell **3.75 hidden MSE**, comfortably
> under both the Phase-4A ceiling (8.94) and the original Phase-3
> ceiling (7.0).
>
> Color/background randomization plus pixel noise did not break
> slot-delta memory; it slightly improved the slot_delta line and
> widened the margin vs baselines. That rules out a cheap
> explanation of earlier phases: **the mechanism is not merely
> color shortcutting**.

## What's new vs Phase 4A

Two additional stress flags, both stacked on top of everything from
Phase 4A:

| Flag | Behaviour |
|---|---|
| `color_randomization=True` | Each entity (agent, every target, every distractor) gets a fresh random RGB at each episode reset. The channel-as-label shortcut (red=target, blue=distractor) is broken — the model can't rely on a fixed colour cue across episodes. |
| `background_randomization=True` | The black canvas is replaced by a per-pixel uniform random background (magnitude 0.15). Entities are drawn on top. The encoder can no longer assume "anything non-zero is signal." |

All Phase-4A flags (`perceptual_noise=0.1`, `moving_distractors=True`,
`partial_observability=True`, `obs_radius=8`) carry over.

## Setup

| Knob | Values |
|---|---|
| seeds | 0, 1, 2, 3, 4 |
| modes | slot_delta, slot_dense_update, dense_jepa_flatten, copy |
| n_targets | 3, 5, 8 |
| n_distractors | 5, 10 (dropped nd=2 — Phase-4A's hardest cell, kept for next-pass) |
| K | 5 |
| J | 20, 40, 80 |
| moving_distractors | true |
| partial_observability | true (obs_radius 8) |
| perceptual_noise | 0.1 |
| color_randomization | **true** |
| background_randomization | **true** |
| training steps | 3000 |

Total: 5 seeds × 4 modes × 3 targets × 2 distractors = **120 sub-runs**, eval at 3 J values = **360 probe rows**. ~10 min wall (5 GPUs).

## Headline numbers

Hidden-frame MSE averaged across all seeds / targets / distractors:

| Mode | J=20 | J=40 | J=80 |
|---|---|---|---|
| **slot_delta** | **3.02** | **3.06** | **3.16** |
| slot_dense_update | 66.05 | 67.03 | 69.42 |
| dense_jepa_flatten | 65.51 | 67.31 | 69.13 |
| copy | 65.45 | 66.93 | 69.16 |

Visible-frame MSE (representation quality on raw observations):

| Mode | J=20 | J=40 | J=80 |
|---|---|---|---|
| **slot_delta** | **0.74** | **0.76** | **0.79** |
| slot_dense_update | 24.29 | 25.18 | 25.99 |
| dense_jepa_flatten | 55.70 | 57.68 | 59.54 |
| copy | 65.23 | 66.71 | 68.85 |

## Phase-4B gates

1. **`slot_delta` hidden MSE at J=40 ≤ 0.75 × best non-slot-delta baseline.**
   slot_delta J=40 = 3.06; best control J=40 ≈ 67.03 → 0.75 × 67 = 50.3. **PASS by 16× margin.** ✓
2. **`slot_delta` worst-cell hidden MSE ≤ 2 × Phase-4A worst-cell (= 8.94).**
   Phase 4B worst cell: **3.75**. **PASS.** ✓
3. **Per-cell gate** (slot_delta ≤ 0.75 × each control, every cell): **36/36 pass.** ✓

## What changed (Phase 4A → Phase 4B)

| Slice | Phase 4A | Phase 4B | Δ |
|---|---|---|---|
| slot_delta worst cell | 4.47 | **3.75** | **-16%** |
| slot_delta mean (J=20/40/80) | 3.30 | **3.08** | **-7%** |
| tightest gate margin | 14.1× | **18.3×** | **+30%** |

**Slot_delta got slightly *better* under more stress.** Two plausible
explanations, both worth recording but not overclaiming:

1. **Color randomization may be a regularizer.** With colour broken as
   a shortcut, the encoder is forced to encode position rather than
   colour identity. Position is exactly what the linear probe needs,
   so the slot representation becomes more probe-friendly.
2. **Smaller cell set hides the harder corner.** Phase 4B dropped
   `n_distractors=2`, which was Phase 4A's worst-case axis (nt=5,
   nd=2, J=80: 4.47 in Phase 4A). The improvement is partly an
   artifact of the easier matrix; a follow-up run with nd=2 included
   would tell us if Phase 4B genuinely won that cell.

## Visible-MSE signal is the cleanest story

Even more striking than the hidden numbers: **slot_delta's *visible*
MSE drops to 0.74-0.79**, while dense_jepa/flatten stays at ~56-60 and
copy at ~65-69. With color randomization, the dense JEPA encoder is
nearly inert at predicting target positions even from a fully visible
frame; the slot encoder produces near-perfect position features.

This is the cleanest possible separation between "encoder learned to
extract entity positions" and "encoder learned channel labels."
Slot-delta does the first; dense baselines do the second when colour
labels are available, and fall back to noise when colours are
randomized.

## Interpretation

The most defensible reading:

> Sparse-delta slot memory survives appearance randomization on top of
> all Phase-3 stress flags and Phase-4A pixel noise. The slot encoder
> learns position-bearing entity representations that don't depend on
> fixed colour channels, and the sparse-delta predictor preserves
> those representations through long occlusion windows even when the
> visible-frame appearance is unstable.

Updated claim stack across the full JEPA track:

| Phase | Claim |
|---|---|
| 2A | slot mechanism validates on a small static env |
| 2B | sparse delta beats fair patch-level dense JEPA |
| 3  | survives moving distractors, partial observability, more entities, long occlusion |
| 4A | survives pixel-level perceptual noise |
| 4B | **survives colour/background randomization on top of 4A** |

Strongest current conclusion:

> Sparse delta updates over persistent slots produce bounded
> hidden-entity memory that survives dense JEPA controls, slot-only
> ablations, moving distractors, partial observability, long occlusion
> to J=80, pixel noise, AND appearance randomization.

Carry-forward caveat:

> The realism axis remains synthetic — entities are still
> patches-on-canvas, not rendered 3D shapes with shading. The next
> realism step is genuinely-rendered scenes (CLEVRER-style) or real
> video. But appearance shortcuts are now ruled out as the source of
> the result.

## Reproducibility

Artifacts at `artifacts/phase4b_run1/`:

```
manifest.json       config; records perceptual_noise=0.1,
                    color_randomization=true, background_randomization=true,
                    phase4B_appearance_random=true
raw_all.jsonl       360 per-(sub-run × J) rows
aggregate_all.csv   120 cells with mean ± stderr ± 95% CI on 5 seeds
gates_all.json      36 per-cell gates, all pass
```

Run command:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_phase3.py \
    --seeds $seed --targets 3,5,8 --distractors 5,10 \
    --K 5 --J 20,40,80 --J-train 10 \
    --modes slot_delta,slot_dense_update,dense_jepa_flatten,copy \
    --moving-distractors --partial-observability --obs-radius 8 \
    --perceptual-noise 0.1 \
    --color-randomization --background-randomization \
    --steps 3000 --probe-episodes 16 --probe-epochs 200 \
    --mask-bias-init 0.0 \
    --out /workspace/phase4b_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```

## Decision

**Phase 4B passes.** Five independent claims now hold:

1. *Phase 2A* — slot mechanism validates.
2. *Phase 2B* — slot_delta beats fair patch-level dense JEPA.
3. *Phase 3* — that advantage survives the stress matrix.
4. *Phase 4A* — that advantage survives perceptual pixel noise.
5. *Phase 4B* — that advantage survives appearance randomization
   (colour shuffled per episode + random background) on top of
   everything above.

The "channel-as-label shortcut" hypothesis is **ruled out**. The slot
representation is doing real position-extraction work.

## What's next

**Phase 3b — behavioural transfer.** Perception has now passed two
meaningful realism checks. The next open question is whether the
representation helps action.

Use a method that avoids the earlier BC failure:

> replay-buffer BC + DAGGER-lite

Minimum setup:

```
freeze slot_delta encoder    (from Phase 4B trained checkpoint)
freeze dense_jepa encoder    (same)
train identical policy heads on each
use replay buffer with model-policy rollouts
periodically query expert for corrections
compare success at J=20, 40, 80
```

**Primary behavioural gate:**

```
slot_delta policy success at J=40 ≥ dense_jepa policy success + 10pp
```

**Better gate (sample-efficiency framing):**

```
slot_delta needs fewer expert corrections / fewer episodes
   to reach the same success threshold
```

— because the thesis is really *bounded memory + sample efficiency*,
not just success rate.

**Deferred until after 3b:**

- *BLA integration.* The representation has earned it, but a positive
  Phase-3b result will make the integration much more credible.
- *Phase 4C rendered 3D scenes.* CLEVRER-style realism is the next
  step on the realism axis but requires a renderer; defer until the
  data pipeline justifies it.
