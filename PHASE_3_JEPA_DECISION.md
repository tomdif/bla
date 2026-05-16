# Phase 3 (JEPA Track) — Decision document

**Date:** 2026-05-16.
**Status:** ✅ **PASSED STRONGLY — stress-robust world-state memory.**

> Phase 3 did not merely reproduce Phase 2 under scale. It showed
> **bounded forgetting across stress axes**. Slot-delta hidden-state
> error remained low and nearly flat across target count, distractor
> count, and occlusion length. The worst slot-delta cell across the
> full stress matrix was **3.50 hidden MSE**, while every gate passed
> with at least a **20.6× margin**.

## Scope

Phase 2 validated the slot-delta mechanism on a small synthetic env
(`PHASE_2_JEPA_DECISION.md`). Phase 3 asked the falsifiable question:
does that result survive on harder versions of the same env?

The pod sweep is the answer.

## Setup

**Hardware:** RunPod 8× RTX PRO 6000 Blackwell (96 GB each), torch 2.8 + CUDA 12.8.

**Matrix:**

| Knob | Values |
|---|---|
| seeds | 0, 1, 2, 3, 4 |
| modes | slot_delta, slot_dense_update, dense_jepa_flatten, copy |
| n_targets | 3, 5, 8 |
| n_distractors | 2, 5, 10 |
| K (visible window) | 5 |
| J (hidden window, eval) | 10, 20, 40, 80 |
| moving_distractors | true |
| partial_observability | true (obs_radius 8) |
| training steps | 3000 |

Total: 5 seeds × 4 modes × 3 targets × 3 distractors × 1 K = **180 sub-runs**, eval at 4 J values = **720 probe rows**.

The orchestrator ran 5 seeds in parallel, one per GPU, ~14 min wall time.

**Hardware-specific config knob:** `mask_bias_init` had to change from
Phase-2's `-2.0` (local CPU, torch 2.4) to `0.0` here (Blackwell, torch
2.8) to avoid mask=0 collapse. The manifest records both — see
"Reproducibility" below.

## Headline numbers

Hidden-frame MSE averaged across all seeds / targets / distractors:

| Mode | J=10 | J=20 | J=40 | J=80 |
|---|---|---|---|---|
| **slot_delta** | **0.71** | **2.72** | **2.61** | **2.72** |
| slot_dense_update | 62.33 | 67.38 | 66.23 | 69.08 |
| dense_jepa_flatten | 64.75 | 66.96 | 65.75 | 67.94 |
| copy | 65.74 | 67.13 | 65.76 | 68.01 |

## Gates

Per-cell gate: `hidden_MSE[slot_delta] ≤ 0.75 × hidden_MSE[control]`
for both controls (`slot_dense_update`, `dense_jepa_flatten`).

```
total gates: 72
passed:      72
pass rate:   100.0 %
```

Tightest cell (smallest slot_delta margin over a control):

```
nt=3 nd=2 J=80 vs dense_jepa_flatten:   20.6x margin   PASSED
nt=3 nd=2 J=80 vs slot_dense_update:    20.7x margin   PASSED
```

## Stress-axis breakdown (slot_delta hidden MSE, averaged across other axes)

| Axis | Values | Result |
|---|---|---|
| n_targets | 3 → 5 → 8 | 2.52 → 2.07 → 1.97 |
| n_distractors | 2 → 5 → 10 | 2.06 → 2.31 → 2.19 |
| J | 10 → 20 → 40 → 80 | 0.71 → 2.72 → 2.61 → 2.72 |

Three things to note (phrased carefully so we don't overclaim):

1. **Forgetting is bounded.** Hidden MSE jumps from J=10 to J=20 (0.71
   → 2.72) and then plateaus all the way out to J=80. The mechanism
   pays a one-time cost going from short to medium occlusion, then
   stops compounding. Dense / delta-less baselines did not have this
   property in Phase 2.
2. **Increasing target count did not degrade slot-delta memory.**
   Performance slightly *improved* with more targets, likely because
   the linear probe had richer and more regular slot-position signal.
   We are *not* claiming "more entities always helps" — only that the
   tested range (3 → 8) did not break the mechanism.
3. **Slot-delta memory remains robust under increased distractor
   pressure**, including moving distractors and partial observability.
   2 vs 10 distractors moves the number by 0.13 — within seed-to-seed
   variation. Distractor identity-swap was a plausible failure mode for
   slot systems; it does not appear to break this configuration.

## Worst cells (top 5 hardest configurations for slot_delta)

| n_targets | n_distractors | J | hidden MSE ± CI95 | hidden/visible ratio |
|---|---|---|---|---|
| 3 | 2 | 80 | 3.50 ± 1.54 | 3.12 |
| 3 | 10 | 80 | 3.23 ± 1.02 | 3.12 |
| 3 | 2 | 20 | 3.18 ± 0.68 | 3.34 |
| 3 | 2 | 40 | 3.10 ± 1.11 | 3.34 |
| 8 | 5 | 20 | 3.07 ± 1.38 | 3.75 |

Even the worst case (3.50) is **~20× better** than the corresponding
baseline cell (~70). The slot mechanism does not have a known
breakdown axis in the tested matrix.

## Interpretation

Phase 2 was: "sparse delta is load-bearing in the easy version."
Phase 3 is: "sparse delta is load-bearing in the hard version, the
moving-distractor version, the partial-observability version, the
many-targets version, and the long-occlusion version, simultaneously."

The most defensible conclusion:

> Sparse delta updates over persistent slots produce a robust
> hidden-entity memory mechanism in this controlled world-model
> setting. The mechanism survives increased entity count, distractor
> pressure, moving distractors, partial observability, and long
> occlusion intervals up to J=80.

The caveat we have to carry forward:

> The result is still synthetic and probe-based. Phase 4 must test
> whether the same mechanism survives rendered image observations and
> then real video.

What Phase 3 **doesn't** show: behavioural transfer (policy success
rate under occlusion). That was inconclusive in Phase 2 because of
standard BC distribution-shift problems. A separate Phase-3b track
should test this using a stronger control regime (replay-buffer BC or
DAGGER on top of the frozen representation). The *representation*
claim is now solid; the *behavioural* claim is still open.

## Reproducibility

Artifacts at `artifacts/phase3_run1/`:

```
manifest.json       full config; records both phase2_reference (CPU/torch 2.4,
                    mask_bias_init=-2.0) and phase3_pod_default (GPU/torch 2.8,
                    mask_bias_init=0.0), with git_commit + timestamp + command
raw_all.jsonl       720 per-(sub-run × J) rows
aggregate_all.csv   180 per-(mode, K, n_targets, n_distractors, J) rows
                    with mean ± stderr ± 95% CI on 5 seeds
gates_all.json      72 gate decisions with margins
```

The manifest names the hardware-specific mask_bias_init explicitly, so
future runs can't silently mix the local-CPU Phase-2 setting with the
pod Phase-3 setting.

Run command:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_phase3.py \
    --seeds $seed --targets 3,5,8 --distractors 2,5,10 \
    --K 5 --J 10,20,40,80 --J-train 10 \
    --modes slot_delta,slot_dense_update,dense_jepa_flatten,copy \
    --moving-distractors --partial-observability --obs-radius 8 \
    --steps 3000 --probe-episodes 16 --probe-epochs 200 \
    --mask-bias-init 0.0 \
    --out /workspace/phase3_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```

## Decision

**Phase 3 passes.** Three independent claims now hold:

1. *Phase 2A*: slot mechanism validates on a small static env.
2. *Phase 2B*: slot_delta beats fair patch-level dense JEPA.
3. *Phase 3*: that advantage survives every stress dimension we tested,
   at a minimum margin of 20.6× across 72 gates.

## What's next

Three tracks open up; they can be pursued in parallel since they don't
share infrastructure:

- **Phase 3b — behavioural transfer.** Wire slot_delta into a policy
  head with proper BC (replay buffer or DAGGER) and measure success
  rate under occlusion. Tests whether the representational win
  translates to behavioural success.
- **Phase 4 — perceptual realism.** Replace the toy patch renderer
  with rendered scene images (CLEVRER-style or similar), then real
  video. The mechanism *should* survive given that it works on
  patch-level features already; this checks whether real perceptual
  noise / lighting / occluders break it.
- **Integration into BLA.** The slot-delta module is ready to slot in
  as the persistent-state / working-memory layer of the broader BLA
  stack. The procedural-core / hybrid-memory / verifier pieces from
  earlier phases already exist; Phase 3 gives them an actual memory
  substrate with bounded forgetting.

The Phase 3 result removes the "unvalidated mechanism" caveat from the
broader architecture.
