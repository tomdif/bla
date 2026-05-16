# Phase 3 (JEPA Track) — Decision document

**Date:** 2026-05-16.
**Status:** ✅ **PASSED — stress-robust world-state memory.**

> The slot-delta mechanism survives every stress dimension we threw at
> it: 8× more entities, 5× more distractors, moving distractors,
> partial observability, and 8× longer occlusion. Across **72 gates ×
> 5 seeds × 36 stress cells**, the advantage over the strongest fair
> baseline never drops below **20.6×**.

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

Three things to note:

1. **Forgetting is bounded.** Hidden MSE jumps from J=10 to J=20 (0.71
   → 2.72) and then plateaus all the way out to J=80. The mechanism
   isn't compounding error through occlusion — it's preserving state.
2. **More entities don't hurt.** Going from 3 to 8 targets actually
   slightly *lowers* mean MSE — likely because more entities give the
   linear probe a richer per-frame signal.
3. **Distractor count is flat noise.** 2 vs 10 distractors moves the
   number by 0.13 — within seed-to-seed variation.

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

The result graduates from *Phase 2 mechanism validated* to **Phase 3
stress-robust world-state memory**. The KAM-JEPA core claim survives
contact with realistic stress.

What this **doesn't** show: behavioural transfer (policy success rate
under occlusion). That was inconclusive in Phase 2 because of standard
BC distribution-shift problems. A separate Phase-3b track should test
this using a stronger control regime (replay-buffer BC or DAGGER on
top of the frozen representation) — but the *representation* claim is
now solid.

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

Two parallel tracks open up:

- **Phase 3b — behavioural transfer.** Wire slot_delta into a policy
  head with proper BC (replay buffer or DAGGER) and measure success
  rate under occlusion. Tests whether the representational win
  translates to behavioural success.
- **Integration into BLA.** The slot-delta module is ready to slot in
  as the persistent-state / working-memory layer of the broader BLA
  stack. The procedural-core / hybrid-memory / verifier pieces from
  earlier phases already exist; this gives them an actual memory
  substrate with bounded forgetting.

Either is a natural next step. The Phase 3 result removes the
"unvalidated mechanism" caveat from the broader architecture.
