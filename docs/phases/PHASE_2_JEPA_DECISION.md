# Phase 2 (JEPA Track) — Decision document

**Date:** 2026-05-16.
**Status:** ✅ **PASSED — core mechanism validated.**

> **Persistent slots only become useful world-state memory when updates
> are sparse and delta-based.**

The ablation is the centerpiece of this phase. Slot binding alone does
not produce the memory effect: it produces an inert "slotified token soup"
that performs the same as a flat patch-level dense JEPA. Sparse delta
updates over those slots are what turn the representation into a
persistent ledger that survives occlusion.

## Scope

Phase 2 of the JEPA / world-model track of B.L.A. tests the load-bearing
claim of the KAM-JEPA design: **persistent slots + sparse causal delta
prediction should preserve entity state under occlusion better than
dense latent JEPA**. The earlier phase was Phase 0 / 1 (the substrate
fix + spatiotemporal-JEPA SIGReg bug — see `docs/01_jepa.md`).

The Phase 2 design was deliberately minimal:

- Untyped 16-slot binding (no affordance heads, no typed commitments).
- Soft `change_mask = sigmoid(logits)`; bounded `delta = δ·tanh(Δ)`.
- L1 sparsity penalty on the mask + Bernoulli-entropy bimodality penalty.
- One synthetic env (multi-target navigate with visible/hidden cycles).
- One probe-based metric (representational, not behavioural).

## Setup

Occluded multi-target navigate env:
```
image_size=32, patch_size=4, n_targets=3, n_distractors=2
visible_steps K=5, hidden_steps J∈{5,10,20,40}
episode_length 24, batch_size 4, 3000 self-supervised steps
```

Hyperparameters selected on a CPU sweep:
```
sparsity_weight   = 5e-3
bimodal_weight    = 1e-3
mask_bias_init    = -2.0
delta_scale       = 0.1
n_slots           = 16
slot_iters        = 3
```

Modes compared:
| Mode | Encoder training signal | State for probe |
|---|---|---|
| `slot_delta` | next-slot prediction + L1 sparsity + bimodal entropy | flat(slots) ∈ ℝ^{16·64} |
| `dense_jepa/flatten` | patch-masked JEPA (BLAJEPAModel) + SIGReg | flat(patches) ∈ ℝ^{64·64} |
| `dense_jepa/mean` | (same encoder as above) | mean(patches) ∈ ℝ^{64} |
| `dense` (legacy) | next pooled-embedding prediction | pooled ∈ ℝ^{64} |
| `copy` | same-frame consistency, no predictor | pooled ∈ ℝ^{64} |
| `slot_dense_update` (ablation) | next-slot prediction (no mask, no bounded delta) | flat(slots) |

Probe: single `nn.Linear(state_dim → 2·n_targets)`, trained on visible
frames only, evaluated on hidden frames. No recurrence, no visibility
flag.

## Results

### Hidden-frame MSE (lower = better representational memory)

| Mode | J=5 | J=10 | J=20 | J=40 |
|---|---|---|---|---|
| **slot_delta** | **0.92** | **2.11** | **3.87** | **4.50** |
| slot_dense_update *(ablation)* | 61.31 | 74.10 | 68.01 | 72.79 |
| dense_jepa/flatten *(fair baseline)* | 55.24 | 67.62 | 62.23 | 66.59 |
| dense_jepa/mean | 60.57 | 73.31 | 66.87 | 71.35 |
| dense (legacy) | 60.56 | 73.30 | 66.87 | 71.35 |
| copy | 60.57 | 73.31 | 66.87 | 71.35 |

### Visible-frame MSE (representation quality on raw observations)

| Mode | J=5 | J=10 | J=20 | J=40 |
|---|---|---|---|---|
| **slot_delta** | **0.98** | **1.45** | **1.35** | **1.63** |
| slot_dense_update | 48.98 | 53.85 | 28.01 | 29.08 |
| dense_jepa/flatten | 46.46 | 54.15 | 49.95 | 54.77 |
| dense_jepa/mean | 61.00 | 73.09 | 66.18 | 71.33 |

### Degradation slope (J=5 → J=40, hidden MSE)

| Mode | Δ |
|---|---|
| slot_delta | +3.58 |
| slot_dense_update | +11.48 |
| dense_jepa/flatten | +11.35 |

### Slot-side phase 1 metrics (slot_delta, end of training)

```
mask_mean        = 0.18  (≤ 0.30 target)
mask_max         = 0.43
slot stability   = 0.995 cosine across hidden→visible boundaries
slot↔target ID   = 0.91  (≥ 0.90 target)
prediction loss  = 0.014 (3× drop from initial 0.045)
```

## Gates

1. **Baseline-must-be-real:** `dense_jepa/flatten` visible MSE meaningfully better than naive `dense`. **46.5 vs 61.0 at J=5 → 24% improvement. ✓**
2. **`slot_delta` hidden MSE at J=20 ≤ 0.75 × dense_jepa/flatten hidden MSE.** **3.87 vs 0.75×62.23=46.7 → 12× margin. ✓**
3. **`slot_delta` degradation slope ≤ 0.75 × dense_jepa/flatten slope.** **3.58 vs 0.75×11.35=8.5 → 31% of dense slope. ✓**
4. **Ablation: `slot_delta` beats `slot_dense_update`.** **3.87 vs 68.01 at J=20 → 17.6× better. ✓**

All four gates pass.

## Core finding

**Sparse causal delta prediction is the load-bearing mechanism.**

At J=20, slot_delta achieves:
- 17.6× lower hidden MSE than `slot_dense_update`
- 16.1× lower hidden MSE than `dense_jepa/flatten`

The hierarchy is clean:

```
slot binding alone:    insufficient   (slot_dense_update ≈ dense_jepa)
dense slot updating:   unstable / forgetful
sparse delta updating: load-bearing
```

## Interpretation

Dense updates overwrite the ledger. Sparse deltas preserve it.

This is the commitment-memory idea in minimal empirical form:
- maintain stable commitments by default,
- update only the slots the action causally affects,
- avoid unnecessary drift through hidden windows.

Two specific corollaries:

1. **Slot binding alone is not the mechanism.** The `slot_dense_update`
   ablation (same slot attention, same encoder, same predictor architecture,
   but the output replaces slots instead of adding a sparse delta) sits at
   roughly the same hidden-MSE as the dense patch-level baseline (68 vs 62
   at J=20). The *combination* of slot binding + sparse delta is what
   produces the 16-17× improvement over both baselines.

2. **Forgetting is bounded.** slot_delta hidden MSE goes 0.92 → 2.11 → 3.87
   → 4.50 across J=5 → 40. It plateaus rather than compounding. That is
   the signature of a real persistent-memory mechanism: state stays
   approximately correct through long occlusion windows instead of drifting
   unboundedly.

Caveat: the result is synthetic and probe-based, not behavioural. A
separate phase-2 attempt at behavioural BC was abandoned because the dense
baseline failed at J=0 (no occlusion) — a policy-distribution-shift
failure orthogonal to the representational question. The probe metric is
the cleaner test of the actual hypothesis at this stage.

## What we built (durable artifacts)

- `system1_jepa/slot.py` — Slot Attention with learned per-slot priors,
  GRU refinement, K iterations.
- `system1_jepa/slot_predictor.py` — `SlotDeltaPredictor` with sparse
  change_mask + bounded delta, soft-mask training, hard-mask diagnostic;
  `update_mode ∈ {delta, dense}` for ablations.
- `system1_jepa/navigate_occlusion.py` — `OccludedMultiTargetNavigateEnv`
  with K-visible/J-hidden cycles + colour-channel distractors.
- `scripts/slot_jepa_train.py` — single-script trainer with five modes,
  CPU-fast, linear-probe eval phase built in.
- `scripts/slot_jepa_compare.sh` — reproduces the table above end-to-end.
- `tests/test_slot_jepa.py` — 8 tests covering slot binding, bounded
  delta, sparsity gradient, env occlusion, end-to-end backprop.
- Spatiotemporal-JEPA bug fix (Phase-2 prep): SIGReg added to
  `SpatiotemporalJEPA.training_loss` (was missing in the previous
  audit; see Phase-2 audit notes).

## What's next (Phase 3 — pod stress test)

The Phase 2 result is small-scale and synthetic. The pod run should
center on the ablation that just produced the decisive finding:

**Comparison matrix:**

```
slot_delta
slot_dense_update
dense_jepa/flatten
copy
```

**Stress dimensions:**

| Knob | Phase 2 (local) | Phase 3 (pod) |
|---|---|---|
| n_targets | 3 | 3, 5, 8 |
| n_distractors | 2 (static) | 2, 5, 10, **moving** |
| J | 5, 10, 20, 40 | 10, 20, 40, **80** |
| K | 5 | 3, 5, 10 |
| seeds | 1 | ≥ 5 |
| observation | rendered patches | + **partial observability + real / rendered scenes** |

Primary metrics carry forward (hidden/visible MSE, ratio, slope).
Secondary diagnostics to add: slot collapse rate, slot swapping rate,
target assignment consistency, per-target MSE, distractor interference.

The headline survives only if **slot_delta still beats slot_dense_update
under the harder envs**. If it does, the broader architectural claim is
durable:

> KAM-JEPA's memory advantage comes from sparse delta updates over
> persistent slots, not slots alone.

If the result survives, the slot-delta module ships as the **memory /
persistent-state layer** of the broader BLA stack. The behavioural-BC
question that was inconclusive in Phase 2 becomes a separate Phase-3b
task: replay-buffer BC or DAGGER on top of the validated state
representation.

## Decision

**Lock Phase 2 as passed.** Proceed to pod scale-up with the ablation
matrix above.
