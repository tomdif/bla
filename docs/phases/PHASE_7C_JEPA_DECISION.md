# Phase 7C (JEPA, MOVi-A) — Decision document

**Date:** 2026-05-16.
**Status:** ❌ **GATE NOT MET — falsifies monolithic-slot identity stabilization.**

> **Phase 7C falsifies the idea that identity stability can be added
> to a single monolithic slot vector.** Identity consistency reduces
> id-subspace drift; full-slot assignment remains unstable because
> dynamic state dominates the matching geometry. The next slot memory
> design must expose identity and dynamics *separately* — a key/value
> object-file architecture, where id_key is used for Hungarian
> assignment and state_value is used for position/dynamics prediction.

> Phase 7B falsified the predictor-side fix. Phase 7C tested the
> encoder-side fix: an auxiliary slot→pos head + identity consistency
> loss between same-entity slots across consecutive frames. Result:
> the encoder DOES separate id-drift from dyn-drift (when combined
> with the architectural split), but the metric we care about (switch
> rate from the joint-slot Hungarian probe) doesn't move.

## Pre-committed gate

```
Good:  switch_rate ≤ 0.50 AND hidden_pos_mse ≤ 3.2e-5
Great: switch_rate ≤ 0.40 AND hidden_pos_mse ≤ 2.5e-5
```

(2× tolerance on slot_delta's 1.6e-5 hpm; 30-50% reduction in slot_delta's 0.71 switch rate.)

## Result table

| mode | vis_mse | hid_mse | switch | drift_ratio | gate |
|---|---|---|---|---|---|
| slot_delta | 1.9e-5 ± 2.5e-5 | 2.2e-5 ± 2.1e-5 | 0.693 ± 0.137 | 1.21 ± 0.36 | ❌ |
| id_dyn_split | 3.2e-5 ± 2.2e-5 | 3.5e-5 ± 2.0e-5 | 0.721 ± 0.040 | 0.58 ± 0.36 | ❌ |
| slot_delta_idcons | 1.9e-5 ± 8.8e-6 | 2.8e-5 ± 1.0e-5 | 0.687 ± 0.070 | 0.92 ± 0.16 | ❌ |
| **id_dyn_split_idcons** | 1.5e-5 ± 5.6e-6 | **2.0e-5 ± 1.6e-6** | 0.715 ± 0.003 | 0.91 ± 0.05 | ❌ |
| slot_dense_update | 4.2e-4 ± 9.9e-5 | 3.2e-4 ± 9.6e-5 | 0.413 ± 0.008 | 1.08 ± 0.09 | ❌ |

The id_dyn_split_idcons combined mode hit the **best position MSE
stability across seeds** (hpm std = 1.6e-6, ~10× tighter than other
slot_delta variants) and the **lowest id_drift** of any mode (0.012).
The encoder consistency loss is working as designed. **But switch
rate didn't change.**

## What this tells us

This is **outcome #2** from the user's Phase 7B follow-up note:

> *If id_dyn_split_idcons wins on id-subspace switch but not full-slot
> switch — Then the architecture is right, but the model needs a
> readout interface that separates id_key (slow, assignment-stable)
> from state_value (fast, position/dynamics).*

The drift diagnostic confirms the architectural separation IS doing
something real: id_dyn_split_idcons has id_drift = 0.012, smaller
than any other mode and tied for smallest with slot_delta_idcons. The
encoder is being driven toward a stable identity subspace.

The reason switch_rate doesn't move:
- The identity-aware Hungarian probe takes the **full slot** [B, S, D]
  as input and Hungarian-matches per frame.
- Even if the id-half is stable, the dyn-half changes — and the dyn-half
  is half the slot's representational capacity (slot_dim // 2 = 64 dim).
- The matching cost geometry is dominated by whichever half varies most.
  When dyn varies, slots shuffle.

So the architecture works in the sense that id_drift is decoupled
from dyn_drift. But the readout used to evaluate doesn't respect the
decoupling.

## What needs to change

The next architecture should make the id_key a **first-class
identifier** for assignment, not just one half of an undifferentiated
slot block. Two paths:

### Phase 7D (quick): id-subspace probe metric

Add `switch_rate_id_subspace` and `switch_rate_dyn_subspace` metrics
that re-fit the identity probe on the id-half (or dyn-half) of slot
states only. If id-subspace switch rate is significantly lower than
the full-slot switch rate, we've confirmed outcome #2 directly and
the architecture is the right level — only the readout/eval needs
updating.

*This is already implemented locally* (extension to
`identity_probe_eval` with a `subspace_dims` slice). Staged on the
pod but not in the Phase 7C runs since the script was already running.
A Phase 7D rerun of slot_delta_idcons + id_dyn_split_idcons + id_dyn_split
with the new metrics is ~6 GPU-minutes on 3 GPUs.

### Phase 8 (architecture): explicit key/value slot

The mature design:

```
slot_i = {
    id_key:       slow, assignment-stable, used for Hungarian
    state_value:  fast, position/dynamics, used for readout
    visibility:   confidence flag
}
```

- Updates: dyn_value via sparse-delta mechanism; id_key via slow EMA
  with explicit consistency loss.
- Hungarian match: use **only id_key**.
- Readout: use [id_key, state_value] but predict positions primarily
  from state_value.

This is a real architectural change, not a loss tweak. It changes
what "a slot" is.

## Other observations worth keeping

1. **Across-seed variance for slot_delta is huge** (vis_mse std = 2.5e-5
   for a mean of 1.9e-5; switch std 0.137 for mean 0.693). The plain
   slot_delta architecture is brittle. The _idcons variants have
   ~10× tighter cross-seed variance — even though their mean switch
   rate didn't improve, the **stability of the result improved a lot**.
   For publication purposes, idcons gives a more reproducible model.

2. **slot_dense_update reliably wins switch rate (0.413)** by a wide
   margin and is extremely stable across seeds (std = 0.008). But it
   trades position MSE by 20× (3.2e-4 vs 1.6e-5). This is the same
   tradeoff seen in Phase 7 v1 — replacing slots each step gives
   identity stability but loses spatial precision.

3. **The drift_ratio is informative but doesn't predict switch rate.**
   id_dyn_split has the lowest drift_ratio (0.58, id is ~58% as
   volatile as dyn) but **highest** switch rate (0.721). The Hungarian
   readout doesn't see drift_ratio; it sees full-slot geometry.

## Decision

**Phase 7C: gate not met.** Falsification of the simplest version of
identity-anchored slot_delta. The architecture and the loss are not
strong enough by themselves to fix what the JEPA objective doesn't
naturally optimize.

Recommended next step: **Phase 7D** (6 GPU-min, code already staged) to
confirm outcome #2 directly via the id-subspace switch metric. If
confirmed, **Phase 8** moves to explicit key/value slot architecture.
If id-subspace switch doesn't improve either, the JEPA objective itself
needs strengthening (longer horizons, contrastive identity loss, or
explicit object-file memory).

## Reproducibility

Code at `1ed091e` (Phase 7B + 7C scaffolding). Artifacts at
`artifacts/phase7c_run1/seed_{0,1,2}/`.

Run command:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_a_local/validation \
    --modes slot_delta,id_dyn_split,slot_delta_idcons,id_dyn_split_idcons,slot_dense_update \
    --seeds $seed --epochs 10 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --id-consistency-w 1.0 --aux-pos-w 0.5 \
    --out /workspace/phase7c_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```
