# Phase 7B (JEPA, MOVi-A) — Decision document

**Date:** 2026-05-16.
**Status:** ❌ **FALSIFIED — predictor-side id_dyn_split alone does not close the identity-stability gap.** The identity problem is upstream, in the encoder.

> Phase 7 v1 left the slot architecture on a real tradeoff: slot_delta
> wins position MSE by 20-40×, slot_dense_update wins identity-switch
> rate. Phase 7B tested whether the simplest architectural fix —
> splitting each slot into [slow id, fast dynamics] and updating them
> with different rules in the predictor — closes the gap.
> **It does not.** The encoder still emits identity-subspace features
> that drift frame-to-frame, and the JEPA loss forces the predictor to
> chase that drift, defeating the EMA on the id half.

## Headline numbers (mean ± std, 3 seeds, 3000 steps each)

| mode | vis_mse | hid_mse | switch_rate ↓ |
|---|---|---|---|
| slot_delta | 9.07e-6 ± 3.1e-6 | 1.60e-5 ± 4.0e-6 | 0.711 ± 0.079 |
| **id_dyn_split** | 9.28e-6 ± 3.6e-6 | 1.27e-5 ± 3.3e-6 | **0.717 ± 0.108** |
| slot_dense_update | 4.39e-4 ± 1.2e-4 | 3.80e-4 ± 1.2e-4 | 0.381 ± 0.082 |
| dense_jepa | 2.05e-4 ± 1.0e-5 | 2.25e-4 ± 5.6e-5 | 0.471 ± 0.055 |
| copy | 1.85e-3 ± 5.0e-4 | 1.85e-3 ± 3.6e-4 | 0.155 ± 0.013 |

**id_dyn_split = slot_delta on every metric.** Switch rate moved from
0.711 to 0.717 — *within seed noise*, not an improvement. Position
MSE moved similarly. The EMA on the id half had no measurable effect
on what we care about.

Per-seed breakdown showed id_dyn_split helped seed 0 (switch 0.75 →
0.60) but hurt seeds 1 and 2 (switch +0.05 to +0.13). Random.

## Why the fix didn't work

The id_dyn_split predictor update was:

```
id_next  = id  +  alpha * id_delta     (slow EMA, alpha = 0.05)
dyn_next = dyn + change_mask * delta   (fast sparse)
```

The intuition: keep id stable across frames. The reality:

1. At training time, the JEPA target is `slots_{t+1}.detach()` —
   the encoder's *own* output at the next frame.
2. If the encoder produces id-half outputs that drift across frames,
   the predictor must produce id-half outputs that also drift to
   minimize JEPA loss.
3. The EMA constraint `id_next = id + alpha * id_delta` only
   constrains the *predictor*. It says nothing about what the encoder
   emits. If the encoder's frame-t+1 id-half is far from the
   encoder's frame-t id-half (because the encoder is unstable in that
   subspace), the predictor learns large `id_delta` values to make
   the EMA-smoothed prediction match anyway.
4. So the "slow id" is cosmetic — the actual encoder-output id
   trajectory is still jittery, and the probe (which sees encoder
   outputs, not predictor outputs) reads the same instability.

This is the user-stated failure mode:

```
encoder id-half drifts
→ predictor is trained to chase drift
→ EMA/slow-id update becomes cosmetic
→ identity stability does not improve
```

## What this falsifies

The architectural claim "splitting the slot into slow and fast halves
fixes identity stability" — **false at this scale and config**,
because the split was only applied to the *predictor*, not to the
*encoder*. The fix must be encoder-side, not predictor-side.

## What this preserves

slot_delta's strong position MSE result is intact (1.6e-5 hidden,
roughly matching v2). The architecture's spatial-accuracy advantage
isn't an artifact of the predictor's particular structure — it
appears to be driven by the JEPA loss's encoder pressure to produce
spatially-tight slot bindings. That's a real result that survives
Phase 7B.

## What Phase 7C tests

Encoder-side identity consistency loss:

```
L_id_cons = mean_{entity e visible at t-1 and t}
             ||slot_id(t)[match_e(t)]
                - stopgrad(slot_id(t-1)[match_e(t-1)])||²
```

where `match_e(t)` is the slot Hungarian-matched to entity e at
frame t (using an auxiliary slot→position head trained inline by
MSE on GT positions). This is option #1 from the user's Phase 7B
follow-up note, with GT-based matching as the oracle-clean variant.

Phase 7C will run 5 modes × 3 seeds:
- slot_delta (control)
- id_dyn_split (control, Phase 7B's mode)
- slot_delta_idcons (slot_delta + encoder identity consistency)
- id_dyn_split_idcons (combined)
- slot_dense_update (control)

Each run also reports the **id_drift / dyn_drift** diagnostic:
mean of ||slot_id(t) - slot_id(t-1)[matched]|| vs same for dyn-half.
If id_drift ≈ dyn_drift, the split is still cosmetic. If
id_drift << dyn_drift, the encoder is being driven to a stable
identity subspace.

Pre-committed Phase 7C gate (locked here, before the result):

```
Good result:  switch_rate ≤ 0.50  AND  hidden_pos_mse ≤ 3.2e-5
Great result: switch_rate ≤ 0.40  AND  hidden_pos_mse ≤ 2.5e-5
```

(2× tolerance on slot_delta's 1.6e-5 hidden MSE; 30-50% reduction
in slot_delta's 0.71 switch rate.)

## Methodology note for the memory

This is a useful example of [[feedback-joint-metric-vs-single-axis]]
in action: id_dyn_split passed a *narrow* version of the gate
("predictor smooths the id half") but failed the actually-measured
identity-stability metric. A predictor-only architectural change
can't fix an encoder-output property. Future architecture-tweak
proposals should be checked against where the property actually
lives.

## Reproducibility

Code at `188647f` (training orchestrator) + new `id_consistency.py`
module (will land with Phase 7C commit). Artifacts at
`artifacts/phase7b_run1/seed_{0,1,2}/`.

Run command:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_a_local/validation \
    --modes slot_delta,id_dyn_split,slot_dense_update,dense_jepa,copy \
    --seeds $seed --epochs 10 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --out /workspace/phase7b_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```
