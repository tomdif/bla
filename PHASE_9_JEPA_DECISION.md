# Phase 9 (OF-JEPA generalization to MOVi-D) — Decision document

**Date:** 2026-05-16.
**Status:** ⚠️ **PARTIAL — OF-JEPA wins identity stability and visible-state accuracy on MOVi-D, but fails identity-conditioned hidden-state prediction because occluded files keep binding to visible proposals. slot_delta's low anonymous hidden MSE reflects opportunistic Hungarian rematching, NOT persistent object memory. Phase 9B (visibility-gated null binding + transition-only updates for occluded files) is now load-bearing.**

> Phase 9 tested whether OF-JEPA v0 transfers from MOVi-A's simple
> shapes to MOVi-D's harder visual regime: HDRI-projected backgrounds,
> Google Scanned Object meshes (real shapes/textures), and 11-23
> instances per scene (vs 3-10 in MOVi-A) with ~5% natural occlusion.
> Result: identity binding transfers cleanly (switch 0.093, cos_gap
> 0.48, diversity 2.02). But the architecture has no occlusion
> lifecycle — when an entity is hidden, the matcher still routes its
> file to *some* visible proposal, which corrupts the file's state.
> The fix is structural and comes in Phase 9B.

## Headline numbers (3 seeds, 3000 steps each, MOVi-D val split)

| Mode | vis_mse | hid_mse (anon) | switch ↓ | diversity ↓ | h/v ratio | cos_gap ↑ |
|---|---|---|---|---|---|---|
| **of_jepa_v0** | 1.7e-5 ± 5.6e-6 | **4.2e-2 ± 2.5e-2** | **0.093 ± 0.012** | **2.02 ± 0.10** | 2780 | **0.477 ± 0.03** |
| slot_delta | 9.2e-6 ± 3.2e-6 | **1.25e-5 ± 2.5e-6** | 0.800 ± 0.045 | 6.28 ± 0.10 | 1.42 | ~0 |
| slot_dense_update | 2.9e-4 ± 4.7e-5 | 2.92e-4 ± 3.7e-5 | 0.602 ± 0.025 | 5.54 ± 0.16 | 1.01 | ~0 |

## The subtle metric issue

Read straight off the table, slot_delta "wins" hidden_pos_mse by
**3360×** (1.25e-5 vs 4.2e-2). But this is **not** evidence that
slot_delta has better persistent object memory. It's a metric
artifact:

1. The hidden_pos_mse metric uses **anonymous per-frame Hungarian
   matching** — it relabels slot→entity assignments fresh each frame
   to minimize MSE.

2. slot_delta has **no persistent identity binding** (slot diversity
   6.28 — each slot is shuffled across 6+ entities per episode). When
   entity *e* is occluded at frame *t*, the per-frame Hungarian just
   routes it to whichever slot happens to predict a nearby position.
   The MSE stays low because the assignment is opportunistic.

3. OF-JEPA has **persistent identity binding** (diversity 2.02). When
   entity *e* is occluded at frame *t*, the Sinkhorn matcher tries
   to bind entity *e*'s designated file to some visible proposal —
   the closest available — and that file's state_value gets
   corrupted by the wrong proposal's content. The probe then sees
   garbage at that file's index and reports high MSE.

**slot_delta's anonymous tracking score is good. Its object-file
memory score is zero**, because there is no object-file. OF-JEPA
is being held to a harder standard — and is currently failing it
on occluded frames.

## What this falsifies and what it confirms

**Falsified:** "OF-JEPA v0 is a complete object-file world model."
It isn't — it lacks the occlusion lifecycle. The architecture
binds files to visible proposals always, which corrupts the file
when its entity is occluded.

**Confirmed:** Identity binding transfers from MOVi-A's simple regime
to MOVi-D's HDRI-real-object regime. The persistent-memory mechanism
is dataset-robust:
- switch_rate 0.093 vs slot_delta 0.800 — 8.6× better identity stability.
- cos_gap 0.477 vs slot_delta ≈ 0 — id_keys remain structurally separated.
- diversity 2.02 vs 6.28 — files stay near-uniquely bound.

**Confirmed:** OF-JEPA's visible-state quality is intact:
visible_position_mse 1.7e-5, only 1.85× higher than slot_delta's
9.2e-6 — well within the joint-gate tolerance.

So the failure is specifically **at the moment a file's entity
becomes occluded**, not in identity binding or visible state.

## The architectural fix (Phase 9B)

The next-version object file needs three pieces:

```
ObjectFile_i = {
    id_key,              # persistent address (unchanged across occlusion)
    state_value,         # dynamic state (evolved via transition during occlusion)
    visibility_belief,   # per-frame logit: is my entity visible right now?
    [appearance_key,     # v2: re-id signature after gap]
}
```

Update rule:

```
match_confidence = sinkhorn(memory, proposals + NULL_column)

if match_confidence is high:
    state_value += sparse_delta(matched_proposal)
    id_key  ← EMA toward matched_id
    visibility ← visible
elif match_confidence is low (file → NULL):
    state_value ← transition_model(state_value)
    id_key      ← unchanged (KEEP the persistent address)
    visibility  ← occluded
```

The critical invariant: **never overwrite a persistent object
file with an unrelated visible proposal just because the true
object is hidden.**

Phase 9B implementation (already drafted in `system1_jepa/of_jepa.py`
`ObjectFileMemoryV1`):

1. **NULL column in Sinkhorn**: a learnable bias score appended to
   the proposal grid. Files can route assignment mass to the null
   column → `match_confidence = 1 - null_mass` becomes the gate.
2. **transition_model MLP**: small 2-layer head that predicts
   state_value forward without observation, used when match_confidence
   is low.
3. **visibility_head**: per-file logit predicting current visibility,
   supervised against GT during training.

## The new headline metric for Phase 9B

The standard `hidden_pos_mse` (anonymous Hungarian per frame) is
the wrong metric for an object-file architecture. Phase 9B adds:

> **identity_conditioned_hidden_mse** — for each file, find its modal
> entity assignment across the episode. Then measure per-frame MSE
> between probe(file) and GT_pos(file's modal entity), regardless of
> per-frame visibility. This measures: "does the file STAY bound to
> its entity across time, even when the entity becomes occluded?"

This is what object-file memory actually does. It can't be gamed by
anonymous rematching.

Code in `system1_jepa/identity_probe.py:identity_conditioned_position_eval`,
wired into the trainer for the Phase 9B sweep.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 | ✅ | slot_delta strong spatial state memory under stress |
| 7 v1 | ⚠ | slot_delta vs slot_dense_update tradeoff identified |
| 7B-D | ❌ | slot-content interventions falsified (4 attempts) |
| 8A | ❌ | contrastive loss collapses content at any effective λ |
| 8C | ✅ | OF-JEPA on MOVi-A: switch 0.002, joint gate passed |
| 8D | ✅ | OF-JEPA under stress (≥8 ent, stride=4): BOTH axes won |
| **9** | **⚠** | **OF-JEPA on MOVi-D: identity binding transfers, but lacks occlusion lifecycle. Phase 9B required.** |

## Decision

Phase 9 v0 passes the **identity stability transfer** test (switch
0.093, cos_gap 0.48, diversity 2.02 all hold under MOVi-D's harder
regime) but **fails the joint gate** on occluded frames because v0
has no occlusion lifecycle.

Phase 9B's visibility-gated update rule is now load-bearing, not
optional. Code is already drafted; the sweep launches next.

## Reproducibility

Code at Phase 8D-era commit + `--max-entities` flag (MOVi-D has
≤19 instances, so n_slots=20). Artifacts at
`artifacts/phase9_run1/seed_{0,1,2}/`.

Run command:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_d_local/validation \
    --modes of_jepa_v0,slot_delta,slot_dense_update \
    --seeds $seed --epochs 20 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --of-jepa-w 1.0 --of-pos-w 10.0 \
    --n-slots 20 --slot-dim 128 --max-entities 25 \
    --out /workspace/phase9_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```
