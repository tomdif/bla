# Phase 13 (CLEVRER) — Decision document

**Date:** 2026-05-17.
**Status:**
- **13.3 ✅ STRONG PASS** — OF-JEPA identity binding transfers cleanly from MOVi-A to CLEVRER.
- **13.4 ⚠ NEGATIVE** — Relation graph does **not** improve collision prediction beyond what pair-readouts already extract from OF-JEPA's object-file state.

> Two distinct results in Phase 13. The first (architectural transfer)
> is the headline positive: OF-JEPA's identity mechanism — switch rate
> 0.0014, slot diversity 1.035, cos_gap 0.462 — reproduces almost
> exactly on a harder external benchmark with real video + real
> collisions. The second (relations as collision-prediction input) is
> a clean negative: the relation graph trained on top of frozen
> OF-JEPA adds essentially nothing to a pair-MLP collision head. Both
> findings sharpen the architecture's load-bearing surface.

---

## Phase 13.3 — OF-JEPA identity transfer to CLEVRER

3 seeds × 3000 steps on 500 CLEVRER training episodes (32 frames each,
stride 4 from the native 128-frame 5 s clips).

| Metric | seed0 | seed1 | seed2 | mean | MOVi-A reference |
|---|---|---|---|---|---|
| **switch_rate** | 0.0024 | 0.0015 | 0.0003 | **0.0014** | 0.002 (same!) |
| **slot_diversity** | 1.050 | 1.042 | 1.012 | **1.035** | 1.05 |
| **cos_gap** | 0.456 | 0.467 | 0.464 | **0.462** | 0.45 |
| id_visible_mse | 0.058 | 0.060 | 0.060 | 0.059 | 2.5e-5 |
| id_hidden_mse | 0.574 | 0.539 | 0.734 | 0.616 | 4.2e-5 |
| id_h/v ratio | 9.91 | 9.00 | 12.24 | 10.4 | 1.51 |

**Identity binding axes (switch, diversity, cos_gap) match MOVi-A
within tolerance.** The persistent-id-prototype mechanism is dataset-
robust: real video + GSO-style objects + collisions don't break it.

Position MSE is in a different absolute regime on CLEVRER (3000×
higher visible MSE than MOVi-A) because CLEVRER's dynamics + smaller
objects + faster motion + denser occlusion are substantially harder.
Most importantly, the **identity-conditioned** h/v ratio of 10.4 is
high (vs MOVi-A's 1.5) — meaning the architecture maintains object
identity across CLEVRER's occlusions but the slot's spatial state
loses precision under the harder occlusion regime.

This is exactly the expected transfer profile:

> *Identity binding is structural and transfers; spatial precision is
> task-dependent and degrades on harder dynamics.*

## Phase 13.4 — relation graph collision prediction

The setup tests whether adding a learned relation graph on top of
frozen OF-JEPA improves collision-event prediction (predict whether
each object pair collides within the next k=4 frames).

- Baseline: pair-MLP on `concat(slot_i + slot_j, |slot_i − slot_j|)`
- +Relations: same pair-MLP plus a softmax-aggregated relation-message
  feature derived from the trained relation graph

| Metric | baseline | +relations | Δ | verdict |
|---|---|---|---|---|
| collision AUC | 0.8447 | 0.8442 | −0.1% | ⚠ no effect |
| collision AP | 0.1618 | 0.1628 | +0.6% | ⚠ no effect |
| positive_rate | 3.75% | 3.75% | — | (collisions rare) |
| n_pairs | 49,748 | 49,748 | — | held-out eval |

**Net effect: zero.** OF-JEPA's object-file state alone predicts
collisions at AUC 0.84. The relation graph is informationally redundant
with what a pair-MLP can already extract from the same slot states.

## Why the relation graph helped in Phase 12 but not Phase 13.4

The two experiments differed in baseline input structure:

| Phase | Baseline input | Relations input | Lift |
|---|---|---|---|
| 12 (MOVi) | single slot per entity → future pos | + neighbor msg | +5–9% |
| 13.4 (CLEVRER) | pair (slot_i, slot_j) → collision | + neighbor msg | ~0% |

Phase 12's baseline operated on a single slot, so neighbor messages
gave the readout STRICTLY MORE information than it had before. Phase
13.4's baseline already operated on pairs, so the relation message
(a softmax-weighted summary of OTHER pairs) added a strictly less
informative signal than the pair the readout already saw directly.

**Architectural lesson:** the relation graph is a useful representational
projection of pairwise object-file structure. It helps when downstream
readouts can't directly access pairs. It doesn't help when they can.

For BLA's System-2 bridge (which reads object-file pairs explicitly),
the relation graph adds little. For System-2 reasoning over single
entity properties (where pair information would otherwise be lost),
relations are valuable.

## What this lets us claim and not claim

**Claim (locked):**
- OF-JEPA v0 transfers cleanly to CLEVRER on identity binding.
- The identity-as-address architecture is dataset-robust beyond MOVi.
- Object-file states alone predict collisions at AUC 0.84 — strong.

**Do NOT claim:**
- "Relation graph improves CLEVRER prediction."
- "Relations are load-bearing for collision dynamics."
- Any "OF-JEPA + relations beats baselines" headline.

The Phase 12 "good not strong" verdict on relations holds: relations
are useful as a System-2-readable structure but they are not load-
bearing when the consumer already accesses pairs.

## What's still open

| Question | What would test it |
|---|---|
| Does OF-JEPA improve future-state prediction at long k on CLEVRER? | Phase 12-style future_pos_mse comparison at k=8, 16 |
| Do relations help on tasks System-2 can't do as pairs? | Counterfactual / causal queries that require message-passing |
| Can we get position MSE down on CLEVRER? | Longer training, larger encoder, or denser frame stride |
| Does identity stay tight under MOVi-E camera motion? | The Phase 9B-shelved lifecycle test on streaming data |

The Phase 10–20 plan's biggest remaining test is **action-conditioned
dynamics** (Phase 14), where relations may finally become load-bearing
because the action graph naturally pairs an actor with a target
object.

## Reproducibility

- Phase 13.3: `python3 scripts/slot_jepa_clevrer_train.py --cache /workspace/clevrer_local/train --seeds 0,1,2 --max-steps 3000 --jepa-stride 4`
- Phase 13.4: `python3 scripts/clevrer_collision_predict.py --cache /workspace/clevrer_local/train --seed 0 --of-jepa-steps 1500 --rel-epochs 10 --head-epochs 20 --k-frames 4`

Code at this commit. Artifacts:
- `artifacts/phase13_clevrer/seed{0,1,2}.json` — identity transfer numbers
- `artifacts/phase13b_collision/seed0.json` — collision comparison

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 (JEPA) | ✅ | slot_delta spatial state |
| 7 v1–8A | ❌ | content-side identity fixes falsified |
| 8C | ✅ | OF-JEPA v0 joint pass on MOVi-A |
| 8D | ✅ | wins BOTH axes under MOVi-A stress |
| 9 | ✅ | (corrected) MOVi-D identity transfer |
| 9B | ❌ | v1 visibility-gating is a regression |
| 10 | ✅ | subpackage refactor regression on 4090 |
| 12 | ✅ good | relation readout AUC 0.87; 5–9% lift on single-slot readouts |
| **13.3** | **✅** | **OF-JEPA identity binding transfers cleanly to CLEVRER** |
| **13.4** | **⚠** | **Relations don't help when pair info is already in the readout** |
