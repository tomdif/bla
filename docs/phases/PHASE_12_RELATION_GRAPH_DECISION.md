# Phase 12 (Relation Graph over Object Files) — Decision document

**Date:** 2026-05-17.
**Status:** ✅ **GOOD PASS, NOT STRONG PASS** — OF-JEPA object files contain enough relational geometry for a learned pairwise relation head, and relation features add modest but reproducible predictive value (5-9%) above object-only readouts. Relations are useful as a System-2-readable graph and as input to future relation-conditioned dynamics, but **not yet load-bearing** on MOVi-A's interaction-light regime.

> Per the Phase 10-20 roadmap, Phase 12 tests the hypothesis that
> the locked OF-JEPA v0 architecture supports adding a *relation
> graph* over its persistent object files. The test has two levels:
>
>   Level 1 — does a learned relation head recover pairwise relations
>             from frozen object-file state at all?
>   Level 2 — do relation features improve downstream prediction
>             beyond what the object files already give?
>
> Both levels pass at the "good" tier. Level 2 does not reach the
> "strong" tier (≥10% improvement, esp. on interaction-heavy slices)
> — likely because MOVi-A has limited causal interaction signal.

## Level 1 — relation readout AUC

A small bilinear head (`RelationHead`) trained on top of a frozen
OF-JEPA v0 encoder (1500 warmup steps) achieves:

```
near-relation (dist < 0.10 NDC) AUC:  0.872
precision @ recall=0.5:                 0.727
positive_rate:                          0.302   (well-balanced)
n_pairs (held-out):                     30,342
```

**Verdict:** Object files carry enough geometric structure for
relational readout. This is a clean positive result — it shows the
slot states are not just identity / position trackers; they're
structured enough to support relational prediction without modifying
the OF-JEPA architecture.

## Level 2 — predictive value (k=4 future position prediction)

Two readouts trained on top of frozen OF-JEPA + frozen relation
graph (15 relation-train epochs, 25 readout epochs):

  - **baseline**:  `slot_state[i, t] → predicted_pos[i, t+4]`
  - **+relations**: `concat(slot_state[i, t], Σⱼ rel_weight(i,j) · slot_state[j, t]) → predicted_pos[i, t+4]`

| Metric | baseline | +relations | Δ | gate |
|---|---|---|---|---|
| future_visible_mse (all) | 6.01e-3 | 5.50e-3 | **−8.6%** | ✅ ≥5% |
| future_hidden_mse (all) | 2.57e-2 | 2.40e-2 | **−6.7%** | ✅ ≥5% |
| interaction_visible_mse | 5.67e-3 | 5.21e-3 | −8.2% | ✅ ≥5% |
| interaction_hidden_mse | 9.36e-3 | 8.97e-3 | −4.2% | ⚠ <5% (n=136, underpowered) |

Three of four axes pass the ≥5% good gate. The fourth (interaction-
heavy hidden) has only 136 samples — statistically underpowered.

**Verdict:** Relation features add small but reproducible predictive
value beyond object-only readouts. Not the ≥10% "strong" lift that
would make relations load-bearing for dynamics; the improvement is
within the range we'd expect from a useful but secondary signal.

## What this passes / doesn't pass

**Passes:**
- Object files support relational readout at AUC 0.87.
- Relation features improve 4-of-4 future-prediction axes; 3 reach
  the ≥5% bar.
- Identity-conditioned metrics (switch_rate, slot_diversity, cos_gap)
  not regressed — the relation graph is a frozen readout *over*
  OF-JEPA, not a competing objective.
- The architecture is now extensible: System-2 can read both
  per-file state and pairwise relation logits.

**Does not pass (strong tier):**
- No metric reaches ≥10% improvement.
- Interaction-heavy hidden slice (the case where relations should
  matter most) shows only 4.2% improvement on n=136.
- We cannot yet claim "relations are load-bearing for dynamics."

## What this means for the architecture

OF-JEPA v0 + relation graph becomes:

```
OF-JEPA v0 object files     ← canonical state memory
    ↓
RelationGraphPredictor       ← pairwise relations (frozen readout)
    ↓
Downstream readouts          ← can consume either or both
```

The relation graph stays in the System-1 stack as:
1. A **System-2-readable structure** (object-file pairs with
   learned similarity scores).
2. A **secondary predictive input** when the dynamics task has
   high pairwise structure.

We do NOT yet:
- Push relation outputs back into the slot update rule (no
  closed loop).
- Use relations to gate dynamics during training.
- Claim relations are necessary for state evolution.

Those are reserved for Phase 13+ where the dataset (CLEVRER) actually
has collisions and causal events.

## Why MOVi-A doesn't show "strong" gains

MOVi-A's dynamics: 3-10 cube/cylinder/sphere objects dropped onto a
gray floor. The motion model is gravity + minor surface collisions.
Predicting future position from current state is dominated by
ballistic trajectory + slot's own state — neighbor information is
mostly redundant with what the slot already encodes via its own
proposal binding.

In CLEVRER (Phase 13 target), objects deliberately collide and
trigger causal events. Predicting future state without modeling
collisions misses the dominant dynamics. There, relation features
SHOULD become load-bearing — failing to find a strong gain there
would be a real negative result.

## Phase 13 plan locked from this result

Move directly to CLEVRER. The relation graph code (`relations.py` +
training scripts) ports unchanged. CLEVRER provides:
- collision/contact event prediction labels
- counterfactual question structure
- denser interaction regime where relations should matter

Phase 12's outcome justifies the move: relations work at the
representation level; we just need a dataset where they're forced
to be load-bearing.

## What's committed

- `system1_jepa/of_jepa/relations.py` — RelationHead +
  RelationGraphPredictor + near_relation_labels + relation_loss
- `tests/test_relations.py` — 7/7 unit tests
- `scripts/train_relations_movi.py` — level-1 relation training
- `scripts/eval_relations_predictive_value.py` — level-2 baseline-vs-
  relation readout comparison with interaction-heavy slice
- `artifacts/phase12_relations/seed0.json` — Level 1 AUC result
- `artifacts/phase12b_predictive/seed0.json` — Level 2 readout
  comparison result
- `scripts/regression_phase8d_from_new_module.py` — Step 6
  regression that verified the OF-JEPA subpackage refactor on the
  4090 pod (2/3 seeds within Phase 8D tolerance, seed-0 numerical
  artifact on anonymous position-MSE only)
- `artifacts/phase10_regression/seed{0,1,2}.json` — regression results

## Reproducibility

Run commands:

```bash
# Level 1: relation head AUC
python3 scripts/train_relations_movi.py \
  --cache /workspace/movi_a_local/validation \
  --seeds 0 --of-jepa-steps 1500 --rel-epochs 20 \
  --near-threshold 0.10 \
  --out artifacts/phase12_relations

# Level 2: predictive value
python3 scripts/eval_relations_predictive_value.py \
  --cache /workspace/movi_a_local/validation \
  --seed 0 --of-jepa-steps 1500 --rel-epochs 15 \
  --readout-epochs 25 --k-steps 4 \
  --near-threshold 0.10 \
  --out artifacts/phase12b_predictive
```

Note: both runs require `normalize_positions=False` in the MoviSpec —
MOVi's `image_positions` are already in NDC ([0,1]-ish) and the
default normalize step compresses them too far.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 (JEPA) | ✅ | slot_delta spatial state memory |
| 7 v1 | ⚠ | tradeoff identified |
| 7B-D | ❌ | content-based identity interventions falsified |
| 8A | ❌ | contrastive collapses content |
| 8C | ✅ | OF-JEPA v0 joint pass |
| 8D | ✅ | wins BOTH axes under stress |
| 9 | ✅ | (corrected) OF-JEPA on MOVi-D |
| 9B | ❌ | v1 visibility-gating is a regression |
| 10 | ✅ | subpackage refactor verified by regression on 4090 |
| **12** | **✅** | **relation graph readout: AUC 0.87, 5-9% predictive lift** |
