# Phase 7D (JEPA, MOVi-A) — Decision document

**Date:** 2026-05-16.
**Status:** ❌ **GATE NOT MET — falsifies "the architecture is right, just read it from the id_key."** id_subspace switch rate ≈ full-slot switch rate (≈ 0.68-0.70). The JEPA objective itself does not naturally reward persistent identity, regardless of where in the slot we look.

> Phase 7C showed encoder consistency loss reduces id_drift but
> doesn't move the full-slot switch rate. The hypothesis (user's
> outcome #2) was that the architectural id/dyn separation is doing
> identity-stabilizing work in the id subspace, but the readout
> mixes the two halves. Phase 7D tested this directly by Hungarian-
> matching ONLY the id-half of the slot (id_key) and ONLY the
> dyn-half (state_value), as separate eval probes.
>
> The result decisively falsifies the readout-interface hypothesis:
> **id_key switch rate is statistically indistinguishable from
> full-slot switch rate.** The id subspace doesn't actually carry
> more *identity-distinguishing* information than the full slot —
> it carries the same geometry, just on a sub-axis.

## Results table (3 seeds, mean ± std)

| mode | sw_full | sw_id_key | sw_dyn | hpm_full | hpm_id | hpm_dyn |
|---|---|---|---|---|---|---|
| slot_delta | 0.682 ± 0.149 | — | — | 2.18e-5 | — | — |
| id_dyn_split | 0.738 ± 0.067 | **0.698 ± 0.036** | 0.730 ± 0.046 | 2.70e-5 | **1.23e-5** | 1.20e-4 |
| id_dyn_split_idcons | 0.705 ± 0.025 | **0.684 ± 0.025** | 0.714 ± 0.014 | 2.78e-5 | 2.91e-5 | 3.25e-5 |
| slot_dense_update | 0.401 ± 0.008 | — | — | 3.19e-4 | — | — |

## Three findings

### 1. id_key switch rate ≈ full-slot switch rate

```
id_dyn_split_idcons:  sw_full = 0.705 ≈ sw_id_key = 0.684 ≈ sw_dyn = 0.714
id_dyn_split:         sw_full = 0.738 ≈ sw_id_key = 0.698 ≈ sw_dyn = 0.730
```

The id subspace and dyn subspace each shuffle slots-to-entities at
roughly the same rate as the full slot. **There is no
identity-stabilizing subspace.** The architecture's id/dyn
separation doesn't create a sub-axis where slot bindings are
persistent; it creates a sub-axis where slot bindings happen to be
useful for *different* things (position info lives more cleanly in
the id half, see #2 below), but bindings are equally shuffling
everywhere.

### 2. The id subspace carries position info — well

```
id_dyn_split: hpm_id_key = 1.23e-5 vs hpm_full = 2.70e-5 vs hpm_dyn = 1.20e-4
```

When we restrict the linear probe input to the id-half (64-dim
subset of the 128-dim slot), it gets *better* position MSE than the
full-slot probe (1.23e-5 vs 2.70e-5). The id subspace is the more
spatially-precise half. The dyn subspace, despite being where the
sparse-delta updates fire, gets 10× worse position MSE.

That is the opposite of what we wanted: we hoped id_key would
encode entity *identity* (and dyn would encode position/state),
but the encoder learned to put position info into both halves —
and *more cleanly* into the half we labelled "id".

### 3. JEPA loss doesn't reward persistent identity

The end-to-end story across Phase 7B, 7C, 7D:

| Phase | Fix | Outcome |
|---|---|---|
| 7B | predictor-side slow EMA on id half | switch unchanged |
| 7C | + encoder consistency loss + aux head | switch unchanged, id_drift drops |
| 7D | + id_key Hungarian (read only the id half) | switch unchanged |

The intervention space "tweak the slot architecture / loss /
readout" is exhausted. None of these moves the switch rate.

**Why:** the JEPA objective is `||predict(slots_t) - slots_{t+1}||²`.
This rewards slots that are predictable across frames. It says
nothing about whether slot_i at frame t is bound to the same entity
as slot_i at frame t+1. If at every frame the encoder re-assigns
slot_i to whichever entity is currently easiest to predict from
slot_i's previous content, JEPA loss is minimized and switch rate
stays at chance.

The slot-permutation invariance of SlotAttention is, ironically,
working *against* identity persistence here. Slot indices are
exchangeable at each frame, so the encoder has no reason to keep
the same slot bound to the same entity.

## What this falsifies

Three increasingly broad claims are now falsified at this scale:

1. *"Predictor-side EMA fixes identity"* — Phase 7B, false.
2. *"Encoder-side consistency loss fixes identity"* — Phase 7C, false.
3. *"Read the id half and identity becomes stable"* — Phase 7D, false.

All three assumed that identity is an emergent property that we
can extract by structuring the model's outputs differently. The
data says: the model's outputs don't encode identity persistently
because the training signal doesn't ask for it.

## What remains intact

- `slot_delta`'s position-MSE strength (~2e-5 hidden) survives every
  ablation. The spatial-state-memory claim is robust.
- `slot_dense_update`'s identity-stability advantage (sw 0.40) is also
  robust but comes with a 20× position MSE penalty.
- The JEPA + identity-aware Hungarian probe + drift diagnostic
  *pipeline* is mature: it can discriminate modes reliably, surface
  decoupled metrics, and falsify hypotheses cleanly. This is what
  let us trace the failure mode in three phases instead of declaring
  a vague "didn't work."

## What this means for the architecture path

The JEPA objective is **insufficient** for identity-stable slot
binding. The next round of work needs to change the training signal,
not the model structure. Three candidate paths, in order of
implementation cost:

### A. Identity-supervised contrastive loss (cheapest)

For each pair (slot_i(t), slot_j(t+1)) Hungarian-matched to the same
GT entity ID, pull them together; for pairs matched to *different*
entity IDs, push them apart. This is a contrastive analog of the
identity consistency loss — it explicitly rewards persistent
binding, not just frame-local stability.

### B. Slot-permutation-fixed binding (medium)

Replace SlotAttention's exchangeable-slot init with persistent slot
prototypes. Each slot has a learned identity vector that biases the
attention's query — slot 0 is always "look for X-like things", slot
1 always "look for Y-like things". The model loses general
permutation invariance but gains an identity-axis the encoder can
learn against.

### C. Object-file memory (most ambitious)

Each slot becomes an explicit `{id_key, state_value, visibility}`
namedtuple with its own attention head for matching incoming
observations to existing slots vs spawning a new slot. This is the
"hashtable-of-objects" mental model the user proposed. It requires
designing an explicit slot-spawn/slot-release mechanism, but it
matches how cognitive scientists describe object files in human
visual cognition.

## Decision

Phase 7D closes the slot-architecture-only experimental line. The
identity-stability gap on rendered video is not a slot-structure
problem; it's a training-signal problem.

For Phase 8 the right move is **A** (identity-contrastive loss) as
a single cheap experiment to confirm that *any* identity-aware
signal fixes the gate, before committing to the larger architectural
investment of B or C. If A passes, B and C become genuine architectural
choices rather than reactions to a confused metric.

For the broader BLA project, the result locks the JEPA System-1 track at:

> *"slot-delta JEPA provides strong spatial state memory on rendered
> video, but identity-persistent object binding requires an explicit
> identity training signal beyond the JEPA prediction loss. The slot
> mechanism is a useful spatial-memory substrate; it is not, on its
> own, an object-file memory."*

This is honest and publishable. It also tightens the BLA integration
question: when Phase 7's slot_delta lands in the System-1/2 bridge,
expect to ship with a contrastive identity head, not bare JEPA.

## Reproducibility

Code at `bf1rvqee8`-era commit (Phase 7C + subspace probe extension).
Artifacts at `artifacts/phase7d_run1/seed_{0,1,2}/`.

Run command:

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_a_local/validation \
    --modes slot_delta,id_dyn_split,id_dyn_split_idcons,slot_dense_update \
    --seeds $seed --epochs 10 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --id-consistency-w 1.0 --aux-pos-w 0.5 \
    --out /workspace/phase7d_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```
