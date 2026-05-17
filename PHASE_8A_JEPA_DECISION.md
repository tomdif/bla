# Phase 8A (JEPA, MOVi-A) — Decision document (final, two-pass)

**Date:** 2026-05-16.
**Status:** ❌ **GATE NOT MET — collapse-tradeoff confirmed across λ ∈ {0.1, 0.3, 1.0, 3.0}.** No setting of the identity-contrastive weight simultaneously preserves position MSE and reduces switch rate. The contrastive loss can buy identity stability only at the price of collapsing the slot's content representation.

> Phase 8A tested "add an explicit identity-contrastive objective on
> id_key" as the cheapest way to inject identity supervision into the
> JEPA slot pipeline. v1 ran at λ=0.1 across 4 modes; v2 swept
> λ ∈ {0.3, 1.0, 3.0} (and 10.0 was queued but cancelled) on the
> contrastive mode only. The λ sweep was cut short after the
> collapse-tradeoff pattern was clearly established — extending to
> λ=10.0 would only confirm worse collapse.

## λ-sweep result table (mean across available seeds, 3000 steps each)

| λ | n_seeds | sw_full | sw_id_key | hpm | cos_gap |
|---|---|---|---|---|---|
| 0.1 | 3 | 0.691 | 0.691 | 2.5e-5 | (not measured v1) |
| 0.3 | 3 | 0.306 | 0.314 | 2.8e-4 | 0.028 |
| 1.0 | 3 | **0.102** | 0.103 | **7.5e-4** | 0.121 |
| 3.0 | 1 | 0.998 | 0.988 | 4.2e-4 | 0.029 |

Pre-committed gate: switch_rate ≤ 0.50 AND hpm ≤ 5e-5.

## The collapse-tradeoff pattern

Three regimes show up cleanly:

1. **λ = 0.1: nothing changes.** Contrastive loss is too weak to
   shift the encoder; result indistinguishable from no contrastive
   (switch ≈ 0.69, hpm ≈ 2.5e-5). Same fail as Phase 7C.

2. **λ ∈ {0.3, 1.0}: identity wins, content loses.** Switch rate
   drops impressively (0.10 at λ=1.0!) and cos_gap shows real
   discrimination (0.12). But position MSE blows up 30-50× — the
   encoder is being forced to encode identity into the slot vector
   at the expense of state. Joint gate fails.

3. **λ = 3.0: catastrophic over-collapse.** Switch rate jumps to
   chance (0.998) and cos_gap collapses to 0.029. The contrastive
   loss has so dominated the encoder that all slot states are
   essentially identical, and any "stability" is from constant noise.
   λ=10.0 would be even worse.

## The intermediate finding: λ=1.0 cos_gap = 0.121, switch = 0.10

This is the single most positive data point in the entire Phase 7+8
arc — the encoder IS producing somewhat-discriminable id_keys
across same/different entities (cos_gap = 0.121) and the switch
rate IS strongly suppressed (0.10, well below the 0.50 gate). The
mechanism *can* work.

But position MSE is 7.5e-4 vs slot_delta's 1.6e-5 — **47× worse**.
You can't extract this as a "λ=1.0 wins" result because the joint
metric tracks failure correctly: stability bought at the cost of
content.

This is the user's predicted outcome from the Phase 8A pre-launch:

> *If 8A improves switch but hurts MSE — Then there is a tradeoff.
> Next architecture should separate id_key trained contrastively
> from state_value trained predictively more aggressively.*

Exactly that. The contrastive loss can move switch rate, but the
slot vector is shared between identity and content. Forcing one
displaces the other.

## What this closes

Phase 8A closes the **slot-content-as-identity** experimental line.
All four interventions on the standard slot architecture are
falsified:

| Phase | Intervention | Result |
|---|---|---|
| 7B | predictor-side slow EMA | switch unchanged |
| 7C | encoder consistency loss | id_drift drops, switch unchanged |
| 7D | id-subspace Hungarian readout | sw_id ≈ sw_full |
| 8A | identity-contrastive loss | only at content-cost (joint gate fails) |

The four together prove that *content-based slot identity is the
wrong abstraction.* Identity is not a feature that can be carved
out of the same vector that carries dynamic state.

## What this opens

The user-proposed first-principles redesign:

> **Object-File JEPA (OF-JEPA): identity as an address, state as
> content.** Persistent memory cells query observations, are bound
> to specific objects by differentiable assignment, and update
> id_key as an address (slow EMA) and state_value as content
> (sparse delta). Identity comes from the *binding mechanism*,
> not from the content vector.

This becomes Phase 8C.

## Memory entry added

[[feedback-prediction-vs-assignment]] — *prediction learns state,
assignment learns identity, do not expect one loss to learn both.*

## Reproducibility

Code at `Phase 8A scaffolding` commit + identity_contrastive_loss +
cosine_diagnostic in `id_consistency.py`. Partial artifacts at
`artifacts/phase8a_run1/` (v1, λ=0.1, full) and
`artifacts/phase8a_v2_run1/` (partial, λ ∈ {0.3, 1.0} complete,
λ=3.0 seed 0 only).

Run command (v2):

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup bash -c '
    for lam in 0.3 1.0 3.0 10.0; do
      python3 scripts/slot_jepa_movi_train.py \
        --modes id_dyn_split_idcons_idcontrast --seeds '$seed' \
        --max-steps 3000 --id-contrastive-w $lam ...
    done
  ' &
done
```
