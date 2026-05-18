# Phase 18ν — Scheduled aux loss (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18κ Regime 2 revealed an inversion:
- **In-distribution**: supervised > geo > end2end (Phase 18λ-v2)
- **OOD goal-distance**: end2end > geo > supervised (Phase 18κ R2)

The supervised aux loss is a **distribution-dependent inductive
bias**: helpful in-dist (strong prior on training geometry),
constraining OOD (binds adapter to training-distribution geometry).

The architectural fix is to **mix or schedule** the auxiliary loss
so the adapter gets the geometry-target structure during early
training (in-dist sharpness) but is freed late in training
(OOD generalization).

## The question

> Can a scheduled / pretrain-finetune training procedure match
> supervised in-distribution AND match-or-beat end2end out-of-
> distribution?

## Variants

| Variant | Procedure | Reference Phase |
|---|---|---|
| **A. supervised** | Adapter trained slot→geo MSE; VH trained on adapter output with episode_imp MSE | Phase 18λ-v2 (reference) |
| **B. end2end** | Adapter+VH trained jointly with episode_imp MSE only | Phase 18λ-v2 (reference) |
| **C. annealed** | Same architecture as end2end, but loss = λ(t)·geo_MSE + (1−λ(t))·value_MSE, where λ linearly anneals 1.0 → 0.0 over training | NEW |
| **D. pretrain+ft** | Phase 1 (1000 steps): adapter trained on geo MSE only. Phase 2 (1000 steps): adapter+VH fine-tuned jointly on value MSE (small lr) with optional small residual geo regularizer | NEW (user's preferred candidate) |

All variants use the same 10-dim latent dim and same MLP widths for
fair architectural comparison.

## Pre-committed gates

Evaluated at TWO eval distributions per seed:
- **In-distribution**: goal-dist ∈ [0.05, 0.08]
- **OOD**: goal-dist ∈ [0.10, 0.15] (Phase 18κ R2 distribution)

```
G1 (in-distribution capacity).
   mean(combined_sum_scheduled) >= 0.95 × mean(combined_sum_supervised)
       (scheduled recipe captures supervised's in-dist performance
        within 5%)

G2 (OOD generalization).
   mean(combined_sum_scheduled) >= 0.95 × mean(combined_sum_end2end on OOD)
       (scheduled recipe captures end2end's OOD performance within 5%)

G3 (stability).
   std(combined_sum_scheduled) across seeds <= 1.25 × std(end2end)
       (no worse seed variance than end2end)

G4 (adapter quality diagnostic).
   adapter geo-recovery Spearman across seeds >= 0.40
       (the adapter still extracts geometric structure, even if
        less than supervised's 0.50)
```

The "scheduled" recipe in the gates is whichever of C / D performs
best across both regimes. If neither variant passes both G1 and G2,
neither captures both regimes; if both pass, prefer D
(pretrain+ft) as the user's identified strongest candidate.

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3 + G4** | Scheduled aux is the unified recipe; supersedes both supervised and end2end. Lock as the new default. |
| G1 + G3 + G4 (G2 fail) | Scheduled aux matches supervised in-dist but not end2end OOD. Use supervised for in-dist deployment. |
| G2 + G3 + G4 (G1 fail) | Scheduled aux matches end2end OOD but not supervised in-dist. Use scheduled aux for OOD deployment. |
| 0/4 or 1/4 | Scheduled approach doesn't help; keep distribution-conditional choice from Phase 18κ R2. |

## What this phase is NOT

- Not a new architecture (same End2EndAdapterValue / supervised
  adapter classes; only training procedure changes).
- Not a new task (Stack push, as before).
- Not yet multi-seed locked — single 3-seed run first; expand to
  5 seeds if results are promising.

## Implementation sketch

New training routines in `system1_jepa/geometry_adapter.py`:

```python
def train_end2end_with_aux_schedule(
    model, slots, goals, plans, labels, target_geo,
    *,
    schedule_kind: str,     # "linear_anneal" or "pretrain_ft"
    steps: int = 2000,
    pretrain_steps: int = 1000,  # for pretrain_ft
    aux_weight_init: float = 1.0,
    aux_weight_final: float = 0.0,
    aux_residual_weight: float = 0.05,  # for pretrain_ft late phase
    ...
) -> stats
```

New script `scripts/phase18nu_scheduled_aux.py`:
- Per seed: load cache; train 4 heads (A, B, C, D).
- Eval at both distributions (8 modes × 30 ep × 2 dists × 3 seeds).

Pod budget per seed: 4 heads × 3 min = 12 min training + 8 modes ×
30 × 2 dists × 15s = 60 min eval = 72 min wall. Three seeds parallel
= 72 min wall.

## Reproducibility

Pre-committed gates: this file.

Decision doc: `docs/phases/PHASE_18N_SCHEDULED_AUX_DECISION.md`
(after gates evaluated).

Artifacts: `artifacts/phase18nu_multi/{aggregate.json,
summary_seed{0,1,2}.json}`.

## Sibling memory

- `[[bla-locked-planning-recipe]]` — three co-locked peers; 18ν
  tests whether a unified recipe replaces the choose-by-deployment
  pattern
- `[[aux-loss-distribution-dependent]]` — Phase 18κ R2 lesson;
  18ν is the constructive follow-up
- `[[engineered-aux-loss-useful-inductive-bias]]` — Phase 18λ-v2;
  qualified as in-distribution only
