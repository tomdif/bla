# Phase D3 — Cross-Task Doctrine Validation (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Precommit — predictions locked before any compute.**
**Source spec:** `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` (commit `c33b556`).

## Purpose

The BLA System-1 architecture spec (D2) made falsifiable claims about
which recipe wins in which deployment regime. Phase D3 puts those
claims to a real test by running on **new** robosuite tasks beyond
the Stack-push (Phases 14–18) and Lift (Phase 18κ R3) tasks that
generated the doctrine.

This document **locks the predictions** before any compute, so that
positive results aren't rationalized after the fact and negative
results aren't quietly retro-fit.

## The doctrine being tested

From `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §5:

```
Scripted FSM prior (noisy, headroom)
  → Recipes A / B / C / D family (light CEM + value head)
Expert demo prior (narrow, contact-sensitive)
  → Recipe E (demo_no_cem, no perturbation)
```

And the locked applicability rule:

> When the prior is an expert demonstration manifold, the correct
> search budget is zero or near-zero structured search.

## Tasks under test

Candidates (robosuite + robomimic-compatible):

| Task | Contact-sensitive? | Scripted FSM viable? | Expert demos? | Regime |
|---|---|---|---|---|
| **PickPlaceCan** | YES (grasp + place) | Hard (multi-stage) | YES (robomimic ph) | **demo-prior** |
| Door | YES (handle grasp + rotate) | Hard (rotational constraint) | YES (robomimic ph) | **demo-prior** |
| NutAssembly | YES (peg alignment) | Very hard (precise) | YES (robomimic ph) | **demo-prior** |
| (control) Stack push | NO (push only) | YES (FSM works) | n/a (already done) | **FSM-prior**, validated |

All three new candidates fall in the **demo-prior** regime by the
doctrine's classification. They are all contact-sensitive and have
robomimic expert demos available.

### Pilot scope (this precommit)

**Pilot task:** PickPlaceCan (closest to the Stack/Lift family the
arc was built on; robomimic demos exist).

**Stretch:** if PickPlaceCan pilot confirms the prediction, run
the same protocol on Door and/or NutAssembly. If PickPlaceCan
*falsifies*, escalate to a careful debug before adding tasks.

## Pre-committed predictions

### Headline prediction (the doctrine's core claim on PickPlaceCan)

```
P0 (HEADLINE):
  On PickPlaceCan, Recipe E (demo_no_cem) outperforms every
  CEM-augmented mode at n=5 pilot and n=30 main.
```

**Falsification criterion (P0):** if any CEM-augmented mode
(phase17_locked, combined_sum_supervised, combined_sum_geo,
combined_sum_end2end) beats demo_no_cem at n=30 mean by more than
**+0.05 absolute improvement**, the headline doctrine is falsified
for PickPlaceCan.

### Supporting predictions

```
P1: demo_no_cem (Recipe E) achieves nonzero success rate on
    PickPlaceCan at n=30 (i.e., the demo prior + per-dim sigma
    masking transfers as engineering substrate).

P2: naive_cem (no prior, no value head) is at floor (success ≤ 0.05).

P3: phase17_locked (CEM + Phase 17 predictor + no value head) is
    intermediate: it benefits from the predictor scoring but is
    hurt by CEM perturbing the demo prior. Expected:
    floor < phase17_locked < demo_no_cem.

P4: combined_sum_supervised (Recipe B with retrained adapter on
    PickPlaceCan rollouts) does NOT recover demo_no_cem-level
    performance. The "value head doesn't rescue CEM around demos"
    finding from Phase 18κ R3 should replicate.

P5: variance pattern: demo_no_cem has the LOWEST std across
    seeds; CEM-augmented modes have wider variance.
```

### What would PARTIALLY falsify the doctrine

```
F1: If combined_sum_* modes match demo_no_cem within ±0.02 on
    PickPlaceCan, the doctrine's "CEM hurts demo priors" claim
    weakens to "CEM is neutral around demo priors." The regime
    map collapses to two regions instead of two strong claims.

F2: If demo_no_cem fails to lift / place at all (success near 0),
    the demo-replay machinery is task-specific and the doctrine's
    cross-task scope shrinks to "tasks similar to robosuite Stack/Lift."

F3: If phase17_locked beats demo_no_cem, the predictor's transferable
    scoring outweighs the demo manifold's preservation — interesting
    counter-finding worth its own decision doc.
```

### What would COMPLETELY falsify the doctrine

```
F4: If demo_no_cem is WORSE than naive_cem on PickPlaceCan (no
    prior at all beats demo prior), the "transferable object is
    the demonstration manifold" claim is wrong.
```

## Experimental protocol

### Modes evaluated

Pilot (n=5 eps, 1 seed): 5 modes
- `demo_no_cem` — Recipe E
- `phase17_locked` — CEM + Phase 17 predictor only
- `combined_sum_supervised` — Recipe B with task-specific adapter
- `naive_cem` — no prior, CEM only (floor)
- `gt_oracle_pickplace` (if implementable) — scripted FSM oracle

Main (n=30 eps × 3 seeds): same 5 modes.

### Per-dim CEM sigma

```
sigma = [σ_motion, σ_motion, σ_motion, σ_motion, σ_motion, σ_motion, 0]
σ_motion = 0.12 (Phase 17 locked value)
σ_gripper = 0 (locked from Phase 18κ R3 CEM-preserves-semantic-channels)
```

### Improvement metric

Two candidate definitions:
- **lift_phase**: did the can leave the table by ≥ 0.05 m? → 1.0 / 0.0
- **place_phase**: did the can land in the target bin? → 1.0 / 0.0

The headline gate uses `place_phase` (binary success). `lift_phase`
is a secondary diagnostic.

### Compute budget

- Pilot: 5 modes × n=5 = 25 episodes × ~30 s/episode ≈ 12 min.
- Main: 5 modes × n=30 × 3 seeds = 450 episodes ≈ 4 hours.

Total budget: ~5 hours of pod GPU time for full PickPlaceCan
validation. Stretch to Door / NutAssembly only if PickPlaceCan
confirms (otherwise debug first).

## What this precommit is NOT

- Not a guarantee that demo replay works on PickPlaceCan
  out-of-the-box. The demos may need their own gripper-mask /
  per-dim-sigma calibration.
- Not a comparison against a from-scratch task-specific BC policy.
  The doctrine is about *transfer of the BLA recipe family*, not
  about absolute peak performance.
- Not a claim about Door/NutAssembly. Those tasks are stretch
  goals contingent on PickPlaceCan confirming.

## Falsification ledger (filled in by decision doc, NOT now)

```
P0 HEADLINE: ___ confirmed / falsified
P1: ___
P2: ___
P3: ___
P4: ___
P5: ___
F1..F4: ___ which (if any) triggered
```

## Locked

This precommit is locked at commit-time. Any later interpretation
or post-hoc re-framing of the predictions must be flagged in the
decision doc as "post-hoc," not as "predicted by the precommit."
