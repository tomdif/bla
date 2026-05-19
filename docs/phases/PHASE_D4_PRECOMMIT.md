# Phase D4 — NutAssemblySquare Doctrine Validation (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Gates locked before run.**
**Parent:** `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` regime map.
**Sibling:** `docs/phases/PHASE_D3_MAIN_DECISION.md` (PickPlaceCan, commit `7588e5f`).

## Pivot from Door

The original D4 plan named Door for "articulated dynamics."
**Robomimic v141 registry does not ship Door PH demos** (only
lift / can / square / transport / tool_hang). Pivoted to
**NutAssemblySquare** (precise-insertion task) per user choice
2026-05-19. Square tests a meaningfully different constraint
structure from PickPlaceCan: precise peg insertion vs grasp-and-
place.

## Purpose

Second cross-task validation of the demo-prior doctrine on a task
the regime map was not built on. PickPlaceCan was the first
external test (Phase D3-main); Square is the second.

If demo_no_cem wins again on Square, Recipe E becomes a
**multi-task cross-validated** doctrine. If it fails, the regime
map gets a scope limit at insertion-class tasks.

## Pilot result (locked already, n=5 × 1 seed)

```
Mode             imp_mean  success
demo_no_cem        0.60     3/5   ⭐
phase17_locked     0.20     1/5
naive_cem          0.00     0/5
```

Δ = +0.40. Falsification threshold ±0.05. Cleared by 8×.

## Main protocol

```
Task     : robosuite NutAssemblySquare
Modes    : demo_no_cem, phase17_locked, naive_cem
Seeds    : 0, 1, 2 (parallel on GPUs 0/1/2)
Episodes : 30 per seed per mode
Init     : state-matched (env.sim.set_state_from_flattened from
           demo.states[0]); demo bank = {1, 2, 3, 4, 5}
           (5 of 17 working demos screened — all 17 lift on
           state-matched reset; the bank of 5 mirrors D3 protocol)
Sigma    : per-dim, gripper masked to 0
Plan H   : 30 stride-4 = 120 env steps per plan (apples-to-apples
           with D3-main)
```

## Pre-committed gates

```
G1 (effect size):  demo_no_cem mean improvement ≥ phase17_locked + 0.10
G2 (success):      demo_no_cem success rate ≥ phase17_locked + 10pp
G3 (consistency):  demo_no_cem beats phase17_locked on ≥ 2 of 3 seeds
G4 (floor):        naive_cem near-floor (mean improvement ≤ 0.10)
```

Strong-pass:

```
SP:  demo_no_cem beats phase17_locked by ≥ 0.25 improvement AND
     ≥ 25 percentage-point success on the 3-seed mean.
```

## Pre-committed predictions

Based on the pilot and the D3-main precedent:

```
demo_no_cem:    expected imp 0.55–0.75 (lower than D3-main's 0.72
                because demo timing varies across the 5-demo bank
                and plan_horizon=30 cuts off some lifts).

phase17_locked: expected imp 0.15–0.30 (similar to D3-main's 0.16;
                CEM noise occasionally helps when demo timing
                doesn't fit horizon).

naive_cem:      expected imp ~0.00 (floor; CEM without prior
                cannot generate coordinated grasp+lift+insert).
```

## Falsification scenarios

```
F1: combined CEM modes match demo_no_cem within ±0.02 → CEM-is-
    neutral-around-demos claim weakens, regime map collapses.

F2: demo_no_cem ≤ 0.10 → demo-replay machinery breaks on Square
    (e.g., insertion specifically demands trajectory precision the
    demo bank can't preserve under state-matched reset alone).

F3: phase17_locked beats demo_no_cem → predictor transfer
    outweighs demo manifold preservation. Counter-doctrine result.

F4: demo_no_cem < naive_cem → demo prior makes things worse than
    no prior at all. Hard falsification.
```

## Compute budget

~10 minutes wall time for the 3-seed parallel run (matches D3-main).

## Locked
