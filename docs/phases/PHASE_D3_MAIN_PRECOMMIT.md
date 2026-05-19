# Phase D3-main — PickPlaceCan Main Sweep (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Gates locked before run.**
**Parents:**
- `docs/phases/PHASE_D3_PRECOMMIT.md` (pilot precommit, commit `91821ba`)
- `docs/phases/PHASE_D3_PILOT_DECISION.md` (pilot result, commit `df52972`)

## Purpose

The pilot (n=5, 1 seed) confirmed P0 at Δ=+0.60. The main run
tightens error bars and tests whether the effect holds across
seeds and at 6× more episodes per mode.

## Protocol

```
Task     : robosuite PickPlaceCan
Modes    : demo_no_cem, phase17_locked, naive_cem
Seeds    : 0, 1, 2 (parallel on GPUs 0/1/2)
Episodes : 30 per seed per mode
Init     : state-matched (env.sim.set_state_from_flattened from
           demo.states[0]); demo bank = {5, 8, 10, 13, 16}
           (5 demos that lift can on own init state)
Sigma    : per-dim, gripper masked to 0 (locked from Phase 18κ R3)
```

CEM hyperparameters: `K=32, iters=1, σ_motion=0.12, σ_floor=0.05`.

## Pre-committed gates

```
G1 (effect size):  demo_no_cem mean improvement ≥ phase17_locked + 0.10
G2 (success):      demo_no_cem success rate ≥ phase17_locked + 10pp
G3 (consistency):  demo_no_cem beats phase17_locked on ≥ 2 of 3 seeds
G4 (floor):        naive_cem near-floor (mean improvement ≤ 0.10)
```

Strong-pass criterion:

```
SP:  demo_no_cem beats phase17_locked by ≥ 0.25 improvement AND
     ≥ 25 percentage-point success on the 3-seed mean.
```

## Pre-committed predictions

Based on the pilot:

```
demo_no_cem:    expected imp 0.70–0.85, success 0.65–0.85, low/zero variance
                across seeds (demo replay is deterministic; only the choice
                of demo per episode varies, and that's controlled by ep_id
                modulo |demo_bank|).

phase17_locked: expected imp 0.15–0.35, success 0.10–0.30, moderate variance.

naive_cem:      expected imp ~0.00, success ~0.00, low variance (CEM
                without prior cannot generate coordinated grasp).
```

## Falsification ledger (filled in by decision doc)

```
G1 (Δ_imp ≥ +0.10):     ___ pass / fail / by how much
G2 (Δ_success ≥ +10pp): ___ pass / fail / by how much
G3 (2/3 seeds):         ___ pass / fail
G4 (naive ≤ 0.10):      ___ pass / fail

Strong pass triggered?  ___ yes / no
```

## Stretch contingent on pass

If main run passes all gates:
1. Add **Door** as a second cross-task test (same protocol).
2. Optional: train PickPlaceCan-specific value heads for P4
   (combined_sum_* modes don't recover demo_no_cem).

If main run partially fails:
- Document which gate failed and what it tells us about the doctrine
  scope.

If main run fully fails (G1/G2 inverted):
- The pilot was a small-n artifact. The doctrine's cross-task scope
  is narrower than claimed. Write a retraction.

## Locked
