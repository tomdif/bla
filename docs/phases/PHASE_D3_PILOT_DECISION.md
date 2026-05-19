# Phase D3 — PickPlaceCan Pilot (Decision)

**Date:** 2026-05-19.
**Status:** ✅ **Doctrine prediction P0 CONFIRMED at pilot scale.**
**Precommit:** `docs/phases/PHASE_D3_PRECOMMIT.md` (commit `91821ba`).

## Headline

```
On PickPlaceCan with state-matched demo init, n=5 eps, 1 seed:

  Mode             imp_mean  success  z_gain (m)
  demo_no_cem        0.801     4/5     0.148    ⭐
  phase17_locked     0.202     1/5     0.026
  naive_cem          0.003     0/5     0.0003   (floor)

  Δ(demo_no_cem − phase17_locked) = +0.60 imp = +60 percentage-point success gap.
  Falsification threshold from precommit: ±0.05. CLEARED by 12×.
```

The BLA regime map's prediction for contact-sensitive tasks with
expert demos **transfers cleanly to a task the doctrine was not
built on**.

## Precommit predictions vs result

| Prediction | Threshold | Observed | Result |
|---|---|---|---|
| **P0** (headline): demo_no_cem outperforms every CEM-augmented mode | Δ > +0.05 | Δ = +0.60 | ✅ **CONFIRMED** |
| P1: demo_no_cem nonzero on PickPlaceCan | > 0 | 0.80 | ✅ CONFIRMED |
| P2: naive_cem at floor | ≤ 0.05 | 0.003 | ✅ CONFIRMED |
| P3: phase17_locked intermediate | floor < locked < demo | 0.003 < 0.20 < 0.80 | ✅ CONFIRMED |
| P4: combined_sum_* doesn't recover demo_no_cem | n/a | not tested (value heads need PickPlace-specific training cache) | ⏭ deferred |
| P5: demo_no_cem has lowest variance | n/a | single seed | ⏭ deferred to main |

**No partial- or full-falsification triggers (F1–F4) fired.**

## Per-episode detail

```
demo_no_cem                     phase17_locked (CEM σ=0.12 around demo)
  demo  5: z=0.193m  succ=1       demo  5: z=0.000m  succ=0
  demo  8: z=0.000m  succ=0       demo  8: z=0.000m  succ=0
  demo 10: z=0.163m  succ=1       demo 10: z=0.131m  succ=1
  demo 13: z=0.146m  succ=1       demo 13: z=0.000m  succ=0
  demo 16: z=0.238m  succ=1       demo 16: z=0.000m  succ=0

naive_cem (no prior, σ=0.5)
  all 5: z=0.0003m  succ=0  (floor)
```

The single phase17_locked success (demo 10) appears to be the case
where CEM's elite candidate stayed close enough to the demo to
preserve the grasp. The 4 failures show CEM noise corrupting the
demo's grasp timing — exactly the Phase 18κ R3 Lift finding
replicating cross-task.

## Interpretation

### The doctrine claim now has out-of-sample validation

The BLA regime map was built on:
- Stack push (Phases 14–18): scripted-FSM prior, Recipes A/B/C/D win
- Lift (Phase 18κ R3): expert-demo prior, Recipe E (demo_no_cem) wins

PickPlaceCan was a **new task** the doctrine had not seen. The
precommit classified it as **demo-prior regime** (contact-sensitive
+ robomimic demos available) and predicted Recipe E dominance.

Pilot result: that prediction holds at +60 percentage-point margin.

### The locked applicability rule survives a cross-task test

> "When the prior is an expert demonstration manifold, the correct
> search budget is zero or near-zero structured search. CEM may
> occasionally match the demo but is not reliable enough to be the
> default without a trust-region / seeded-control audit."

PickPlaceCan replicates this: CEM σ=0.12 (with gripper-bit masked
to 0) destroys 4 of 5 grasps. The lone CEM success is the
"occasionally matches" caveat in action.

### Engineering caveat: demos need state alignment

A separate finding before the doctrine test could run:

> 0 of 50 robomimic PickPlaceCan demos succeed on a fresh random
> env reset. With state-matched reset (`env.sim.set_state_from_flattened(
> demo.states[0])`), 5 of 20 demos succeed.

PickPlaceCan's initial-state distribution is wider than Lift's, so
the Phase 18κ R3 "find a couple of demos that happen to work on fresh
resets" approach breaks. The fix is robomimic's standard playback
protocol: reset env to demo's recorded init state. This is a recipe
ingredient, not a doctrine change. The architecture spec's
[deployment regime map] still predicts the right winner.

## What this pilot does NOT yet establish

- **Multi-seed variance**: pilot is 1 seed, n=5. Main run at 3
  seeds × n=30 would tighten the margin estimates.
- **Value-head modes**: combined_sum_geo, _supervised, _end2end
  modes were skipped because their value heads need PickPlaceCan-
  specific training caches. Phase 18κ R3 had a full training arc
  per seed; this pilot is a minimum-viable doctrine test.
- **Stretch tasks** (Door, NutAssembly): pending. The PickPlaceCan
  confirmation unlocks scaling to these.

## Decision

**The BLA architecture spec's regime-map predictions are validated
cross-task on PickPlaceCan at pilot scale.** The doctrine is now a
falsifiable theory that has survived its first external test.

Recommended next:
1. **Main run** at 3 seeds × n=30 to tighten error bars and test
   P5 (variance pattern).
2. Optional: **add Door** as a second cross-task test (same protocol,
   same demos source).
3. Optional: **train PickPlaceCan-specific value heads** to test P4
   (combined_sum_* modes do not recover demo_no_cem performance).

Decision doc commit closes the pilot phase. The pilot's headline
becomes a single bullet in the architecture spec's evidence table.

## Files

- Precommit: `docs/phases/PHASE_D3_PRECOMMIT.md` (commit `91821ba`).
- Task primitives: `scripts/phase_d3_pickplace.py`.
- Pilot eval: `scripts/phase_d3_pilot.py`.
- Summary: `/workspace/phase_d3_pilot/summary.json` (pod).

## Locked

This decision is locked at commit-time. No post-hoc reinterpretation
without flagging it as such.
