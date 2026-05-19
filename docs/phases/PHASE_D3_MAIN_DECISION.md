# Phase D3-main — PickPlaceCan Main Sweep (Decision)

**Date:** 2026-05-19.
**Status:** ✅ **STRONG PASS — all 4 gates cleared, strong-pass criterion triggered.**
**Precommit:** `docs/phases/PHASE_D3_MAIN_PRECOMMIT.md` (commit `c74ef97`).
**Pilot decision:** `docs/phases/PHASE_D3_PILOT_DECISION.md` (commit `df52972`).

## Headline

> **Phase D3-main confirms the demo-prior doctrine on PickPlaceCan.
> `demo_no_cem` beats the locked OF-JEPA + CEM planner by +0.564
> improvement and +56.7pp success across 3 seeds × 30 episodes,
> while naive CEM remains at floor. The transferable object is the
> demonstration manifold, not action-space search around it.**

## Three-seed aggregate (n=30 per seed per mode)

| Mode | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| **demo_no_cem** | imp 0.722 / 70% | imp 0.683 / 67% | imp 0.767 / 77% | **0.724 ± 0.043 / 71% ± 5pp** |
| phase17_locked | imp 0.107 / 10% | imp 0.171 / 13% | imp 0.202 / 20% | 0.160 ± 0.048 / 14% ± 5pp |
| naive_cem | 0.003 / 0% | 0.003 / 0% | 0.003 / 0% | 0.003 / 0% (floor) |

Pilot result (n=5) had Δ=+0.60. Main run (n=30 × 3 seeds) has
**Δ=+0.564**. Effect size held at scale.

## Gate evaluation

| Gate | Threshold | Observed | Result |
|---|---|---:|---|
| **G1** (Δ_imp ≥ +0.10) | +0.10 | **+0.564** | ✅ PASS by 5.6× |
| **G2** (Δ_succ ≥ +10pp) | +10pp | **+56.7pp** | ✅ PASS by 5.7× |
| **G3** (demo > locked on ≥ 2/3 seeds) | 2/3 | **3/3** | ✅ PASS |
| **G4** (naive_cem ≤ 0.10) | ≤ 0.10 | 0.003 | ✅ PASS |
| **Strong-pass** (Δ≥+0.25 AND Δ_succ≥+25pp) | both | Δ=+0.564, Δ_succ=+56.7pp | ✅ **YES** |

## Variance pattern (precommit P5 confirmed)

```
demo_no_cem    σ = 0.043   (most reliable nontrivial mode)
phase17_locked σ = 0.048   (CEM noise consistent with prior runs)
naive_cem      σ = 0.000   (deterministic floor)
```

`demo_no_cem` is **not only the highest-performing mode but also
the most reliable**. The recipe is doing what the doctrine
predicts: the demonstration manifold is the stable policy object,
and search around it adds variance without adding signal.

## What this proves

The D3 doctrine claim now spans **two independent contact-sensitive
manipulation tasks**:

```
Lift           (Phase 18κ R3, 3 seeds × n=30 + 1 rerun)
PickPlaceCan   (Phase D3-main, 3 seeds × n=30)
```

Across both tasks, with different cube counts (1 vs 1 + distractors),
different success criteria (z-gain vs lift-and-place), different demo
banks (lift_replay vs can_v141), and different state-init protocols
(fresh-reset-screen vs state-matched-init), the result is the same:

> In demo-prior / contact-sensitive regimes, the demonstration
> manifold is the transferable object. CEM exploration around it
> is not the default.

The effect size on PickPlaceCan (Δ=+0.564) is even larger than on
Lift (Δ≈+0.10 vs `phase17_locked` at the four-run aggregate). This
suggests the doctrine *strengthens* on tasks where the demo is more
critical (PickPlaceCan grasp-and-place vs Lift's grasp-and-lift).

## Updated deployment regime map

```text
Stack / FSM-prior regime:
  OF-JEPA + action predictor + light CEM + value guidance.

Demo-prior / contact-sensitive regime:
  demo_no_cem is the DEFAULT.
  Do not add CEM unless there is a calibrated trust-region reason
  to do so.

OOD goal-shift regime:
  end2end or pretrain+ft adapter remains preferred candidate.
```

This is now a deployment-conditional doctrine that has survived
out-of-sample testing on two tasks. Update target:
`docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §5 should incorporate
this stronger statement.

## What this does NOT yet establish

- **P4 (combined_sum_\* doesn't recover demo_no_cem)** — value-head
  modes require PickPlaceCan-specific training caches; deferred.
- **Door / NutAssembly** — not yet tested. PickPlaceCan and Lift are
  both pick-style tasks. Door tests articulated dynamics.
- **Tasks where CEM SHOULD win** — the doctrine predicts FSM-prior
  regimes (Stack push family) favor Recipes A-D. This was validated
  by Phases 14-18 originally; no need to re-test, but a fresh
  FSM-prior task could strengthen the regime map's other branch.

## Decision

**The doctrine's cross-task validation is now strong.** The recipe
map has predictive power outside its original task family.

Locking three downstream actions:

1. **Update the architecture spec** (`BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md`)
   to reflect the strengthened regime map.
2. **D4 — Door regime-map validation** as the next stretch. Predict
   the regime, lock predictions, then run.
3. **Defer P4 value-head recovery test.** The doctrine is strong
   enough that the falsification test "can value-guided CEM ever
   recover `demo_no_cem` in demo-prior regimes?" can wait until
   after the regime map is mapped to more tasks.

## Why D4 (Door) next, not P4

Door adds **contact + articulated dynamics** — a different
constraint structure than Lift / PickPlaceCan. If `demo_no_cem`
wins on Door too, Recipe E becomes a strong cross-task doctrine.
If it fails, the regime map becomes more precise (the doctrine has
a scope limit at articulated tasks). Either outcome is high-value.

P4 is a finer-grained falsification test that's worth running
later, after the regime map's coverage is clearer.

## Files

- Precommit: `docs/phases/PHASE_D3_MAIN_PRECOMMIT.md` (commit `c74ef97`)
- Decision: this file
- Task primitives: `scripts/phase_d3_pickplace.py`
- Pilot/main eval: `scripts/phase_d3_pilot.py`
- Pod artifacts: `/workspace/phase_d3_main_seed{0,1,2}/summary.json`

## Locked
