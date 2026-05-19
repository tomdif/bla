# Phase Scale-1 — Task-Breadth Regime-Map Benchmark (Precommit)

**Date locked:** 2026-05-19.
**Status:** 🔒 **Router predictions locked before any compute on new tasks.**
**Parents:**
- `docs/BLA_SCALING_ROADMAP.md` (priority order, commit `55074ca`)
- `bla/routing/recipe_router.py` (router commit `668e02c`)

## Purpose

Phase Scale-1 is the doctrine-validating move. Take the BLA
architecture spec's regime map, encode it as the deployed
`recipe_router` function, and check that the router predicts
the empirically-winning recipe on a 6–10 task suite **before**
running each task.

> **The differentiating metric: did the router pick the right
> recipe before any compute on the task?**

## Pass criteria

```
G_pass:   ≥ 70% of tasks: router's predicted recipe matches the
          empirical winner (or near-winner, within ±0.05 imp).
G_strong: ≥ 80%.
```

This is a multi-task gate, not a per-task gate. Individual task
failures are recorded but don't invalidate the doctrine if the
overall accuracy clears the threshold.

## Task suite

7 tasks total — 6 have existing validation as sunk evidence, 1
requires new compute (ToolHang) in this session.

| # | Task | TaskDescriptor | Router → | Status |
|---|---|---|---|---|
| 1 | Lift (single cube) | `demo, contact, narrow_init` | **E1** | ✓ validated (Phase 18κ R3) |
| 2 | PickPlaceCan | `demo, contact, wide_init` | **E2** | ✓ validated (Phase D3-main) |
| 3 | NutAssemblySquare | `demo, contact, wide_init` | **E2** | ✓ validated (Phase D4) |
| 4 | **ToolHang** | `demo, contact, wide_init` | **E2** | **NEW — this session** |
| 5 | Stack push (sim feats) | `fsm, sim_true` | **A** | ✓ validated (Phase 18η-multi) |
| 6 | Stack push (slots only) | `fsm` | **B** | ✓ validated (Phase 18λ) |
| 7 | Stack OOD goal-distance | `fsm, ood` | **D** | partial (Phase 18κ R2: D best of schedule variants) |

## Pre-committed router calls (locked verbatim)

```python
from bla.routing import TaskDescriptor, recipe_router

# 1. Lift
recipe_router(TaskDescriptor(
    prior_kind="demo", contact_sensitive=True,
    init_distribution_wide=False, task_name="Lift"))
# Expected: Recipe.E1

# 2. PickPlaceCan
recipe_router(TaskDescriptor(
    prior_kind="demo", contact_sensitive=True,
    init_distribution_wide=True, task_name="PickPlaceCan"))
# Expected: Recipe.E2

# 3. NutAssemblySquare
recipe_router(TaskDescriptor(
    prior_kind="demo", contact_sensitive=True,
    init_distribution_wide=True, task_name="NutAssemblySquare"))
# Expected: Recipe.E2

# 4. ToolHang (NEW)
recipe_router(TaskDescriptor(
    prior_kind="demo", contact_sensitive=True,
    init_distribution_wide=True, task_name="ToolHang"))
# Expected: Recipe.E2

# 5. Stack push with sim-true features
recipe_router(TaskDescriptor(
    prior_kind="fsm", contact_sensitive=False,
    sim_true_features=True, task_name="StackPush"))
# Expected: Recipe.A

# 6. Stack push without sim-true features
recipe_router(TaskDescriptor(
    prior_kind="fsm", contact_sensitive=False,
    sim_true_features=False, task_name="StackPush-NoSimFeats"))
# Expected: Recipe.B

# 7. Stack OOD goal-distance shift
recipe_router(TaskDescriptor(
    prior_kind="fsm", contact_sensitive=False,
    out_of_distribution=True, sim_true_features=True,
    task_name="StackPush-OOD"))
# Expected: Recipe.D
```

All seven routes are deterministic. The router is at commit
`668e02c`; if any of these calls returns a different recipe at
benchmark scoring time, that's a router regression, not a
doctrine update.

## New compute this session: ToolHang

ToolHang is robomimic v141's longest task (~600 action steps per
demo, 5× longer than Lift/Can/Square). 200 demos available.

**ToolHang test protocol** (mirrors D3/D4):

```
Modes  : demo_no_cem, phase17_locked, naive_cem
Seeds  : 0, 1, 2 (parallel on GPUs 0/1/2)
Init   : state-matched (env.sim.set_state_from_flattened
         from demo.states[0])
Demos  : screen first 20 demos for which lift the tool on
         state-matched reset; use a bank of 5 working ones
Plan H : 50 stride-4 = 200 env steps (longer than D3/D4's
         30 stride-4 = 120 because ToolHang demos are longer)
Sigma  : per-dim, σ_motion=0.12, σ_gripper=0 (locked from
         search-budget-zero-around-expert-demos)
Episodes per mode per seed: 30 (main) / 5 (pilot)
```

ToolHang success metric: tool z-gain ≥ 0.05m (lift phase) for
the doctrine test. Full hang-success deferred.

## Pre-committed ToolHang predictions

```
demo_no_cem:    expected imp 0.40–0.70 (lower than D3/D4 because
                ToolHang is harder and demos may not all clear the
                lift gate; the doctrine claim is still that
                demo_no_cem beats CEM-around-demo).

phase17_locked: expected imp 0.10–0.30 (CEM around the demo
                destroys grasp timing, as in Lift / Can / Square).

naive_cem:      expected imp ~0.00 (floor).
```

Falsification specific to ToolHang:

```
F1: demo_no_cem ≤ 0.20 AND phase17_locked ≤ 0.20 → ToolHang is
    just too hard for our predictor + plan_horizon; result
    inconclusive; doctrine status unchanged but ToolHang doesn't
    count toward Scale-1.

F2: phase17_locked > demo_no_cem + 0.05 → doctrine has a scope
    limit at long-horizon tasks (≥600 steps). Router would need
    a horizon-aware rule.

F3: demo_no_cem ≥ 0.30 AND phase17_locked < demo_no_cem - 0.10 →
    doctrine holds and Scale-1 has 4 of 4 demo-prior validations.
```

## Scale-1 aggregate scoring rule

For each of the 7 tasks, the "empirical winner" is the recipe
that has the highest 3-seed mean improvement (or matches within
±0.05 of the top mode). Match if router prediction = empirical
winner. Score = # matches / # tasks.

```
G_pass:   ≥ 70%  (≥ 5 of 7 tasks)
G_strong: ≥ 80%  (≥ 6 of 7 tasks)
```

Given 6 of 7 tasks already have validation that supports the
router's prediction, ToolHang's outcome is the swing vote:

- ToolHang confirms (E2 wins) → 7 of 7 = 100% (strong pass)
- ToolHang falsifies (F2: phase17_locked wins) → 6 of 7 = 86% (still strong pass on the rest)
- ToolHang inconclusive (F1) → exclude from denominator: 6 of 6 = 100%

So even a ToolHang falsification doesn't kill Scale-1; it would
just refine the regime map with a horizon-aware rule.

## What this precommit is NOT

- Not a claim that ToolHang's specific demos transfer cleanly
  on a fresh env reset (we already know from D3/D4 that
  state-matched init is needed for wide-init tasks).
- Not a claim that the Phase 17 predictor scores ToolHang
  trajectories well (it was trained on Stack push); the doctrine
  test only requires that CEM around the demo hurts on average,
  regardless of score quality.
- Not Phase Scale-2 — model capacity is locked OUT until Scale-1
  passes.

## Locked
