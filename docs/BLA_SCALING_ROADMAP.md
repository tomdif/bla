# BLA Scaling Roadmap — From Research Arc to Deployable System

**Date locked:** 2026-05-19.
**Parent:** `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` (architecture
spec, doctrine-locked at commit `a500012` after Phase D4 close).
**Status:** Strategy — no compute commitments. Operational details
(funding, hiring, 90-day) are the project owner's planning; this
doc records the architectural priority order.

## One-sentence strategy

> Scale BLA by turning its discoveries into a regime-aware object-
> file world-model system: OF-JEPA object files for state, structured
> adapters for value, learned/demo priors for action, and a **regime
> router** that decides when to search and when to stay on the
> demonstration manifold.

The headline differentiating metric:

> **Show that BLA can choose the right recipe before running a new
> task** — regime prediction accuracy.

## Priority order

```
1. Consolidate architecture.                  [DONE — architecture spec + recipe map]
2. Validate recipe map across more tasks.     [partial — 3 demo-prior tasks done; need breadth]
3. Build rolling-window / stateful runtime.   [pending — D1b → stateful encode_step]
4. Learn priors / demo retrieval.             [pending — replace CEM with demo retrieval]
5. Move to real-world BLA-Forge.              [pending — physical testbed]
6. Then scale model size and data.            [explicitly DO NOT lead with this]
```

Anti-pattern to avoid: **jumping to bigger neural networks**. The
current advantage is architectural (object files + recipe selection
+ regime map). Scale should preserve that advantage, not bury it.

## What the doctrine has already established

```
Lift              demo_no_cem wins
PickPlaceCan      demo_no_cem wins strongly
NutAssemblySquare demo_no_cem wins strongly, lowest variance

Recipe map predicts the right winner across 3 independent
contact-sensitive task families with zero falsification triggers.
```

This is no longer just a finding; it is a deployment rule.

## §1 The Recipe Router (new component)

Turn the regime map into a deployable decision system:

```
if task is FSM-prior / broad action search:
    use OF-JEPA + light CEM + value head + combined_sum

if task is demo-prior / contact-sensitive:
    use demo_no_cem (Recipe E)

if task has OOD goal shift:
    use end2end or pretrain+ft adapter (Recipe C / D)

if simulator-true features are available:
    use engineered geo or supervised adapter (Recipe A / B)
```

Target package structure (next session deliverable):

```
bla/
  system1/              OF-JEPA v0 + action predictor
  planning/             CEM with prior + value scoring
  recipes/              A / B / C / D / E definitions
  routing/              recipe_router(task_descriptor) → recipe
  eval/                 metrics + benchmark harness
docs/
  BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md
  BLA_SCALING_ROADMAP.md                ← this file
  phases/PHASE_*.md
```

## §2 Phase Scale-1: task-breadth benchmark

Pre-commit recipe predictions on a 6–10 task suite, then run:

```
Task                  Predicted recipe
Lift                  demo_no_cem      ← already validated
PickPlaceCan          demo_no_cem      ← already validated
NutAssemblySquare     demo_no_cem      ← already validated
Door                  TBD (demo_no_cem or trust-region demo)
ToolHang              TBD
TwoArmLift            TBD (or simpler variant)
NutAssemblyRound      demo_no_cem
PickPlace (full)      demo_no_cem
Stack (push variant)  OF-JEPA + value + light CEM   ← FSM-prior
Push                  OF-JEPA + value + light CEM   ← FSM-prior
OOD goal-distance     end2end / pretrain+ft
```

**Pass condition:** recipe map predicts winner or near-winner on
≥70% of tasks. **Strong pass:** ≥80% correct recipe selection.

This is the most important scaling milestone. It converts BLA from
"we found some good recipes" into "we have a theory of embodied
world-model deployment."

## §3 Streaming runtime ladder (Phase D1b → D3-runtime)

Already in `BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §3. The ladder:

```
v0: batched encode             ← works today (Demo A validates)
v1: rolling window (K=5–8)     ← D1b, next engineering step
v2: stateful encode_step       ← required for real robotics
```

Make v2 a top engineering priority **after** the consolidation +
Phase Scale-1 task-breadth benchmark.

## §4 From scripted/demo priors to learned priors

The doctrine says: don't replace demos with CEM. The scaling claim:
**learn how to retrieve, time-warp, and lightly adapt demos**, not
how to search around them.

Four proposal-policy variants to learn:

```
Policy A: broad FSM proposal           (FSM-prior regime)
Policy B: demo manifold replay/retrieval (demo-prior regime; the
          most important to scale Recipe E)
Policy C: residual around demonstration   (small structured trust region)
Policy D: OOD / end2end value-guided      (OOD regime)
```

Policy B is the highest-leverage. Sketch:

```
demo retrieval prior:
  given (current state, goal),
    retrieve nearest demo SEGMENT in the bank,
    optionally time-warp / lightly adapt.
```

This scales Recipe E without re-introducing destructive CEM noise.

## §5 Data engine

Three labeled datasets, not just "more frames":

```
A. Broad intervention data
   random pushes / scripted pushes / failed pushes / contact + no-
   contact / goal-directed variations
   → for OF-JEPA + action predictor

B. Expert demo manifold data
   Lift / PickPlace / NutAssembly / Door / ToolHang demos
   → for Recipe E + demo retrieval

C. OOD regime data
   goal-distance / object-size / friction / camera / init-distribution shifts
   → for end2end / pretrain+ft adapters
```

Per-episode labels needed for the recipe router:

```
regime type
chosen recipe
actual outcome
counterfactual candidates
whether search helped or hurt
```

## §6 Model capacity (LAST, not first)

Scale capacity only after the regime map holds at task-breadth.
Suggested progression:

```
near-term:    8–32 object files, hidden 256–768
next scale:   32–64 files, hidden 768–1024, multi-camera, longer T,
              learned proposal policies
serious:      64–256 files (hierarchical), video encoder front-end,
              stateful encode_step, multi-task incl. real data
```

**Locked lesson:** *scale capacity through structured memory and
routing, not raw slot count.* This is the durable lesson from
Phases 7–9.

## §7 Real-world: BLA-Forge

Physical testbed (deferred concrete; this is the project owner's
build plan):

```
fabrication:       Prusa CORE One INDX or equivalent
intervention:      Avid CNC / open gantry
sensing:           overhead camera + toolhead camera
end-effectors:     pusher / gripper / magnet / suction / probe
fixtures:          printed objects + jigs
```

**Real-world goal — NOT general robotics.** It is:

> Can the same recipe map predict when to use search versus demo
> replay in the real world?

Start with the tasks where BLA already has doctrine: push to target,
pick/place, peg/nut insertion, tool / ramp, occlusion tracking,
counterfactual push demos.

## §8 BLA benchmark suite

A benchmark that reflects this thesis (not generic robotics
benchmarks). Working name:

> **BLA Object-File World Model Benchmark**

Metrics:

```
identity-conditioned tracking
object-file stability
action-conditioned prediction
recipe selection accuracy         ← the unique metric
planner improvement
demo-manifold reliability
search harm rate                  ← novel; capture CEM-destructive cases
OOD adaptation
streaming runtime stability
```

Core comparisons:

```
OF-JEPA stack (BLA)
dense JEPA
slot_delta / slot_dense_update
Dreamer-style latent baseline
demo_no_cem  (BC-flavor baseline)
CEM-only
BC-only
```

## §9 90-day target sequence

(Owner's plan; recorded but not driven by Claude:)

```
Weeks 1–2:   architecture doc; clean code; recipe router; rolling-window demo
Weeks 3–5:   5–8 task regime-map benchmark; precommit predictions
Weeks 6–8:   learned demo retrieval prior; residual adaptation
Weeks 9–12:  BLA-Forge testbed; stateful encode_step prototype
```

## Bottom line

The arc landed where it was supposed to. Scaling proceeds by:

```
1. Consolidate architecture.            DONE
2. Validate recipe map across tasks.    NEXT
3. Build streaming runtime.             follow
4. Learn demo retrieval priors.         follow
5. Real-world BLA-Forge.                follow
6. Then scale model size/data.          do not lead with this
```

Concrete next technical milestone:

> **Show that BLA can choose the right recipe BEFORE running a new
> task.** Recipe-selection accuracy is the differentiating metric.

Until that's validated at task-breadth (Phase Scale-1), no model
size scaling.
