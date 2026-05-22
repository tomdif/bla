# BLA-Forge — Real-World Testbed Spec

**Date:** 2026-05-19.
**Status:** v0 planning document. Locks the scope, gates, and
interfaces before any hardware decisions are made.
**Parent docs:**
- `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` (architecture)
- `docs/BLA_SCALING_ROADMAP.md` §6 (priority order: real-world
  before model capacity scaling)
- `docs/phases/PHASE_DR3_DECISION.md` (why we're moving off
  simulator)

## Purpose

The simulator phase has reached its natural boundary. DR1+DR2+DR3
exhausted the metric / coverage / rerank axes; the remaining
variance in Recipe E is execution stochasticity specific to
robosuite. The architecture is now mature enough that real-world
testing is the next bottleneck, not premature.

**BLA-Forge is NOT a general robotics project.** Its single
purpose is:

> Does the BLA recipe map predict when to use search vs demo
> replay under real physical variance — the same way it does in
> simulator?

If yes, BLA's doctrine generalizes beyond a specific simulator
noise model. If no, the simulator validations were over-fit to
robosuite's stochasticity, and the doctrine needs revision.

## What this doc is NOT

- A hardware purchase order. Part numbers below are placeholders.
- A build manual. The owner does the physical build.
- A claim about timelines or budget.
- A robotics PhD curriculum. We are testing one architectural
  claim on the smallest hardware that can test it.

## §1 First physical benchmark — minimum-viable tasks

Start with 3–4 tasks that have direct simulator analogues. Do
**not** start with full pick-and-place or humanoid manipulation.

```
Task 1 — Push to target
  Object: printed block on a flat surface
  Action: pushing with a passive pusher tip
  Sim analogue: Stack push (Phases 14–18), FSM-prior regime
  Predicted recipe (router): A or B (FSM-prior + sim-true features
    or supervised adapter respectively, depending on whether
    sim-true features = recorded human-engineered geometry are
    available)

Task 2 — Pick / place simple object
  Object: printed cube or pre-existing object with grasp affordance
  End-effector: simple gripper (or magnet for ferrous proxy)
  Sim analogue: Lift / PickPlaceCan
  Predicted recipe: E2_FAST or E2_STABLE
    (E1 narrow init / E2 wide init depending on the setup)

Task 3 — Peg/nut insertion
  Object: printed peg + matching slot
  Sim analogue: NutAssemblySquare
  Predicted recipe: E2_FAST or E2_STABLE

Task 4 — Occlusion tracking demo (no manipulation required)
  Object: two distinguishable colored blocks; an opaque occluder
    passes between them and the camera
  Sim analogue: Demo A (Phase D1 legibility) + 9B visibility
  Falsification check: do the object-file slots track identity
    through a real occlusion event?
```

Tasks 1–3 are doctrine tests; Task 4 is a runtime/perception test.

## §2 Pre-committed gates

These are the falsifiable claims that justify BLA-Forge as a
phase, before any hardware exists. **These should not be edited
after the build begins** without flagging the change.

```
G1 (doctrine transfer):
   On Task 2 or 3 (contact-sensitive with demos):
     E2_FAST or E2_STABLE achieves success rate ≥ random-CEM + 10pp.

G2 (recipe E2 variance ordering):
   E2_STABLE success-rate std (across runs) < E2_FAST std,
   matching the simulator finding.

G3 (rolling-window runtime fidelity):
   On the moving cube of Task 1 or 2:
     OF-JEPA rolling-window K=5 mean per-frame cubeA decode error
     ≤ 5× simulator number (allow real-world camera/calibration
     overhead). Target: ≤ 25 mm.

G4 (router correctness):
   On all three contact-sensitive tasks (2, 3, plus Task 1 if
   it lands in the FSM-prior regime), the recipe_router selects
   the empirically-winning recipe on ≥ 70% of tasks. Same
   threshold as Phase Scale-1.
```

Strong-pass: all 4 gates clear AND demonstration video makes the
identity tracking visible (Task 4 + on-screen slot labels during
Tasks 1–3).

## §3 Hardware stack (placeholder structure)

Owner fills in specifics; this captures the minimum architectural
ingredients.

```
3.1  Fabrication:
       Prusa CORE One INDX  (or equivalent 3D printer)
     Purpose: print objects, fixtures, occluders, jigs.

3.2  Intervention platform:
       Cartesian gantry  (Avid CNC style) OR equivalent 3-DOF arm
     Workspace: ≥ 300 × 300 × 150 mm
     Repeatability: ≤ 1 mm
     End-effectors interchangeable (toolhead mount):
       - passive pusher tip
       - simple parallel gripper
       - electromagnet (for ferrous proxy objects)
       - suction cup (for flat objects)
       - probe (for touch-only diagnostics)

3.3  Sensing:
       Overhead camera:  USB or GigE, ≥ 1080p, fixed mount
       Toolhead camera (optional v0): for close-range context

3.4  Compute:
       Local workstation w/ GPU (≥ RTX 3060 class) for OF-JEPA
       inference at K=5 rolling window. Latency target: ≤ 50 ms
       per perception step.

3.5  Object set:
       Printed cubes (multiple sizes, identical except color)
       Pegs + matching slots (loose fit, tight fit)
       Curved tool + hanging frame (ToolHang analogue, deferred)
       Occluders (printed flat panels)
```

## §4 Camera geometry + calibration procedure

```
4.1  Overhead camera mount:
       Fixed pose; nominally normal to workspace plane.
       Height: 60-100 cm above table.
       FOV: covers full gantry workspace + 20% margin.

4.2  Intrinsics:
       Standard checkerboard calibration (OpenCV calibrateCamera).
       Persist as JSON: camera_matrix, distortion_coeffs.

4.3  Extrinsics (camera ↔ gantry frame):
       Move gantry to 4-corner + center fiducial positions; click
       fiducial centers in image; solvePnP for rigid transform.
       Persist as JSON: T_cam_to_gantry.

4.4  Sanity checks:
       After calibration, gantry-move a fiducial to 10 known
       (x, y) positions; image reprojection error ≤ 2 px.

4.5  Re-calibrate after physical disturbance:
       Bumped camera, replaced lens, moved table → recalibrate.
```

## §5 Software interface (binds to simulator artifacts)

```
5.0  Perception layer (BF-0.7 / BF-0.8 / BF-0.9 / BF-0.10 / BF-0.11 locked):

     Canonical pipeline:
       BF-0.2 fiducial detection (AprilTag / Charuco on table)
         → initial SAM 2.1 click seed at fiducial-projected pixel
         → SAM 2.1 Hiera-Tiny video predictor (27 FPS on H100,
           ~2 cm 3D pose accuracy, identity-aware mask propagation)
         → mask centroid + plane projection (or RGB-D unprojection)
         → if mask_area == 0 for >= silence_threshold consecutive
           frames AND fiducial visible at current pose:
              re-seed SAM via add_new_points_or_box(clear_old_points=True)
              at fiducial-projected pixel for current frame
         → FiducialDetection-shaped output feeds §5.1 OF-JEPA wrapper
         → reset SAM state at episode boundary (memory grows
           ~0.75 MB/frame in current sam2 release; cap by re-init
           between picks, NOT one session across all episodes)

     Implementation: bla.forge.SAMPerception
       backends: "mock_static" (CPU dev, deterministic) | "sam2.1"
       fiducial_fallback_fn: Callable[[int, int], Optional[(u, v)]]
         enables the watchdog pattern; defaults to None (vanilla
         tracking, no re-seed).

     LOCKED CLAUSE (2026-05-22, post BF-0.11):
       Fiducials are MANDATORY for initial identity seeding and
       recovery until learned identity re-acquisition is validated.
       SAM 2.1 alone cannot reassociate when the target moves
       significantly during occlusion (BF-0.10 failure mode); the
       fiducial channel provides the rescue anchor (BF-0.11 fix).

5.1  OF-JEPA runtime (D1b locked):
       Mode: rolling-window K=5.
       Input: 128x128 RGB crops from the overhead camera, scaled
       to the model's expected resolution.
       Output per frame: slot_states [S, slot_dim] + decoded slot
       positions (via slot_to_pos_aux).

5.2  Recipe router (Scale-0 deployed):
       from bla.routing import TaskDescriptor, recipe_router
       decision = recipe_router(TaskDescriptor(
           prior_kind=..., contact_sensitive=...,
           init_distribution_wide=..., sim_true_features=False,
           task_name="forge_task_2_pickplace"))
       → returns Recipe + RecipeConfig + rationale string

5.3  Demo retrieval (DR1 + DR3 locked):
       bla.recipes.DemoRetriever
       Bank: real-world demos collected per task (separate from
       robomimic sim demos).
       Key: same shape as PickPlaceCan
         (obj_x, obj_y, eef_x, eef_y, obj_z, eef_z)
       in gantry-frame coordinates.
       Outcome score: real-world success metric per demo.
       Modes:
         E2_FAST   = retriever.retrieve(key, k=1)
         E2_STABLE = retriever.retrieve_constrained_rerank(
                       key, k=5, filter_ratio=1.25)

5.4  Action execution:
       Gantry receives (x, y, z, gripper_open) per env-step at
       control frequency matching robosuite's (~20 Hz).
       Action sequences from demo replay are NOT gripper-noised
       (Phase 18κ R3 lesson — gripper bit must be deterministic).
```

## §6 Data schema (per-episode logging)

Persist every episode to disk for offline analysis. **The data
schema is more important than the choice of hardware** because
the schema is what makes per-episode diagnostics possible.

```
episode_record = {
    "ep_id": int,
    "timestamp": ISO-8601 string,
    "task": str,                      # "push" / "pickplace" / "insert" / "occlude"
    "router_decision": {
        "recipe": Recipe,
        "rationale": str,
        "task_descriptor": TaskDescriptor,
    },
    "frames": list of compressed images, one per timestep,
    "slot_states": [T, S, slot_dim] float32 array (the encoder
                    output, post rolling K=5 inference),
    "decoded_positions": [T, S, 2] float32 (slot_to_pos_aux),
    "gantry_actions": [T, action_dim] float32,
    "gantry_states": [T, gantry_state_dim] float32 (encoder pose
                       feedback),
    "outcome": {
        "success": bool,
        "improvement": float in [0, 1],
        "metric_name": str,
        "notes": str,
    },
    "retrieved_demo": {
        "demo_id": int or None,
        "nn_distance": float,
        "filter_passed": [int],         # for constrained rerank
    },
    "safety_events": list of {
        "timestep": int, "reason": str, "action": str},
    "perturbations": list of any manual interventions,
}
```

## §7 Real-world demo bank

Collect demos by teleoperation (joystick / direct gantry control)
on the same hardware. Each demo:

```
demo_record = {
    "demo_id": int,
    "task": str,
    "initial_state": {                   # ground-truth gantry frame
        "object_pose": (x, y, z, theta),
        "eef_pose": (x, y, z, theta),
        "gripper_open": float,
    },
    "actions": [T, action_dim] float32,
    "achieved_outcome": float,           # used as outcome_score
                                          # in DemoRetriever
    "collector_notes": str,
}
```

Aim for ≥ 30 demos per task during initial commissioning. Tag
each demo with its outcome (graspy / messy / failed); the
DR3-locked outcome_score field carries this directly.

## §8 Safety constraints

Non-negotiable:

```
8.1  Workspace bounds enforcement:
       gantry firmware-level limits (hard stops) at ±20 mm beyond
       the calibrated workspace.

8.2  Force/current limits:
       motor current threshold for stall detection; trigger
       e-stop on contact-force events above N (TBD per hardware).

8.3  Emergency stop:
       large red button accessible to operator; cuts gantry power.

8.4  Camera-only mode for unsupervised runs:
       perception logging only; no gantry motion when operator
       is not at the workstation.

8.5  No autonomous overnight runs without watchdog:
       any unattended session has a separate watchdog process
       that triggers e-stop if no "alive" heartbeat for >60 s.

8.6  Object set restriction:
       only printed/known objects on the workspace. No people
       or sharp objects in the workspace during autonomous runs.
```

## §9 Anti-patterns (locked from prior phases)

```
9.1  Do not bring CEM back to demo-prior tasks.
     Phase D3/D4/Scale-1/DR1 cross-validated CEM around expert
     demos as destructive. The real-world test should preserve
     E2_FAST and E2_STABLE without action-space search around
     them.

9.2  Do not scale model size before BLA-Forge.
     Per scaling roadmap §6. Model capacity scaling is last,
     after real-world tests establish whether the doctrine
     transfers.

9.3  Do not use raw OF-JEPA slot states as retrieval keys.
     Phase DR2 falsified this; use engineered geometry from the
     overhead camera + gantry feedback.

9.4  Do not skip calibration when results look "noisy".
     If real-world gates fail, the first diagnostic is camera
     calibration drift, not model failure.

9.5  Do not chase the simulator's remaining variance.
     Phase DR3 falsified the bank-coverage hypothesis; the
     residual simulator variance is execution stochasticity
     specific to robosuite. Real-world has a different noise
     model and is its own test.

9.6  Do not start with full pick-and-place.
     Task 1 (push) is the cheapest first test that exercises
     the perception + control pipeline end-to-end.
```

## §10 Per-task success metrics

```
Task 1 (Push to target):
  Success: object center within R mm of target after T seconds.
    R = 20 mm initially (relaxes with friction calibration).
  Improvement: 1 - clip(distance_to_target / start_distance, 0, 1).

Task 2 (Pick / place):
  Success: object lifted ≥ 30 mm and placed within target zone.
  Improvement: 1 if both lift + place succeeded, partial credit
    for lift-only.

Task 3 (Peg insertion):
  Success: peg fully seated in slot (gantry probe-test confirms
    end-state depth).
  Improvement: graded by insertion depth.

Task 4 (Occlusion tracking):
  Success: object-file slot bound to the occluded object's identity
    at t=0 still decodes within X mm of true position at re-emergence.
  Falsification: identity switch between the two cubes during
    occlusion event.
```

## §11 Phase sequence

```
Phase BF-0 — hardware bring-up (no doctrine claims):
  build gantry, mount camera, calibrate, run Task 4 (passive
  occlusion tracking — perception only).

Phase BF-1 — Task 1 push (FSM-prior regime):
  collect ~30 push demos; run Recipe A or B; verify gates G3/G4.
  This is the simplest contact event under real friction.

Phase BF-2 — Task 2 pick/place (demo-prior regime):
  collect ~30 pick/place demos; deploy E2_FAST + E2_STABLE;
  verify all 4 gates.

Phase BF-3 — Task 3 peg insertion (demo-prior, precision):
  ~30 insertion demos; test E2_STABLE specifically (precision
  task should favor lower-variance recipe).

Phase BF-4 — falsification / cross-task router check:
  use the recipe_router on an unseen real-world task without
  retraining; pre-commit prediction; run.

ONLY after BF-2 or later passes: consider model capacity scaling
or learned retrieval embeddings.
```

## §12 Open questions to resolve during BF-0 / BF-1

```
Q1: What is the real-world equivalent of state-matched reset?
    Robosuite's set_state_from_flattened doesn't have a physical
    analogue. The real-world demo retrieval bank's "init state"
    will be the OBSERVED initial state (camera + gantry feedback),
    not a sim state.

Q2: How does outcome_score get recorded in deployment?
    Each real-world demo has a measurable outcome (e.g., final
    object position). Outcome_score := the recorded success
    metric. No bank-build re-simulation needed.

Q3: Does rolling-window K=5 latency budget hold under real
    camera framerate?
    Robosuite renders frames on demand; real camera streams at
    30 Hz. The K=5 window needs the last 5 frames buffered.
    Latency budget: 50 ms per perception cycle.

Q4: Slot identity stability under physical occlusion?
    Phase 9B's visibility-gated v1 was deferred. Re-test on real
    occlusion (Task 4) before assuming it works.

Q5: How wide is the real-world initial-state distribution?
    Determines whether E1 (narrow init) or E2 (wide init) variants
    are the right defaults. The router can pick correctly only
    if we measure this empirically.
```

## §13 Locked

Until BLA-Forge BF-0 hardware bring-up begins, this spec is the
canonical pointer to what the real-world phase will test. Edits
to gates G1-G4 must be flagged as post-hoc and require an
explicit decision-doc.

## Files

- This spec: `docs/BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md`
- Architecture: `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md`
- Scaling roadmap: `docs/BLA_SCALING_ROADMAP.md` §6
- Doctrine memory: see `[[search-budget-zero-around-expert-demos]]`,
  `[[doctrine-validated-cross-task]]`, `[[execution-stochasticity-not-metric]]`
