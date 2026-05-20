"""BF-1.1 — mock deployment loop: ties the BLA-Forge harness together.

Runs the full production inner cycle in the mock environment:

  scene (mock_fiducials)
    → perception (RollingObjectFileTracker)
    → retrieval query (caller-supplied key_fn over the scene)
    → DemoRetriever.retrieve(query) → DemoState
    → action selection (Recipe E2_FAST: replay retrieved demo's
      action_seq one step at a time; "search budget 0 around expert
      demos" — see feedback/search-budget-zero-around-expert-demos)
    → safety gate (SafetyMonitor)
    → EpisodeLogger.append_step + safety_event hooks
    → break on halt OR end of demo action sequence

Output: a real-shape EpisodeRecord with `retrieved_demo` populated by
the actual retriever, not a hardcoded stub. This is the function the
hardware-arrival session will exercise first; the mock backends just
swap out underneath.

The deployment loop does NOT introduce a new recipe. It uses Recipe E
(demo_no_cem) which is the locked default for contact-sensitive tasks
per the doctrine-validated-cross-task memory.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from bla.forge.calibration import CalibrationBundle, mock_calibration
from bla.forge.demo_bank import DemoBank
from bla.forge.episode import EpisodeLogger, EpisodeRecord
from bla.forge.fiducials import mock_fiducials, fiducial_to_world
from bla.forge.rolling_tracker import RollingObjectFileTracker
from bla.forge.safety import (
    SafetyMonitor,
    mock_velocity_limits,
    mock_workspace_bounds,
    safety_decision_to_event,
)
from bla.recipes import DemoRetriever, DemoState


def build_mock_deployment_loop(
    *,
    bank: DemoBank,
    key_fn: Callable[[dict], np.ndarray],
    query_world_xy_per_id: dict[int, tuple[float, float]],
    ep_id: int = 0,
    task: str = "pickplace",
    bundle: Optional[CalibrationBundle] = None,
    K: int = 5,
    n_slots: int = 6,
    slot_dim: int = 128,
    action_dim: int = 7,
    gantry_state_dim: int = 9,
    include_safety: bool = True,
    safety_breach_at_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> EpisodeRecord:
    """Run the full mock deployment cycle and return an EpisodeRecord.

    Args:
      bank                  preloaded DemoBank (BF-1.0). Must be non-empty.
      key_fn                builds the retrieval query from a dict of
                            the form {fiducial_id_int: world_xy_array}.
                            Must match the key_fn used at bank-indexing
                            time so query and bank live in the same
                            feature space.
      query_world_xy_per_id scene specification for the mock fiducials:
                            {fid_id: (x, y)} in world meters. The
                            retrieval query is built by passing this
                            (after fiducial→world projection) to key_fn.
      max_steps             cap the rollout length. None = run the full
                            retrieved demo's action_seq.

    Returns an EpisodeRecord with:
      - frames, slot_states, decoded_positions populated from perception
      - gantry_actions = the retrieved demo's actions (one per step)
      - gantry_states = synthetic (ee_xyz tracks the cube + small noise)
      - retrieved_demo metadata with the chosen demo_id and distance
      - router_decision = E2_FAST (demo_no_cem locked default)
      - safety_events if the SafetyMonitor flagged anything
      - outcome.success = True unless safety halted the rollout

    Raises if the bank is empty or the key_fn produces a key shape that
    doesn't match the bank's indexed keys.
    """
    if len(bank) == 0:
        raise ValueError("Bank is empty; nothing to retrieve from")

    if bundle is None:
        bundle = mock_calibration()

    # 1. Build the retriever index from the bank.
    states = bank.to_demo_states(
        key_fn=lambda rec: _record_to_key(rec, key_fn),
    )
    retriever = DemoRetriever()
    retriever.build_index(states)

    # 2. Set the scene + build the query.
    fids = mock_fiducials(
        ids=tuple(query_world_xy_per_id.keys()),
        world_xy_per_id={k: tuple(map(float, v))
                          for k, v in query_world_xy_per_id.items()},
        bundle=bundle,
    )
    # Project to world so the query is in the SAME space as
    # initial_state["fiducials"][...] — matches bank-side key_fn.
    query_scene = {f.id: np.array(fiducial_to_world(f, bundle).world_xy,
                                       dtype=np.float32)
                       for f in fids}
    query_key = np.asarray(key_fn(query_scene), dtype=np.float32)
    [chosen] = retriever.retrieve(query_key, k=1)
    nn_dist = float(np.linalg.norm(query_key - chosen.key))

    # 3. Stand up perception + (optional) safety.
    tracker = RollingObjectFileTracker(
        bundle, K=K, n_slots=n_slots, slot_dim=slot_dim,
        backend="mock_static",
    )
    monitor: Optional[SafetyMonitor] = None
    if include_safety:
        monitor = SafetyMonitor(
            mock_workspace_bounds(), mock_velocity_limits(),
            deadman_timeout_s=60.0,
        )

    logger = EpisodeLogger(
        ep_id=ep_id, task=task,
        router_decision={
            "recipe": "E2_FAST",
            "rationale": "demo-prior contact-sensitive (locked default)",
            "task_descriptor": {"prior_kind": "demo",
                                  "contact_sensitive": True},
        },
        retrieved_demo={
            "demo_id": int(chosen.demo_id),
            "nn_distance": nn_dist,
            "filter_passed": [int(chosen.demo_id)],
        },
    )

    # 4. Inner loop: replay the retrieved demo's actions.
    H, W = bundle.intrinsics.image_size_wh[1], bundle.intrinsics.image_size_wh[0]
    rng = np.random.RandomState(ep_id)
    actions = chosen.action_seq
    n_steps = actions.shape[0] if max_steps is None else min(actions.shape[0],
                                                                  max_steps)
    # Object stays roughly where the scene placed it (no scripted sweep
    # this time — the action stream is the retrieved demo, not synthetic).
    # We add a small drift so decoded_positions has nonzero variation,
    # mirroring real-world object micro-motion.
    base_xy = np.array(query_world_xy_per_id[next(iter(query_world_xy_per_id))],
                          dtype=np.float32)
    halted = False
    for t in range(n_steps):
        drift = rng.uniform(-0.002, 0.002, size=2).astype(np.float32)
        cube_xy = base_xy + drift
        step_fids = mock_fiducials(
            ids=tuple(query_world_xy_per_id.keys()),
            world_xy_per_id={
                fid_id: (tuple(cube_xy.tolist()) if i == 0
                          else tuple(map(float, xy)))
                for i, (fid_id, xy) in enumerate(query_world_xy_per_id.items())
            },
            bundle=bundle,
        )
        frame = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
        obs = tracker.step(frame, step_fids)

        action = actions[t].astype(np.float32)
        # Synthetic gantry pose: hover 10cm above cube
        ee_xyz = np.array([float(cube_xy[0]), float(cube_xy[1]), 0.10],
                              dtype=np.float32)
        if safety_breach_at_step is not None and t == safety_breach_at_step:
            ee_xyz = ee_xyz + np.array([0.30, 0.0, 0.0], dtype=np.float32)
        gantry_state = np.zeros(gantry_state_dim, dtype=np.float32)
        gantry_state[:3] = ee_xyz
        gantry_state[3:] = rng.uniform(-0.1, 0.1,
                                            size=gantry_state_dim - 3).astype(np.float32)

        if monitor is not None:
            monitor.tick()
            decision = monitor.decide(
                timestep=t, pose_xyz=ee_xyz.astype(np.float64),
            )
            if decision.reason != "ok":
                logger.log_safety_event(**safety_decision_to_event(decision))
            if decision.action == "halt":
                halted = True
                logger.append_step(
                    frame=frame, obs=obs,
                    gantry_action=action, gantry_state=gantry_state,
                )
                break

        logger.append_step(
            frame=frame, obs=obs,
            gantry_action=action, gantry_state=gantry_state,
        )

    if halted:
        logger.set_outcome(
            success=False, improvement=0.0, metric_name="cube_z_gain",
            notes="Mock deployment halted by SafetyMonitor.",
        )
    else:
        # We replayed a demo to completion; treat as success at the
        # demo's own outcome_score. Real-deploy will measure for real.
        logger.set_outcome(
            success=True, improvement=float(chosen.outcome_score),
            metric_name="cube_z_gain",
            notes=f"Mock deployment; replayed demo_id={chosen.demo_id}.",
        )
    return logger.finalize()


def _record_to_key(rec, key_fn) -> np.ndarray:
    """Adapt a DemoRecord → key_fn(scene dict).

    The deployment loop's key_fn takes a {fiducial_id_int: world_xy
    np.ndarray} dict (the scene at deploy time). DemoBank's records
    store initial_state as {"fiducials": {"<id_str>": [x, y, z]}}.
    This converts one to the other so the SAME key_fn drives both
    indexing and querying.
    """
    scene = {}
    for k, v in rec.initial_state.get("fiducials", {}).items():
        scene[int(k)] = np.array(v[:2], dtype=np.float32)
    return key_fn(scene)
