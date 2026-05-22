"""BF-1.5 — SAM-driven deployment loop.

Variant of build_mock_deployment_loop where perception comes from a real
SAM 2.1 video predictor instead of mock_fiducials(). The rest of the
chain (rolling_tracker → retriever → action replay → safety → logger) is
identical.

Use case: you have a pre-rendered video sitting on disk (e.g. the BF-0.7
PickPlaceCan demo) and want to run BF-1.x end-to-end with real perception
to validate the wire interface before hardware arrives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from bla.forge.calibration import CalibrationBundle
from bla.forge.demo_bank import DemoBank
from bla.forge.episode import EpisodeLogger, EpisodeRecord
from bla.forge.fiducials import fiducial_to_world
from bla.forge.rolling_tracker import RollingObjectFileTracker
from bla.forge.safety import (
    SafetyMonitor,
    mock_velocity_limits,
    mock_workspace_bounds,
    safety_decision_to_event,
)
from bla.forge.sam_perception import (
    SAMPerception, SAMSeed, FiducialFallbackFn,
)
from bla.recipes import DemoRetriever


def build_sam_deployment_loop(
    *,
    bank: DemoBank,
    key_fn: Callable[[dict], np.ndarray],
    sam_video_path: Path | str,
    seeds: list[SAMSeed],
    bundle: CalibrationBundle,
    ep_id: int = 0,
    task: str = "pickplace",
    K: int = 5,
    n_slots: int = 6,
    slot_dim: int = 128,
    action_dim: int = 7,
    gantry_state_dim: int = 9,
    include_safety: bool = True,
    safety_breach_at_step: Optional[int] = None,
    max_steps: Optional[int] = None,
    sam_backend: str = "sam2.1",
    sam_model: str = "facebook/sam2.1-hiera-tiny",
    world_plane_z: float = 0.0,
    fiducial_fallback_fn: Optional[FiducialFallbackFn] = None,
    silence_threshold: int = 3,
) -> EpisodeRecord:
    """Run the deployment cycle with SAM 2.1 perception and return an
    EpisodeRecord.

    Args:
      sam_video_path  directory of JPEG frames that SAM will track. SAM
                      uses these instead of the random-noise frames the
                      mock loop generates.
      seeds           list of SAMSeed(obj_id, pixel_uv). The obj_id of
                      each seed must match a fiducial_id used to index
                      the bank with key_fn.
      bundle          required (not optional) — must match the camera that
                      produced sam_video_path.
      world_plane_z   z-coordinate of the world plane the perception
                      projects pixels to. For BF-0.7 PickPlaceCan, set to
                      0.86 (can rest height); for a real gantry table, 0.
      fiducial_fallback_fn  optional BF-0.11 watchdog callback:
                      (frame_idx, obj_id) → (u, v) or None. When SAM's
                      mask area drops to 0 for >= silence_threshold
                      consecutive frames, SAMPerception will call this
                      to obtain a fresh seed pixel and re-prompt itself.
                      In real deployment this wraps BF-0.2 detect_fiducials.
      silence_threshold  consecutive zero-mask frames before watchdog
                      triggers a re-seed.

    Other args mirror build_mock_deployment_loop.
    """
    if len(bank) == 0:
        raise ValueError("Bank is empty; nothing to retrieve from")

    # --- 1. Build the retriever from the bank ---
    states = bank.to_demo_states(
        key_fn=lambda rec: _record_to_key(rec, key_fn),
    )
    retriever = DemoRetriever()
    retriever.build_index(states)

    # --- 2. Stand up SAM perception (with optional BF-0.11 watchdog) ---
    sam = SAMPerception(
        video_path=sam_video_path, seeds=seeds,
        backend=sam_backend, sam_model=sam_model,
        fiducial_fallback_fn=fiducial_fallback_fn,
        silence_threshold=silence_threshold,
    )

    # --- 3. Build query from frame 0 detections ---
    f0_detections = sam.detect(0)
    f0_world = {
        f.id: np.array(fiducial_to_world(f, bundle, world_plane_z=world_plane_z).world_xy,
                          dtype=np.float32)
        for f in f0_detections if f.confidence > 0
    }
    query_key = np.asarray(key_fn(f0_world), dtype=np.float32)
    [chosen] = retriever.retrieve(query_key, k=1)
    nn_dist = float(np.linalg.norm(query_key - chosen.key))

    # --- 4. Stand up tracker + safety + logger ---
    tracker = RollingObjectFileTracker(
        bundle, K=K, n_slots=n_slots, slot_dim=slot_dim,
        backend="mock_static", world_plane_z=world_plane_z,
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

    # --- 5. Inner loop: replay the retrieved demo's actions, perception
    #        from SAM per step ---
    H = bundle.intrinsics.image_size_wh[1]
    W = bundle.intrinsics.image_size_wh[0]
    rng = np.random.RandomState(ep_id)
    actions = chosen.action_seq
    n_demo_steps = actions.shape[0]
    n_sam_frames = len(sam)
    if n_sam_frames == 0:
        # mock backend — clip by demo length only
        n_steps = n_demo_steps if max_steps is None else min(n_demo_steps,
                                                                  max_steps)
    else:
        n_steps = min(n_demo_steps, n_sam_frames)
        if max_steps is not None: n_steps = min(n_steps, max_steps)

    halted = False
    for t in range(n_steps):
        # Real RGB frame from the video — for the tracker's frame buffer
        if n_sam_frames > 0:
            frame_path = Path(sam_video_path) / f"{t:05d}.jpg"
            frame = np.asarray(Image.open(frame_path).convert("RGB"))
        else:
            frame = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)

        # Per-step SAM detections (filtered to confidence > 0)
        step_fids = [f for f in sam.detect(t) if f.confidence > 0]

        # Update rolling tracker
        obs = tracker.step(frame, step_fids)

        # Synthetic gantry pose: hover above the primary tracked object,
        # using the rolling-tracker's decoded world position (NOT
        # ground-truth — this is the same indirection the real system
        # uses)
        primary_obj_id = seeds[0].obj_id
        primary_slot_idx = next(
            (s for s, oid in obs.slot_to_object_id.items()
             if oid == primary_obj_id), None)
        if primary_slot_idx is not None and not np.isnan(
                obs.decoded_positions_world[primary_slot_idx, 0]):
            tracked_xy = obs.decoded_positions_world[primary_slot_idx]
        else:
            tracked_xy = np.zeros(2, dtype=np.float64)
        ee_xyz = np.array([float(tracked_xy[0]), float(tracked_xy[1]), 0.10],
                              dtype=np.float32)
        if safety_breach_at_step is not None and t == safety_breach_at_step:
            ee_xyz = ee_xyz + np.array([0.30, 0.0, 0.0], dtype=np.float32)

        action = actions[t].astype(np.float32)
        gantry_state = np.zeros(gantry_state_dim, dtype=np.float32)
        gantry_state[:3] = ee_xyz
        gantry_state[3:] = rng.uniform(
            -0.1, 0.1, size=gantry_state_dim - 3).astype(np.float32)

        if monitor is not None:
            monitor.tick()
            decision = monitor.decide(
                timestep=t, pose_xyz=ee_xyz.astype(np.float64))
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
            notes="SAM deployment halted by SafetyMonitor.",
        )
    else:
        logger.set_outcome(
            success=True, improvement=float(chosen.outcome_score),
            metric_name="cube_z_gain",
            notes=f"SAM deployment; replayed demo_id={chosen.demo_id}.",
        )
    return logger.finalize()


def _record_to_key(rec, key_fn) -> np.ndarray:
    scene = {}
    for k, v in rec.initial_state.get("fiducials", {}).items():
        scene[int(k)] = np.array(v[:2], dtype=np.float32)
    return key_fn(scene)
