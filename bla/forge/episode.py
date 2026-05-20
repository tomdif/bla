"""BF-0.4 — episode JSON logger + real-world demo bank schema.

Per BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md §6 (episode record) and §7
(demo record).

Two record types:

  EpisodeRecord  — one rollout. Captures everything needed to reproduce
                   or analyze it: router decision, per-step frames /
                   slot states / decoded positions / gantry actions /
                   gantry states, outcome, retrieved-demo metadata,
                   safety events, perturbations.

  DemoRecord     — one teleoperated demonstration in the BLA-Forge
                   real-world demo bank. Convertible to bla.recipes
                   DemoState so the existing DemoRetriever plugs in
                   unchanged.

Storage strategy:

  - Small fixed-size metadata (router decision, outcome, ids, notes,
    safety events, schema version) → inline JSON.
  - Bulky per-step arrays (frames, slot_states, decoded_positions,
    gantry_actions, gantry_states) → sidecar .npz at the same path
    prefix as the .json. The JSON records the .npz filename relatively
    so the pair moves together.

  This matches how robomimic / many robotics frameworks separate
  metadata from bulk data. Tests cover both inline-only mode (handy
  for synthetic episodes) and sidecar mode.

Schema versioning is lightweight: `EPISODE_SCHEMA_VERSION` /
`DEMO_SCHEMA_VERSION` at module scope; each record carries it; loaders
will warn (not crash) on mismatch and let downstream code handle.

The first BF-0 deliverable, per the user's locked plan:

  A mocked camera/gantry loop that produces the EXACT same episode JSON
  schema the real BLA-Forge hardware will produce later.

`build_mock_episode_loop` at the bottom of this file is that loop.
"""
from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bla.forge.calibration import CalibrationBundle
from bla.forge.fiducials import FiducialDetection, mock_fiducials
from bla.forge.rolling_tracker import ObservationStep, RollingObjectFileTracker


EPISODE_SCHEMA_VERSION = "1.0"
DEMO_SCHEMA_VERSION = "1.0"

# Bulky arrays that get offloaded to .npz when use_sidecar=True
_BULKY_ARRAY_FIELDS = (
    "frames",
    "slot_states",
    "decoded_positions",
    "gantry_actions",
    "gantry_states",
)


# ============================================================================
# DemoRecord
# ============================================================================

@dataclass
class DemoRecord:
    """One teleoperated real-world demonstration. Per spec §7."""
    demo_id: int
    task: str
    initial_state: dict   # object_pose, eef_pose, gripper_open
    actions: np.ndarray   # [T, action_dim] float32
    achieved_outcome: float
    collector_notes: str = ""
    schema_version: str = DEMO_SCHEMA_VERSION

    def __post_init__(self):
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be [T, action_dim], got shape {self.actions.shape}")

    def to_demo_state(self, key: np.ndarray, init_state_flat: Optional[np.ndarray] = None):
        """Convert to bla.recipes.DemoState for the DemoRetriever pipeline.

        Args:
          key: the retrieval key vector for this demo (geometry / V-JEPA /
               whatever the deployment chooses).
          init_state_flat: optional flattened mujoco/gantry state for
               set_state_from_flattened-style reset. None for real-world
               where state-matched reset doesn't apply.
        """
        from bla.recipes import DemoState   # late import to avoid circular
        return DemoState(
            key=np.asarray(key, dtype=np.float32),
            action_seq=self.actions,
            init_state=init_state_flat,
            demo_id=int(self.demo_id),
            outcome_score=float(self.achieved_outcome),
            metadata={
                "task": self.task,
                "initial_state": self.initial_state,
                "collector_notes": self.collector_notes,
                "schema_version": self.schema_version,
            },
        )


def save_demo(record: DemoRecord, path: str | Path) -> Path:
    """Write a demo as JSON (small enough that sidecar isn't worthwhile)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": record.schema_version,
        "demo_id": int(record.demo_id),
        "task": record.task,
        "initial_state": record.initial_state,
        "actions": record.actions.tolist(),
        "achieved_outcome": float(record.achieved_outcome),
        "collector_notes": record.collector_notes,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_demo(path: str | Path) -> DemoRecord:
    path = Path(path)
    d = json.loads(path.read_text())
    if d.get("schema_version") != DEMO_SCHEMA_VERSION:
        warnings.warn(
            f"DemoRecord schema version mismatch: file has "
            f"{d.get('schema_version')!r}, code expects "
            f"{DEMO_SCHEMA_VERSION!r}", stacklevel=2)
    return DemoRecord(
        demo_id=int(d["demo_id"]),
        task=d["task"],
        initial_state=d["initial_state"],
        actions=np.asarray(d["actions"], dtype=np.float32),
        achieved_outcome=float(d["achieved_outcome"]),
        collector_notes=d.get("collector_notes", ""),
        schema_version=d.get("schema_version", DEMO_SCHEMA_VERSION),
    )


# ============================================================================
# EpisodeRecord
# ============================================================================

@dataclass
class EpisodeRecord:
    """One real-world (or mocked) rollout. Per spec §6.

    Per-step arrays are time-aligned: index t in any of frames /
    slot_states / decoded_positions / gantry_actions / gantry_states
    refers to the same physical timestep.
    """
    ep_id: int
    timestamp: str        # ISO-8601
    task: str             # "push" / "pickplace" / "insert" / "occlude"
    router_decision: dict   # recipe (str), rationale, task_descriptor
    outcome: dict           # success, improvement, metric_name, notes
    retrieved_demo: dict    # demo_id, nn_distance, filter_passed
    safety_events: list = field(default_factory=list)
    perturbations: list = field(default_factory=list)
    # Per-step arrays. May be empty (no rollout yet) or partially filled.
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))
    slot_states: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    decoded_positions: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    gantry_actions: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    gantry_states: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self):
        # Ensure arrays are numpy
        self.frames = np.asarray(self.frames)
        self.slot_states = np.asarray(self.slot_states, dtype=np.float32)
        self.decoded_positions = np.asarray(self.decoded_positions, dtype=np.float32)
        self.gantry_actions = np.asarray(self.gantry_actions, dtype=np.float32)
        self.gantry_states = np.asarray(self.gantry_states, dtype=np.float32)


def save_episode(
    record: EpisodeRecord,
    path: str | Path,
    *,
    use_sidecar: bool = True,
) -> Path:
    """Write an EpisodeRecord to JSON (+ optional .npz sidecar for bulky data).

    Args:
      path: target .json path. The sidecar (if used) goes to
            path.with_suffix(".npz") at the same prefix.
      use_sidecar: True (default) → bulky arrays go to .npz; JSON has
                   only metadata + an "arrays_sidecar" reference. False →
                   bulky arrays inline as nested lists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_payload: dict[str, Any] = {
        "schema_version": record.schema_version,
        "ep_id": int(record.ep_id),
        "timestamp": record.timestamp,
        "task": record.task,
        "router_decision": record.router_decision,
        "outcome": record.outcome,
        "retrieved_demo": record.retrieved_demo,
        "safety_events": record.safety_events,
        "perturbations": record.perturbations,
    }

    if use_sidecar:
        sidecar_path = path.with_suffix(".npz")
        np.savez_compressed(
            sidecar_path,
            frames=record.frames,
            slot_states=record.slot_states,
            decoded_positions=record.decoded_positions,
            gantry_actions=record.gantry_actions,
            gantry_states=record.gantry_states,
        )
        metadata_payload["arrays_sidecar"] = sidecar_path.name
    else:
        for fld in _BULKY_ARRAY_FIELDS:
            arr = getattr(record, fld)
            metadata_payload[fld] = arr.tolist()

    path.write_text(json.dumps(metadata_payload, indent=2))
    return path


def load_episode(path: str | Path) -> EpisodeRecord:
    """Load an EpisodeRecord; auto-detects sidecar via 'arrays_sidecar' key."""
    path = Path(path)
    d = json.loads(path.read_text())
    if d.get("schema_version") != EPISODE_SCHEMA_VERSION:
        warnings.warn(
            f"EpisodeRecord schema version mismatch: file has "
            f"{d.get('schema_version')!r}, code expects "
            f"{EPISODE_SCHEMA_VERSION!r}", stacklevel=2)

    arrays: dict[str, np.ndarray] = {}
    if "arrays_sidecar" in d:
        sidecar_path = path.parent / d["arrays_sidecar"]
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Episode references sidecar {sidecar_path} but file is missing")
        with np.load(sidecar_path, allow_pickle=False) as npz:
            for fld in _BULKY_ARRAY_FIELDS:
                arrays[fld] = npz[fld]
    else:
        for fld in _BULKY_ARRAY_FIELDS:
            arrays[fld] = np.asarray(d.get(fld, []))

    return EpisodeRecord(
        ep_id=int(d["ep_id"]),
        timestamp=d["timestamp"],
        task=d["task"],
        router_decision=d["router_decision"],
        outcome=d["outcome"],
        retrieved_demo=d["retrieved_demo"],
        safety_events=d.get("safety_events", []),
        perturbations=d.get("perturbations", []),
        frames=arrays["frames"],
        slot_states=arrays["slot_states"],
        decoded_positions=arrays["decoded_positions"],
        gantry_actions=arrays["gantry_actions"],
        gantry_states=arrays["gantry_states"],
        schema_version=d.get("schema_version", EPISODE_SCHEMA_VERSION),
    )


# ============================================================================
# EpisodeLogger — accumulator class for the streaming case
# ============================================================================

class EpisodeLogger:
    """Accumulate per-step data into an EpisodeRecord during a rollout.

    Usage:

        logger = EpisodeLogger(
            ep_id=0, task="pickplace", timestamp=...,
            router_decision={...}, retrieved_demo={...},
        )
        for frame, fids, action in stream:
            obs = tracker.step(frame, fids)
            logger.append_step(
                frame=frame,
                obs=obs,
                gantry_action=action,
                gantry_state=current_gantry_state,
            )
        logger.set_outcome(success=True, improvement=1.0,
                             metric_name="cube_z_gain", notes="")
        record = logger.finalize()
        save_episode(record, "ep_0.json")
    """

    def __init__(
        self,
        ep_id: int,
        task: str,
        timestamp: Optional[str] = None,
        router_decision: Optional[dict] = None,
        retrieved_demo: Optional[dict] = None,
    ):
        self.ep_id = ep_id
        self.task = task
        self.timestamp = timestamp or datetime.datetime.utcnow().isoformat() + "Z"
        self.router_decision = router_decision or {}
        self.retrieved_demo = retrieved_demo or {}
        self.outcome: dict = {}
        self.safety_events: list[dict] = []
        self.perturbations: list[dict] = []
        # Per-step accumulators (lists; stacked at finalize time)
        self._frames: list[np.ndarray] = []
        self._slot_states: list[np.ndarray] = []
        self._decoded_positions: list[np.ndarray] = []
        self._gantry_actions: list[np.ndarray] = []
        self._gantry_states: list[np.ndarray] = []

    def append_step(
        self,
        *,
        frame: np.ndarray,
        obs: ObservationStep,
        gantry_action: np.ndarray,
        gantry_state: np.ndarray,
    ) -> None:
        """Record one timestep. All inputs must be time-aligned to the same
        physical step. The ObservationStep comes from a BF-0.3
        RollingObjectFileTracker; gantry_action/state come from the
        motion-control layer (mocked in BF-0.4, real in BLA-Forge)."""
        self._frames.append(np.asarray(frame))
        self._slot_states.append(np.asarray(obs.slot_states, dtype=np.float32))
        self._decoded_positions.append(
            np.asarray(obs.decoded_positions_world, dtype=np.float32))
        self._gantry_actions.append(np.asarray(gantry_action, dtype=np.float32))
        self._gantry_states.append(np.asarray(gantry_state, dtype=np.float32))

    def set_outcome(
        self,
        *,
        success: bool,
        improvement: float,
        metric_name: str,
        notes: str = "",
    ) -> None:
        if not 0.0 <= improvement <= 1.0:
            raise ValueError(
                f"improvement must be in [0, 1], got {improvement}")
        self.outcome = {
            "success": bool(success),
            "improvement": float(improvement),
            "metric_name": str(metric_name),
            "notes": str(notes),
        }

    def log_safety_event(
        self, *, timestep: int, reason: str, action: str,
    ) -> None:
        self.safety_events.append({
            "timestep": int(timestep),
            "reason": str(reason),
            "action": str(action),
        })

    def log_perturbation(self, info: dict) -> None:
        self.perturbations.append(dict(info))

    def finalize(self) -> EpisodeRecord:
        """Stack accumulators into a complete EpisodeRecord."""
        if not self.outcome:
            warnings.warn(
                "EpisodeLogger.finalize() called before set_outcome(); "
                "outcome will be empty dict.", stacklevel=2)
        n = len(self._frames)
        # Stack arrays (or use zero-length arrays if no steps recorded)
        def _stack(lst, dtype):
            if not lst:
                return np.zeros((0,), dtype=dtype)
            return np.stack(lst, axis=0).astype(dtype)

        return EpisodeRecord(
            ep_id=self.ep_id,
            timestamp=self.timestamp,
            task=self.task,
            router_decision=self.router_decision,
            outcome=self.outcome,
            retrieved_demo=self.retrieved_demo,
            safety_events=self.safety_events,
            perturbations=self.perturbations,
            frames=_stack(self._frames, np.uint8) if self._frames
                else np.zeros((0,), dtype=np.uint8),
            slot_states=_stack(self._slot_states, np.float32),
            decoded_positions=_stack(self._decoded_positions, np.float32),
            gantry_actions=_stack(self._gantry_actions, np.float32),
            gantry_states=_stack(self._gantry_states, np.float32),
        )

    @property
    def n_steps(self) -> int:
        return len(self._frames)


# ============================================================================
# The mocked camera/gantry loop (the BF-0 first deliverable)
# ============================================================================

def build_mock_episode_loop(
    *,
    ep_id: int = 0,
    task: str = "pickplace",
    n_steps: int = 30,
    bundle: Optional[CalibrationBundle] = None,
    K: int = 5,
    n_slots: int = 6,
    slot_dim: int = 128,
    action_dim: int = 7,
    gantry_state_dim: int = 9,
    router_decision: Optional[dict] = None,
    retrieved_demo: Optional[dict] = None,
) -> EpisodeRecord:
    """Run a fully synthetic camera/gantry/object-file loop.

    No real hardware, no real OF-JEPA encoder; everything is mock-backed
    via the BF-0.1/0.2/0.3 mock modules. The point: produce an
    EpisodeRecord of the EXACT shape the real hardware will produce.

    Hardware-arrival path: swap the mock_fiducials() call for
    detect_fiducials(real_image, family="aruco_4x4_50"), swap the
    mock_static backend for "ofjepa", and swap the synthetic gantry
    action/state for real motion-controller readbacks. The EpisodeRecord
    contract is unchanged.
    """
    if bundle is None:
        from bla.forge.calibration import mock_calibration
        bundle = mock_calibration()

    tracker = RollingObjectFileTracker(
        bundle, K=K, n_slots=n_slots, slot_dim=slot_dim,
        backend="mock_static",
    )
    logger = EpisodeLogger(
        ep_id=ep_id,
        task=task,
        router_decision=router_decision or {
            "recipe": "E2_FAST",
            "rationale": "demo-prior contact-sensitive task (mocked)",
            "task_descriptor": {"prior_kind": "demo",
                                  "contact_sensitive": True},
        },
        retrieved_demo=retrieved_demo or {
            "demo_id": 0, "nn_distance": 0.0, "filter_passed": [0],
        },
    )

    # Synthetic frame canvas size
    H, W = bundle.intrinsics.image_size_wh[1], bundle.intrinsics.image_size_wh[0]
    rng = np.random.RandomState(ep_id)

    # Pretend the object is at a fixed start; sweep it across the workspace
    # over n_steps so decoded_positions has real motion (and slot_states pose
    # halves vary).
    start_xy = np.array([-0.10, -0.05])
    end_xy = np.array([+0.10, +0.05])

    for t in range(n_steps):
        frac = t / max(n_steps - 1, 1)
        cube_xy = start_xy + frac * (end_xy - start_xy)
        fids = mock_fiducials(
            ids=(0, 1),
            world_xy_per_id={0: tuple(cube_xy.tolist()),
                                1: (0.10, -0.10)},
            bundle=bundle,
        )
        frame = rng.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
        obs = tracker.step(frame, fids)
        # Mocked gantry action / state — zeros plus small noise, just to
        # exercise the shapes
        action = rng.uniform(-0.05, 0.05, size=action_dim).astype(np.float32)
        action[6] = 1.0   # gripper close (per Phase 18κ R3 convention)
        gantry_state = rng.uniform(-0.5, 0.5,
                                            size=gantry_state_dim).astype(np.float32)
        logger.append_step(
            frame=frame, obs=obs,
            gantry_action=action, gantry_state=gantry_state,
        )

    logger.set_outcome(
        success=True, improvement=1.0, metric_name="cube_z_gain",
        notes="Synthetic mocked-loop episode; not real hardware.",
    )
    return logger.finalize()
