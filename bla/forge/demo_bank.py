"""BF-1.0 — demo collection harness + on-disk demo bank.

Closes the loop between BF-0.4's `DemoRecord` storage layer and
`bla.recipes.DemoRetriever`:

  - `DemoCollector`  : streaming accumulator that turns a teleop session
                       into a `DemoRecord` (mirrors EpisodeLogger's
                       lifecycle: start → append_action → finalize).

  - `DemoBank`       : directory-backed store that scans a folder of
                       saved `DemoRecord` JSONs, validates them, and
                       exposes them as a `list[DemoState]` ready for
                       `DemoRetriever.build_index(...)`.

Mock-first design (matches BF-0.1 / 0.2 / 0.3 / 0.5):

  build_mock_demo(...)      synthesizes a complete DemoRecord using
                            mock_calibration + mock_fiducials + a
                            scripted action sequence. Lets us populate
                            a demo bank pre-hardware so the retrieval
                            path can be tested end-to-end.

When real teleop arrives the only thing that changes is the source of
`initial_state` (BF-0.2 mock_fiducials → BF-0.2 detect_fiducials) and
the action stream (synthetic → real spacemouse / VR readback). The
`DemoRecord` / `DemoBank` / `DemoRetriever` contracts stay fixed.

A key design note: the retrieval *key* shape is task-dependent (e.g.
6-D for PickPlaceCan: cube_xy, eef_xy, cube_z, eef_z) so `DemoBank`
does NOT pick a key for you. It exposes raw `DemoRecord`s; the caller
supplies a `key_fn(record) → np.ndarray` to turn the bank into
retrievable `DemoState`s. This keeps `DemoBank` reusable across tasks.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from bla.forge.calibration import CalibrationBundle
from bla.forge.episode import (
    DEMO_SCHEMA_VERSION,
    DemoRecord,
    load_demo,
    save_demo,
)
from bla.forge.fiducials import FiducialDetection, fiducial_to_world
from bla.recipes import DemoState


# ---------- collector ----------
class DemoCollector:
    """Streaming accumulator for one teleoperated demo.

    Lifecycle:

        coll = DemoCollector(demo_id=0, task="pickplace", action_dim=7)
        coll.start(initial_fiducials=fids, bundle=bundle,
                   extra={"gripper_open": 1.0})
        for action in teleop_stream:
            coll.append_action(action)
        record = coll.finalize(achieved_outcome=0.85,
                               collector_notes="clean lift, no drop")
        save_demo(record, "/path/to/demo_0000.json")

    The `initial_state` dict captured at `start(...)` is what the
    retrieval system will later condition on. We snapshot it as
    {fiducial_id → [world_x, world_y, world_z]} plus any caller-supplied
    extras (gripper state, target marker, etc.).
    """

    def __init__(
        self,
        *,
        demo_id: int,
        task: str,
        action_dim: int = 7,
    ):
        if action_dim < 1:
            raise ValueError(f"action_dim must be ≥ 1, got {action_dim}")
        self.demo_id = int(demo_id)
        self.task = str(task)
        self.action_dim = int(action_dim)
        self._initial_state: Optional[dict] = None
        self._actions: list[np.ndarray] = []
        self._started = False
        self._finalized = False

    # ---------- start ----------
    def start(
        self,
        *,
        initial_fiducials: Sequence[FiducialDetection],
        bundle: CalibrationBundle,
        extra: Optional[dict] = None,
    ) -> None:
        """Snapshot the initial scene state from fiducial detections.

        Produces an `initial_state` dict of the form:

            {
              "fiducials": { fiducial_id_str: [x, y, z] world meters },
              ...extra keys merged in...
            }

        Stored as a dict (not a numpy array) so it's JSON-friendly and
        human-readable. Callers map this into a retrieval key later via
        DemoBank.to_demo_states(key_fn=...).
        """
        if self._started:
            raise RuntimeError("DemoCollector.start() called twice")
        if self._finalized:
            raise RuntimeError("DemoCollector already finalized")
        fid_world: dict[str, list[float]] = {}
        for f in initial_fiducials:
            pose = fiducial_to_world(f, bundle)
            x, y = float(pose.world_xy[0]), float(pose.world_xy[1])
            # z = table plane (0.0 in gantry frame); fiducials are on objects
            # sitting ON the table. Real teleop will eventually have 6-DoF
            # pose estimation; for now we record the world xy + z=0.
            fid_world[str(f.id)] = [x, y, 0.0]
        state: dict = {"fiducials": fid_world}
        if extra:
            for k, v in extra.items():
                if k == "fiducials":
                    raise ValueError(
                        "extra cannot override 'fiducials' key")
                state[k] = v
        self._initial_state = state
        self._started = True

    # ---------- streaming ----------
    def append_action(self, action: np.ndarray) -> None:
        """Append one action vector. Shape must match action_dim."""
        if not self._started:
            raise RuntimeError(
                "DemoCollector.append_action() before start()")
        if self._finalized:
            raise RuntimeError("DemoCollector already finalized")
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (self.action_dim,):
            raise ValueError(
                f"action must be shape ({self.action_dim},), got {arr.shape}")
        self._actions.append(arr)

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    # ---------- finalize ----------
    def finalize(
        self,
        *,
        achieved_outcome: float,
        collector_notes: str = "",
    ) -> DemoRecord:
        """Pack the accumulated stream into a DemoRecord.

        Raises if `start()` was never called or no actions accumulated.
        The `achieved_outcome` is the demo-quality scalar used by
        DemoRetriever.retrieve_rerank_by_outcome — higher means better
        demonstration.
        """
        if not self._started:
            raise RuntimeError("DemoCollector.finalize() before start()")
        if self._finalized:
            raise RuntimeError("DemoCollector already finalized")
        if len(self._actions) == 0:
            raise RuntimeError(
                "DemoCollector.finalize() with no appended actions")
        if not (0.0 <= float(achieved_outcome) <= 1.0):
            raise ValueError(
                f"achieved_outcome must be in [0,1], got {achieved_outcome}")
        actions = np.stack(self._actions, axis=0).astype(np.float32)
        rec = DemoRecord(
            demo_id=self.demo_id,
            task=self.task,
            initial_state=dict(self._initial_state),
            actions=actions,
            achieved_outcome=float(achieved_outcome),
            collector_notes=str(collector_notes),
        )
        self._finalized = True
        return rec


# ---------- DemoBank ----------
@dataclass
class DemoBank:
    """Directory-backed collection of saved `DemoRecord` JSONs.

    Usage:

        bank = DemoBank.from_directory("/path/to/demos")
        states = bank.to_demo_states(
            key_fn=lambda r: np.array([r.initial_state["fiducials"]["0"][0],
                                        r.initial_state["fiducials"]["0"][1],
                                        r.initial_state["fiducials"]["1"][0],
                                        r.initial_state["fiducials"]["1"][1]],
                                       dtype=np.float32),
        )
        retriever = DemoRetriever()
        retriever.build_index(states)

    Demo files are loaded eagerly at construction. Re-scan with
    `reload()` if the directory changes; this is intentionally simple
    for the pre-hardware phase (no file-watcher / no caching layer).

    The bank is task-agnostic; `key_fn` is the only task-specific bit.
    """
    records: list[DemoRecord] = field(default_factory=list)
    directory: Optional[Path] = None

    # ---------- construction ----------
    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        *,
        task_filter: Optional[str] = None,
        min_outcome: float = 0.0,
        skip_invalid: bool = True,
    ) -> "DemoBank":
        """Scan a directory for *.json demo files and load them.

        Args:
          task_filter   if set, drop demos whose `task` field doesn't match.
          min_outcome   drop demos with achieved_outcome strictly below
                        this threshold (default 0.0 → keep everything).
          skip_invalid  on a load failure (corrupt JSON, schema mismatch,
                        shape error), warn and skip rather than raise.
                        Set False to make the bank construction strict.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"DemoBank directory does not exist: {directory}")
        records: list[DemoRecord] = []
        for p in sorted(directory.glob("*.json")):
            try:
                rec = load_demo(p)
            except Exception as e:
                if skip_invalid:
                    warnings.warn(
                        f"DemoBank: skipping {p.name}: {e}",
                        stacklevel=2)
                    continue
                raise
            if task_filter is not None and rec.task != task_filter:
                continue
            if rec.achieved_outcome < min_outcome:
                continue
            records.append(rec)
        return cls(records=records, directory=directory)

    # ---------- helpers ----------
    def __len__(self) -> int:
        return len(self.records)

    def by_demo_id(self, demo_id: int) -> DemoRecord:
        """Look up a demo by its id. O(N); fine for hundreds of demos."""
        for r in self.records:
            if r.demo_id == demo_id:
                return r
        raise KeyError(f"No demo with demo_id={demo_id}")

    def next_demo_id(self) -> int:
        """Return one greater than the max existing demo_id (0 if empty).

        For the operator-side collection workflow: after recording a
        demo, save it as `demo_{next_demo_id()}.json` to avoid id
        collisions.
        """
        if not self.records:
            return 0
        return max(r.demo_id for r in self.records) + 1

    def add(self, record: DemoRecord, *, save: bool = True) -> Path:
        """Add a DemoRecord to the bank.

        If `save=True` (default) and the bank has a directory, write
        the record to disk as `demo_{id:04d}.json`. Either way, append
        to the in-memory list. Returns the written path (or '' if
        save=False).
        """
        if any(r.demo_id == record.demo_id for r in self.records):
            raise ValueError(
                f"DemoBank already contains demo_id={record.demo_id}")
        self.records.append(record)
        if save and self.directory is not None:
            path = self.directory / f"demo_{record.demo_id:04d}.json"
            save_demo(record, path)
            return path
        return Path("")

    def to_demo_states(
        self,
        key_fn: Callable[[DemoRecord], np.ndarray],
        *,
        init_state_fn: Optional[Callable[[DemoRecord], np.ndarray]] = None,
    ) -> list[DemoState]:
        """Convert the bank into DemoStates for DemoRetriever.

        Args:
          key_fn          maps a DemoRecord to its retrieval key
                          (np.ndarray, shape [D]). Task-specific.
          init_state_fn   optional; maps a DemoRecord to a flattened
                          init-state vector for state-matched env reset.
                          None = init_state stays None in DemoState
                          (fresh-reset deployments, including the mock
                          loop and BLA-Forge).

        Raises if `key_fn` returns inconsistent shapes across records
        (DemoRetriever requires a single 2-D key matrix).
        """
        states: list[DemoState] = []
        ref_shape: Optional[tuple[int, ...]] = None
        for r in self.records:
            k = np.asarray(key_fn(r), dtype=np.float32)
            if ref_shape is None:
                ref_shape = k.shape
            elif k.shape != ref_shape:
                raise ValueError(
                    f"key_fn returned inconsistent shapes: "
                    f"{ref_shape} vs {k.shape} for demo_id={r.demo_id}")
            init = init_state_fn(r) if init_state_fn else None
            states.append(DemoState(
                key=k,
                action_seq=r.actions,
                init_state=init,
                demo_id=r.demo_id,
                outcome_score=r.achieved_outcome,
                metadata={"task": r.task, "notes": r.collector_notes},
            ))
        return states


# ---------- mock demo builder ----------
def build_mock_demo(
    *,
    demo_id: int,
    task: str = "pickplace",
    n_actions: int = 20,
    cube_xy: tuple[float, float] = (0.0, 0.0),
    bundle: Optional[CalibrationBundle] = None,
    achieved_outcome: float = 0.85,
    collector_notes: str = "synthetic mock demo",
    seed: Optional[int] = None,
) -> DemoRecord:
    """Build a synthetic DemoRecord using mock fiducials + scripted actions.

    The point: populate a DemoBank pre-hardware so retrieval can be
    tested end-to-end without recording anything real. Hardware-arrival
    swap: replace the mock_fiducials call with detect_fiducials on a
    captured frame and replace the scripted action sequence with the
    real teleop stream.
    """
    from bla.forge.calibration import mock_calibration
    from bla.forge.fiducials import mock_fiducials

    if bundle is None:
        bundle = mock_calibration()
    rng = np.random.RandomState(demo_id if seed is None else seed)

    fids = mock_fiducials(
        ids=(0, 1),
        world_xy_per_id={0: tuple(map(float, cube_xy)),
                          1: (0.10, -0.10)},
        bundle=bundle,
    )

    coll = DemoCollector(demo_id=demo_id, task=task, action_dim=7)
    coll.start(
        initial_fiducials=fids, bundle=bundle,
        extra={"gripper_open": 1.0},
    )
    # Scripted action sequence: small randomized deltas, gripper closes
    # halfway through (mock pick-up).
    for t in range(n_actions):
        action = rng.uniform(-0.02, 0.02, size=7).astype(np.float32)
        action[6] = 1.0 if t > n_actions // 2 else 0.0
        coll.append_action(action)
    return coll.finalize(
        achieved_outcome=achieved_outcome,
        collector_notes=collector_notes,
    )
