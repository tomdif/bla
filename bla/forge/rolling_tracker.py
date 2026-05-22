"""BF-0.3 — table-coordinate transform + rolling K=5 OF-JEPA inference wrapper.

Per BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md §5.1 and the locked BF-0.3
deliverable: a streaming wrapper around D1b's batched K=5 encode that
buffers the last K camera frames and emits per-step slot states +
decoded world positions.

Mock-first design (matches BF-0.1 / BF-0.2):

  RollingObjectFileTracker(backend="mock_static")
    - Consumes BF-0.2 FiducialDetection list per step
    - Maps fiducials → world via BF-0.1 CalibrationBundle
    - Maintains persistent fiducial-id → slot-index binding (identity)
    - Produces deterministic slot states + decoded_positions_world
    - Right SHAPE for BF-0.4 episode logger to consume immediately

  RollingObjectFileTracker(backend="ofjepa")
    - NotImplementedError until the Phase-14 / Phase-17 encoder is
      available on real-world hardware. Same API; hardware-arrival
      becomes a backend swap, not a rewrite.

The mock backend deliberately encodes:
  - identity persistence (same ID → same slot across all steps where
    it's visible)
  - graceful occlusion (slot keeps last-known position with NaN
    fallback policy controllable by caller)
  - rolling buffer semantics (last K frames + fiducial lists held;
    earlier frames are forgotten)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bla.forge.calibration import CalibrationBundle
from bla.forge.fiducials import (
    FiducialDetection,
    fiducial_to_world,
    resolve_duplicate_ids,
)


# ---------- dataclass: one streamed observation ----------
@dataclass
class ObservationStep:
    """Per-step output of the rolling-window tracker.

    Fields:
      timestep                     monotonic step counter (0-indexed)
      slot_states                  [n_slots, slot_dim] — the OF-JEPA-style
                                   slot embeddings produced by the encoder
                                   backend (deterministic for mock).
      decoded_positions_world      [n_slots, 2] in meters; gantry frame.
                                   NaN for slots that have no current binding.
      slot_to_object_id            {slot_index: fiducial_id} for slots
                                   bound to a detected fiducial THIS step.
      confidence                   [n_slots] in [0, 1]; 0 for unbound slots,
                                   matches fiducial confidence for bound slots.
      buffer_len                   how many frames are in the rolling buffer
                                   right now (≤ K); useful for callers that
                                   gate on a warmed-up buffer.
    """
    timestep: int
    slot_states: np.ndarray
    decoded_positions_world: np.ndarray
    slot_to_object_id: dict[int, int]
    confidence: np.ndarray
    buffer_len: int


# ---------- the wrapper ----------
class RollingObjectFileTracker:
    """Buffer the last K frames; emit per-step object-file observations.

    Production-shape API. The mock backend lets BF-0.4 consume real-shape
    output immediately; the ofjepa backend wires in the real Phase-14
    encoder when hardware arrives.

    Args:
      bundle:        BF-0.1 CalibrationBundle for pixel↔world transforms.
      K:             rolling window size. Defaults to 5 (D1b's locked
                     rolling-window default).
      n_slots:       number of object-file slots. Matches Phase-14 OF-JEPA
                     default of 6.
      slot_dim:      slot embedding dim. Matches Phase-14 default of 128.
      backend:       "mock_static" (default) | "ofjepa" (raises until
                     hardware-side model is wired in).
      missing_policy: "nan" (default) | "carry" — what to do when a
                     previously-bound slot's fiducial isn't seen this step.
                     "nan" marks decoded_positions_world[s] = NaN; "carry"
                     keeps the last known position.

    Usage:

        tracker = RollingObjectFileTracker(bundle, K=5)
        for frame, detections in stream:
            step = tracker.step(frame, detections)
            # step.slot_states, step.decoded_positions_world, ...
    """

    def __init__(
        self,
        bundle: CalibrationBundle,
        *,
        K: int = 5,
        n_slots: int = 6,
        slot_dim: int = 128,
        backend: str = "mock_static",
        missing_policy: str = "nan",
        world_plane_z: float = 0.0,
    ):
        if K < 1:
            raise ValueError(f"K must be ≥ 1, got {K}")
        if n_slots < 1:
            raise ValueError(f"n_slots must be ≥ 1, got {n_slots}")
        if slot_dim < 1:
            raise ValueError(f"slot_dim must be ≥ 1, got {slot_dim}")
        if backend not in ("mock_static", "ofjepa"):
            raise ValueError(f"unknown backend: {backend!r}")
        if missing_policy not in ("nan", "carry"):
            raise ValueError(f"unknown missing_policy: {missing_policy!r}")

        self.bundle = bundle
        self.K = K
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.backend = backend
        self.missing_policy = missing_policy
        self.world_plane_z = float(world_plane_z)

        self._frame_buf: deque = deque(maxlen=K)
        self._fid_buf: deque = deque(maxlen=K)
        self._step_idx = -1
        # Persistent identity binding: fiducial_id → slot_index
        self._id_to_slot: dict[int, int] = {}
        # Carry state for missing_policy="carry"
        self._last_positions = np.full((n_slots, 2), np.nan)

    # ---------- public state ----------
    @property
    def buffer_len(self) -> int:
        return len(self._frame_buf)

    @property
    def step_count(self) -> int:
        return self._step_idx + 1

    @property
    def identity_bindings(self) -> dict[int, int]:
        """Snapshot of {fiducial_id: slot_index} bindings."""
        return dict(self._id_to_slot)

    def reset(self):
        """Clear the rolling buffer and all identity bindings."""
        self._frame_buf.clear()
        self._fid_buf.clear()
        self._step_idx = -1
        self._id_to_slot.clear()
        self._last_positions = np.full((self.n_slots, 2), np.nan)

    # ---------- main step ----------
    def step(
        self,
        frame: np.ndarray,
        fiducials: Optional[list[FiducialDetection]] = None,
    ) -> ObservationStep:
        """Add one frame + fiducial detections; emit ObservationStep."""
        if frame.ndim not in (2, 3):
            raise ValueError(
                f"frame must be H×W (grayscale) or H×W×3 (BGR), "
                f"got shape {frame.shape}")

        self._step_idx += 1
        self._frame_buf.append(frame)
        fids = fiducials if fiducials is not None else []
        # Defensive: collapse duplicate IDs by highest confidence
        # (matches BF-0.2's policy; downstream slot binding assumes unique IDs)
        fids = resolve_duplicate_ids(fids, strategy="highest_confidence")
        self._fid_buf.append(fids)

        # Update identity bindings for newly-seen IDs
        self._update_bindings(fids)

        # Backend dispatch
        if self.backend == "mock_static":
            return self._encode_mock_static(fids)
        elif self.backend == "ofjepa":
            return self._encode_ofjepa()
        # unreachable: __init__ validates
        raise AssertionError(self.backend)

    # ---------- internal: identity binding ----------
    def _update_bindings(self, fids: list[FiducialDetection]) -> None:
        """Assign a slot index to each newly-seen fiducial ID.

        Strategy: first-come-first-served. Once an ID is bound to a slot
        it KEEPS that slot for the lifetime of the tracker. If we run out
        of slots, new IDs are dropped (logged in confidence=0; not bound).

        This is the mock realization of the "identity-as-address" property
        (Phase 8C lesson): the slot index IS the persistent address.
        """
        used = set(self._id_to_slot.values())
        for f in fids:
            if f.id in self._id_to_slot:
                continue
            # Find the lowest unused slot index
            for s in range(self.n_slots):
                if s not in used:
                    self._id_to_slot[f.id] = s
                    used.add(s)
                    break
            # If no free slots, this ID is dropped silently (real BLA-Forge
            # would log a warning; for now: silent so tests are clean).

    # ---------- internal: mock_static backend ----------
    def _encode_mock_static(
        self, fids: list[FiducialDetection],
    ) -> ObservationStep:
        slot_states = np.zeros((self.n_slots, self.slot_dim), dtype=np.float64)
        decoded = np.full((self.n_slots, 2), np.nan, dtype=np.float64)
        slot_to_id: dict[int, int] = {}
        confidence = np.zeros(self.n_slots, dtype=np.float64)

        # Map each visible fiducial to its persistent slot
        for f in fids:
            if f.id not in self._id_to_slot:
                continue  # dropped (no free slot at binding time)
            s = self._id_to_slot[f.id]
            pose = fiducial_to_world(f, self.bundle, world_plane_z=self.world_plane_z)
            decoded[s] = pose.world_xy
            slot_to_id[s] = f.id
            confidence[s] = f.confidence
            # Deterministic mock slot state: a per-ID embedding stable across
            # all steps (the "identity" component) plus a small position-
            # dependent term (so identity is preserved but state reflects
            # current pose).
            slot_states[s] = self._mock_slot_state(f.id, pose.world_xy)
            # Cache for "carry" policy
            self._last_positions[s] = pose.world_xy

        # Apply missing_policy for slots NOT in slot_to_id
        if self.missing_policy == "carry":
            for s in range(self.n_slots):
                if s not in slot_to_id and not np.isnan(self._last_positions[s, 0]):
                    decoded[s] = self._last_positions[s]
                    # State retains last identity embedding (no fresh position term)
                    # but flagged with confidence 0
                    # We re-emit the mock state at last position
                    # Find what id was at that slot
                    last_id = next(
                        (i for i, idx in self._id_to_slot.items() if idx == s),
                        None,
                    )
                    if last_id is not None:
                        slot_states[s] = self._mock_slot_state(
                            last_id, self._last_positions[s])

        return ObservationStep(
            timestep=self._step_idx,
            slot_states=slot_states,
            decoded_positions_world=decoded,
            slot_to_object_id=slot_to_id,
            confidence=confidence,
            buffer_len=self.buffer_len,
        )

    def _mock_slot_state(self, fid_id: int, world_xy: np.ndarray) -> np.ndarray:
        """Deterministic synthetic slot embedding.

        Half of slot_dim is a per-ID embedding (the "identity address",
        stable across all observations of this ID). The other half encodes
        the world pose (so callers that decode-via-projection see a
        consistent representation).

        This is not a real OF-JEPA output; it's a placeholder that respects
        the architectural commitment "identity = persistent address" so
        BF-0.4's logger and downstream consumers don't have to special-case
        the mock vs real encoder.
        """
        half = self.slot_dim // 2
        rng = np.random.RandomState(int(fid_id) + 1)
        identity_component = rng.randn(half)
        pose_component = np.zeros(self.slot_dim - half)
        # Pose as a low-frequency sinusoidal embedding (Fourier-style)
        # so that nearby poses → similar features.
        freqs = np.linspace(1.0, 8.0, (self.slot_dim - half) // 2)
        x, y = float(world_xy[0]), float(world_xy[1])
        for i, f in enumerate(freqs):
            pose_component[2 * i] = np.sin(f * x)
            pose_component[2 * i + 1] = np.sin(f * y)
        out = np.concatenate([identity_component, pose_component])
        # If slot_dim was odd, pad the last entry with 0 (rare; defensive)
        if out.size < self.slot_dim:
            out = np.concatenate([out, np.zeros(self.slot_dim - out.size)])
        return out

    # ---------- internal: ofjepa backend ----------
    def _encode_ofjepa(self) -> ObservationStep:
        raise NotImplementedError(
            "ofjepa backend pending hardware-side OF-JEPA encoder. The "
            "production-side wiring will buffer the last K frames, call "
            "model.encode_video on the K-frame stack, take the FINAL "
            "timestep's slot_states (per D1b §3), and decode positions "
            "via model.slot_to_pos_aux + the calibration's image→world "
            "projection. Until then, use backend='mock_static'."
        )
