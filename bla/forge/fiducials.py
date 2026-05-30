"""BF-0.2 — AprilTag / Charuco fiducial detection → object poses.

Per the locked BF-0.2 spec: detector/pose source, NOT a policy layer.
Output feeds BF-0.3 (rolling K=5 observation stream) and BF-0.4
(episode JSON logger / demo bank schema).

Mock-first pattern (same as BF-0.1 calibration):
  - real backend: `detect_fiducials(image, family=...)` uses OpenCV
    aruco (and optionally pupil_apriltags for AprilTag families).
  - mock backend: `mock_fiducials(...)` synthesizes deterministic
    detections for testing + downstream BF-0.3/0.4 dev work before
    hardware arrives.
  - both return the same `FiducialDetection` shape.

The pixel → world handoff goes through BF-0.1's CalibrationBundle:

    detections = detect_fiducials(image, family="aruco_4x4_50")
    world_poses = [fiducial_to_world(d, bundle) for d in detections]
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from bla.forge.calibration import (
    CalibrationBundle,
    project_image_to_world_plane,
    project_world_to_image,
)


# ---------- dataclasses ----------
@dataclass
class FiducialDetection:
    """One detected fiducial in image space.

    Fields:
      id              integer tag ID
      family          tag family name (e.g. "aruco_4x4_50", "apriltag_36h11",
                      "charuco")
      pixel_corners   [4, 2] float pixel coords of the 4 tag corners
                      in order: top-left, top-right, bottom-right, bottom-left
                      (OpenCV aruco convention)
      center_px       [2] float; convenience = pixel_corners.mean(axis=0)
      confidence      float ∈ [0, 1]; mock backends set to 1.0
    """
    id: int
    family: str
    pixel_corners: np.ndarray
    center_px: np.ndarray
    confidence: float = 1.0

    def __post_init__(self):
        self.pixel_corners = np.asarray(self.pixel_corners, dtype=np.float64)
        self.center_px = np.asarray(self.center_px, dtype=np.float64)
        if self.pixel_corners.shape != (4, 2):
            raise ValueError(
                f"pixel_corners must be 4×2, got {self.pixel_corners.shape}")
        if self.center_px.shape != (2,):
            raise ValueError(
                f"center_px must be (2,), got {self.center_px.shape}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class FiducialWorldPose:
    """One detected fiducial mapped to the world (gantry) frame.

    Fields:
      id           integer tag ID (matches FiducialDetection)
      world_xy     [2] float; tag center in world meters at the
                   calibration's reference plane (table top, typically z=0)
      yaw          optional float, radians; rotation about world +z derived
                   from the corner ordering. None if the tag is too small
                   or degenerate (e.g. all four corners coincident).
      confidence   passthrough from FiducialDetection
    """
    id: int
    world_xy: np.ndarray
    yaw: Optional[float] = None
    confidence: float = 1.0

    def __post_init__(self):
        self.world_xy = np.asarray(self.world_xy, dtype=np.float64)
        if self.world_xy.shape != (2,):
            raise ValueError(
                f"world_xy must be (2,), got {self.world_xy.shape}")


# ---------- serialization ----------
def detections_to_json(
    detections: list[FiducialDetection],
    path: str | Path,
    *,
    frame_id: Optional[int] = None,
    image_size_wh: Optional[tuple[int, int]] = None,
) -> Path:
    """Round-trippable JSON for a list of detections (single frame)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_id": frame_id,
        "image_size_wh": list(image_size_wh) if image_size_wh else None,
        "detections": [
            {
                "id": int(d.id),
                "family": d.family,
                "pixel_corners": d.pixel_corners.tolist(),
                "center_px": d.center_px.tolist(),
                "confidence": float(d.confidence),
            }
            for d in detections
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def detections_from_json(path: str | Path) -> list[FiducialDetection]:
    path = Path(path)
    d = json.loads(path.read_text())
    return [
        FiducialDetection(
            id=int(item["id"]),
            family=item["family"],
            pixel_corners=np.asarray(item["pixel_corners"]),
            center_px=np.asarray(item["center_px"]),
            confidence=float(item["confidence"]),
        )
        for item in d["detections"]
    ]


# ---------- pixel → world handoff ----------
def fiducial_to_world(
    detection: FiducialDetection,
    bundle: CalibrationBundle,
    world_plane_z: float = 0.0,
) -> FiducialWorldPose:
    """Map one detection's center + corners to a FiducialWorldPose.

    Yaw is derived from the angle of the (top-left → top-right) edge in
    world coords after projection through the table plane. If the
    projected edge has near-zero length (degenerate tag), yaw is None.
    """
    center_world_3d = project_image_to_world_plane(
        detection.center_px, bundle, world_plane_z=world_plane_z)
    center_world_xy = center_world_3d[:2]

    # Project top-left and top-right corners; angle of that edge in world is
    # the yaw of the tag.
    tl_world = project_image_to_world_plane(
        detection.pixel_corners[0], bundle, world_plane_z=world_plane_z)
    tr_world = project_image_to_world_plane(
        detection.pixel_corners[1], bundle, world_plane_z=world_plane_z)
    edge = tr_world[:2] - tl_world[:2]
    edge_len = float(np.linalg.norm(edge))
    yaw = float(np.arctan2(edge[1], edge[0])) if edge_len > 1e-6 else None

    return FiducialWorldPose(
        id=detection.id,
        world_xy=center_world_xy,
        yaw=yaw,
        confidence=detection.confidence,
    )


# ---------- duplicate-ID resolver ----------
def resolve_duplicate_ids(
    detections: list[FiducialDetection],
    *,
    strategy: str = "highest_confidence",
) -> list[FiducialDetection]:
    """Deduplicate detections with the same tag ID.

    strategies:
      "highest_confidence": keep the detection with max confidence per ID;
                            ties broken by earliest position in the input
                            (stable).
      "reject": raise ValueError on any duplicate ID — appropriate when a
                 duplicate indicates a sensor/detector bug, not noise.
    """
    if strategy == "reject":
        seen = set()
        for d in detections:
            if d.id in seen:
                raise ValueError(f"Duplicate fiducial ID {d.id} rejected")
            seen.add(d.id)
        return list(detections)
    if strategy == "highest_confidence":
        best: dict[int, FiducialDetection] = {}
        for d in detections:
            if d.id not in best or d.confidence > best[d.id].confidence:
                best[d.id] = d
        # Preserve original order (deterministic)
        kept_ids = set(best.keys())
        out = []
        added = set()
        for d in detections:
            if d.id in kept_ids and d.id not in added:
                out.append(best[d.id])
                added.add(d.id)
        return out
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------- mock backend ----------
def mock_fiducials(
    ids: tuple[int, ...] = (0, 1, 2, 3),
    world_xy_per_id: Optional[dict[int, tuple[float, float]]] = None,
    bundle: Optional[CalibrationBundle] = None,
    tag_size_m: float = 0.04,
    family: str = "mock_aruco_4x4_50",
    confidence: float = 1.0,
) -> list[FiducialDetection]:
    """Synthesize deterministic FiducialDetection list for development.

    For each ID, places a tag at a known world (x, y) and projects its
    corners through `bundle` into pixel space. Without a bundle, falls
    back to placing corners directly in pixel coords on a default
    640×480 image.

    Defaults: 4 tags arranged at the 4 corners of a 20×20 cm workspace
    centered at (0, 0).
    """
    if world_xy_per_id is None:
        # Default workspace corners (in meters), suitable for the BF-0.1
        # mock_calibration default (workspace centered at origin).
        default_layout = {
            0: (-0.10, -0.10),
            1: (+0.10, -0.10),
            2: (+0.10, +0.10),
            3: (-0.10, +0.10),
        }
        world_xy_per_id = {i: default_layout[i] for i in ids if i in default_layout}
        # Fill any extra IDs at the origin so caller always gets full coverage
        for i in ids:
            if i not in world_xy_per_id:
                world_xy_per_id[i] = (0.0, 0.0)

    half = tag_size_m / 2.0
    detections: list[FiducialDetection] = []

    if bundle is None:
        # Pixel-only fallback: synthesize at fixed pixel positions on a
        # 640×480 canvas. Useful for tests that don't care about world coords.
        canvas_w, canvas_h = 640, 480
        for tag_id in ids:
            cx = canvas_w // 2 + (tag_id - len(ids) / 2 + 0.5) * 80
            cy = canvas_h // 2
            corners_px = np.array([
                [cx - 20, cy - 20],   # top-left
                [cx + 20, cy - 20],   # top-right
                [cx + 20, cy + 20],   # bottom-right
                [cx - 20, cy + 20],   # bottom-left
            ], dtype=np.float64)
            detections.append(FiducialDetection(
                id=int(tag_id),
                family=family,
                pixel_corners=corners_px,
                center_px=corners_px.mean(axis=0),
                confidence=confidence,
            ))
        return detections

    # Project tag corners through the calibration
    for tag_id in ids:
        wx, wy = world_xy_per_id[tag_id]
        # Tag lies flat on the table (z=0); corners in OpenCV aruco order
        # (top-left, top-right, bottom-right, bottom-left) — we use
        # +y = "up" in world, so top corners are +y.
        corners_world = np.array([
            [wx - half, wy + half, 0.0],
            [wx + half, wy + half, 0.0],
            [wx + half, wy - half, 0.0],
            [wx - half, wy - half, 0.0],
        ])
        corners_px = project_world_to_image(corners_world, bundle)
        center_world = np.array([wx, wy, 0.0])
        center_px = project_world_to_image(center_world, bundle)
        detections.append(FiducialDetection(
            id=int(tag_id),
            family=family,
            pixel_corners=corners_px,
            center_px=center_px,
            confidence=confidence,
        ))
    return detections


# ---------- real OpenCV backend ----------
ARUCO_DICT_NAMES = {
    "aruco_4x4_50": "DICT_4X4_50",
    "aruco_4x4_100": "DICT_4X4_100",
    "aruco_4x4_250": "DICT_4X4_250",
    "aruco_5x5_50": "DICT_5X5_50",
    "aruco_5x5_100": "DICT_5X5_100",
    "aruco_5x5_250": "DICT_5X5_250",
    "aruco_6x6_50": "DICT_6X6_50",
    "aruco_6x6_100": "DICT_6X6_100",
    "aruco_6x6_250": "DICT_6X6_250",
}


def detect_fiducials(
    image: np.ndarray,
    *,
    family: str = "aruco_4x4_50",
) -> list[FiducialDetection]:
    """Run OpenCV aruco detection on a BGR or grayscale image.

    Requires opencv-contrib-python. Returns a list of FiducialDetection;
    empty list when nothing is detected.

    For AprilTag families (apriltag_36h11, etc.), opencv-contrib also
    ships APRILTAG_* dicts but with limited tag-family support. For
    serious AprilTag use, install pupil-apriltags and add a backend.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "detect_fiducials requires opencv-contrib-python; install with "
            "`pip install opencv-contrib-python` or use mock_fiducials.")
    if family not in ARUCO_DICT_NAMES:
        raise ValueError(
            f"Unknown family '{family}'. Known: {sorted(ARUCO_DICT_NAMES)}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAMES[family]))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners_list, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    detections: list[FiducialDetection] = []
    for corners, tag_id in zip(corners_list, ids.flatten()):
        # corners has shape (1, 4, 2)
        c = corners.reshape(4, 2).astype(np.float64)
        detections.append(FiducialDetection(
            id=int(tag_id),
            family=family,
            pixel_corners=c,
            center_px=c.mean(axis=0),
            confidence=1.0,   # opencv aruco doesn't expose a confidence score
        ))
    return detections


# ---------- streaming detector ABC ----------
class FiducialDetector(abc.ABC):
    """Streaming detector interface: frame in, detections out.

    The existing module-level `detect_fiducials` / `mock_fiducials`
    functions are stateless and one-shot. Real-time deployment wants a
    detector that can hold per-stream state (tag-family selection, frame
    counter, smoothing, lost-and-reacquired logic). Tests + BF-0
    development want a deterministic mock that produces a SCRIPTED
    sequence of detection sets, not a single static snapshot.

    Pattern (matches the BF-0.3 RollingObjectFileTracker backend swap):

        detector: FiducialDetector = MockFiducialDetector(...)  # for dev
        # detector = OpenCVArucoDetector(family="aruco_4x4_50")  # hardware

        for frame in camera_stream:
            fids = detector.step(frame)
            obs = tracker.step(frame, fids)

    Subclasses MUST implement `step`. `reset` defaults to a no-op.
    """

    @abc.abstractmethod
    def step(self, frame: np.ndarray) -> list[FiducialDetection]:
        """Consume one frame; emit zero or more detections."""

    def reset(self) -> None:
        """Clear any per-stream state (frame counter, smoothing buffer)."""


class OpenCVArucoDetector(FiducialDetector):
    """Thin streaming wrapper around `detect_fiducials`.

    Holds an OpenCV ArucoDetector across frames (so the detector object
    is constructed once, not per-frame) and tracks a monotonic step
    counter for diagnostics. Behaviorally identical to calling
    `detect_fiducials(frame, family=...)` per frame.
    """

    def __init__(self, family: str = "aruco_4x4_50"):
        if family not in ARUCO_DICT_NAMES:
            raise ValueError(
                f"Unknown family '{family}'. Known: {sorted(ARUCO_DICT_NAMES)}")
        self.family = family
        self._step_idx = -1
        # Lazy-construct on first step so import doesn't require cv2.
        self._detector = None

    def _build_detector(self):
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "OpenCVArucoDetector requires opencv-contrib-python; install "
                "with `pip install opencv-contrib-python` or use "
                "MockFiducialDetector for testing.")
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, ARUCO_DICT_NAMES[self.family]))
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(aruco_dict, params), cv2

    def step(self, frame: np.ndarray) -> list[FiducialDetection]:
        if self._detector is None:
            self._detector, self._cv2 = self._build_detector()
        self._step_idx += 1

        cv2 = self._cv2
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame)
        corners_list, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []
        out: list[FiducialDetection] = []
        for corners, tag_id in zip(corners_list, ids.flatten()):
            c = corners.reshape(4, 2).astype(np.float64)
            out.append(FiducialDetection(
                id=int(tag_id),
                family=self.family,
                pixel_corners=c,
                center_px=c.mean(axis=0),
                confidence=1.0,
            ))
        return out

    def reset(self) -> None:
        self._step_idx = -1

    @property
    def step_count(self) -> int:
        return self._step_idx + 1


# Type alias for the scripted-mock per-step generator.
# Given (step_idx, frame) → list of (id, world_xy) the mock should emit
# this step. Use to encode temporal patterns like "tag 2 disappears at
# step 30" or "all four tags drift along a trajectory".
StepScript = Callable[[int, np.ndarray], "list[tuple[int, tuple[float, float]]]"]


class MockFiducialDetector(FiducialDetector):
    """Deterministic mock detector producing scripted detection streams.

    Two modes:
      - static layout: same set of (id, world_xy) on every step.
      - scripted: caller supplies a `script(step_idx, frame) → [(id, xy)...]`
        function. Use this to simulate occlusion, motion, new objects
        entering the scene, etc.

    Requires a CalibrationBundle so world (x, y) is projected into pixel
    coords for the FiducialDetection (matches `mock_fiducials` behavior).

    Args:
      bundle:          calibration for world→pixel projection.
      static_layout:   {id: (x, y)} in world meters. Used as the every-step
                       layout when `script` is None.
      script:          optional callable returning per-step (id, (x, y))
                       pairs. Overrides static_layout when given.
      tag_size_m:      physical tag edge length, for pixel-corner synthesis.
      family:          family string carried on each emitted detection.
      confidence:      passthrough confidence on each detection.
    """

    def __init__(
        self,
        *,
        bundle,  # CalibrationBundle; not imported at top to avoid cycle
        static_layout: Optional[dict[int, tuple[float, float]]] = None,
        script: Optional[StepScript] = None,
        tag_size_m: float = 0.04,
        family: str = "mock_aruco_4x4_50",
        confidence: float = 1.0,
    ):
        if static_layout is None and script is None:
            raise ValueError(
                "MockFiducialDetector requires either `static_layout` or "
                "`script`. Pass at least one.")
        self.bundle = bundle
        self.static_layout = dict(static_layout) if static_layout else None
        self.script = script
        self.tag_size_m = float(tag_size_m)
        self.family = family
        self.confidence = float(confidence)
        self._step_idx = -1

    def step(self, frame: np.ndarray) -> list[FiducialDetection]:
        self._step_idx += 1
        if self.script is not None:
            entries = list(self.script(self._step_idx, frame))
        else:
            entries = list(self.static_layout.items())  # type: ignore[union-attr]

        if not entries:
            return []
        ids = tuple(int(e[0]) for e in entries)
        layout = {int(i): (float(x), float(y)) for i, (x, y) in entries}
        return mock_fiducials(
            ids=ids,
            world_xy_per_id=layout,
            bundle=self.bundle,
            tag_size_m=self.tag_size_m,
            family=self.family,
            confidence=self.confidence,
        )

    def reset(self) -> None:
        self._step_idx = -1

    @property
    def step_count(self) -> int:
        return self._step_idx + 1
