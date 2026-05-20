"""BLA-Forge real-world testbed software harness.

Per `docs/BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md` and the locked
BF-0 sub-phase order (`feedback`/`project_bla_bf0_software_prep`):

  BF-0.1 — camera calibration + table coordinate transform
  BF-0.2 — AprilTag / Charuco fiducial detection → object poses
  BF-0.3 — table-transform + rolling K=5 OF-JEPA inference wrapper
  BF-0.4 — episode JSON logger + demo bank schema
  BF-0.5 — safety / e-stop watchdog scaffold

Design principle: every interface that touches real hardware is
implemented mock-first, with the production-shape API. Hardware
arrival = sensor swap, not a new software project.

This subpackage stays small and dependency-light. Heavy lifting
(OF-JEPA, robosuite) lives upstream in `bla.recipes` / scripts.
"""
from bla.forge.calibration import (
    CameraIntrinsics,
    CameraExtrinsics,
    CalibrationBundle,
    save_calibration,
    load_calibration,
    project_world_to_image,
    project_image_to_world_plane,
    reprojection_error,
    estimate_intrinsics_from_checkerboard,
    estimate_extrinsics_from_corners,
    mock_calibration,
)
from bla.forge.fiducials import (
    FiducialDetection,
    FiducialWorldPose,
    detect_fiducials,
    mock_fiducials,
    fiducial_to_world,
    resolve_duplicate_ids,
    detections_to_json,
    detections_from_json,
)
from bla.forge.rolling_tracker import (
    ObservationStep,
    RollingObjectFileTracker,
)
from bla.forge.episode import (
    EpisodeRecord,
    DemoRecord,
    EpisodeLogger,
    save_episode,
    load_episode,
    save_demo,
    load_demo,
    build_mock_episode_loop,
    EPISODE_SCHEMA_VERSION,
    DEMO_SCHEMA_VERSION,
)

__all__ = [
    # BF-0.1
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CalibrationBundle",
    "save_calibration",
    "load_calibration",
    "project_world_to_image",
    "project_image_to_world_plane",
    "reprojection_error",
    "estimate_intrinsics_from_checkerboard",
    "estimate_extrinsics_from_corners",
    "mock_calibration",
    # BF-0.2
    "FiducialDetection",
    "FiducialWorldPose",
    "detect_fiducials",
    "mock_fiducials",
    "fiducial_to_world",
    "resolve_duplicate_ids",
    "detections_to_json",
    "detections_from_json",
    # BF-0.3
    "ObservationStep",
    "RollingObjectFileTracker",
    # BF-0.4
    "EpisodeRecord",
    "DemoRecord",
    "EpisodeLogger",
    "save_episode",
    "load_episode",
    "save_demo",
    "load_demo",
    "build_mock_episode_loop",
    "EPISODE_SCHEMA_VERSION",
    "DEMO_SCHEMA_VERSION",
]
