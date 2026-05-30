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
    FiducialDetector,
    FiducialWorldPose,
    MockFiducialDetector,
    OpenCVArucoDetector,
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
from bla.forge.safety import (
    WorkspaceBounds,
    VelocityLimits,
    SafetyDecision,
    SafetyMonitor,
    mock_workspace_bounds,
    mock_velocity_limits,
    safety_decision_to_event,
)
from bla.forge.demo_bank import (
    DemoCollector,
    DemoBank,
    build_mock_demo,
)
from bla.forge.deploy_loop import build_mock_deployment_loop
from bla.forge.sam_perception import SAMPerception, SAMSeed, FiducialFallbackFn
from bla.forge.deploy_loop_sam import build_sam_deployment_loop
from bla.forge.outcome import (
    cube_xy_displacement_score,
    no_safety_events_score,
    episode_length_fraction_score,
    combine_scores,
    compute_outcome,
    episode_to_demo,
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
    "FiducialDetector",
    "FiducialWorldPose",
    "MockFiducialDetector",
    "OpenCVArucoDetector",
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
    # BF-0.5
    "WorkspaceBounds",
    "VelocityLimits",
    "SafetyDecision",
    "SafetyMonitor",
    "mock_workspace_bounds",
    "mock_velocity_limits",
    "safety_decision_to_event",
    # BF-1.0
    "DemoCollector",
    "DemoBank",
    "build_mock_demo",
    # BF-1.1
    "build_mock_deployment_loop",
    # BF-1.5 / BF-0.11
    "SAMPerception",
    "SAMSeed",
    "FiducialFallbackFn",
    "build_sam_deployment_loop",
    # BF-1.2
    "cube_xy_displacement_score",
    "no_safety_events_score",
    "episode_length_fraction_score",
    "combine_scores",
    "compute_outcome",
    "episode_to_demo",
]
