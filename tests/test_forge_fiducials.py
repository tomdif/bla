"""Unit tests for bla.forge.fiducials (BF-0.2).

Validates the mock fiducial detection contract independently of OpenCV.
Real-backend (`detect_fiducials`) is exercised only if opencv-contrib
is installed; otherwise that path is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    CalibrationBundle,
    FiducialDetection,
    FiducialWorldPose,
    detections_from_json,
    detections_to_json,
    fiducial_to_world,
    mock_calibration,
    mock_fiducials,
    project_image_to_world_plane,
    project_world_to_image,
    resolve_duplicate_ids,
)


def test_fiducial_detection_validates_shape():
    with pytest.raises(ValueError, match="pixel_corners must be 4×2"):
        FiducialDetection(
            id=0, family="x",
            pixel_corners=np.zeros((3, 2)),
            center_px=np.zeros(2),
        )
    with pytest.raises(ValueError, match="center_px must be"):
        FiducialDetection(
            id=0, family="x",
            pixel_corners=np.zeros((4, 2)),
            center_px=np.zeros(3),
        )
    with pytest.raises(ValueError, match="confidence must be in"):
        FiducialDetection(
            id=0, family="x",
            pixel_corners=np.zeros((4, 2)),
            center_px=np.zeros(2),
            confidence=1.5,
        )


def test_fiducial_world_pose_validates_shape():
    with pytest.raises(ValueError, match="world_xy must be"):
        FiducialWorldPose(id=0, world_xy=np.zeros(3))


def test_mock_fiducials_returns_4_default_tags():
    """Default mock returns one detection per default ID."""
    detections = mock_fiducials()
    ids = sorted(d.id for d in detections)
    assert ids == [0, 1, 2, 3]
    for d in detections:
        assert d.pixel_corners.shape == (4, 2)
        assert d.center_px.shape == (2,)
        assert d.family == "mock_aruco_4x4_50"
        assert 0.0 <= d.confidence <= 1.0


def test_mock_fiducials_center_equals_corner_mean():
    """Invariant: center = mean of 4 corners."""
    detections = mock_fiducials()
    for d in detections:
        np.testing.assert_allclose(d.center_px, d.pixel_corners.mean(axis=0),
                                          atol=1e-6)


def test_mock_fiducials_with_calibration_projects_through():
    """Detection centers project consistently with the calibration."""
    bundle = mock_calibration()
    # Place 4 tags at known world positions
    layout = {
        0: (-0.10, -0.10),
        1: (+0.10, -0.10),
        2: (+0.10, +0.10),
        3: (-0.10, +0.10),
    }
    detections = mock_fiducials(ids=(0, 1, 2, 3),
                                       world_xy_per_id=layout,
                                       bundle=bundle)
    for d in detections:
        wx, wy = layout[d.id]
        # Round-trip the detection's center pixel → world
        world_recovered = project_image_to_world_plane(d.center_px, bundle)
        np.testing.assert_allclose(world_recovered[:2], [wx, wy], atol=1e-6)


def test_mock_fiducials_stable_across_calls():
    """Determinism: same args → identical detections."""
    bundle = mock_calibration()
    d1 = mock_fiducials(bundle=bundle)
    d2 = mock_fiducials(bundle=bundle)
    for a, b in zip(d1, d2):
        assert a.id == b.id
        np.testing.assert_array_equal(a.pixel_corners, b.pixel_corners)
        np.testing.assert_array_equal(a.center_px, b.center_px)


def test_mock_fiducials_no_bundle_falls_back_to_pixel_layout():
    """Without a CalibrationBundle, mock places tags directly in pixel
    space on a 640×480 canvas. Useful for non-geometric tests."""
    detections = mock_fiducials(ids=(0, 1, 2, 3))
    assert len(detections) == 4
    for d in detections:
        # Within the 640×480 canvas
        assert 0 <= d.center_px[0] <= 640
        assert 0 <= d.center_px[1] <= 480


def test_fiducial_to_world_recovers_center_and_yaw():
    """fiducial_to_world should give back the same center we placed at."""
    bundle = mock_calibration()
    layout = {7: (0.05, -0.03)}
    detections = mock_fiducials(ids=(7,), world_xy_per_id=layout,
                                       bundle=bundle)
    pose = fiducial_to_world(detections[0], bundle)
    assert pose.id == 7
    np.testing.assert_allclose(pose.world_xy, [0.05, -0.03], atol=1e-6)
    # mock_fiducials places top-left and top-right with the same y in world,
    # so yaw should be 0 (edge points along world +x).
    assert pose.yaw is not None
    np.testing.assert_allclose(pose.yaw, 0.0, atol=1e-6)


def test_fiducial_to_world_handles_yawed_tag():
    """A 30°-rotated tag (constructed manually) should produce yaw=30°."""
    bundle = mock_calibration()
    # Place a tag at the origin, then rotate its corners by 30° in world
    yaw_rad = np.deg2rad(30.0)
    half = 0.02
    R = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                     [np.sin(yaw_rad),  np.cos(yaw_rad)]])
    local_corners = np.array([
        [-half, +half],
        [+half, +half],
        [+half, -half],
        [-half, -half],
    ])
    world_corners = local_corners @ R.T   # rotate by yaw
    world_corners_3d = np.hstack([world_corners, np.zeros((4, 1))])
    pixel_corners = project_world_to_image(world_corners_3d, bundle)
    detection = FiducialDetection(
        id=42, family="custom",
        pixel_corners=pixel_corners,
        center_px=pixel_corners.mean(axis=0),
    )
    pose = fiducial_to_world(detection, bundle)
    assert pose.yaw is not None
    # mock_calibration uses image-y-down + world-y-up, so the projected
    # world-frame edge angle should match the world-frame yaw.
    np.testing.assert_allclose(pose.yaw, yaw_rad, atol=1e-3)


def test_resolve_duplicate_ids_highest_confidence():
    """Keep the highest-confidence detection per ID."""
    bundle = mock_calibration()
    base = mock_fiducials(bundle=bundle)
    # Add a low-confidence duplicate of ID 1
    base_copy = list(base)
    base_copy.append(FiducialDetection(
        id=1, family=base[0].family,
        pixel_corners=base[0].pixel_corners,
        center_px=base[0].center_px,
        confidence=0.3,
    ))
    deduped = resolve_duplicate_ids(base_copy, strategy="highest_confidence")
    # Should have 4 detections (original IDs 0,1,2,3); the high-confidence
    # ID 1 wins.
    assert sorted(d.id for d in deduped) == [0, 1, 2, 3]
    id1 = next(d for d in deduped if d.id == 1)
    assert id1.confidence == 1.0   # original high-conf wins


def test_resolve_duplicate_ids_reject_raises():
    """Reject mode raises on duplicates."""
    bundle = mock_calibration()
    base = list(mock_fiducials(bundle=bundle))
    base.append(base[0])   # duplicate ID 0
    with pytest.raises(ValueError, match="Duplicate fiducial ID 0"):
        resolve_duplicate_ids(base, strategy="reject")


def test_resolve_duplicate_ids_no_duplicates_passes_through():
    """Both strategies leave already-unique input unchanged in identity."""
    bundle = mock_calibration()
    base = mock_fiducials(bundle=bundle)
    deduped_a = resolve_duplicate_ids(base, strategy="highest_confidence")
    deduped_b = resolve_duplicate_ids(base, strategy="reject")
    assert [d.id for d in deduped_a] == [d.id for d in base]
    assert [d.id for d in deduped_b] == [d.id for d in base]


def test_resolve_duplicate_ids_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        resolve_duplicate_ids([], strategy="bogus")


def test_detections_json_roundtrip(tmp_path: Path):
    """Save → load gives back the same detection list."""
    bundle = mock_calibration()
    detections = mock_fiducials(bundle=bundle)
    path = tmp_path / "dets.json"
    detections_to_json(detections, path, frame_id=42,
                            image_size_wh=(640, 480))
    raw = json.loads(path.read_text())
    assert raw["frame_id"] == 42
    assert raw["image_size_wh"] == [640, 480]
    assert len(raw["detections"]) == len(detections)
    loaded = detections_from_json(path)
    assert len(loaded) == len(detections)
    for a, b in zip(detections, loaded):
        assert a.id == b.id
        assert a.family == b.family
        np.testing.assert_array_equal(a.pixel_corners, b.pixel_corners)
        np.testing.assert_array_equal(a.center_px, b.center_px)
        assert a.confidence == b.confidence


def test_detections_json_empty_list(tmp_path: Path):
    """Save/load with zero detections (e.g. nothing visible) is well-defined."""
    path = tmp_path / "empty.json"
    detections_to_json([], path)
    loaded = detections_from_json(path)
    assert loaded == []


def test_missing_fiducial_handled_cleanly():
    """If a query ID isn't in the bank, downstream code should not crash.

    The contract: mock_fiducials returns ONLY the ids passed in. Asking
    for a subset works; the caller is responsible for handling "ID X not
    detected this frame."""
    bundle = mock_calibration()
    detections = mock_fiducials(ids=(0, 2), bundle=bundle)
    ids = sorted(d.id for d in detections)
    assert ids == [0, 2]
    # Confirm we can ask for the one that IS detected without error
    by_id = {d.id: d for d in detections}
    assert 0 in by_id
    assert 1 not in by_id   # not detected this frame


def test_fiducial_to_world_passes_confidence_through():
    """Confidence is a passthrough metadata field through world projection."""
    bundle = mock_calibration()
    layout = {0: (0.0, 0.0)}
    detections = mock_fiducials(ids=(0,), world_xy_per_id=layout,
                                       bundle=bundle, confidence=0.42)
    pose = fiducial_to_world(detections[0], bundle)
    assert pose.confidence == 0.42
