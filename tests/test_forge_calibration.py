"""Unit tests for bla.forge.calibration (BF-0.1).

Validates the mock-first calibration harness against known synthetic
geometry. These tests don't require OpenCV — they only exercise the
projection math and serialization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    CalibrationBundle,
    CameraIntrinsics,
    CameraExtrinsics,
    load_calibration,
    mock_calibration,
    project_image_to_world_plane,
    project_world_to_image,
    reprojection_error,
    save_calibration,
)


def test_mock_calibration_default_shape():
    """Default mock returns a sane intrinsics + extrinsics bundle."""
    bundle = mock_calibration()
    assert bundle.intrinsics.camera_matrix.shape == (3, 3)
    assert bundle.intrinsics.distortion_coeffs.shape == (5,)
    assert bundle.extrinsics.T_cam_to_world.shape == (4, 4)
    assert bundle.metadata["source"] == "mock"


def test_mock_camera_at_known_height():
    """Default mock has camera at (0, 0, 0.80) looking down."""
    bundle = mock_calibration()
    T = bundle.extrinsics.T_cam_to_world
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.80], atol=1e-9)


def test_world_origin_projects_to_image_center():
    """World (0, 0, 0) under default top-down mock → image center."""
    bundle = mock_calibration(image_size_wh=(640, 480))
    px = project_world_to_image(np.array([0.0, 0.0, 0.0]), bundle)
    np.testing.assert_allclose(px, [320.0, 240.0], atol=1e-6)


def test_world_x_shift_moves_pixel_right():
    """+x world should map to +x (right) in image."""
    bundle = mock_calibration(image_size_wh=(640, 480))
    px_center = project_world_to_image(np.array([0.0, 0.0, 0.0]), bundle)
    px_right = project_world_to_image(np.array([0.05, 0.0, 0.0]), bundle)
    assert px_right[0] > px_center[0]   # x increases
    np.testing.assert_allclose(px_right[1], px_center[1], atol=1e-6)


def test_world_y_shift_moves_pixel_up_in_image_coords():
    """+y world → smaller image y (image y-down convention; world y-up).

    Default mock_calibration uses image y-down + world y-up, so a +y world
    shift produces a SMALLER pixel y (moving "up" in the image)."""
    bundle = mock_calibration(image_size_wh=(640, 480))
    px_center = project_world_to_image(np.array([0.0, 0.0, 0.0]), bundle)
    px_up = project_world_to_image(np.array([0.0, 0.05, 0.0]), bundle)
    assert px_up[1] < px_center[1]


def test_inverse_projection_recovers_world_xy():
    """Project a world point to pixel; invert through table plane;
    should recover the original world (x, y) within tight tolerance."""
    bundle = mock_calibration()
    world_in = np.array([0.04, -0.03, 0.0])
    px = project_world_to_image(world_in, bundle)
    world_out = project_image_to_world_plane(px, bundle, world_plane_z=0.0)
    np.testing.assert_allclose(world_out[:2], world_in[:2], atol=1e-6)
    np.testing.assert_allclose(world_out[2], 0.0, atol=1e-9)


def test_inverse_projection_batched():
    """N world points round-trip correctly through batched API."""
    bundle = mock_calibration()
    world_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.10, 0.05, 0.0],
        [-0.08, 0.07, 0.0],
        [0.02, -0.04, 0.0],
    ])
    pixels = project_world_to_image(world_pts, bundle)
    assert pixels.shape == (4, 2)
    recovered = project_image_to_world_plane(pixels, bundle, world_plane_z=0.0)
    assert recovered.shape == (4, 3)
    np.testing.assert_allclose(recovered[:, :2], world_pts[:, :2], atol=1e-6)


def test_reprojection_error_zero_for_self_projection():
    """If we project then ask for reprojection error against the same
    pixels, the error must be 0 (perfect by construction)."""
    bundle = mock_calibration()
    world_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [-0.05, -0.05, 0.0],
    ])
    pixels = project_world_to_image(world_pts, bundle)
    err = reprojection_error(world_pts, pixels, bundle)
    assert err < 1e-6


def test_reprojection_error_nonzero_for_perturbed_pixels():
    """A 5px deliberate offset gives a 5px reprojection error."""
    bundle = mock_calibration()
    world_pts = np.array([[0.0, 0.0, 0.0]])
    pixels = project_world_to_image(world_pts, bundle)
    perturbed = pixels + np.array([[5.0, 0.0]])
    err = reprojection_error(world_pts, perturbed, bundle)
    np.testing.assert_allclose(err, 5.0, atol=1e-6)


def test_save_load_roundtrip(tmp_path: Path):
    """Calibration JSON is round-trippable."""
    bundle = mock_calibration(image_size_wh=(800, 600), fov_deg=70.0)
    path = tmp_path / "calibration.json"
    save_calibration(bundle, path)
    assert path.exists()
    # Verify JSON is human-readable
    raw = json.loads(path.read_text())
    assert "intrinsics" in raw and "extrinsics" in raw
    # Round-trip
    loaded = load_calibration(path)
    np.testing.assert_array_equal(
        loaded.intrinsics.camera_matrix, bundle.intrinsics.camera_matrix)
    np.testing.assert_array_equal(
        loaded.extrinsics.T_cam_to_world, bundle.extrinsics.T_cam_to_world)
    assert loaded.metadata["source"] == "mock"


def test_world_grid_10_corners_reprojection_under_2px():
    """The BF-0.1 sanity gate from BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md §4.4:
    gantry-move fiducial to 10 known positions; image reprojection error
    ≤ 2 px. With the synthetic mock calibration this should be ~0."""
    bundle = mock_calibration()
    # 10 world positions covering a 30×30 cm workspace
    rng = np.random.RandomState(0)
    world_pts = rng.uniform(-0.15, 0.15, size=(10, 2))
    world_pts = np.hstack([world_pts, np.zeros((10, 1))])
    pixels = project_world_to_image(world_pts, bundle)
    err = reprojection_error(world_pts, pixels, bundle)
    assert err < 2.0   # BF-spec gate


def test_camera_intrinsics_rejects_wrong_shape():
    """Catch bad input early so calibration JSONs from real cameras
    don't silently corrupt downstream pipelines."""
    with pytest.raises(ValueError, match="camera_matrix must be 3×3"):
        CameraIntrinsics(
            camera_matrix=np.zeros((2, 2)),
            distortion_coeffs=np.zeros(5),
            image_size_wh=(640, 480),
        )
    with pytest.raises(ValueError, match="invalid size"):
        CameraIntrinsics(
            camera_matrix=np.eye(3),
            distortion_coeffs=np.zeros(3),
            image_size_wh=(640, 480),
        )


def test_camera_extrinsics_rejects_wrong_shape():
    with pytest.raises(ValueError, match="T_cam_to_world must be 4×4"):
        CameraExtrinsics(T_cam_to_world=np.eye(3))


def test_T_world_to_cam_is_actual_inverse():
    """The T_world_to_cam property must compose to identity with the
    forward transform."""
    bundle = mock_calibration()
    T_cw = bundle.extrinsics.T_cam_to_world
    T_wc = bundle.extrinsics.T_world_to_cam
    np.testing.assert_allclose(T_cw @ T_wc, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(T_wc @ T_cw, np.eye(4), atol=1e-9)
