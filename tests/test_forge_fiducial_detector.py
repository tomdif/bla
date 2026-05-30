"""Unit tests for the FiducialDetector ABC + MockFiducialDetector.

Covers the streaming-detector seam that lets BF-0 development proceed
on the frame→detections boundary without hardware. Real OpenCV path
(OpenCVArucoDetector) is exercised lightly via constructor + lazy-init
checks; full pixel-detection coverage lives in opencv-side smoke tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from bla.forge import (
    FiducialDetection,
    FiducialDetector,
    MockFiducialDetector,
    OpenCVArucoDetector,
    RollingObjectFileTracker,
    mock_calibration,
)


# ---------- helpers ----------
def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


# ---------- abc enforcement ----------
def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        FiducialDetector()  # type: ignore[abstract]


def test_concrete_subclass_must_implement_step():
    class Half(FiducialDetector):
        pass
    with pytest.raises(TypeError):
        Half()  # type: ignore[abstract]


# ---------- MockFiducialDetector: static layout ----------
def test_mock_static_layout_basic():
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle,
        static_layout={0: (0.0, 0.0), 1: (0.05, 0.0)},
    )
    fids = det.step(_frame())
    assert len(fids) == 2
    assert {f.id for f in fids} == {0, 1}
    assert all(isinstance(f, FiducialDetection) for f in fids)


def test_mock_static_layout_deterministic_across_steps():
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle, static_layout={7: (0.02, -0.01)})
    a = det.step(_frame())
    b = det.step(_frame())
    assert a[0].id == b[0].id == 7
    np.testing.assert_allclose(a[0].pixel_corners, b[0].pixel_corners)


def test_mock_step_count_advances():
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle, static_layout={0: (0.0, 0.0)})
    assert det.step_count == 0
    det.step(_frame())
    det.step(_frame())
    assert det.step_count == 2


def test_mock_reset_clears_counter():
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle, static_layout={0: (0.0, 0.0)})
    det.step(_frame())
    det.step(_frame())
    det.reset()
    assert det.step_count == 0


# ---------- MockFiducialDetector: scripted ----------
def test_mock_scripted_per_step_variation():
    """Script encodes a fiducial disappearing at step 2."""
    bundle = mock_calibration()

    def script(step_idx, frame):
        if step_idx < 2:
            return [(0, (0.0, 0.0)), (1, (0.05, 0.0))]
        return [(0, (0.0, 0.0))]  # tag 1 occluded

    det = MockFiducialDetector(bundle=bundle, script=script)
    visible_step0 = {f.id for f in det.step(_frame())}
    visible_step1 = {f.id for f in det.step(_frame())}
    visible_step2 = {f.id for f in det.step(_frame())}
    assert visible_step0 == {0, 1}
    assert visible_step1 == {0, 1}
    assert visible_step2 == {0}


def test_mock_scripted_motion():
    """World position changes per step → pixel position changes."""
    bundle = mock_calibration()

    def script(step_idx, frame):
        return [(0, (0.001 * step_idx, 0.0))]

    det = MockFiducialDetector(bundle=bundle, script=script)
    pos0 = det.step(_frame())[0].center_px.copy()
    for _ in range(10):
        det.step(_frame())
    pos10 = det.step(_frame())[0].center_px
    assert not np.allclose(pos0, pos10)


def test_mock_scripted_empty_detection_set():
    """A script returning [] should yield no detections that step."""
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle,
        script=lambda i, f: [(0, (0.0, 0.0))] if i == 0 else [],
    )
    assert len(det.step(_frame())) == 1
    assert det.step(_frame()) == []


# ---------- MockFiducialDetector: input validation ----------
def test_mock_requires_layout_or_script():
    with pytest.raises(ValueError, match="static_layout"):
        MockFiducialDetector(bundle=mock_calibration())


# ---------- OpenCVArucoDetector: construction (no cv2 required) ----------
def test_opencv_detector_unknown_family_rejected():
    with pytest.raises(ValueError, match="Unknown family"):
        OpenCVArucoDetector(family="bogus")


def test_opencv_detector_lazy_construction():
    """Constructing should not import cv2; only .step() requires it."""
    det = OpenCVArucoDetector(family="aruco_4x4_50")
    assert det._detector is None
    assert det.family == "aruco_4x4_50"
    assert det.step_count == 0


# ---------- integration: detector → tracker pipeline ----------
def test_detector_to_tracker_round_trip():
    """End-to-end: mock detector → RollingObjectFileTracker.

    Verifies the streaming seam: the test never calls mock_fiducials
    directly. The detector emits per-step detections; the tracker binds
    them to slots and decodes world positions.
    """
    bundle = mock_calibration()
    det = MockFiducialDetector(
        bundle=bundle,
        static_layout={0: (0.03, 0.01), 1: (-0.03, -0.01)},
    )
    tracker = RollingObjectFileTracker(bundle, n_slots=4, slot_dim=16, K=3)

    for _ in range(5):
        frame = _frame()
        fids = det.step(frame)
        obs = tracker.step(frame, fids)

    # Identity-as-address: tag 0 and tag 1 should each bind to one slot
    # and stay there.
    bindings = tracker.identity_bindings
    assert set(bindings.keys()) == {0, 1}
    # Slot 0 and slot 1 should be in use (first-come-first-served).
    assert set(bindings.values()) == {0, 1}
    # Last obs should have decoded world positions for both bound slots.
    bound_slots = list(obs.slot_to_object_id.keys())
    assert len(bound_slots) == 2
    for s in bound_slots:
        assert not np.isnan(obs.decoded_positions_world[s, 0])


def test_detector_to_tracker_handles_occlusion():
    """Scripted occlusion: tag 1 disappears after step 2."""
    bundle = mock_calibration()

    def script(step_idx, frame):
        if step_idx < 3:
            return [(0, (0.02, 0.0)), (1, (-0.02, 0.0))]
        return [(0, (0.02, 0.0))]

    det = MockFiducialDetector(bundle=bundle, script=script)
    tracker = RollingObjectFileTracker(
        bundle, n_slots=4, slot_dim=16, K=5, missing_policy="nan")

    last_obs = None
    for _ in range(6):
        frame = _frame()
        fids = det.step(frame)
        last_obs = tracker.step(frame, fids)

    # Tag 1's slot remains in the identity binding (address persistence)
    bindings = tracker.identity_bindings
    assert 1 in bindings
    # ...but with missing_policy="nan", its current decoded pos is NaN.
    slot_for_1 = bindings[1]
    assert np.isnan(last_obs.decoded_positions_world[slot_for_1, 0])
    # Tag 0 is still observed.
    assert 0 in last_obs.slot_to_object_id.values()
