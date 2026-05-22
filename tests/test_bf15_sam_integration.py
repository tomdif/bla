"""BF-1.5 unit tests — SAMPerception wire interface + SAM deployment loop.

Real SAM 2.1 with CUDA is exercised by the pod-side integration script
at scripts/bf15_sam_integration_pod.py. This file tests the in-process
mock backend so the wire interface is verifiable without GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    DemoBank, DemoRecord, build_mock_demo, save_demo,
)
from bla.forge.calibration import mock_calibration
from bla.forge.fiducials import FiducialDetection
from bla.forge.sam_perception import SAMPerception, SAMSeed
from bla.forge.deploy_loop_sam import build_sam_deployment_loop


# ---------- SAMPerception (mock backend) ----------

def test_sam_perception_mock_emits_fiducial_detections():
    sam = SAMPerception(
        video_path="",
        seeds=[SAMSeed(obj_id=1, pixel_uv=(240.0, 200.0)),
               SAMSeed(obj_id=2, pixel_uv=(100.0, 300.0))],
        backend="mock_static",
    )
    dets = sam.detect(frame_idx=0)
    assert len(dets) == 2
    assert all(isinstance(d, FiducialDetection) for d in dets)
    ids = sorted(d.id for d in dets)
    assert ids == [1, 2]
    d0 = next(d for d in dets if d.id == 1)
    assert d0.pixel_corners.shape == (4, 2)
    assert d0.center_px.shape == (2,)
    np.testing.assert_allclose(d0.center_px, [240.0, 200.0])
    assert d0.confidence == 1.0
    assert d0.family == "sam2_track"


def test_sam_perception_mock_deterministic_across_frames():
    sam = SAMPerception(
        video_path="", seeds=[SAMSeed(obj_id=1, pixel_uv=(240.0, 200.0))],
        backend="mock_static",
    )
    d_a = sam.detect(0)
    d_b = sam.detect(42)
    np.testing.assert_allclose(d_a[0].center_px, d_b[0].center_px)
    assert d_a[0].confidence == d_b[0].confidence == 1.0


def test_sam_perception_rejects_empty_seeds():
    with pytest.raises(ValueError, match="at least one seed"):
        SAMPerception(video_path="", seeds=[], backend="mock_static")


def test_sam_perception_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        SAMPerception(video_path="", seeds=[SAMSeed(1, (0.0, 0.0))],
                       backend="bogus")


def test_sam_perception_accepts_watchdog_args():
    """BF-0.11: SAMPerception takes fiducial_fallback_fn + silence_threshold
    parameters; mock backend doesn't exercise them but the API surface
    must be present for the sam2.1 backend to use."""
    def fid_fn(frame_idx, obj_id):
        return (100.0, 200.0)
    sam = SAMPerception(
        video_path="",
        seeds=[SAMSeed(obj_id=1, pixel_uv=(50.0, 50.0))],
        backend="mock_static",
        fiducial_fallback_fn=fid_fn,
        silence_threshold=5,
    )
    assert sam.fiducial_fallback_fn is fid_fn
    assert sam.silence_threshold == 5
    # Reseed events list initializes empty
    assert sam.reseed_events == []


# ---------- build_sam_deployment_loop end-to-end (mock backend) ----------

def _build_bank_with_one_demo(tmp_path: Path) -> DemoBank:
    rec = build_mock_demo(demo_id=0, cube_xy=(0.0, 0.0),
                            achieved_outcome=0.5)
    save_demo(rec, tmp_path / "demo_0000.json")
    return DemoBank.from_directory(tmp_path)


def test_sam_loop_mock_runs_end_to_end(tmp_path: Path):
    """Wire interface: SAM-mock perception → tracker → retrieval → logger."""
    bank = _build_bank_with_one_demo(tmp_path)
    bundle = mock_calibration()
    from bla.forge.calibration import project_world_to_image
    seed_px = project_world_to_image(np.array([0.0, 0.0, 0.0]), bundle)

    def key_fn(scene):
        if 0 not in scene:
            return np.zeros(2, dtype=np.float32)
        return scene[0].astype(np.float32)

    record = build_sam_deployment_loop(
        bank=bank, key_fn=key_fn,
        sam_video_path="",
        seeds=[SAMSeed(obj_id=0, pixel_uv=(float(seed_px[0]),
                                              float(seed_px[1])))],
        bundle=bundle,
        ep_id=0, task="pickplace",
        sam_backend="mock_static",
        max_steps=20,
        include_safety=False,
    )
    assert record is not None
    assert record.retrieved_demo is not None
    assert record.retrieved_demo["demo_id"] == 0
    assert len(record.gantry_actions) > 0
    assert len(record.gantry_actions) <= 20
    assert record.router_decision["recipe"] == "E2_FAST"
    decoded = np.asarray(record.decoded_positions)
    assert decoded.ndim == 3 and decoded.shape[2] == 2


def test_sam_loop_mock_retrieval_nn_distance_is_small(tmp_path: Path):
    """SAM-derived query key should match the bank key for the matching
    demo to ~mm precision (mock backend is deterministic)."""
    bank = _build_bank_with_one_demo(tmp_path)
    bundle = mock_calibration()
    from bla.forge.calibration import project_world_to_image
    seed_px = project_world_to_image(np.array([0.0, 0.0, 0.0]), bundle)

    def key_fn(scene):
        if 0 not in scene:
            return np.zeros(2, dtype=np.float32)
        return scene[0].astype(np.float32)

    record = build_sam_deployment_loop(
        bank=bank, key_fn=key_fn,
        sam_video_path="",
        seeds=[SAMSeed(obj_id=0, pixel_uv=(float(seed_px[0]),
                                              float(seed_px[1])))],
        bundle=bundle, ep_id=0,
        sam_backend="mock_static", max_steps=5,
        include_safety=False,
    )
    assert record.retrieved_demo["nn_distance"] < 0.01
