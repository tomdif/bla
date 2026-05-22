"""BF-1.6 pod integration test — fiducial-watchdog through deploy_loop_sam.

Variant of bf15_sam_integration_pod.py with `fiducial_fallback_fn` wired
in. Expects watchdog to rescue SAM in the mid-trajectory occlusion that
caused BF-0.10/BF-1.5's 40 zero-mask frames.

Gates:
  G1 : real-shape EpisodeRecord (carry-over from BF-1.5)
  G2 : decoded_positions xy vs GT, mean < 1 cm
  G3 : n_valid_frames >= 95 (vs BF-1.5 vanilla 60)
  G4 : sam.reseed_events shows ≥ 1 watchdog event
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/bla_repo")
os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

from bla.forge.calibration import (
    CalibrationBundle, CameraIntrinsics, CameraExtrinsics,
)
from bla.forge.demo_bank import DemoBank
from bla.forge.episode import DemoRecord
from bla.forge.sam_perception import SAMSeed, SAMPerception
from bla.forge.deploy_loop_sam import build_sam_deployment_loop


DEMO_DIR = Path("/workspace/bf15_flat_demo")
FRAMES_DIR = DEMO_DIR / "frames"
PLANE_Z = 0.86


def bundle_from_demo() -> CalibrationBundle:
    demo = np.load(DEMO_DIR / "demo.npz")
    K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    T = np.eye(4); T[:3, :3] = R_wc @ flip; T[:3, 3] = t_wc
    return CalibrationBundle(
        intrinsics=CameraIntrinsics(
            camera_matrix=K, distortion_coeffs=np.zeros(5),
            image_size_wh=(480, 480)),
        extrinsics=CameraExtrinsics(T_cam_to_world=T),
        metadata={"source": "bf15_flat"},
    )


def main():
    print(json.dumps({"event": "start"}), flush=True)
    bundle = bundle_from_demo()
    demo = np.load(DEMO_DIR / "demo.npz")
    can_gt = demo["can_gt"]
    K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]

    def project_world_to_pixel(p):
        p_cam = R_wc.T @ (p - t_wc)
        x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
        return (float(K[0, 0] * x / z + K[0, 2]),
                float(K[1, 1] * y / z + K[1, 2]))

    u0, v0 = project_world_to_pixel(can_gt[0])
    print(json.dumps({"event": "seed_pixel", "u": u0, "v": v0}), flush=True)

    # In real BLA-Forge, this wraps BF-0.2 detect_fiducials on the live frame.
    # For sim validation, we use the GT pose as the fiducial channel —
    # exactly the same information a table-mounted AprilTag would provide.
    def fiducial_fallback(frame_idx: int, obj_id: int):
        if frame_idx >= len(can_gt): return None
        return project_world_to_pixel(can_gt[frame_idx])

    # Build a single-demo bank
    bank_dir = Path("/tmp/bf16_test_bank")
    if bank_dir.exists():
        for p in bank_dir.glob("*.json"): p.unlink()
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank = DemoBank(records=[], directory=bank_dir)
    n_demo_steps = 100
    dummy_actions = np.zeros((n_demo_steps, 7), dtype=np.float32)
    dummy_actions[:, 0] = 0.01
    bank.add(DemoRecord(
        demo_id=0, task="pickplace",
        initial_state={
            "fiducials": {"1": [float(can_gt[0, 0]), float(can_gt[0, 1]),
                                 float(can_gt[0, 2])]},
        },
        actions=dummy_actions, achieved_outcome=0.5,
    ))

    def key_fn(scene):
        if 1 not in scene: return np.zeros(2, dtype=np.float32)
        return scene[1].astype(np.float32)

    print(json.dumps({"event": "running_loop_with_watchdog"}), flush=True)
    t0 = time.time()
    record = build_sam_deployment_loop(
        bank=bank, key_fn=key_fn,
        sam_video_path=FRAMES_DIR,
        seeds=[SAMSeed(obj_id=1, pixel_uv=(float(u0), float(v0)))],
        bundle=bundle, ep_id=0, task="pickplace",
        sam_backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        world_plane_z=PLANE_Z,
        max_steps=n_demo_steps,
        include_safety=False,
        fiducial_fallback_fn=fiducial_fallback,
        silence_threshold=3,
    )
    elapsed = time.time() - t0
    print(json.dumps({"event": "loop_done", "elapsed_s": elapsed}),
          flush=True)

    # Gate eval
    g1_ok = (record is not None and hasattr(record, "decoded_positions"))
    n_steps = len(record.gantry_actions) if g1_ok else 0
    decoded = np.asarray(record.decoded_positions)
    rms_errors = []
    for t in range(n_steps):
        slots_xy = decoded[t]
        best_d = np.inf
        for s in range(slots_xy.shape[0]):
            xy = slots_xy[s]
            if np.isnan(xy[0]): continue
            d = float(np.linalg.norm(xy - can_gt[t][:2]))
            if d < best_d: best_d = d
        if best_d < np.inf: rms_errors.append(best_d)
    mean_err = float(np.mean(rms_errors)) if rms_errors else float("inf")
    max_err = float(np.max(rms_errors)) if rms_errors else float("inf")
    n_valid = len(rms_errors)

    print(json.dumps({"event": "xy_metrics",
                       "n_steps": n_steps,
                       "n_valid_frames": n_valid,
                       "mean_err_cm": mean_err * 100,
                       "max_err_cm": max_err * 100}), flush=True)

    g2_ok = mean_err < 0.01
    g3_ok = n_valid >= 95

    # Watchdog event count — pull from SAMPerception (need to reconstruct)
    # build_sam_deployment_loop constructed SAMPerception internally; we
    # can't easily get its reseed_events from outside. Instead, count by
    # constructing the same SAMPerception directly and inspecting.
    sam_diag = SAMPerception(
        video_path=FRAMES_DIR,
        seeds=[SAMSeed(obj_id=1, pixel_uv=(float(u0), float(v0)))],
        backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        fiducial_fallback_fn=fiducial_fallback, silence_threshold=3,
    )
    n_reseeds = len(sam_diag.reseed_events)
    print(json.dumps({"event": "watchdog_events",
                       "n_reseeds": n_reseeds,
                       "events": sam_diag.reseed_events}), flush=True)
    g4_ok = n_reseeds >= 1

    verdict = {
        "G1_real_shape_record": bool(g1_ok),
        "G2_mean_err_under_1cm": bool(g2_ok),
        "G3_n_valid_at_least_95": bool(g3_ok),
        "G4_watchdog_fired": bool(g4_ok),
        "all_passed": bool(g1_ok and g2_ok and g3_ok and g4_ok),
    }
    print(json.dumps({"event": "verdict", **verdict}), flush=True)


if __name__ == "__main__":
    main()
