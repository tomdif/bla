"""BF-1.5 integration test — run the full SAM-driven deployment loop
on the BF-0.7 PickPlaceCan saved demo and validate end-to-end.

Gates:
  G1 : SAM-driven loop produces a real-shape EpisodeRecord (no crashes)
  G2 : decoded_positions track ground-truth Can xy within 5 cm RMS
  G3 : SAM perception successfully retrieves the (single) demo in the bank
  G4 : episode logs at least N=50 steps (perception doesn't lose the can early)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/bla_repo")
os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

# Force test imports BEFORE building the bundle
from bla.forge.calibration import (
    CalibrationBundle, CameraIntrinsics, CameraExtrinsics,
)
from bla.forge.demo_bank import DemoBank, DemoRecord
from bla.forge.sam_perception import SAMSeed
from bla.forge.deploy_loop_sam import build_sam_deployment_loop


BF07_DIR = Path("/workspace/bf15_flat_demo")  # XY-only demo for plane projection
FRAMES_DIR = BF07_DIR / "frames"


def bundle_from_demo() -> CalibrationBundle:
    """Construct a CalibrationBundle from the demo's camera params."""
    demo = np.load(BF07_DIR / "demo.npz")
    K = demo["K"]
    R_wc = demo["R_wc"]
    t_wc = demo["t_wc"]
    image_size = (480, 480)
    # MuJoCo cam: +x right, +y up, looks -z. Pinhole expects +x right, +y down,
    # +z forward. The R_world_cam matrix here is in MuJoCo cam convention.
    # We need to express it as standard pinhole (the calibration module's
    # K + extrinsics use the OpenCV convention).
    # Flip y and z axes of cam frame to get OpenCV.
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_world_pinhole = R_wc @ flip
    T_cam_to_world = np.eye(4, dtype=np.float64)
    T_cam_to_world[:3, :3] = R_world_pinhole
    T_cam_to_world[:3, 3] = t_wc
    intr = CameraIntrinsics(
        camera_matrix=K, distortion_coeffs=np.zeros(5),
        image_size_wh=image_size,
    )
    extr = CameraExtrinsics(T_cam_to_world=T_cam_to_world)
    return CalibrationBundle(intrinsics=intr, extrinsics=extr,
                              metadata={"source": "bf07_pickplacecan"})


def main():
    print(json.dumps({"event": "start"}), flush=True)
    bundle = bundle_from_demo()

    # The Can rest plane is z = 0.86 in MuJoCo world. For our test we project
    # to that plane so the SAM-derived pixel maps back to the can's actual
    # world position.
    can_plane_z = 0.86
    print(json.dumps({"event": "bundle_built",
                       "image_size_wh": list(bundle.intrinsics.image_size_wh),
                       "fx": bundle.intrinsics.fx, "fy": bundle.intrinsics.fy,
                       "cam_pos_world": bundle.extrinsics.T_cam_to_world[:3, 3].tolist()}),
          flush=True)

    # Seed pixel: the projected GT can position on frame 0
    demo = np.load(BF07_DIR / "demo.npz")
    can_gt = demo["can_gt"]  # (T, 3)
    # Recompute the seed pixel from MuJoCo math to match BF-0.7
    # (we have K, R_wc, t_wc from demo)
    K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
    p_cam = R_wc.T @ (can_gt[0] - t_wc)
    x_pin = p_cam[0]; y_pin = -p_cam[1]; z_pin = -p_cam[2]
    u0 = K[0, 0] * x_pin / z_pin + K[0, 2]
    v0 = K[1, 1] * y_pin / z_pin + K[1, 2]
    print(json.dumps({"event": "seed_computed",
                       "seed_pixel": [float(u0), float(v0)],
                       "can_world_0": can_gt[0].tolist()}), flush=True)

    # Build a tiny demo bank with 1 demo. The demo records the can's initial
    # position so retrieval matches.
    bank_dir = Path("/tmp/bf15_test_bank")
    if bank_dir.exists():
        for p in bank_dir.glob("*.json"): p.unlink()
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank = DemoBank(records=[], directory=bank_dir)
    n_demo_steps = 100
    dummy_actions = np.zeros((n_demo_steps, 7), dtype=np.float32)
    dummy_actions[:, 0] = 0.01
    rec = DemoRecord(
        demo_id=0,
        task="pickplace",
        initial_state={
            "fiducials": {"1": [float(can_gt[0, 0]), float(can_gt[0, 1]),
                                 float(can_gt[0, 2])]},
        },
        actions=dummy_actions,
        achieved_outcome=0.5,
    )
    bank.add(rec)
    print(json.dumps({"event": "bank_built", "n_demos": len(bank)}), flush=True)

    # Same key_fn at indexing time and at query time
    def key_fn(scene: dict) -> np.ndarray:
        # Use fiducial id 1 → its world xy. Single-object case.
        if 1 not in scene:
            # No detection of the target this frame; return a zero key
            return np.zeros(2, dtype=np.float32)
        return scene[1].astype(np.float32)

    # Run with REAL SAM backend
    print(json.dumps({"event": "running_sam_loop"}), flush=True)
    t0 = time.time()
    record = build_sam_deployment_loop(
        bank=bank,
        key_fn=key_fn,
        sam_video_path=FRAMES_DIR,
        seeds=[SAMSeed(obj_id=1, pixel_uv=(float(u0), float(v0)))],
        bundle=bundle,
        ep_id=0, task="pickplace",
        sam_backend="sam2.1",
        sam_model="facebook/sam2.1-hiera-tiny",
        world_plane_z=can_plane_z,
        max_steps=n_demo_steps,
        include_safety=False,  # mock workspace is sized for table origin,
                                # not the BF-0.7 PickPlaceCan world frame
    )
    elapsed = time.time() - t0
    print(json.dumps({"event": "loop_done", "elapsed_s": elapsed}), flush=True)

    # --- Gate eval ---
    # G1: real-shape EpisodeRecord
    g1_ok = (record is not None
             and hasattr(record, "frames")
             and hasattr(record, "slot_states")
             and hasattr(record, "decoded_positions")
             and hasattr(record, "gantry_actions"))

    n_steps = len(record.gantry_actions) if g1_ok else 0
    print(json.dumps({"event": "schema_check", "n_steps": int(n_steps),
                       "g1_ok": bool(g1_ok)}), flush=True)

    # G2: decoded_positions xy vs ground-truth xy RMS (compare per-step
    # to the can_gt at that step, in world xy)
    if g1_ok:
        decoded = np.asarray(record.decoded_positions)  # [T, n_slots, 2]
        # Find the slot that's bound to obj_id=1 over time
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
        rms = float(np.sqrt(np.mean(np.square(rms_errors)))) if rms_errors else float("inf")
        mean_err = float(np.mean(rms_errors)) if rms_errors else float("inf")
        max_err = float(np.max(rms_errors)) if rms_errors else float("inf")
        g2_ok = rms < 0.05
        print(json.dumps({"event": "xy_error",
                           "n_compared": len(rms_errors),
                           "rms_m": rms, "mean_m": mean_err, "max_m": max_err,
                           "g2_ok": bool(g2_ok)}), flush=True)
    else:
        g2_ok = False

    # G3: retrieved demo present in record
    g3_ok = record.retrieved_demo is not None and \
            record.retrieved_demo.get("demo_id") == 0
    print(json.dumps({"event": "retrieval_check",
                       "retrieved_demo": record.retrieved_demo,
                       "g3_ok": bool(g3_ok)}), flush=True)

    # G4: at least 50 steps logged
    g4_ok = n_steps >= 50
    print(json.dumps({"event": "step_count", "n_steps": n_steps,
                       "g4_ok": bool(g4_ok)}), flush=True)

    verdict = {
        "G1_real_shape_record": bool(g1_ok),
        "G2_xy_rms_under_5cm": bool(g2_ok),
        "G3_retrieved_demo_present": bool(g3_ok),
        "G4_at_least_50_steps": bool(g4_ok),
        "all_passed": bool(g1_ok and g2_ok and g3_ok and g4_ok),
    }
    print(json.dumps({"event": "verdict", **verdict}), flush=True)


if __name__ == "__main__":
    main()
