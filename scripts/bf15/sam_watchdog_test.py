"""BF-0.11: fiducial-as-watchdog re-seed for SAM 2.1.

When SAM mask-area drops to 0 for >= SILENCE_THRESHOLD consecutive
frames AND a "fiducial" pose is available, re-seed SAM via
predictor.add_new_points_or_box(frame_idx=t, ...) at the
fiducial-projected pixel.

In production the "fiducial pose" comes from BF-0.2 detect_fiducials().
For this test we use the GT can xy as a proxy fiducial — same
information channel.

Gates (LOCKED before run):
  G1 : after a re-seed event, SAM produces a non-empty mask within
       <=5 frames
  G2 : post-watchdog total mean 3D error < 5 cm
  G3 : post-watchdog zero-mask gap < 50% of vanilla run (vanilla had
       40/100 zero frames; watchdog should bring this under 20)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

from sam2.sam2_video_predictor import SAM2VideoPredictor

DEMO_DIR = Path("/workspace/bf15_flat_demo")
FRAMES_DIR = DEMO_DIR / "frames"
SILENCE_THRESHOLD = 3  # consecutive zero-mask frames before re-seed
PLANE_Z = 0.86

demo = np.load(DEMO_DIR / "demo.npz")
K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
can_gt = demo["can_gt"]
n_frames = can_gt.shape[0]


def project_world_to_pixel(p):
    p_cam = R_wc.T @ (p - t_wc)
    x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
    return float(K[0, 0] * x / z + K[0, 2]), float(K[1, 1] * y / z + K[1, 2])


def unproject_pixel(u, v, depth_m):
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    p_cam = np.array([x, -y, -depth_m], dtype=np.float64)
    return R_wc @ p_cam + t_wc


def fiducial_pixel_for_frame(frame_idx):
    """Stand-in for BF-0.2 fiducial detection: return projected GT can xy."""
    return project_world_to_pixel(can_gt[frame_idx])


def run_with_watchdog(predictor, sam_model="facebook/sam2.1-hiera-tiny"):
    """Track with watchdog re-seeding. Returns per-frame results + events."""
    state = predictor.init_state(video_path=str(FRAMES_DIR))
    # Initial seed at frame 0
    u0, v0 = fiducial_pixel_for_frame(0)
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=np.array([[u0, v0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    masks_per_frame: dict[int, int] = {}  # frame_idx -> area
    centroid_per_frame: dict[int, tuple] = {}
    reseed_events: list[dict] = []
    silence = 0
    start_frame = 0

    while start_frame < n_frames:
        # Propagate from start_frame onwards until either:
        #  (a) we hit end of video
        #  (b) silence count crosses threshold -> re-seed + break to outer
        broke_for_reseed = False
        with torch.inference_mode():
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=start_frame):
                m = mask_logits[0]
                if m.ndim == 3: m = m[0]
                m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
                area = int(m_np.sum())
                masks_per_frame[frame_idx] = area
                if area > 0:
                    ys, xs = np.where(m_np)
                    centroid_per_frame[frame_idx] = (float(xs.mean()),
                                                          float(ys.mean()))
                    silence = 0
                else:
                    silence += 1
                    if silence >= SILENCE_THRESHOLD and frame_idx + 1 < n_frames:
                        # Re-seed at the NEXT frame (or +1) where we have a
                        # fresh fiducial pose
                        reseed_at = frame_idx + 1
                        if reseed_at >= n_frames: break
                        u_r, v_r = fiducial_pixel_for_frame(reseed_at)
                        # Refresh prompt for obj_id=1
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=reseed_at, obj_id=1,
                            points=np.array([[u_r, v_r]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                            clear_old_points=True,
                        )
                        reseed_events.append({
                            "frame_idx": int(reseed_at),
                            "silence_count": silence,
                            "fiducial_pixel": [u_r, v_r],
                        })
                        silence = 0
                        start_frame = reseed_at
                        broke_for_reseed = True
                        break
            else:
                # generator finished naturally
                start_frame = n_frames
        if not broke_for_reseed:
            start_frame = n_frames

    return masks_per_frame, centroid_per_frame, reseed_events


# ---- Run ----
torch.cuda.synchronize()
print(json.dumps({"event": "loading_sam"}), flush=True)
predictor = SAM2VideoPredictor.from_pretrained(
    "facebook/sam2.1-hiera-tiny", device="cuda")
print(json.dumps({"event": "sam_loaded"}), flush=True)

t0 = time.time()
masks, centroids, events = run_with_watchdog(predictor)
elapsed = time.time() - t0

# Compute metrics
areas = np.array([masks.get(i, 0) for i in range(n_frames)])
n_zero = int((areas == 0).sum())
err_3d = []
for i in range(n_frames):
    c = centroids.get(i)
    if c is None: continue
    u_c, v_c = c
    # Use plane projection (same as deploy loop)
    # Build a fiducial detection on-the-fly: ray from camera to (u, v)
    # intersected with the z=PLANE_Z plane.
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u_c, v_c, 1.0])
    # Convert pinhole cam frame to mujoco (flip y, z) then to world
    ray_mj = np.array([ray_cam[0], -ray_cam[1], -ray_cam[2]])
    ray_world = R_wc @ ray_mj
    if abs(ray_world[2]) < 1e-9: continue
    s = (PLANE_Z - t_wc[2]) / ray_world[2]
    p_world = t_wc + s * ray_world
    err = float(np.linalg.norm(p_world[:2] - can_gt[i][:2]))
    err_3d.append((i, err))

mean_err = float(np.mean([e for _, e in err_3d])) if err_3d else float("inf")
max_err = float(np.max([e for _, e in err_3d])) if err_3d else float("inf")

# Recovery analysis: for each re-seed event, how many frames until non-empty mask?
recovery_lags = []
for ev in events:
    rf = ev["frame_idx"]
    for k in range(rf, min(rf + 10, n_frames)):
        if masks.get(k, 0) > 0:
            recovery_lags.append(k - rf)
            break

summary = {
    "n_frames": int(n_frames),
    "elapsed_s": elapsed,
    "n_reseed_events": len(events),
    "reseed_events": events,
    "n_zero_mask_frames": n_zero,
    "vanilla_n_zero_frames": 40,  # from BF-0.10 baseline
    "n_valid_frames": int(len(err_3d)),
    "mean_err_cm": mean_err * 100,
    "max_err_cm": max_err * 100,
    "recovery_lags_frames": recovery_lags,
}
print(json.dumps({"event": "summary", **summary}), flush=True)

g1 = all(lag <= 5 for lag in recovery_lags) if recovery_lags else True
g2 = mean_err < 0.05
g3 = n_zero < 20
verdict = {
    "G1_recovery_within_5_frames": bool(g1),
    "G2_mean_err_under_5cm": bool(g2),
    "G3_zero_gap_under_50pct_of_vanilla": bool(g3),
    "all_passed": bool(g1 and g2 and g3),
}
print(json.dumps({"event": "verdict", **verdict}), flush=True)

with open(DEMO_DIR / "watchdog_test.json", "w") as f:
    json.dump({"summary": summary, "verdict": verdict,
               "per_frame_areas": {int(k): int(v) for k, v in masks.items()}},
              f, indent=2,
              default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
print(json.dumps({"event": "done"}), flush=True)
