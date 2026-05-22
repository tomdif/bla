"""
BF-0.7-followup-2: sweep SAM 2.1 backbones for accuracy × latency tradeoff.

For each backbone (tiny → small → base+ → large): run the same calibration
on the saved Can demo and measure (mean_err_m, p95_ms).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

from sam2.sam2_video_predictor import SAM2VideoPredictor

DEMO_DIR = "/workspace/bf07_sam_calibration"
FRAMES_DIR = f"{DEMO_DIR}/frames"

demo = np.load(f"{DEMO_DIR}/demo.npz")
K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
can_gt = demo["can_gt"]
depth_m_all = demo["depth_m"]
n_frames = depth_m_all.shape[0]


def project_world_to_pixel(p_world, K, R_world_cam, t_world_cam):
    p_cam = R_world_cam.T @ (p_world - t_world_cam)
    x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
    return float(K[0, 0] * x / z + K[0, 2]), float(K[1, 1] * y / z + K[1, 2])


def unproject_pixel_to_world(u, v, depth_m, K, R_world_cam, t_world_cam):
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    z = depth_m
    p_cam = np.array([x, -y, -z], dtype=np.float64)
    return R_world_cam @ p_cam + t_world_cam


u0, v0 = project_world_to_pixel(can_gt[0], K, R_wc, t_wc)


def benchmark(model_id: str):
    print(json.dumps({"event": "loading", "model_id": model_id}), flush=True)
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    state = predictor.init_state(video_path=FRAMES_DIR)
    torch.cuda.synchronize()
    init_state_s = time.time() - t0

    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=np.array([[u0, v0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    errs = []
    per_frame_ms = []
    t_prev = time.time()
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            torch.cuda.synchronize()
            t_now = time.time()
            per_frame_ms.append((t_now - t_prev) * 1000.0)
            t_prev = t_now

            m = mask_logits[0]
            if m.ndim == 3: m = m[0]
            m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
            if m_np.sum() == 0:
                errs.append(np.nan); continue
            ys, xs = np.where(m_np)
            u_c, v_c = float(xs.mean()), float(ys.mean())
            depth_med = float(np.median(depth_m_all[frame_idx][ys, xs]))
            p_est = unproject_pixel_to_world(u_c, v_c, depth_med, K, R_wc, t_wc)
            errs.append(float(np.linalg.norm(p_est - can_gt[frame_idx])))

    per_frame_ms = np.array(per_frame_ms)
    errs = np.array(errs)
    steady = per_frame_ms[1:]
    out = {
        "model_id": model_id,
        "init_state_s": init_state_s,
        "seed_pixel": [u0, v0],
        "steady_mean_ms": float(steady.mean()),
        "steady_p95_ms": float(np.percentile(steady, 95)),
        "steady_max_ms": float(steady.max()),
        "sustained_fps_mean": float(1000.0 / steady.mean()),
        "mean_err_cm": float(np.nanmean(errs) * 100),
        "median_err_cm": float(np.nanmedian(errs) * 100),
        "max_err_cm": float(np.nanmax(errs) * 100),
        "n_valid": int((~np.isnan(errs)).sum()),
        "n_frames": int(len(errs)),
    }
    return out


configs = [
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
]
results = []
for c in configs:
    try:
        r = benchmark(c)
        print(json.dumps({"event": "backbone_done", **r}), flush=True)
        results.append(r)
        torch.cuda.empty_cache()
    except Exception as e:
        print(json.dumps({"event": "backbone_error", "model_id": c,
                           "err": str(e)}), flush=True)
        torch.cuda.empty_cache()

# Pareto summary
print(json.dumps({"event": "summary"}), flush=True)
print(f"{'backbone':<40} {'mean_err_cm':>11} {'max_err_cm':>10} "
      f"{'p95_ms':>8} {'fps':>6}")
for r in results:
    name = r["model_id"].replace("facebook/sam2.1-hiera-", "")
    print(f"{name:<40} {r['mean_err_cm']:>11.2f} {r['max_err_cm']:>10.2f} "
          f"{r['steady_p95_ms']:>8.1f} {r['sustained_fps_mean']:>6.1f}")

with open(f"{DEMO_DIR}/backbone_sweep.json", "w") as f:
    json.dump({"results": results}, f, indent=2)
print(json.dumps({"event": "done"}), flush=True)
