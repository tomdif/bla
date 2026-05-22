"""
BF-0.7-followup: SAM 2.1 latency benchmark on the saved Can demo.

Measures:
  - init_state time (one-shot setup)
  - add_new_points_or_box time (one-shot seed)
  - per-frame propagation latency (mean, p50, p95, max, sustained FPS)

Pre-committed thresholds (locked before run):
  green   : per_frame_p95 < 33 ms   (30 Hz real-time)
  useful  : per_frame_p95 < 100 ms  (10 Hz, BLA-Forge control rate)
  needs_optim : per_frame_p95 > 100 ms
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
MODEL_ID = "facebook/sam2.1-hiera-large"

# Load demo to recover seed pixel
demo = np.load(f"{DEMO_DIR}/demo.npz")
rgb = demo["rgb"]; K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
can_gt = demo["can_gt"]
n_frames = rgb.shape[0]
print(json.dumps({"event": "demo_loaded", "n_frames": int(n_frames),
                   "shape": list(rgb.shape)}), flush=True)

# Project first-frame GT can pose to pixel (same code as calibration)
def project_world_to_pixel(p_world, K, R_world_cam, t_world_cam):
    p_cam = R_world_cam.T @ (p_world - t_world_cam)
    x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


u0, v0 = project_world_to_pixel(can_gt[0], K, R_wc, t_wc)
print(json.dumps({"event": "seed_pixel", "u": u0, "v": v0}), flush=True)

# Time load
t0 = time.time()
predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID, device="cuda")
load_s = time.time() - t0
print(json.dumps({"event": "model_loaded", "load_s": load_s}), flush=True)

# Time init_state
torch.cuda.synchronize()
t0 = time.time()
state = predictor.init_state(video_path=FRAMES_DIR)
torch.cuda.synchronize()
init_state_s = time.time() - t0
print(json.dumps({"event": "init_state", "init_state_s": init_state_s,
                   "per_frame_ms_amortized": (init_state_s / n_frames) * 1000.0}),
      flush=True)

# Time add_new_points_or_box
torch.cuda.synchronize()
t0 = time.time()
predictor.add_new_points_or_box(
    inference_state=state, frame_idx=0, obj_id=1,
    points=np.array([[u0, v0]], dtype=np.float32),
    labels=np.array([1], dtype=np.int32),
)
torch.cuda.synchronize()
seed_s = time.time() - t0
print(json.dumps({"event": "seed", "seed_s": seed_s}), flush=True)

# Time per-frame propagation: iterate the generator and measure wall-clock
# per yield. The first yield is the seed frame which may include redundant
# init work; we'll capture it separately.
per_frame_ms = []
t_prev = time.time()
with torch.inference_mode():
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        torch.cuda.synchronize()
        t_now = time.time()
        per_frame_ms.append((t_now - t_prev) * 1000.0)
        t_prev = t_now

per_frame_ms = np.array(per_frame_ms, dtype=np.float64)
# Drop the first frame (warmup / seed accounting) for the steady-state stats
steady = per_frame_ms[1:]

stats = {
    "n_frames_measured": int(len(per_frame_ms)),
    "first_frame_ms": float(per_frame_ms[0]),
    "steady_mean_ms": float(steady.mean()),
    "steady_median_ms": float(np.median(steady)),
    "steady_p95_ms": float(np.percentile(steady, 95)),
    "steady_max_ms": float(steady.max()),
    "steady_min_ms": float(steady.min()),
    "sustained_fps_mean": float(1000.0 / steady.mean()),
    "sustained_fps_p95": float(1000.0 / np.percentile(steady, 95)),
    "init_state_s": init_state_s,
    "init_state_per_frame_ms": init_state_s / n_frames * 1000.0,
    "seed_s": seed_s,
}
print(json.dumps({"event": "stats", **stats}), flush=True)

# Verdict
p95 = stats["steady_p95_ms"]
if p95 < 33.0:
    verdict = "GREEN_30HZ"
elif p95 < 100.0:
    verdict = "USEFUL_10HZ"
else:
    verdict = "NEEDS_OPTIM"
print(json.dumps({"event": "verdict", "verdict": verdict,
                   "p95_ms": p95,
                   "thresholds": {"green": "p95<33ms", "useful": "p95<100ms"}}),
      flush=True)

with open(f"{DEMO_DIR}/latency.json", "w") as f:
    json.dump({"stats": stats, "verdict": verdict,
               "per_frame_ms": per_frame_ms.tolist()}, f, indent=2)
print(json.dumps({"event": "done"}), flush=True)
