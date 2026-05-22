"""BF-0.12: SAM 2.1 + watchdog sensor-noise robustness sweep.

For each Gaussian noise σ (per-channel, in 0-255 scale):
  1. Synthesize a noisy variant of the flat demo's JPEGs
  2. Run SAM 2.1 Hiera-Tiny + fiducial watchdog
  3. Measure mean 3D error vs GT, n_valid_frames, n_reseed_events
  4. Report

Pre-committed gate: at σ=20 (moderate real-camera noise), mean error
< 2 cm AND n_reseed <= 3.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

sys.path.insert(0, "/workspace/bla_repo")
from bla.forge.sam_perception import SAMPerception, SAMSeed

DEMO_DIR = Path("/workspace/bf15_flat_demo")
SRC_FRAMES = DEMO_DIR / "frames"
OUT_BASE = Path("/workspace/bf12_noise_sweep")
PLANE_Z = 0.86

NOISE_SIGMAS = [0, 5, 10, 20, 40]

demo = np.load(DEMO_DIR / "demo.npz")
K = demo["K"]; R_wc = demo["R_wc"]; t_wc = demo["t_wc"]
can_gt = demo["can_gt"]
n_frames = can_gt.shape[0]


def project_world_to_pixel(p):
    p_cam = R_wc.T @ (p - t_wc)
    x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
    return (float(K[0, 0] * x / z + K[0, 2]),
            float(K[1, 1] * y / z + K[1, 2]))


def unproject_pixel_to_plane(u, v, plane_z):
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_mj = np.array([ray_cam[0], -ray_cam[1], -ray_cam[2]])
    ray_world = R_wc @ ray_mj
    if abs(ray_world[2]) < 1e-9: return None
    s = (plane_z - t_wc[2]) / ray_world[2]
    return t_wc + s * ray_world


def synthesize_noisy(sigma: float, frames_dir: Path):
    """Write noisy variant of each frame to frames_dir as JPEG."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)  # deterministic
    for i in range(n_frames):
        src = np.asarray(Image.open(SRC_FRAMES / f"{i:05d}.jpg").convert("RGB"))
        if sigma > 0:
            noise = rng.normal(0, sigma, size=src.shape)
            noisy = np.clip(src.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            noisy = src
        Image.fromarray(noisy).save(frames_dir / f"{i:05d}.jpg", quality=95)


def fiducial_fallback(frame_idx, obj_id):
    if frame_idx >= n_frames: return None
    return project_world_to_pixel(can_gt[frame_idx])


def evaluate(sigma: float):
    print(json.dumps({"event": "synth_start", "sigma": sigma}), flush=True)
    frames_dir = OUT_BASE / f"sigma_{int(sigma):03d}"
    if not (frames_dir / f"{n_frames - 1:05d}.jpg").exists():
        synthesize_noisy(sigma, frames_dir)
    print(json.dumps({"event": "synth_done", "sigma": sigma}), flush=True)

    u0, v0 = project_world_to_pixel(can_gt[0])
    t0 = time.time()
    sam = SAMPerception(
        video_path=frames_dir,
        seeds=[SAMSeed(obj_id=1, pixel_uv=(u0, v0))],
        backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        fiducial_fallback_fn=fiducial_fallback,
        silence_threshold=3,
    )
    elapsed = time.time() - t0

    # Compute per-frame metrics
    errs = []
    n_valid = 0
    for i in range(n_frames):
        dets = sam.detect(i)
        good = [d for d in dets if d.confidence > 0]
        if not good: continue
        d = good[0]
        u_c, v_c = float(d.center_px[0]), float(d.center_px[1])
        p_world = unproject_pixel_to_plane(u_c, v_c, PLANE_Z)
        if p_world is None: continue
        err = float(np.linalg.norm(p_world[:2] - can_gt[i][:2]))
        errs.append(err)
        n_valid += 1

    result = {
        "sigma": int(sigma),
        "elapsed_s": elapsed,
        "n_frames": int(n_frames),
        "n_valid": int(n_valid),
        "mean_err_cm": float(np.mean(errs) * 100) if errs else float("inf"),
        "median_err_cm": float(np.median(errs) * 100) if errs else float("inf"),
        "max_err_cm": float(np.max(errs) * 100) if errs else float("inf"),
        "n_reseed_events": len(sam.reseed_events),
        "reseed_events": sam.reseed_events,
    }
    print(json.dumps({"event": "result", **result}), flush=True)
    # Free GPU memory between runs
    del sam
    torch.cuda.empty_cache()
    return result


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = []
    for sigma in NOISE_SIGMAS:
        results.append(evaluate(sigma))

    print(json.dumps({"event": "summary"}), flush=True)
    print(f"{'sigma':>6} {'n_valid':>8} {'mean_cm':>9} "
          f"{'max_cm':>8} {'n_reseed':>9}")
    for r in results:
        print(f"{r['sigma']:>6d} {r['n_valid']:>8d} "
              f"{r['mean_err_cm']:>9.2f} {r['max_err_cm']:>8.2f} "
              f"{r['n_reseed_events']:>9d}")

    g_sigma20 = next(r for r in results if r["sigma"] == 20)
    gate_passed = (g_sigma20["mean_err_cm"] < 2.0
                    and g_sigma20["n_reseed_events"] <= 3)
    print(json.dumps({"event": "verdict",
                       "sigma20_mean_err_cm": g_sigma20["mean_err_cm"],
                       "sigma20_n_reseed": g_sigma20["n_reseed_events"],
                       "gate_passed": gate_passed}), flush=True)

    with open(OUT_BASE / "results.json", "w") as f:
        json.dump({"results": results, "gate_passed": gate_passed},
                  f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))


if __name__ == "__main__":
    main()
