"""BF-0.13: SAM 2.1 + watchdog lighting brightness + JPEG quality sweep.

Lighting sweep:
  scale ∈ {0.3, 0.5, 1.0, 1.5, 2.5}
    0.3  = very dim (poor warehouse light)
    0.5  = dim
    1.0  = baseline
    1.5  = bright
    2.5  = overexposed

JPEG quality sweep:
  q ∈ {10, 30, 50, 80, 95}
    10 = severe compression artifacts (network-strained stream)
    30 = visible artifacts
    50 = moderate
    80 = good
    95 = baseline

Gate: at lighting 0.5 AND JPEG q=30, mean error stays < 2 cm.
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
OUT_BASE = Path("/workspace/bf13_image_quality")
PLANE_Z = 0.86

LIGHTING_SCALES = [0.3, 0.5, 1.0, 1.5, 2.5]
JPEG_QUALITIES = [10, 30, 50, 80, 95]

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


def synthesize_lighting(scale: float, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        src = np.asarray(Image.open(SRC_FRAMES / f"{i:05d}.jpg").convert("RGB"))
        scaled = np.clip(src.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        Image.fromarray(scaled).save(frames_dir / f"{i:05d}.jpg", quality=95)


def synthesize_jpeg(quality: int, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        src = Image.open(SRC_FRAMES / f"{i:05d}.jpg").convert("RGB")
        src.save(frames_dir / f"{i:05d}.jpg", quality=quality)


def fiducial_fallback(frame_idx, obj_id):
    if frame_idx >= n_frames: return None
    return project_world_to_pixel(can_gt[frame_idx])


def run_one(frames_dir: Path, label: str):
    u0, v0 = project_world_to_pixel(can_gt[0])
    t0 = time.time()
    sam = SAMPerception(
        video_path=frames_dir,
        seeds=[SAMSeed(obj_id=1, pixel_uv=(u0, v0))],
        backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        fiducial_fallback_fn=fiducial_fallback, silence_threshold=3,
    )
    elapsed = time.time() - t0
    errs, n_valid = [], 0
    for i in range(n_frames):
        dets = sam.detect(i)
        good = [d for d in dets if d.confidence > 0]
        if not good: continue
        d = good[0]
        p_world = unproject_pixel_to_plane(d.center_px[0], d.center_px[1],
                                                 PLANE_Z)
        if p_world is None: continue
        errs.append(float(np.linalg.norm(p_world[:2] - can_gt[i][:2])))
        n_valid += 1
    r = {
        "label": label,
        "elapsed_s": elapsed,
        "n_valid": n_valid,
        "mean_err_cm": float(np.mean(errs) * 100) if errs else float("inf"),
        "median_err_cm": float(np.median(errs) * 100) if errs else float("inf"),
        "max_err_cm": float(np.max(errs) * 100) if errs else float("inf"),
        "n_reseed_events": len(sam.reseed_events),
    }
    del sam; torch.cuda.empty_cache()
    return r


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # --- Lighting ---
    lighting_results = []
    for s in LIGHTING_SCALES:
        d = OUT_BASE / f"lighting_{int(s * 10):03d}"
        if not (d / f"{n_frames - 1:05d}.jpg").exists():
            synthesize_lighting(s, d)
        r = run_one(d, f"scale_{s}")
        r["scale"] = s
        print(json.dumps({"event": "lighting_result", **r}), flush=True)
        lighting_results.append(r)

    # --- JPEG quality ---
    jpeg_results = []
    for q in JPEG_QUALITIES:
        d = OUT_BASE / f"jpeg_q{q:03d}"
        if not (d / f"{n_frames - 1:05d}.jpg").exists():
            synthesize_jpeg(q, d)
        r = run_one(d, f"q_{q}")
        r["quality"] = q
        print(json.dumps({"event": "jpeg_result", **r}), flush=True)
        jpeg_results.append(r)

    print(json.dumps({"event": "lighting_summary"}), flush=True)
    print(f"{'scale':>5} {'n_valid':>8} {'mean_cm':>8} {'max_cm':>8} {'n_re':>5}")
    for r in lighting_results:
        print(f"{r['scale']:>5.1f} {r['n_valid']:>8d} "
              f"{r['mean_err_cm']:>8.2f} {r['max_err_cm']:>8.2f} "
              f"{r['n_reseed_events']:>5d}")

    print(json.dumps({"event": "jpeg_summary"}), flush=True)
    print(f"{'q':>4} {'n_valid':>8} {'mean_cm':>8} {'max_cm':>8} {'n_re':>5}")
    for r in jpeg_results:
        print(f"{r['quality']:>4d} {r['n_valid']:>8d} "
              f"{r['mean_err_cm']:>8.2f} {r['max_err_cm']:>8.2f} "
              f"{r['n_reseed_events']:>5d}")

    # Gate
    L05 = next(r for r in lighting_results if r["scale"] == 0.5)
    JQ30 = next(r for r in jpeg_results if r["quality"] == 30)
    gate_lighting = L05["mean_err_cm"] < 2.0
    gate_jpeg = JQ30["mean_err_cm"] < 2.0
    print(json.dumps({"event": "verdict",
                       "lighting_0.5_mean_cm": L05["mean_err_cm"],
                       "jpeg_q30_mean_cm": JQ30["mean_err_cm"],
                       "gate_lighting_passed": bool(gate_lighting),
                       "gate_jpeg_passed": bool(gate_jpeg),
                       "all_passed": bool(gate_lighting and gate_jpeg)}),
          flush=True)

    with open(OUT_BASE / "results.json", "w") as f:
        json.dump({"lighting": lighting_results, "jpeg": jpeg_results,
                    "gate_lighting_passed": gate_lighting,
                    "gate_jpeg_passed": gate_jpeg},
                  f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))


if __name__ == "__main__":
    main()
