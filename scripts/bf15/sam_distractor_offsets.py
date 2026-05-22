"""BF-0.14 sweep: lookalike-proximity at offsets ∈ {5, 10, 20} cm.

  20 cm = comfortable (sanity baseline)
  10 cm = the gate offset (mean ≤ 1 cm target)
   5 cm = stress-not-gate (find breaking point)

Same parallel-trajectory setup as sam_distractor_proximity.py.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/bla_repo")
from sam_perception_calibration import (
    build_env, find_can_qpos_addr, camera_intrinsics, depth_image_to_meters,
)
from bla.forge.sam_perception import SAMPerception, SAMSeed
from PIL import Image


N_FRAMES = 100
IMG = 480
PLANE_Z = 0.86
OFFSETS_CM = [20, 10, 5]
OUT_BASE = Path("/workspace/bf14_offset_sweep")


def find_milk_qpos_addr(env):
    m = env.sim.model
    for jid in range(m.njnt):
        if (m.body_id2name(m.jnt_bodyid[jid]) == "Milk_main"
                and int(m.jnt_type[jid]) == 0):
            return int(m.jnt_qposadr[jid])
    raise RuntimeError()


def plan_flat_trajectory(can_start, n_frames):
    p0 = np.asarray(can_start, dtype=np.float64).copy()
    p_end = p0 + np.array([0.20, 0.30, 0.0])
    pts = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        t = 0.5 - 0.5 * np.cos(np.pi * t)
        pts.append(p0 * (1 - t) + p_end * t)
    return np.stack(pts)


def render_one(offset_m: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"; frames_dir.mkdir(exist_ok=True)
    env = build_env(IMG); env.reset()
    K, R_wc, t_wc = camera_intrinsics(env, IMG)
    can_qpos = find_can_qpos_addr(env)
    milk_qpos = find_milk_qpos_addr(env)
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    can_traj = plan_flat_trajectory(can_start, N_FRAMES)
    milk_traj = can_traj.copy()
    milk_traj[:, 1] += offset_m
    can_gt, milk_gt = [], []
    for i in range(N_FRAMES):
        env.sim.data.qpos[can_qpos:can_qpos + 3] = can_traj[i]
        env.sim.data.qpos[milk_qpos:milk_qpos + 3] = milk_traj[i]
        for jname in ("Can_joint0", "Milk_joint0"):
            qv = env.sim.model.jnt_dofadr[env.sim.model.joint_name2id(jname)]
            env.sim.data.qvel[qv:qv + 6] = 0.0
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(rgb)).save(
            frames_dir / f"{i:05d}.jpg", quality=95)
        can_gt.append(env.sim.data.get_body_xpos("Can_main").copy())
        milk_gt.append(env.sim.data.get_body_xpos("Milk_main").copy())
    return np.stack(can_gt), np.stack(milk_gt), K, R_wc, t_wc


def evaluate(offset_cm: int):
    print(json.dumps({"event": "render_start", "offset_cm": offset_cm}),
          flush=True)
    out_dir = OUT_BASE / f"off_{offset_cm:03d}cm"
    can_gt, milk_gt, K, R_wc, t_wc = render_one(offset_cm / 100, out_dir)

    def project_world_to_pixel(p):
        p_cam = R_wc.T @ (p - t_wc)
        x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
        return (float(K[0, 0] * x / z + K[0, 2]),
                float(K[1, 1] * y / z + K[1, 2]))

    def unproject_pixel(u, v, plane_z):
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([u, v, 1.0])
        ray_mj = np.array([ray_cam[0], -ray_cam[1], -ray_cam[2]])
        ray_world = R_wc @ ray_mj
        if abs(ray_world[2]) < 1e-9: return None
        s = (plane_z - t_wc[2]) / ray_world[2]
        return t_wc + s * ray_world

    sep_px = []
    for i in range(N_FRAMES):
        cu, cv = project_world_to_pixel(can_gt[i])
        mu, mv = project_world_to_pixel(milk_gt[i])
        sep_px.append(((cu - mu) ** 2 + (cv - mv) ** 2) ** 0.5)

    def fiducial_fallback(frame_idx, obj_id):
        return project_world_to_pixel(can_gt[frame_idx])

    u0, v0 = project_world_to_pixel(can_gt[0])
    t0 = time.time()
    sam = SAMPerception(
        video_path=out_dir / "frames",
        seeds=[SAMSeed(obj_id=1, pixel_uv=(u0, v0))],
        backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        fiducial_fallback_fn=fiducial_fallback, silence_threshold=3,
    )
    elapsed = time.time() - t0

    errs_to_can, drift_diffs = [], []
    n_valid = 0
    for i in range(N_FRAMES):
        dets = sam.detect(i)
        good = [d for d in dets if d.confidence > 0]
        if not good: continue
        d = good[0]
        u_c, v_c = float(d.center_px[0]), float(d.center_px[1])
        p = unproject_pixel(u_c, v_c, PLANE_Z)
        if p is None: continue
        ec = float(np.linalg.norm(p[:2] - can_gt[i][:2]))
        em = float(np.linalg.norm(p[:2] - milk_gt[i][:2]))
        errs_to_can.append(ec)
        drift_diffs.append(em - ec)  # +ve = closer to can (correct)
        n_valid += 1

    r = {
        "offset_cm": offset_cm,
        "elapsed_s": elapsed,
        "n_valid": int(n_valid),
        "n_reseed": len(sam.reseed_events),
        "mean_err_cm": float(np.mean(errs_to_can) * 100) if errs_to_can else float("inf"),
        "max_err_cm": float(np.max(errs_to_can) * 100) if errs_to_can else float("inf"),
        "median_err_cm": float(np.median(errs_to_can) * 100) if errs_to_can else float("inf"),
        "n_frames_closer_to_milk": int(sum(1 for d in drift_diffs if d < 0)),
        "mean_drift_toward_milk_cm": float(np.mean(drift_diffs) * 100)
            if drift_diffs else None,
        "mean_pixel_separation_px": float(np.mean(sep_px)),
    }
    print(json.dumps({"event": "result", **r}), flush=True)
    del sam; torch.cuda.empty_cache()
    return r


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = [evaluate(o) for o in OFFSETS_CM]

    print(json.dumps({"event": "summary"}), flush=True)
    print(f"{'offset':>7} {'sep_px':>7} {'n_valid':>8} "
          f"{'mean':>7} {'max':>7} {'closer_to_milk':>15} {'reseed':>7}")
    for r in results:
        print(f"{r['offset_cm']:>5d}cm "
              f"{r['mean_pixel_separation_px']:>7.1f} "
              f"{r['n_valid']:>8d} "
              f"{r['mean_err_cm']:>7.2f} "
              f"{r['max_err_cm']:>7.2f} "
              f"{r['n_frames_closer_to_milk']:>15d} "
              f"{r['n_reseed']:>7d}")

    # User's gates: at 10cm, mean ≤ 1 cm AND zero identity switches
    r10 = next(r for r in results if r["offset_cm"] == 10)
    g_id = r10["n_frames_closer_to_milk"] == 0
    g_mean = r10["mean_err_cm"] <= 1.0
    print(json.dumps({"event": "verdict_10cm",
                       "G_identity_switches": int(r10["n_frames_closer_to_milk"]),
                       "G_mean_err_under_1cm": bool(g_mean),
                       "G_identity_clean": bool(g_id),
                       "both_passed": bool(g_id and g_mean)}), flush=True)

    with open(OUT_BASE / "results.json", "w") as f:
        json.dump({"results": results}, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))


if __name__ == "__main__":
    main()
