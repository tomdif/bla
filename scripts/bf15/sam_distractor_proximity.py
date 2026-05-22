"""BF-0.14: SAM 2.1 + watchdog parallel-distractor proximity test.

Animate Milk on a parallel trajectory shifted ~10 cm in world-y from
the Can throughout the demo. Milk and Can move together but never
occlude — they sit side-by-side in pixel space.

Tests: does SAM's identity locking confuse the two when they're
spatially proximate but not overlapping?

Gates:
  G1 : 0 identity switches (mask centroid stays within can_pixel ± 30 px,
       not drifting toward milk_pixel)
  G2 : mean 3D error < 2 cm
  G3 : max error < 5 cm
  G4 : n_reseed <= 1
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


OUT = Path("/workspace/bf14_distractor_proximity")
N_FRAMES = 100
IMG = 480
PLANE_Z = 0.86
MILK_Y_OFFSET = 0.10   # 10 cm — close but not occluding


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    env = build_env(IMG); env.reset()
    K, R_wc, t_wc = camera_intrinsics(env, IMG)
    znear = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
    zfar = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
    can_qpos = find_can_qpos_addr(env)
    milk_qpos = find_milk_qpos_addr(env)
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    can_traj = plan_flat_trajectory(can_start, N_FRAMES)
    # Milk on PARALLEL trajectory: same xy motion, shifted +y
    milk_traj = can_traj.copy()
    milk_traj[:, 1] += MILK_Y_OFFSET

    can_gt = []
    milk_gt = []
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
            OUT / "frames" / f"{i:05d}.jpg", quality=95)
        can_gt.append(env.sim.data.get_body_xpos("Can_main").copy())
        milk_gt.append(env.sim.data.get_body_xpos("Milk_main").copy())
    can_gt = np.stack(can_gt); milk_gt = np.stack(milk_gt)
    np.savez_compressed(OUT / "demo.npz", can_gt=can_gt, milk_gt=milk_gt,
                        K=K, R_wc=R_wc, t_wc=t_wc)
    print(json.dumps({"event": "render_done"}), flush=True)

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

    # Diagnostic: pixel separation between can and milk centroids per frame
    sep_px = []
    for i in range(N_FRAMES):
        cu, cv = project_world_to_pixel(can_gt[i])
        mu, mv = project_world_to_pixel(milk_gt[i])
        sep_px.append(float(((cu - mu) ** 2 + (cv - mv) ** 2) ** 0.5))
    print(json.dumps({
        "event": "pixel_separation_diag",
        "mean_separation_px": float(np.mean(sep_px)),
        "min_separation_px": float(np.min(sep_px)),
        "max_separation_px": float(np.max(sep_px)),
    }), flush=True)

    def fiducial_fallback(frame_idx, obj_id):
        if frame_idx >= len(can_gt): return None
        return project_world_to_pixel(can_gt[frame_idx])

    u0, v0 = project_world_to_pixel(can_gt[0])
    t0 = time.time()
    sam = SAMPerception(
        video_path=OUT / "frames",
        seeds=[SAMSeed(obj_id=1, pixel_uv=(u0, v0))],
        backend="sam2.1", sam_model="facebook/sam2.1-hiera-tiny",
        fiducial_fallback_fn=fiducial_fallback, silence_threshold=3,
    )
    elapsed = time.time() - t0

    # Per-frame: drift of SAM centroid toward MILK vs toward CAN ground truth
    errs_to_can = []
    drift_toward_milk = []
    centroids = []
    n_valid = 0
    for i in range(N_FRAMES):
        dets = sam.detect(i)
        good = [d for d in dets if d.confidence > 0]
        if not good:
            centroids.append(None); continue
        d = good[0]
        u_c, v_c = float(d.center_px[0]), float(d.center_px[1])
        centroids.append((u_c, v_c))
        p_world = unproject_pixel(u_c, v_c, PLANE_Z)
        if p_world is None: continue
        err_to_can = float(np.linalg.norm(p_world[:2] - can_gt[i][:2]))
        err_to_milk = float(np.linalg.norm(p_world[:2] - milk_gt[i][:2]))
        # Negative drift = closer to can; positive = drifting toward milk
        drift_toward_milk.append(err_to_milk - err_to_can)
        errs_to_can.append(err_to_can)
        n_valid += 1

    summary = {
        "n_frames": int(N_FRAMES),
        "elapsed_s": elapsed,
        "n_valid": int(n_valid),
        "n_reseed_events": len(sam.reseed_events),
        "mean_err_cm": float(np.mean(errs_to_can) * 100) if errs_to_can else float("inf"),
        "median_err_cm": float(np.median(errs_to_can) * 100) if errs_to_can else float("inf"),
        "max_err_cm": float(np.max(errs_to_can) * 100) if errs_to_can else float("inf"),
        # Drift diagnostic — average distance to milk MINUS distance to can.
        # Positive = SAM is closer to can (correct). Negative = closer to milk (wrong).
        "drift_toward_milk_cm_mean": (
            float(np.mean(drift_toward_milk) * 100) if drift_toward_milk else None),
        "n_frames_closer_to_milk": int(sum(1 for d in drift_toward_milk
                                                if d < 0)),
        "mean_pixel_separation_px": float(np.mean(sep_px)),
    }
    print(json.dumps({"event": "summary", **summary}), flush=True)

    g1 = summary["n_frames_closer_to_milk"] == 0
    g2 = summary["mean_err_cm"] < 2.0
    g3 = summary["max_err_cm"] < 5.0
    g4 = summary["n_reseed_events"] <= 1
    verdict = {
        "G1_no_identity_switches_to_milk": bool(g1),
        "G2_mean_err_under_2cm": bool(g2),
        "G3_max_err_under_5cm": bool(g3),
        "G4_n_reseed_at_most_1": bool(g4),
        "all_passed": bool(g1 and g2 and g3 and g4),
    }
    print(json.dumps({"event": "verdict", **verdict}), flush=True)

    with open(OUT / "result.json", "w") as f:
        json.dump({"summary": summary, "verdict": verdict,
                   "reseed_events": sam.reseed_events,
                   "pixel_separation_px": sep_px,
                   "drift_toward_milk_per_frame_cm": drift_toward_milk},
                  f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "__float__") else str(o))


if __name__ == "__main__":
    main()
