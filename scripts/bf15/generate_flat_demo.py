"""Render a Can demo where the can slides on the z=0.86 plane (no lift).

This is the BLA-Forge plane-projection use case: object moves in XY only,
table-plane assumption is exact.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, "/workspace")
from sam_perception_calibration import (
    build_env, find_can_qpos_addr, camera_intrinsics,
    project_world_to_pixel, depth_image_to_meters,
)
from PIL import Image

OUT = Path("/workspace/bf15_flat_demo")
N_FRAMES = 100
IMG = 480


def plan_flat_trajectory(can_start, n_frames):
    p0 = np.asarray(can_start, dtype=np.float64).copy()
    p_end = p0 + np.array([0.20, 0.30, 0.0])  # XY motion only
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
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    traj = plan_flat_trajectory(can_start, N_FRAMES)
    can_gt = []
    for i in range(N_FRAMES):
        env.sim.data.qpos[can_qpos:can_qpos + 3] = traj[i]
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(rgb)).save(
            OUT / "frames" / f"{i:05d}.jpg", quality=95)
        can_gt.append(env.sim.data.get_body_xpos("Can_main").copy())
    can_gt = np.stack(can_gt)
    np.savez_compressed(OUT / "demo.npz",
                        can_gt=can_gt, K=K, R_wc=R_wc, t_wc=t_wc,
                        znear=znear, zfar=zfar)
    print(json.dumps({"event": "flat_demo_done",
                       "n_frames": N_FRAMES,
                       "can_z_range": [float(can_gt[:, 2].min()),
                                         float(can_gt[:, 2].max())],
                       "xy_disp": float(np.linalg.norm(can_gt[-1, :2] - can_gt[0, :2]))}),
          flush=True)


if __name__ == "__main__":
    main()
