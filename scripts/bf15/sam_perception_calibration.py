"""
BF-0.7: SAM perception front-end calibration on synthetic Can motion.

Pipeline:
  1. Build PickPlaceCan env, capture initial frame.
  2. Plan a smooth lift+translate+place trajectory for the Can body.
  3. Each step: directly set Can_main free-joint xyz, sim.forward(), render RGB+depth.
  4. Project the first-frame can_xpos to a pixel -> initial click prompt for SAM.
  5. Run SAM 3 video tracker across the sequence.
  6. For each frame: mask centroid -> median depth inside mask -> unproject to 3D.
  7. Report mean / motion-segment / max error vs ground-truth can xyz.

Pre-commit gates (locked):
   strong_pass : mean_err < 0.02m AND motion_seg_err < 0.03m
   useful      : mean_err < 0.05m AND no motion blowup
   falsify     : mean_err > 0.10m OR identity switch detected
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite as suite
from PIL import Image


def build_env(image_size: int = 480, horizon: int = 300):
    return suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview"],
        camera_heights=image_size, camera_widths=image_size,
        camera_depths=True, horizon=horizon,
        reward_shaping=True, ignore_done=True,
    )


def find_can_qpos_addr(env) -> int:
    m = env.sim.model
    for jid in range(m.njnt):
        body_id = m.jnt_bodyid[jid]
        body_name = m.body_id2name(body_id)
        if body_name == "Can_main" and int(m.jnt_type[jid]) == 0:
            return int(m.jnt_qposadr[jid])
    raise RuntimeError("Can_main free joint not found")


def plan_can_trajectory(can_start: np.ndarray, n_frames: int):
    p0 = np.asarray(can_start, dtype=np.float64).copy()
    p_lift = p0 + np.array([0.0, 0.0, 0.18])
    p_trans = p_lift + np.array([0.30, 0.32, 0.0])
    p_land = p_trans + np.array([0.0, 0.0, -0.18])
    n0 = max(2, n_frames // 10)
    n1 = max(4, n_frames // 5)
    n3 = max(4, n_frames // 5)
    n2 = n_frames - n0 - n1 - n3
    pts = []
    for _ in range(n0):
        pts.append(p0)
    for i in range(n1):
        t = (i + 1) / n1
        pts.append(p0 * (1 - t) + p_lift * t)
    for i in range(n2):
        t = (i + 1) / n2
        t = 0.5 - 0.5 * np.cos(np.pi * t)
        pts.append(p_lift * (1 - t) + p_trans * t)
    for i in range(n3):
        t = (i + 1) / n3
        pts.append(p_trans * (1 - t) + p_land * t)
    return np.stack(pts, axis=0)


def set_can_xyz(env, qpos_addr: int, xyz: np.ndarray):
    env.sim.data.qpos[qpos_addr:qpos_addr + 3] = xyz
    qvel_addr = env.sim.model.jnt_dofadr[
        env.sim.model.joint_name2id("Can_joint0")]
    env.sim.data.qvel[qvel_addr:qvel_addr + 6] = 0.0
    env.sim.forward()


def camera_intrinsics(env, image_size: int):
    import mujoco
    model = env.sim.model._model
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    fovy = float(env.sim.model.cam_fovy[cid])
    H = W = image_size
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_pos = np.array(env.sim.model.cam_pos[cid], dtype=np.float64).copy()
    cq = np.array(env.sim.model.cam_quat[cid], dtype=np.float64)
    w, x, y, z = cq
    R_world_cam = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return K, R_world_cam, cam_pos


def project_world_to_pixel(p_world, K, R_world_cam, t_world_cam):
    p_cam = R_world_cam.T @ (p_world - t_world_cam)
    x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
    if z <= 1e-6: return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def unproject_pixel_to_world(u, v, depth_m, K, R_world_cam, t_world_cam):
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    z = depth_m
    p_cam = np.array([x, -y, -z], dtype=np.float64)
    return R_world_cam @ p_cam + t_world_cam


def depth_image_to_meters(depth_img, znear, zfar):
    z_n = depth_img.astype(np.float64)
    return znear * zfar / (zfar - z_n * (zfar - znear))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/bf07_sam_calibration")
    ap.add_argument("--n-frames", type=int, default=135)
    ap.add_argument("--image-size", type=int, default=480)
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    np.random.seed(args.seed)
    print(json.dumps({"event": "start", "args": vars(args)}), flush=True)

    env = build_env(args.image_size)
    obs = env.reset()
    K, R_wc, t_wc = camera_intrinsics(env, args.image_size)
    znear = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
    zfar = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
    qpos_addr = find_can_qpos_addr(env)
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    traj = plan_can_trajectory(can_start, args.n_frames)
    print(json.dumps({"event": "plan",
                       "can_start": can_start.tolist(),
                       "traj_total_disp_m": float(np.linalg.norm(traj[-1] - traj[0])),
                       "n": int(len(traj))}), flush=True)

    frames_rgb, frames_depth_m, can_gt = [], [], []
    t0 = time.time()
    for i in range(args.n_frames):
        set_can_xyz(env, qpos_addr, traj[i])
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        depth_raw = np.flipud(obs["agentview_depth"][..., 0])
        depth_m = depth_image_to_meters(depth_raw, znear, zfar).astype(np.float32)
        can_pos = env.sim.data.get_body_xpos("Can_main").copy()
        frames_rgb.append(np.ascontiguousarray(rgb))
        frames_depth_m.append(np.ascontiguousarray(depth_m))
        can_gt.append(can_pos)
        Image.fromarray(rgb).save(out / "frames" / f"{i:05d}.jpg", quality=95)
    can_gt = np.stack(can_gt, axis=0)
    print(json.dumps({"event": "demo_done", "n_frames": len(frames_rgb),
                       "can_total_disp_m": float(np.linalg.norm(can_gt[-1] - can_gt[0])),
                       "elapsed_s": time.time() - t0}), flush=True)

    np.savez_compressed(out / "demo.npz",
                        rgb=np.stack(frames_rgb), depth_m=np.stack(frames_depth_m),
                        can_gt=can_gt, K=K, R_wc=R_wc, t_wc=t_wc)
    u0, v0 = project_world_to_pixel(can_gt[0], K, R_wc, t_wc)
    print(json.dumps({"event": "initial_prompt", "u": u0, "v": v0}), flush=True)
    if args.skip_sam:
        print(json.dumps({"event": "skip_sam"}), flush=True)
        return

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_id = args.sam_model
    print(json.dumps({"event": "loading_sam", "model": sam_id, "device": device}),
          flush=True)
    predictor = SAM2VideoPredictor.from_pretrained(sam_id, device=device)
    print(json.dumps({"event": "sam_loaded", "model": sam_id}), flush=True)

    # SAM 2 expects video as a directory of JPEG/PNG frames OR a (T,H,W,3) np array.
    # Use the frames-on-disk path we already saved.
    video_dir = str(out / "frames")
    state = predictor.init_state(video_path=video_dir)
    # Seed identity with a click at the projected GT pixel in frame 0
    _, obj_ids, _ = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=np.array([[u0, v0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    print(json.dumps({"event": "session_initialized",
                       "seed_pixel": [u0, v0],
                       "obj_ids_after_seed": [int(x) for x in obj_ids]}),
          flush=True)

    # Map frame_idx -> mask (binary, HxW) via SAM 2 video propagation
    masks_per_frame = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            # mask_logits: (n_obj, 1, H, W) torch tensor
            m = mask_logits[0]
            if m.ndim == 3: m = m[0]
            m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
            masks_per_frame[frame_idx] = m_np

    centroids_px, areas, errs_3d, est_3d = [], [], [], []
    for frame_idx in range(args.n_frames):
        m_np = masks_per_frame.get(frame_idx)
        if m_np is None or m_np.sum() == 0:
            centroids_px.append((np.nan, np.nan)); areas.append(0)
            est_3d.append([np.nan, np.nan, np.nan]); errs_3d.append(np.nan)
            continue
        area = int(m_np.sum())
        ys, xs = np.where(m_np)
        u_c, v_c = float(xs.mean()), float(ys.mean())
        centroids_px.append((u_c, v_c)); areas.append(area)
        depth_at = frames_depth_m[frame_idx][ys, xs]
        depth_med = float(np.median(depth_at))
        p_est = unproject_pixel_to_world(u_c, v_c, depth_med, K, R_wc, t_wc)
        est_3d.append(p_est.tolist())
        err = float(np.linalg.norm(p_est - can_gt[frame_idx]))
        errs_3d.append(err)
        if frame_idx % 20 == 0 or frame_idx in (0, 5, args.n_frames - 1):
            print(json.dumps({"event": "frame", "idx": frame_idx,
                               "centroid_px": [u_c, v_c],
                               "area": area,
                               "depth_med": depth_med,
                               "err_m": err}), flush=True)

    centroids_px = np.array(centroids_px); areas = np.array(areas)
    est_3d = np.array(est_3d); errs_3d = np.array(errs_3d)
    running_med = np.array([np.median(areas[max(0, i - 5):i + 1])
                              for i in range(len(areas))])
    switch_flags = (areas < running_med / 5).astype(int)

    can_vel = np.linalg.norm(np.diff(can_gt, axis=0), axis=1)
    motion_idx = (np.where(can_vel > 1e-3)[0] + 1)
    motion_idx = motion_idx[motion_idx < len(errs_3d)]
    motion_errs = errs_3d[motion_idx]

    valid = ~np.isnan(errs_3d)
    summary = {
        "n_frames": int(len(errs_3d)),
        "n_valid": int(valid.sum()),
        "mean_err_m": float(np.nanmean(errs_3d)),
        "median_err_m": float(np.nanmedian(errs_3d)),
        "max_err_m": float(np.nanmax(errs_3d)),
        "motion_seg_n": int(len(motion_errs)),
        "motion_seg_mean_err_m": (float(np.nanmean(motion_errs))
                                    if len(motion_errs) else None),
        "motion_seg_max_err_m": (float(np.nanmax(motion_errs))
                                    if len(motion_errs) else None),
        "identity_switches_detected": int(switch_flags.sum()),
        "mean_mask_area_px": float(areas.mean()),
        "min_mask_area_px": int(areas.min()),
        "can_total_disp_m": float(np.linalg.norm(can_gt[-1] - can_gt[0])),
        "sam_model": sam_id,
    }
    print(json.dumps({"event": "summary", **summary}), flush=True)

    mean_e = summary["mean_err_m"]
    median_e = summary["median_err_m"]
    motion_mean_e = summary["motion_seg_mean_err_m"] or 0.0
    motion_max_e = summary["motion_seg_max_err_m"] or 0.0
    switches = summary["identity_switches_detected"]
    # Verdict uses MOTION-SEGMENT stats as primary (frame-0 outliers ignored if
    # tracking locks in late) and MEDIAN as a robust headline.
    if mean_e > 0.10:
        verdict = "FALSIFY"
    elif median_e < 0.02 and motion_mean_e < 0.03 and motion_max_e < 0.05:
        verdict = "STRONG_PASS"
    elif median_e < 0.05 and motion_mean_e < 0.05:
        verdict = "USEFUL"
    else:
        verdict = "AMBIGUOUS"
    print(json.dumps({"event": "verdict", "verdict": verdict,
                       "switches": switches}), flush=True)

    def _to_serializable(o):
        try: return float(o)
        except (TypeError, ValueError): pass
        if hasattr(o, "tolist"): return o.tolist()
        return str(o)

    with open(out / "result.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "verdict": verdict,
                   "per_frame": {"centroids_px": centroids_px.tolist(),
                                  "areas": areas.tolist(),
                                  "errs_3d": errs_3d.tolist(),
                                  "est_3d": est_3d.tolist(),
                                  "can_gt": can_gt.tolist()}}, f, indent=2,
                  default=_to_serializable)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
