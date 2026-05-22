"""
BF-0.8: SAM 2.1 occlusion stress test.

Same Can pick-and-place trajectory as BF-0.7, but animate Milk_main along the
camera-to-Can line during frames 60-85 to synthesize full visual occlusion.

Pre-committed gates (LOCKED before run):
  G1 : 0 identity switches (mask area collapse > 5x is NOT counted as a switch
       if it occurs WITHIN the occlusion window — that's expected; switches
       OUTSIDE the window count)
  G2 : mask recovers within 5 frames after the Can reappears
  G3 : post-occlusion mean 3D error <= 3 cm
  G4 : no persistent wrong-object lock (post-occlusion centroid must be within
       30 px of GT projected position)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_TOKEN", open("/root/.huggingface/token").read().strip())

import robosuite as suite
from PIL import Image

sys.path.insert(0, "/workspace")
from sam_perception_calibration import (
    build_env, find_can_qpos_addr, plan_can_trajectory,
    camera_intrinsics, project_world_to_pixel,
    unproject_pixel_to_world, depth_image_to_meters,
)

# -- occlusion plan ------------------------------------------------------

OCC_START = 60  # inclusive
OCC_END = 86    # exclusive (so 26-frame occlusion window)
PARK_POS = np.array([10.0, 10.0, 10.0])

def find_milk_qpos_addr(env):
    m = env.sim.model
    for jid in range(m.njnt):
        body_id = m.jnt_bodyid[jid]
        if m.body_id2name(body_id) == "Milk_main" and int(m.jnt_type[jid]) == 0:
            return int(m.jnt_qposadr[jid])
    raise RuntimeError("Milk_main free joint not found")


def plan_milk_pose(frame_idx, can_xyz_now, cam_pos):
    """Return Milk xyz for this frame: either parked or on the cam-to-can line."""
    if frame_idx < OCC_START or frame_idx >= OCC_END:
        return PARK_POS
    # During occlusion: place Milk at 40% from camera to can (closer to camera)
    return 0.6 * cam_pos + 0.4 * can_xyz_now


def set_body_xyz(env, qpos_addr, xyz, joint_name):
    env.sim.data.qpos[qpos_addr:qpos_addr + 3] = xyz
    qvel_addr = env.sim.model.jnt_dofadr[
        env.sim.model.joint_name2id(joint_name)]
    env.sim.data.qvel[qvel_addr:qvel_addr + 6] = 0.0


# -- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/bf08_sam_occlusion")
    ap.add_argument("--n-frames", type=int, default=135)
    ap.add_argument("--image-size", type=int, default=480)
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    np.random.seed(args.seed)
    print(json.dumps({"event": "start", "args": vars(args),
                       "occlusion_window": [OCC_START, OCC_END]}), flush=True)

    env = build_env(args.image_size)
    env.reset()
    K, R_wc, t_wc = camera_intrinsics(env, args.image_size)
    znear = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
    zfar = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
    can_qpos = find_can_qpos_addr(env)
    milk_qpos = find_milk_qpos_addr(env)
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    traj = plan_can_trajectory(can_start, args.n_frames)
    cam_pos = t_wc

    frames_rgb, frames_depth_m, can_gt = [], [], []
    milk_gt_xy = []
    t0 = time.time()
    for i in range(args.n_frames):
        # Set Can on its planned trajectory
        env.sim.data.qpos[can_qpos:can_qpos + 3] = traj[i]
        # Set Milk: parked or occluder pose
        milk_xyz = plan_milk_pose(i, traj[i], cam_pos)
        env.sim.data.qpos[milk_qpos:milk_qpos + 3] = milk_xyz
        # Zero both velocities
        for jname in ("Can_joint0", "Milk_joint0"):
            qv = env.sim.model.jnt_dofadr[
                env.sim.model.joint_name2id(jname)]
            env.sim.data.qvel[qv:qv + 6] = 0.0
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        depth_raw = np.flipud(obs["agentview_depth"][..., 0])
        depth_m = depth_image_to_meters(depth_raw, znear, zfar).astype(np.float32)
        frames_rgb.append(np.ascontiguousarray(rgb))
        frames_depth_m.append(np.ascontiguousarray(depth_m))
        can_gt.append(env.sim.data.get_body_xpos("Can_main").copy())
        milk_gt_xy.append(env.sim.data.get_body_xpos("Milk_main").copy())
        Image.fromarray(rgb).save(out / "frames" / f"{i:05d}.jpg", quality=95)
    can_gt = np.stack(can_gt); milk_gt_xy = np.stack(milk_gt_xy)
    print(json.dumps({"event": "demo_done", "elapsed_s": time.time() - t0,
                       "can_total_disp_m": float(np.linalg.norm(can_gt[-1] - can_gt[0]))}),
          flush=True)

    np.savez_compressed(out / "demo.npz",
                        rgb=np.stack(frames_rgb), depth_m=np.stack(frames_depth_m),
                        can_gt=can_gt, milk_gt=milk_gt_xy,
                        K=K, R_wc=R_wc, t_wc=t_wc,
                        occ_window=np.array([OCC_START, OCC_END]))

    # Verify occlusion actually obscures the can — project both to pixel and
    # measure pixel overlap at peak occlusion frame
    u_can_mid, v_can_mid = project_world_to_pixel(can_gt[OCC_START + 12], K, R_wc, t_wc)
    u_milk_mid, v_milk_mid = project_world_to_pixel(milk_gt_xy[OCC_START + 12], K, R_wc, t_wc)
    print(json.dumps({"event": "occlusion_geometry",
                       "frame": OCC_START + 12,
                       "can_pixel": [u_can_mid, v_can_mid],
                       "milk_pixel": [u_milk_mid, v_milk_mid],
                       "pixel_distance": float(((u_can_mid - u_milk_mid) ** 2 +
                                                  (v_can_mid - v_milk_mid) ** 2) ** 0.5)}),
          flush=True)

    u0, v0 = project_world_to_pixel(can_gt[0], K, R_wc, t_wc)

    # SAM tracking
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.sam_model, device="cuda")
    state = predictor.init_state(video_path=str(out / "frames"))
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=np.array([[u0, v0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    print(json.dumps({"event": "sam_seeded", "seed": [u0, v0],
                       "model": args.sam_model}), flush=True)

    centroids_px, areas, errs_3d, est_3d = [], [], [], []
    with torch.inference_mode():
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = mask_logits[0]
            if m.ndim == 3: m = m[0]
            m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
            area = int(m_np.sum())
            if area == 0:
                centroids_px.append((np.nan, np.nan)); areas.append(0)
                est_3d.append([np.nan, np.nan, np.nan]); errs_3d.append(np.nan)
                continue
            ys, xs = np.where(m_np)
            u_c, v_c = float(xs.mean()), float(ys.mean())
            centroids_px.append((u_c, v_c)); areas.append(area)
            depth_med = float(np.median(frames_depth_m[frame_idx][ys, xs]))
            p_est = unproject_pixel_to_world(u_c, v_c, depth_med, K, R_wc, t_wc)
            est_3d.append(p_est.tolist())
            errs_3d.append(float(np.linalg.norm(p_est - can_gt[frame_idx])))

    centroids_px = np.array(centroids_px)
    areas = np.array(areas)
    est_3d = np.array(est_3d)
    errs_3d = np.array(errs_3d)

    # --- Gate evaluation ----------------------------------------------------
    # Pre-occlusion segment: 0..OCC_START
    # Occlusion segment:    OCC_START..OCC_END
    # Post-occlusion:       OCC_END..end
    pre = slice(0, OCC_START)
    occ = slice(OCC_START, OCC_END)
    post = slice(OCC_END, args.n_frames)

    # G1: identity switches OUTSIDE occlusion window
    # Heuristic: area drop > 5x running median is a switch event
    valid_areas = areas.copy().astype(float)
    valid_areas[areas == 0] = np.nan
    running_med = np.array([np.nanmedian(valid_areas[max(0, i - 5):i + 1])
                              for i in range(len(areas))])
    switch_outside = 0
    for i in range(len(areas)):
        if OCC_START <= i < OCC_END: continue
        if np.isnan(running_med[i]): continue
        if areas[i] < running_med[i] / 5 and areas[i] > 0:
            switch_outside += 1

    # G2: recovery — find the first frame >= OCC_END where mask area >= 50%
    # of pre-occlusion median area
    pre_med_area = float(np.nanmedian(valid_areas[pre]))
    recovery_frame = None
    for i in range(OCC_END, args.n_frames):
        if not np.isnan(valid_areas[i]) and valid_areas[i] >= 0.5 * pre_med_area:
            recovery_frame = i
            break
    frames_to_recover = (recovery_frame - OCC_END) if recovery_frame is not None else None

    # G3: post-occlusion mean error
    post_errs = errs_3d[post]
    post_mean_err = float(np.nanmean(post_errs)) if (~np.isnan(post_errs)).any() else None

    # G4: persistent wrong-object lock test —
    # Compare post-occlusion centroid distance to projected GT centroid.
    # Wrong-lock = > 30px persistently (5+ consecutive frames)
    wrong_lock_frames = 0
    max_consec_wrong = 0
    cur = 0
    for i in range(OCC_END, args.n_frames):
        if np.isnan(centroids_px[i, 0]): continue
        u_gt, v_gt = project_world_to_pixel(can_gt[i], K, R_wc, t_wc)
        d_px = float(((centroids_px[i, 0] - u_gt) ** 2
                       + (centroids_px[i, 1] - v_gt) ** 2) ** 0.5)
        if d_px > 30:
            wrong_lock_frames += 1
            cur += 1
            max_consec_wrong = max(max_consec_wrong, cur)
        else:
            cur = 0

    summary = {
        "n_frames": args.n_frames,
        "occlusion_window": [OCC_START, OCC_END],
        "pre_occlusion": {
            "n": OCC_START,
            "mean_err_cm": float(np.nanmean(errs_3d[pre]) * 100),
            "mean_mask_area_px": pre_med_area,
        },
        "during_occlusion": {
            "n": OCC_END - OCC_START,
            "n_with_mask": int((areas[occ] > 0).sum()),
            "n_empty_masks": int((areas[occ] == 0).sum()),
            "mean_mask_area_px": float(np.nanmean(valid_areas[occ])),
            "min_mask_area_px": int(np.nanmin(np.where(areas[occ] > 0,
                                                          areas[occ], 1e12))) if (areas[occ] > 0).any() else 0,
        },
        "post_occlusion": {
            "n": args.n_frames - OCC_END,
            "mean_err_cm": post_mean_err * 100 if post_mean_err else None,
            "max_err_cm": float(np.nanmax(post_errs) * 100) if (~np.isnan(post_errs)).any() else None,
        },
        "gates": {
            "G1_switches_outside_occlusion": switch_outside,
            "G2_frames_to_recover": frames_to_recover,
            "G2_recovery_frame_absolute": recovery_frame,
            "G3_post_mean_err_cm": post_mean_err * 100 if post_mean_err else None,
            "G4_wrong_lock_frames_total": wrong_lock_frames,
            "G4_max_consec_wrong": max_consec_wrong,
        },
        "sam_model": args.sam_model,
    }
    print(json.dumps({"event": "summary", **summary}), flush=True)

    # Verdict
    g1 = switch_outside == 0
    g2 = (frames_to_recover is not None) and (frames_to_recover <= 5)
    g3 = (post_mean_err is not None) and (post_mean_err < 0.03)
    g4 = max_consec_wrong < 5

    verdict = {
        "G1_passed": bool(g1), "G2_passed": bool(g2),
        "G3_passed": bool(g3), "G4_passed": bool(g4),
        "all_passed": bool(g1 and g2 and g3 and g4),
    }
    print(json.dumps({"event": "verdict", **verdict}), flush=True)

    def _ser(o):
        try: return float(o)
        except (TypeError, ValueError): pass
        if hasattr(o, "tolist"): return o.tolist()
        return str(o)
    with open(out / "result.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "verdict": verdict,
                   "per_frame": {
                       "centroids_px": centroids_px.tolist(),
                       "areas": areas.tolist(),
                       "errs_3d": errs_3d.tolist(),
                       "est_3d": est_3d.tolist(),
                       "can_gt": can_gt.tolist()}}, f, indent=2, default=_ser)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
