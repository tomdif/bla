"""
BF-0.9: SAM 2.1 partial occlusion stress test.

Same Can pick-and-place trajectory. Milk_main animated to PARTIALLY occlude
the Can — offset perpendicular to the camera-Can line so it only covers a
fraction of the Can's pixel area. Three sub-windows test 25 / 50 / 75% target
overlap.

Pre-committed gates:
  G1 : 0 identity switches outside the occlusion window
  G2 : mask area stays > 25% of pre-occlusion median during partial occlusion
       (i.e. SAM does NOT abstain — tracks the visible portion)
  G3 : centroid drift from projected GT < 15 px during partial occlusion
  G4 : post-occlusion mean 3D error < 3 cm
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

# Occlusion sub-windows (target overlap fraction, frame range)
SUB_WINDOWS = [
    (0.25, 55, 65),
    (0.50, 65, 75),
    (0.75, 75, 85),
]
OCC_START = SUB_WINDOWS[0][1]
OCC_END = SUB_WINDOWS[-1][2]
PARK = np.array([10.0, 10.0, 10.0])


def find_milk_qpos_addr(env):
    m = env.sim.model
    for jid in range(m.njnt):
        body_id = m.jnt_bodyid[jid]
        if m.body_id2name(body_id) == "Milk_main" and int(m.jnt_type[jid]) == 0:
            return int(m.jnt_qposadr[jid])
    raise RuntimeError()


def overlap_fraction_for_frame(frame_idx):
    for frac, lo, hi in SUB_WINDOWS:
        if lo <= frame_idx < hi:
            return frac
    return None


def plan_milk_pose_partial(frame_idx, can_xyz_now, cam_pos, overlap_frac):
    """
    Place Milk on the camera-Can line at 40% from camera, then offset
    perpendicular by (1 - overlap_frac) * (Milk_apparent_width).

    Milk box is roughly 8 cm wide / 14 cm tall. At depth ~1.4m, apparent
    width ~33 px. To get target_overlap_frac mask overlap with the Can,
    offset Milk laterally by ~ (1 - target_overlap_frac) * Milk_half_width
    in world coords.

    Empirically tune the offset scale by world-frame y to land in the
    correct pixel range. We use perpendicular = (0, +y, 0) since the
    camera is on the +x side; lateral motion in y is roughly screen-x.
    """
    base = 0.6 * cam_pos + 0.4 * can_xyz_now
    # Lateral offset: positive y is right of can in screen space
    # 1.0 = full occlusion (no offset); 0.0 = no occlusion (large offset)
    lateral_m = (1.0 - overlap_frac) * 0.10  # tune
    return base + np.array([0.0, lateral_m, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/bf09_sam_partial_occ")
    ap.add_argument("--n-frames", type=int, default=135)
    ap.add_argument("--image-size", type=int, default=480)
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    print(json.dumps({"event": "start", "args": vars(args),
                       "occlusion_window": [OCC_START, OCC_END],
                       "sub_windows": SUB_WINDOWS}), flush=True)

    env = build_env(args.image_size)
    env.reset()
    K, R_wc, t_wc = camera_intrinsics(env, args.image_size)
    znear = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
    zfar = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
    can_qpos = find_can_qpos_addr(env)
    milk_qpos = find_milk_qpos_addr(env)
    can_start = env.sim.data.get_body_xpos("Can_main").copy()
    traj = plan_can_trajectory(can_start, args.n_frames)

    frames_rgb, frames_depth_m, can_gt, milk_gt = [], [], [], []
    for i in range(args.n_frames):
        env.sim.data.qpos[can_qpos:can_qpos + 3] = traj[i]
        overlap = overlap_fraction_for_frame(i)
        if overlap is None:
            env.sim.data.qpos[milk_qpos:milk_qpos + 3] = PARK
        else:
            env.sim.data.qpos[milk_qpos:milk_qpos + 3] = plan_milk_pose_partial(
                i, traj[i], t_wc, overlap)
        for jname in ("Can_joint0", "Milk_joint0"):
            qv = env.sim.model.jnt_dofadr[env.sim.model.joint_name2id(jname)]
            env.sim.data.qvel[qv:qv + 6] = 0.0
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        depth_raw = np.flipud(obs["agentview_depth"][..., 0])
        depth_m = depth_image_to_meters(depth_raw, znear, zfar).astype(np.float32)
        frames_rgb.append(np.ascontiguousarray(rgb))
        frames_depth_m.append(np.ascontiguousarray(depth_m))
        can_gt.append(env.sim.data.get_body_xpos("Can_main").copy())
        milk_gt.append(env.sim.data.get_body_xpos("Milk_main").copy())
        Image.fromarray(rgb).save(out / "frames" / f"{i:05d}.jpg", quality=95)
    can_gt = np.stack(can_gt); milk_gt = np.stack(milk_gt)
    print(json.dumps({"event": "demo_done"}), flush=True)

    np.savez_compressed(out / "demo.npz",
                        rgb=np.stack(frames_rgb), depth_m=np.stack(frames_depth_m),
                        can_gt=can_gt, milk_gt=milk_gt,
                        K=K, R_wc=R_wc, t_wc=t_wc)

    u0, v0 = project_world_to_pixel(can_gt[0], K, R_wc, t_wc)
    print(json.dumps({"event": "seed_pixel", "u": u0, "v": v0}), flush=True)

    # SAM
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.sam_model, device="cuda")
    state = predictor.init_state(video_path=str(out / "frames"))
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=np.array([[u0, v0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    centroids_px, areas, errs_3d, est_3d = [], [], [], []
    centroid_drift_px = []
    with torch.inference_mode():
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = mask_logits[0]
            if m.ndim == 3: m = m[0]
            m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
            area = int(m_np.sum())
            u_gt, v_gt = project_world_to_pixel(can_gt[frame_idx], K, R_wc, t_wc)
            if area == 0:
                centroids_px.append((np.nan, np.nan)); areas.append(0)
                est_3d.append([np.nan, np.nan, np.nan]); errs_3d.append(np.nan)
                centroid_drift_px.append(np.nan); continue
            ys, xs = np.where(m_np)
            u_c, v_c = float(xs.mean()), float(ys.mean())
            centroids_px.append((u_c, v_c)); areas.append(area)
            drift = float(((u_c - u_gt) ** 2 + (v_c - v_gt) ** 2) ** 0.5)
            centroid_drift_px.append(drift)
            depth_med = float(np.median(frames_depth_m[frame_idx][ys, xs]))
            p_est = unproject_pixel_to_world(u_c, v_c, depth_med, K, R_wc, t_wc)
            est_3d.append(p_est.tolist())
            errs_3d.append(float(np.linalg.norm(p_est - can_gt[frame_idx])))

    centroids_px = np.array(centroids_px); areas = np.array(areas)
    est_3d = np.array(est_3d); errs_3d = np.array(errs_3d)
    centroid_drift_px = np.array(centroid_drift_px)

    pre = slice(0, OCC_START); occ = slice(OCC_START, OCC_END)
    post = slice(OCC_END, args.n_frames)

    pre_med_area = float(np.median(areas[pre][areas[pre] > 0])) if (areas[pre] > 0).any() else 0
    # Per-sub-window stats
    per_sub = {}
    for frac, lo, hi in SUB_WINDOWS:
        seg_areas = areas[lo:hi]; seg_drift = centroid_drift_px[lo:hi]
        seg_errs = errs_3d[lo:hi]
        per_sub[f"{int(frac * 100)}pct_overlap"] = {
            "frames": [lo, hi],
            "mean_area_px": float(np.nanmean(seg_areas)),
            "min_area_px": int(seg_areas.min()),
            "n_zero_mask": int((seg_areas == 0).sum()),
            "area_fraction_of_pre_median": float(np.nanmean(seg_areas) / pre_med_area) if pre_med_area > 0 else None,
            "mean_drift_px": float(np.nanmean(seg_drift)) if (~np.isnan(seg_drift)).any() else None,
            "max_drift_px": float(np.nanmax(seg_drift)) if (~np.isnan(seg_drift)).any() else None,
            "mean_err_cm": float(np.nanmean(seg_errs) * 100) if (~np.isnan(seg_errs)).any() else None,
            "max_err_cm": float(np.nanmax(seg_errs) * 100) if (~np.isnan(seg_errs)).any() else None,
        }

    # Gate evaluation
    valid_areas = areas.astype(float).copy(); valid_areas[areas == 0] = np.nan
    running_med = np.array([np.nanmedian(valid_areas[max(0, i - 5):i + 1])
                              for i in range(len(areas))])
    switch_outside = 0
    for i in range(len(areas)):
        if OCC_START <= i < OCC_END: continue
        if np.isnan(running_med[i]): continue
        if areas[i] < running_med[i] / 5 and areas[i] > 0: switch_outside += 1

    occ_mean_area = float(np.nanmean(areas[occ]))
    g2_area_frac = occ_mean_area / pre_med_area if pre_med_area > 0 else 0

    occ_drift_mean = float(np.nanmean(centroid_drift_px[occ]))
    occ_drift_max = float(np.nanmax(centroid_drift_px[occ]))

    post_mean_err = float(np.nanmean(errs_3d[post]))

    summary = {
        "pre_occlusion_mean_area_px": pre_med_area,
        "per_sub_window": per_sub,
        "occ_overall": {
            "n_zero_mask_frames": int((areas[occ] == 0).sum()),
            "mean_area_px": occ_mean_area,
            "area_fraction_of_pre": g2_area_frac,
            "mean_centroid_drift_px": occ_drift_mean,
            "max_centroid_drift_px": occ_drift_max,
            "mean_err_cm": float(np.nanmean(errs_3d[occ]) * 100),
        },
        "post_occlusion": {
            "mean_err_cm": post_mean_err * 100,
            "max_err_cm": float(np.nanmax(errs_3d[post]) * 100),
        },
        "gates": {
            "G1_switches_outside_occlusion": switch_outside,
            "G2_occ_area_fraction_of_pre": g2_area_frac,
            "G3_max_centroid_drift_px": occ_drift_max,
            "G4_post_mean_err_cm": post_mean_err * 100,
        },
    }
    print(json.dumps({"event": "summary", **summary}), flush=True)

    g1 = switch_outside == 0
    g2 = g2_area_frac > 0.25
    g3 = occ_drift_max < 15.0
    g4 = post_mean_err < 0.03
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
                   "per_frame": {"areas": areas.tolist(),
                                  "centroid_drift_px": centroid_drift_px.tolist(),
                                  "errs_3d": errs_3d.tolist()}},
                  f, indent=2, default=_ser)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
