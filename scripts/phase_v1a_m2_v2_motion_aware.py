"""Phase V1a-M2 v2 — motion-aware Cosmos identity-preservation probe.

V1a-M2 v1 attempted world-to-pixel projection but the rollouts had
zero object motion, making the cosines noise. v2 sidesteps projection
entirely by using the temporal-variance structure of the encoded
latents:

  1. Replay a FULL Can demo (~120 frames). The demo lifts the can
     substantially around frames 50-80, giving real vertical +
     horizontal motion that we can detect.
  2. Encode the rollout with Cosmos in chunks (CV4x8x8 wants T ≡ 1
     mod 4 per chunk).
  3. Compute per-spatial-cell temporal variance of the latent. The
     top-K highest-variance cells are where the model's
     representation is changing — these are the cells where objects
     are visible / moving.
  4. At each high-variance cell, ask:
       Is the feature evolving SMOOTHLY (high consecutive-frame
       cosines) — meaning the cell is "tracking" something coherently?
       Or is the feature JUMPING (low cosines) — meaning the cell is
       just position-bound and content changes as different objects
       pass through?
  5. Compute "start vs end" cosine for high-variance cells: when the
     can has been moved substantially, does that cell's feature
     differ a lot (the can has left, replaced by background) or
     stay similar (the cell's feature is intrinsically background)?

Gate (heuristic; this is exploratory):
  - Top-K high-variance cells should have consecutive-frame cosine
    in a "smooth motion" range (~0.7-0.95). Below ~0.5 = chaotic,
    not useful for identity. Above 0.99 = nothing changed at that
    cell.
  - These cells' start-vs-end cosine should differ from their
    average background cosine — telling us they actually responded
    to motion, not just noise.

This is NOT proof of identity preservation. It's a diagnostic: if
Cosmos features are at minimum responsive to object motion in a
SMOOTH way, identity binding is at least conceivable. If they're
chaotic or invariant, V1a is dead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


COSMOS_RES = 256
COSMOS_LATENT_SPATIAL = 32
COSMOS_LATENT_CHANNELS = 16


def collect_full_can_demo(image_size: int = COSMOS_RES,
                              demo_id: int = 5,
                              demo_dir: str = "/workspace/robomimic_can_replay"
                              ) -> dict:
    """Replay the full Can demo on its state-matched init.

    Demo 5 lifts the can robustly during the lift phase. We replay
    ALL actions to get the full trajectory including grasp+lift.
    """
    import robosuite as rs
    demo_path = Path(demo_dir) / f"ep_{demo_id:05d}.npz"
    d = np.load(demo_path)
    actions = d["actions"]
    init_state = d["init_state"]

    n_actions = len(actions)
    env = rs.make("PickPlaceCan", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview",
                    camera_heights=image_size, camera_widths=image_size,
                    horizon=n_actions + 30)
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    obs = env._get_observations()

    frames = [obs["agentview_image"]]
    can_xyz = [env.sim.data.get_body_xpos("Can_main").copy()]
    eef_xyz = [env.sim.data.get_body_xpos("gripper0_right_eef").copy()]

    for a in actions:
        obs, _, _, _ = env.step(a)
        frames.append(obs["agentview_image"])
        can_xyz.append(env.sim.data.get_body_xpos("Can_main").copy())
        eef_xyz.append(env.sim.data.get_body_xpos("gripper0_right_eef").copy())

    return {
        "frames": np.stack(frames, axis=0),
        "can_xyz_world": np.stack(can_xyz, axis=0).astype(np.float32),
        "eef_xyz_world": np.stack(eef_xyz, axis=0).astype(np.float32),
    }


@torch.no_grad()
def encode_chunked(encoder, frames_THWC: np.ndarray, chunk_frames: int = 17,
                      device: str = "cuda") -> np.ndarray:
    """Encode a long video by chunking into T ≡ 1 (mod 4) segments.

    Returns latent [16, total_T_lat, 32, 32] as float32 numpy.
    Uses NON-OVERLAPPING chunks; concatenates along temporal axis.
    """
    T = frames_THWC.shape[0]
    # T_lat per chunk for chunk_frames=17 is 5; per chunk_frames=21 is 6.
    chunks_out = []
    i = 0
    while i + chunk_frames <= T:
        chunk = frames_THWC[i:i + chunk_frames]
        x = torch.from_numpy(chunk).float() / 127.5 - 1.0
        x = x.permute(3, 0, 1, 2).unsqueeze(0).to(device).to(torch.bfloat16)
        (lat,) = encoder.encode(x)
        chunks_out.append(lat[0].float().cpu().numpy())
        i += chunk_frames
    return np.concatenate(chunks_out, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                    default="/root/cosmos_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit")
    p.add_argument("--out", default="/root/bla/runs/phase_v1a_m2_v2")
    p.add_argument("--demo-id", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k-variance-cells", type=int, default=16)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    # 1. Replay full Can demo
    rollout = collect_full_can_demo(COSMOS_RES, args.demo_id)
    frames = rollout["frames"]
    can = rollout["can_xyz_world"]
    can_displacement = np.linalg.norm(can - can[0:1], axis=1)
    can_z_change = can[:, 2] - can[0, 2]
    print(json.dumps({
        "event": "rollout",
        "n_frames": int(frames.shape[0]),
        "can_max_displacement_m": float(can_displacement.max()),
        "can_max_z_change_m": float(can_z_change.max()),
        "can_first_xyz": can[0].tolist(),
        "can_last_xyz": can[-1].tolist(),
        "frame_when_can_lifted": int(np.argmax(can_z_change > 0.03))
            if (can_z_change > 0.03).any() else -1,
    }), flush=True)

    # 2. Load encoder
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    encoder = CausalVideoTokenizer(checkpoint_enc=args.ckpt)

    # 3. Encode in chunks
    t0 = time.time()
    latents = encode_chunked(encoder, frames, chunk_frames=17, device=args.device)
    encode_elapsed = time.time() - t0
    C, T_lat, H, W = latents.shape
    print(json.dumps({"event": "encoded",
                       "latent_shape": [C, T_lat, H, W],
                       "encode_elapsed_s": round(encode_elapsed, 1)}), flush=True)

    # 4. Per-spatial-cell temporal variance.
    # Flatten spatial → [H*W] variance.
    lat_flat = latents.reshape(C, T_lat, H * W)
    # Per-cell variance: per channel std over time, summed across channels.
    cell_variance = lat_flat.std(axis=1).sum(axis=0)  # [H*W]
    top_idx = np.argsort(cell_variance)[::-1][:args.top_k_variance_cells]
    bot_idx = np.argsort(cell_variance)[:args.top_k_variance_cells]
    print(json.dumps({"event": "variance",
                       "top_cell_variance": [float(cell_variance[i])
                                                for i in top_idx[:5]],
                       "bot_cell_variance": [float(cell_variance[i])
                                                for i in bot_idx[:5]]}),
           flush=True)

    # 5. For each top-K high-variance cell, compute:
    #    - consecutive-frame cosine (smoothness of trajectory)
    #    - start-vs-end cosine
    # And compare against background (low-variance) cells.
    def cos(a, b, eps=1e-6):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

    def cell_stats(idx_set):
        consec, startend = [], []
        for ci in idx_set:
            r, c = ci // W, ci % W
            traj = latents[:, :, r, c]  # [16, T_lat]
            for t in range(T_lat - 1):
                consec.append(cos(traj[:, t], traj[:, t + 1]))
            startend.append(cos(traj[:, 0], traj[:, -1]))
        return {
            "consecutive_cos_mean": float(np.mean(consec)),
            "consecutive_cos_median": float(np.median(consec)),
            "consecutive_cos_min": float(np.min(consec)),
            "startend_cos_mean": float(np.mean(startend)),
            "startend_cos_median": float(np.median(startend)),
        }

    high_var_stats = cell_stats(top_idx)
    low_var_stats = cell_stats(bot_idx)
    print(json.dumps({"event": "high_variance_cells", **high_var_stats}),
           flush=True)
    print(json.dumps({"event": "low_variance_cells (background ref)",
                       **low_var_stats}), flush=True)

    # 6. Diagnose
    diag = {
        "high_var_consecutive_smooth": high_var_stats["consecutive_cos_mean"] >= 0.70,
        "high_var_endpoints_differ": (
            high_var_stats["startend_cos_mean"]
            < low_var_stats["startend_cos_mean"] - 0.1
        ),
        "low_var_endpoints_stable": low_var_stats["startend_cos_mean"] >= 0.95,
    }

    # Heuristic verdict
    if diag["high_var_consecutive_smooth"] and diag["high_var_endpoints_differ"]:
        recommendation = (
            "PROCEED with caution — Cosmos features at active cells "
            "evolve smoothly (suggesting they could carry trackable "
            "identity), and they differ at start vs end (responding to "
            "motion). Build the Cosmos→OF-JEPA slot-attention bridge "
            "and test downstream id_h_mse to confirm."
        )
    elif diag["high_var_consecutive_smooth"]:
        recommendation = (
            "AMBIGUOUS — features are smooth but don't clearly "
            "differentiate start/end. Could be that the can returned "
            "to a similar position by demo's end. Inspect rollout "
            "timing or pick a different demo with clearer net motion."
        )
    else:
        recommendation = (
            "DO NOT PROCEED — Cosmos features at active cells are not "
            "smooth (chaotic across consecutive frames). Identity "
            "binding on these features is unlikely to succeed."
        )

    verdict = {**diag, "recommendation": recommendation}
    print(json.dumps({"event": "verdict", **verdict}), flush=True)

    summary = {
        "args": vars(args),
        "rollout": {
            "n_frames": int(frames.shape[0]),
            "can_max_displacement_m": float(can_displacement.max()),
            "can_max_z_change_m": float(can_z_change.max()),
        },
        "latent_shape": [C, T_lat, H, W],
        "high_variance_stats": high_var_stats,
        "low_variance_stats": low_var_stats,
        "verdict": verdict,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
