"""Phase V1a-M2 — Cosmos-Tokenizer identity-preservation probe.

The task description for V1a flags M2 (identity preservation under
occlusion) as the critical metric:

  G0 only proved interface compat (temporal stability under rolling
  windows). G0 did NOT prove representational quality. If M2 fails,
  V1a is dead even though G0 passed.

Rather than building the full Cosmos → learned projection → OF-JEPA
slot attention pipeline first, this probe asks the cheaper question:

  Do Cosmos's spatial latent features at an object's location encode
  a CONSISTENT, identity-discriminating signal as the object moves?

Two diagnostics on a 13-frame robosuite rollout with cube motion:

  Test A — object-tracking stability:
    At each frame, get GT cube position (mujoco) → map to Cosmos
    latent spatial position. Extract 16-D feature at cube's location.
    Compute cube-feature cosine across consecutive frames.
    HIGH cosine = identity-bound (good; cube's representation
                                  travels with the cube)
    LOW cosine = position-bound (bad; feature changes as cube moves
                                  through space)

  Test B — object vs background discrimination:
    At each frame, also extract feature at a fixed table position
    away from the cube. Compute cube-vs-table cosine.
    LOW cosine = features distinguish cube from background (good)
    HIGH cosine = features are uniform; can't pull out per-object
                  signal (bad)

Combined gate (V1a-M2 pass):
  cube_consecutive_cosine_mean ≥ 0.90  AND  cube_vs_table_cosine_mean ≤ 0.50
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
COSMOS_LATENT_SPATIAL = 32  # 256/8 = 32 (CV4x8x8 spatial compression)
COSMOS_LATENT_CHANNELS = 16


def collect_can_demo_rollout(image_size: int = COSMOS_RES,
                                  n_frames: int = 17,
                                  demo_id: int = 5,
                                  demo_dir: str = "/workspace/robomimic_can_replay"
                                  ) -> dict:
    """Replay a known-working robomimic Can demo to get real can motion.

    Demo 5 is one of the 24 working demos verified during V1b screen.
    Apply its first n_frames actions on its own state-matched init.
    """
    import robosuite as rs
    demo_path = Path(demo_dir) / f"ep_{demo_id:05d}.npz"
    d = np.load(demo_path)
    actions = d["actions"]
    init_state = d["init_state"]

    env = rs.make("PickPlaceCan", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview",
                    camera_heights=image_size, camera_widths=image_size,
                    horizon=n_frames + 30)
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    obs = env._get_observations()

    frames = [obs["agentview_image"]]
    can_xy = [env.sim.data.get_body_xpos("Can_main")[:2].copy()]
    eef_xy = [env.sim.data.get_body_xpos("gripper0_right_eef")[:2].copy()]

    n_apply = min(n_frames - 1, len(actions))
    for t in range(n_apply):
        obs, _, _, _ = env.step(actions[t])
        frames.append(obs["agentview_image"])
        can_xy.append(env.sim.data.get_body_xpos("Can_main")[:2].copy())
        eef_xy.append(env.sim.data.get_body_xpos("gripper0_right_eef")[:2].copy())

    return {
        "frames": np.stack(frames, axis=0),
        "cube_xy_world": np.stack(can_xy, axis=0).astype(np.float32),
        "eef_xy_world": np.stack(eef_xy, axis=0).astype(np.float32),
    }


def world_to_latent_pixel(
    world_xy: np.ndarray, latent_h: int = COSMOS_LATENT_SPATIAL,
    latent_w: int = COSMOS_LATENT_SPATIAL,
) -> tuple[int, int]:
    """Map world (x, y) to (lat_row, lat_col) in the Cosmos latent grid.

    Crude approximation: robosuite Lift's tabletop spans roughly
    [-0.25, +0.25] in x and [-0.25, +0.25] in y under the default
    agentview camera. The image is a top-down view (mostly).

    Tuned by inspecting that cube_xy ~ (0, 0) renders near the image
    center. We use a linear remapping with empirically reasonable
    extents. This is good enough to identify which latent cell the
    cube occupies for the cosine probe.

    Note: agentview is angled, not strictly top-down — but for this
    probe we only need the cube's location to land in a CONSISTENT
    latent cell across frames, not perfect projection.
    """
    # Simple linear scaling: world x ∈ [-0.30, 0.30] → image x ∈ [0, 256] →
    # latent x ∈ [0, 32]. Agentview camera is angled, so we apply a small
    # offset bias to compensate.
    x, y = float(world_xy[0]), float(world_xy[1])
    # Image rows correspond to world -y direction (camera angled down).
    # Image cols correspond to world +x.
    # These constants are approximations; the protocol assumes consistency,
    # not absolute correctness.
    norm_x = np.clip((x + 0.30) / 0.60, 0.0, 1.0)
    norm_y = np.clip((y + 0.30) / 0.60, 0.0, 1.0)
    col = int(norm_x * (latent_w - 1))
    row = int((1.0 - norm_y) * (latent_h - 1))
    return row, col


@torch.no_grad()
def encode_with_cosmos(encoder, frames_THWC: np.ndarray,
                          device: str = "cuda") -> np.ndarray:
    """Encode T frames at 256×256 → latent [16, T_lat, 32, 32] as float32 np."""
    x = torch.from_numpy(frames_THWC).float() / 127.5 - 1.0   # → [-1, 1]
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to(device).to(torch.bfloat16)
    (lat,) = encoder.encode(x)
    # lat: [1, 16, T_lat, 32, 32]
    return lat[0].float().cpu().numpy()  # [16, T_lat, 32, 32]


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                    default="/root/cosmos_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit")
    p.add_argument("--out", default="/root/bla/runs/phase_v1a_m2")
    p.add_argument("--n-frames", type=int, default=17,
                    help="Cosmos needs T ≡ 1 (mod 4); T=17 → 5 latent frames")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cube-stability-threshold", type=float, default=0.90,
                    help="Mean consecutive-frame cosine for cube feature")
    p.add_argument("--cube-vs-table-threshold", type=float, default=0.50,
                    help="Mean cube-vs-empty-table cosine (lower = better)")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    # 1. Collect a PickPlaceCan demo rollout with real can motion (the
    # robomimic demo reliably grasps and lifts the can, giving us 5+ cm
    # of vertical motion + significant horizontal motion as the EEF
    # carries it).
    rollout = collect_can_demo_rollout(COSMOS_RES, args.n_frames)
    frames = rollout["frames"]
    cube_xy = rollout["cube_xy_world"]
    print(json.dumps({"event": "rollout",
                       "n_frames": int(frames.shape[0]),
                       "cube_xy_first": cube_xy[0].tolist(),
                       "cube_xy_last": cube_xy[-1].tolist(),
                       "cube_displacement_m": float(
                           np.linalg.norm(cube_xy[-1] - cube_xy[0]))}),
           flush=True)

    # 2. Load Cosmos encoder
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    encoder = CausalVideoTokenizer(checkpoint_enc=args.ckpt)

    # 3. Encode rollout
    t0 = time.time()
    latents = encode_with_cosmos(encoder, frames, args.device)  # [16, T_lat, 32, 32]
    encode_elapsed = time.time() - t0
    C, T_lat, H_lat, W_lat = latents.shape
    print(json.dumps({"event": "encoded",
                       "latent_shape": [C, T_lat, H_lat, W_lat],
                       "encode_elapsed_s": round(encode_elapsed, 2)}),
           flush=True)

    # 4. Build mapping from latent frame → source frame group.
    # CV4x8x8 with causal 4× temporal compression: latent frame i covers
    # source frames [4(i-1)+1, ..., 4i] for i >= 1, with latent frame 0
    # covering just source frame 0.
    # Simpler approximation: latent frame i corresponds to the MIDPOINT
    # of its 4-frame source window. For T_in=17 → T_lat=5, we have:
    #   lat 0 ← src 0     (or [0..3])
    #   lat 1 ← src ~2.5  (representative midpoint of frames 1-4)
    #   lat 2 ← src ~6.5
    #   lat 3 ← src ~10.5
    #   lat 4 ← src ~14.5
    # We'll use latent frame i → source frame min(4i, T_in-1).
    n_src = frames.shape[0]
    lat_to_src = [min(4 * i, n_src - 1) for i in range(T_lat)]

    # 5. Test A: cube-feature stability across consecutive latent frames
    cube_features = []
    table_features = []
    cube_latent_cells = []
    for lat_idx, src_idx in enumerate(lat_to_src):
        cube_world = cube_xy[src_idx]
        row, col = world_to_latent_pixel(cube_world)
        cube_latent_cells.append((row, col))
        cube_feat = latents[:, lat_idx, row, col]   # [16]
        # Table feature: fixed empty-table location far from cube; use
        # latent corner (0, 0) which corresponds to upper-left of view,
        # far from where the cube is.
        table_feat = latents[:, lat_idx, 2, 2]
        cube_features.append(cube_feat)
        table_features.append(table_feat)

    consecutive_cube_cos = []
    for t in range(len(cube_features) - 1):
        consecutive_cube_cos.append(cosine(cube_features[t], cube_features[t + 1]))
    cube_vs_table_cos = []
    for t in range(len(cube_features)):
        cube_vs_table_cos.append(cosine(cube_features[t], table_features[t]))

    stats = {
        "n_latent_frames": T_lat,
        "lat_to_src_indices": lat_to_src,
        "cube_latent_cells": cube_latent_cells,
        "consecutive_cube_cos": consecutive_cube_cos,
        "consecutive_cube_cos_mean": float(np.mean(consecutive_cube_cos)),
        "consecutive_cube_cos_min": float(np.min(consecutive_cube_cos)),
        "cube_vs_table_cos": cube_vs_table_cos,
        "cube_vs_table_cos_mean": float(np.mean(cube_vs_table_cos)),
        "cube_vs_table_cos_max": float(np.max(cube_vs_table_cos)),
    }

    print(json.dumps({"event": "test_a_stability",
                       "cube_consecutive_cos_mean": stats[
                           "consecutive_cube_cos_mean"],
                       "cube_consecutive_cos_min": stats[
                           "consecutive_cube_cos_min"],
                       "passes_stability": stats[
                           "consecutive_cube_cos_mean"] >= args.cube_stability_threshold}),
           flush=True)
    print(json.dumps({"event": "test_b_discrimination",
                       "cube_vs_table_cos_mean": stats[
                           "cube_vs_table_cos_mean"],
                       "cube_vs_table_cos_max": stats[
                           "cube_vs_table_cos_max"],
                       "passes_discrimination": stats[
                           "cube_vs_table_cos_mean"] <= args.cube_vs_table_threshold}),
           flush=True)

    passes_stability = stats[
        "consecutive_cube_cos_mean"] >= args.cube_stability_threshold
    passes_discrimination = stats[
        "cube_vs_table_cos_mean"] <= args.cube_vs_table_threshold
    passes_m2 = passes_stability and passes_discrimination

    verdict = {
        "passes_stability": passes_stability,
        "passes_discrimination": passes_discrimination,
        "passes_m2": passes_m2,
        "recommendation": (
            "PROCEED — Cosmos features support identity preservation; "
            "build full slot-attention bridge"
            if passes_m2 else
            "DO NOT PROCEED with default V1a — Cosmos VAE features at "
            "object position do not preserve identity-like signal. "
            "Either V1a is dead, or V1a must use a different layer of "
            "Cosmos (intermediate features, not the bottleneck latent)."
        ),
    }
    print(json.dumps({"event": "verdict", **verdict}), flush=True)

    summary = {
        "args": vars(args),
        "rollout": {
            "n_frames": int(frames.shape[0]),
            "cube_displacement_m": float(
                np.linalg.norm(cube_xy[-1] - cube_xy[0])),
        },
        "stats": stats,
        "verdict": verdict,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
