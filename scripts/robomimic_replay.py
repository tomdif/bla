"""Phase 14.5b — Replay robomimic demonstrations through robosuite to
produce per-frame images + GT positions, in the same .npz format that
`RobosuiteDataset` consumes.

The robomimic 'raw' hdf5 contains states + actions per demo, but no
images. We recreate the env from env_args, restore the initial state,
step through the recorded actions, and re-render at 128x128.

For Lift: 2 entities (cube + eef). We write cube_a_pos = cube_pos,
cube_b_pos = zeros, entity_visibility = [True, False, True] so the
dataset still has shape [T, 3, *] and a stationary 'phantom' cubeB is
masked out of training.

Usage:
    python scripts/robomimic_replay.py \\
        --hdf5 /workspace/robomimic_data/demo_v141.hdf5 \\
        --out  /workspace/robomimic_lift_replay \\
        --image-size 128 --horizon 80 --max-demos 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite as rs


def build_env_for_replay(env_args: dict, image_size: int):
    """Recreate env from robomimic env_args, overriding rendering for image obs.

    Robosuite v1.5+ changed the controller API: legacy 'OSC_POSE' configs
    saved in robomimic v1.4.1 datasets no longer load. We drop the legacy
    controller_configs and use robosuite's default Panda controller
    (which is OSC-compatible — robomimic actions are 7-DOF [Δx,Δy,Δz,
    Δrx,Δry,Δrz, gripper] which the default also accepts).
    """
    ek = dict(env_args["env_kwargs"])
    ek["has_renderer"] = False
    ek["has_offscreen_renderer"] = True
    ek["use_camera_obs"] = True
    ek["use_object_obs"] = True
    ek["camera_names"] = "agentview"
    ek["camera_heights"] = image_size
    ek["camera_widths"] = image_size
    ek.pop("controller_configs", None)
    robots = ek.pop("robots", "Panda")
    env = rs.make(env_args["env_name"], robots=robots, **ek)
    return env


def replay_one(env, states: np.ndarray, actions: np.ndarray, horizon: int):
    """State-restore replay: per frame, set mujoco state from the recorded
    trajectory, re-render. Avoids the v1.4→v1.5 controller mismatch — the
    (state, action, next_state) triples are preserved exactly from the
    original robomimic recording; we only resample the image at our
    target resolution.
    """
    T = min(horizon, len(actions))
    env.reset()

    image_size = env.camera_heights[0] if isinstance(env.camera_heights, list) else env.camera_heights
    video = np.zeros((T, image_size, image_size, 3), dtype=np.uint8)
    acts = np.zeros((T, env.action_dim), dtype=np.float32)
    cube_a = np.zeros((T, 3), dtype=np.float32)
    eef = np.zeros((T, 3), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)

    for t in range(T):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        obs = env._get_observations()
        video[t] = obs["agentview_image"]
        cube_a[t] = obs["cube_pos"]
        eef[t] = obs["robot0_eef_pos"]
        acts[t] = actions[t]
        rewards[t] = 0.0

    cube_b = np.zeros_like(cube_a)
    entity_visibility = np.array([1, 0, 1], dtype=bool)  # cubeA, cubeB(absent), eef
    return {
        "video": video,
        "actions": acts,
        "cube_a_pos": cube_a,
        "cube_b_pos": cube_b,
        "eef_pos": eef,
        "rewards": rewards,
        "entity_visibility": entity_visibility,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--max-demos", type=int, default=200)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.hdf5, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))
        demo_keys = demo_keys[: args.max_demos]
        print(f"Replaying {len(demo_keys)} demos from {args.hdf5}", flush=True)

        env = build_env_for_replay(env_args, args.image_size)
        manifest = []
        cube_disp = []
        for i, dk in enumerate(demo_keys):
            d = f["data"][dk]
            states = d["states"][:]
            actions = d["actions"][:].astype(np.float32)
            ep = replay_one(env, states, actions, args.horizon)
            ep_path = out / f"ep_{i:05d}.npz"
            np.savez_compressed(ep_path, **ep)
            d_a = float(np.linalg.norm(ep["cube_a_pos"][-1] - ep["cube_a_pos"][0]))
            cube_disp.append(d_a)
            manifest.append({
                "ep_id": i, "src_demo": dk, "file": ep_path.name,
                "T": int(ep["video"].shape[0]),
                "final_reward": float(ep["rewards"][-1]),
                "cube_a_disp": d_a,
            })
            if (i + 1) % 10 == 0:
                sz = sum(os.path.getsize(out / m["file"]) for m in manifest) / 1e6
                print(f"[{i + 1}/{len(demo_keys)}] cached, {sz:.1f} MB total, "
                      f"mean cube disp: a={np.mean(cube_disp):.3f}", flush=True)
        env.close()

    with open(out / "manifest.json", "w") as fp:
        json.dump({
            "n_episodes": len(manifest),
            "task": env_args["env_name"],
            "robot": "Panda",
            "policy": "robomimic_ph",
            "horizon": args.horizon, "action_dim": 7,
            "cube_a_disp_mean": float(np.mean(cube_disp)),
            "cube_b_disp_mean": 0.0,  # cubeB absent in Lift
            "entity_visibility": [1, 0, 1],
            "src_hdf5": args.hdf5,
            "episodes": manifest,
        }, fp, indent=2)
    print(f"\nDone. {len(manifest)} episodes. policy=robomimic_ph "
          f"mean cube disp: a={np.mean(cube_disp):.3f}m")


if __name__ == "__main__":
    main()
