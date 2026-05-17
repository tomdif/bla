"""Collect random-policy rollouts on a robosuite task.

For Phase 14: we treat the robot end-effector + each object as "objects"
the encoder will discover via slot-attention. Per-step records the
action vector + GT object positions for supervision.

Per episode .npz:
    video           [T, H, W, 3] uint8
    actions         [T, action_dim] float32  (control input at each step)
    cube_a_pos      [T, 3] float32 (cubeA world position)
    cube_b_pos      [T, 3] float32
    eef_pos         [T, 3] float32 (end-effector world position)
    rewards         [T] float32

Usage:
    python scripts/robosuite_collect_rollouts.py \\
        --task Stack --n-episodes 200 --horizon 80 \\
        --out /workspace/robosuite_local/stack
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite as rs


def collect_one(env, horizon: int, seed: int) -> dict:
    """One random-action rollout. Returns dict of per-step arrays."""
    np.random.seed(seed)
    obs = env.reset()
    T = horizon
    video = np.zeros((T, 128, 128, 3), dtype=np.uint8)
    actions = np.zeros((T, env.action_dim), dtype=np.float32)
    cube_a = np.zeros((T, 3), dtype=np.float32)
    cube_b = np.zeros((T, 3), dtype=np.float32)
    eef = np.zeros((T, 3), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    for t in range(T):
        video[t] = obs["agentview_image"]
        cube_a[t] = obs["cubeA_pos"]
        cube_b[t] = obs["cubeB_pos"]
        eef[t] = obs["robot0_eef_pos"]
        action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
        actions[t] = action
        obs, reward, done, _ = env.step(action)
        rewards[t] = reward
        if done:
            # Pad rest of arrays with last value (rare for random policy in 80 steps).
            for u in range(t + 1, T):
                video[u] = video[t]; actions[u] = 0
                cube_a[u] = cube_a[t]; cube_b[u] = cube_b[t]
                eef[u] = eef[t]; rewards[u] = 0
            break
    return {
        "video": video, "actions": actions,
        "cube_a_pos": cube_a, "cube_b_pos": cube_b, "eef_pos": eef,
        "rewards": rewards,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Stack")
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--out", required=True)
    p.add_argument("--robot", default="Panda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    env = rs.make(
        args.task, robots=args.robot,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=True,
        camera_names="agentview", camera_heights=128, camera_widths=128,
        horizon=args.horizon,
    )

    manifest = []
    for i in range(args.n_episodes):
        ep = collect_one(env, args.horizon, seed=i)
        ep_path = out / f"ep_{i:05d}.npz"
        np.savez_compressed(ep_path, **ep)
        manifest.append({
            "ep_id": i, "file": ep_path.name,
            "T": int(ep["video"].shape[0]),
            "final_reward": float(ep["rewards"][-1]),
        })
        if (i + 1) % 10 == 0:
            sz = sum(os.path.getsize(out / m["file"]) for m in manifest) / 1e6
            print(f"[{i + 1}/{args.n_episodes}] cached, {sz:.1f} MB total", flush=True)

    with open(out / "manifest.json", "w") as f:
        json.dump({
            "n_episodes": len(manifest),
            "task": args.task, "robot": args.robot,
            "horizon": args.horizon, "action_dim": int(env.action_dim),
            "episodes": manifest,
        }, f, indent=2)
    env.close()
    print(f"\nDone. {len(manifest)} episodes. action_dim={env.action_dim}")


if __name__ == "__main__":
    main()
