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


def _random_action(env, t: int, ep_state: dict, obs) -> np.ndarray:
    return np.random.uniform(-1, 1, env.action_dim).astype(np.float32)


def _scripted_push_action(env, t: int, ep_state: dict, obs) -> np.ndarray:
    """Approach + contact + push policy.

    Two-phase scripted policy that produces informative (action, effect)
    pairs by driving the EE into the cube rather than hovering near it:

      Phase A (approach):  horizontal dist > contact radius
                            → aim at the cube + small vertical offset above
                            → gripper open
      Phase B (contact):   horizontal dist ≤ contact radius
                            → drive z downward to press into cube
                            → gripper closing (negative action sign in OSC)

    Phase 14.5 v1 (hover-only) hit ~0.04m displacement; this version
    targets ≥ 0.06–0.10m by adding the contact phase.

    Per-episode state:
      target_idx     — 0/1 which cube is currently the target
      switch_step    — when to switch targets next
    """
    if "target_idx" not in ep_state:
        _scripted_push_pick_target(ep_state, t)

    if t >= ep_state["switch_step"]:
        _scripted_push_pick_target(ep_state, t)

    target = obs["cubeA_pos"] if ep_state["target_idx"] == 0 else obs["cubeB_pos"]
    eef = obs["robot0_eef_pos"]
    horiz_dist = float(np.linalg.norm((target - eef)[:2]))

    a = np.zeros(env.action_dim, dtype=np.float32)
    contact_radius = 0.05   # slightly larger than cube width
    sweep_dir = ep_state["sweep_dir"]   # 2D unit vector for horizontal follow-through
    if horiz_dist > contact_radius:
        # Approach: aim at the side of the cube along sweep direction so
        # we land just off-center with momentum already in sweep_dir.
        # Aim at cube_pos - 0.05 * sweep_dir, at cube height.
        approach_target = target.copy()
        approach_target[:2] -= 0.05 * sweep_dir
        approach_target[2] += 0.005   # barely above table to slide laterally
        delta = approach_target - eef
        a[:3] = np.clip(delta * 10.0, -1, 1)
        a[6] = +1.0    # gripper closed = small effective tip for pushing
    else:
        # Contact / drive-through: push laterally in sweep_dir at full gain
        # so the EE sweeps PAST the cube position, dragging it sideways.
        a[0:2] = sweep_dir * 1.0
        a[2] = -0.3    # mild downward to maintain contact with cube side
        a[6] = +1.0
    a[3:6] = 0.0
    # Exploration noise σ=0.20: keeps action-discrimination non-trivial
    # without drowning out the scripted lateral sweep.
    a = a + np.random.normal(0, 0.20, env.action_dim).astype(np.float32)
    return np.clip(a, -1, 1)


def _scripted_push_pick_target(ep_state: dict, t: int):
    """(Re-)sample target cube and sweep direction."""
    ep_state["target_idx"] = int(np.random.randint(2))
    angle = float(np.random.uniform(0, 2 * np.pi))
    ep_state["sweep_dir"] = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    ep_state["switch_step"] = t + int(np.random.randint(20, 35))


_POLICIES = {
    "random": _random_action,
    "scripted_push": _scripted_push_action,
}


def collect_one(env, horizon: int, seed: int, policy: str = "random") -> dict:
    """One rollout under the chosen policy. Returns dict of per-step arrays."""
    np.random.seed(seed)
    obs = env.reset()
    T = horizon
    policy_fn = _POLICIES[policy]
    ep_state: dict = {}
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
        action = policy_fn(env, t, ep_state, obs)
        actions[t] = action
        obs, reward, done, _ = env.step(action)
        rewards[t] = reward
        if done:
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
    p.add_argument("--policy", default="random", choices=list(_POLICIES.keys()),
                    help="Action policy. 'random'=uniform[-1,1] (Phase 14.3 baseline). "
                          "'scripted_push'=OSC toward cube + noise (Phase 14.5).")
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
    cube_disp_a, cube_disp_b = [], []
    for i in range(args.n_episodes):
        ep = collect_one(env, args.horizon, seed=i, policy=args.policy)
        ep_path = out / f"ep_{i:05d}.npz"
        np.savez_compressed(ep_path, **ep)
        # Episode-level diagnostic: total cube displacement.
        d_a = float(np.linalg.norm(ep["cube_a_pos"][-1] - ep["cube_a_pos"][0]))
        d_b = float(np.linalg.norm(ep["cube_b_pos"][-1] - ep["cube_b_pos"][0]))
        cube_disp_a.append(d_a); cube_disp_b.append(d_b)
        manifest.append({
            "ep_id": i, "file": ep_path.name,
            "T": int(ep["video"].shape[0]),
            "final_reward": float(ep["rewards"][-1]),
            "cube_a_disp": d_a, "cube_b_disp": d_b,
        })
        if (i + 1) % 10 == 0:
            sz = sum(os.path.getsize(out / m["file"]) for m in manifest) / 1e6
            print(f"[{i + 1}/{args.n_episodes}] cached, {sz:.1f} MB total, "
                  f"mean cube disp: a={np.mean(cube_disp_a):.3f} b={np.mean(cube_disp_b):.3f}",
                  flush=True)

    with open(out / "manifest.json", "w") as f:
        json.dump({
            "n_episodes": len(manifest),
            "task": args.task, "robot": args.robot, "policy": args.policy,
            "horizon": args.horizon, "action_dim": int(env.action_dim),
            "cube_a_disp_mean": float(np.mean(cube_disp_a)),
            "cube_b_disp_mean": float(np.mean(cube_disp_b)),
            "episodes": manifest,
        }, f, indent=2)
    env.close()
    print(f"\nDone. {len(manifest)} episodes. action_dim={env.action_dim} policy={args.policy}")
    print(f"Mean cube displacement: cubeA={np.mean(cube_disp_a):.3f}m  cubeB={np.mean(cube_disp_b):.3f}m")


if __name__ == "__main__":
    main()
