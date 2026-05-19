"""Phase D3 — PickPlaceCan task primitives.

Mirrors phase18k_r3_lift.py structure for the PickPlaceCan task.

Provides:
- build_env_pickplace: robosuite PickPlaceCan env
- sample_pickplace_goal: derive a goal target from initial scene
- state_features_pickplace: 10-dim engineered geo (same shape as Lift / Stack)
- pickplace_improvement: 0/1 success metric (can lift + place)
- find_can_slot: Hungarian-style slot→entity matching
- rollout_demo_pickplace_prior: replay robomimic can demo actions

Demos are extracted on first use from /workspace/robomimic_data/can_demo_v141.hdf5
and cached as /workspace/robomimic_can_replay/ep_XXXXX.npz.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


# Target heights (matched to robosuite PickPlaceCan defaults)
LIFT_TARGET_Z_GAIN = 0.10   # 10 cm above table — "lifted" gate
CAN_TABLE_Z = 0.82          # nominal table surface


def build_env_pickplace(image_size: int, horizon: int):
    """robosuite PickPlaceCan with offscreen rendering at image_size."""
    import robosuite as rs
    return rs.make("PickPlaceCan", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview", camera_heights=image_size,
                    camera_widths=image_size, horizon=horizon)


def get_can_pos(obs: dict) -> np.ndarray:
    """Robosuite PickPlaceCan exposes the can position under 'Can_pos'."""
    return np.asarray(obs["Can_pos"], dtype=np.float32)


def get_eef_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def sample_pickplace_goal(obs: dict, ep_id: int) -> tuple[np.ndarray, float]:
    """Goal target = the bin position the can should land in.

    For our doctrine test we don't need the actual bin location — the
    success metric is "can lifted ≥ LIFT_TARGET_Z_GAIN above table."
    Returns (initial_can_xy, lift_target_height) for state-features
    encoding consistency.
    """
    can_init_xy = get_can_pos(obs)[:2].copy()
    return can_init_xy, float(LIFT_TARGET_Z_GAIN)


def state_features_pickplace(obs: dict, goal_xy: np.ndarray) -> np.ndarray:
    """10-dim engineered geometry for PickPlaceCan.

    Components (matching Lift's shape):
      can_xy  (2)
      eef_xy  (2)
      eef_z   (1)
      can_z   (1)
      goal_xy (2)  ← initial can xy (used as anchor for normalization)
      push_dir(2)  ← eef→can direction (consistency placeholder)
    """
    can_pos = get_can_pos(obs)
    eef_pos = get_eef_pos(obs)
    can_xy = can_pos[:2].astype(np.float32)
    eef_xy = eef_pos[:2].astype(np.float32)
    eef_z = float(eef_pos[2])
    can_z = float(can_pos[2])
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    diff = (can_xy - eef_xy).astype(np.float32)
    norm = float(np.linalg.norm(diff))
    push_dir = (diff / max(norm, 1e-9) if norm > 1e-9
                  else np.zeros(2, dtype=np.float32))
    return np.concatenate([
        can_xy, eef_xy, [eef_z], [can_z], goal_xy, push_dir
    ]).astype(np.float32)


def pickplace_improvement(can_z_start: float, can_z_end: float,
                            target_z_gain: float = LIFT_TARGET_Z_GAIN) -> float:
    """Normalized can z-gain. 0 = no lift, 1.0 = lifted ≥ target.

    For PickPlaceCan, this measures the lift phase only. Full place-in-bin
    success is a stricter metric reported separately.
    """
    gain = max(0.0, can_z_end - can_z_start)
    return float(min(1.0, gain / max(target_z_gain, 1e-9)))


def pickplace_place_success(obs: dict) -> bool:
    """Did the can land in its target bin (robosuite's own success check)?"""
    try:
        return bool(obs.get("Can_in_target_bin", False))
    except Exception:
        return False


def find_can_slot(model, slot_state, can_xy_norm: np.ndarray,
                    eef_xy_norm: np.ndarray) -> int:
    """Hungarian-match slot → can. Two-entity variant of find_cubeA_slot."""
    from system1_jepa.identity_probe import hungarian_assign
    pred_pos = model.slot_to_pos_aux(slot_state.unsqueeze(0))[0
                                                                  ].detach().cpu().numpy()
    gt_pos = np.stack([can_xy_norm, eef_xy_norm])  # [2, 2]
    rows, cols, _ = hungarian_assign(pred_pos, gt_pos)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if int(c) == 0:
            return int(r)
    return int(np.argmin(np.linalg.norm(pred_pos - can_xy_norm, axis=1)))


# ---------- Demo extraction + replay ----------
_DEMO_CACHE: dict = {}
_DEMO_REPLAY_DIR = "/workspace/robomimic_can_replay"
_DEMO_SOURCE = "/workspace/robomimic_data/can_demo_v141.hdf5"


def extract_demos_to_cache(source: str = _DEMO_SOURCE,
                              out_dir: str = _DEMO_REPLAY_DIR,
                              n_demos: int = 20) -> list[str]:
    """First-time setup: extract actions + states[0] from robomimic hdf5 to npz.

    Saves both `actions` and `init_state` (initial mujoco qpos+qvel) so
    demos can be replayed with state-matched env resets. The init state
    is the full mujoco state vector that can be passed to env.sim's
    `set_state_from_flattened`.
    """
    import h5py
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(source, "r")
    demos = sorted(f["data"].keys(),
                     key=lambda k: int(k.split("_")[1]))[:n_demos]
    # Always re-extract if any file is missing init_state (older format)
    needs_rebuild = False
    for i in range(n_demos):
        p = out / f"ep_{i:05d}.npz"
        if not p.exists():
            needs_rebuild = True; break
        try:
            with np.load(p) as d:
                if "init_state" not in d.files:
                    needs_rebuild = True; break
        except Exception:
            needs_rebuild = True; break
    if not needs_rebuild:
        f.close()
        return [str(out / f"ep_{i:05d}.npz") for i in range(n_demos)]

    paths = []
    for i, dname in enumerate(demos):
        actions = f[f"data/{dname}/actions"][:].astype(np.float32)
        # states[0] is the initial flattened sim state for set_state_from_flattened
        init_state = f[f"data/{dname}/states"][0].astype(np.float64)
        path = out / f"ep_{i:05d}.npz"
        np.savez(path, actions=actions, init_state=init_state)
        paths.append(str(path))
    f.close()
    return paths


def reset_env_to_demo_init(env, demo_path: str) -> dict:
    """Set robosuite env's mujoco state to a demo's initial state, return obs."""
    env.reset()
    d = np.load(demo_path)
    init_state = d["init_state"]
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    # Re-derive obs
    obs = env._get_observations()
    return obs


def load_demo_actions(demo_dir: str = _DEMO_REPLAY_DIR,
                         demo_ids: tuple = tuple(range(20))
                         ) -> list[np.ndarray]:
    """Load action sequences from extracted PickPlaceCan demos."""
    key = (demo_dir, demo_ids)
    if key in _DEMO_CACHE:
        return _DEMO_CACHE[key]
    # Ensure cache exists
    if not Path(demo_dir).exists() or len(list(Path(demo_dir).glob(
            "ep_*.npz"))) < max(demo_ids) + 1:
        extract_demos_to_cache(out_dir=demo_dir,
                                  n_demos=max(demo_ids) + 1)
    actions_list = []
    for ep_id in demo_ids:
        path = f"{demo_dir}/ep_{ep_id:05d}.npz"
        d = np.load(path)
        actions_list.append(d["actions"].astype(np.float32))
    _DEMO_CACHE[key] = actions_list
    return actions_list


def rollout_demo_pickplace_prior(env, obs: dict, goal_xy: np.ndarray,
                                    H: int, stride: int,
                                    demo_dir: str = _DEMO_REPLAY_DIR,
                                    demo_ids: tuple = tuple(range(20)),
                                    rng: np.random.RandomState | None = None
                                    ) -> np.ndarray:
    """Demo-replay scripted prior for PickPlaceCan.

    Picks a random robomimic demo, applies its first H*stride actions to
    the env-cloned state, restores. Returns the stride-subsampled action
    sequence the planner will execute.
    """
    demos = load_demo_actions(demo_dir, demo_ids)
    if rng is None:
        rng = np.random.RandomState()
    demo = demos[rng.randint(len(demos))]
    saved = env.sim.get_state()
    actions = []
    for t in range(H):
        demo_idx = min(t * stride, len(demo) - 1)
        a = demo[demo_idx]
        actions.append(a)
        for _ in range(stride):
            inner_idx = min(t * stride, len(demo) - 1)
            inner_a = demo[inner_idx]
            _ = env.step(inner_a)
    env.sim.set_state(saved); env.sim.forward()
    return np.stack(actions)


# ---------- smoke test ----------
def _smoke():
    """Print a quick smoke summary if run directly."""
    import json
    paths = extract_demos_to_cache(n_demos=5)
    demos = load_demo_actions(demo_ids=(0, 1, 2, 3, 4))
    info = {"event": "smoke_demos_loaded",
              "paths": [str(p) for p in paths],
              "demo_action_shapes": [a.shape for a in demos],
              "demo_action_lens": [int(a.shape[0]) for a in demos]}
    print(json.dumps(info, default=str), flush=True)

    env = build_env_pickplace(image_size=128, horizon=400)
    obs = env.reset()
    can0 = get_can_pos(obs)
    eef0 = get_eef_pos(obs)
    print(json.dumps({"event": "smoke_env_reset",
                       "can_pos": can0.tolist(),
                       "eef_pos": eef0.tolist(),
                       "image_shape": obs["agentview_image"].shape,
                       "obs_keys_sample": [k for k in obs.keys()
                                              if "Can" in k or "can" in k or "target" in k]
                       }), flush=True)

    # Run one demo replay end-to-end and check can_z
    demo = demos[0]
    n_steps = min(len(demo), 80)
    for a in demo[:n_steps]:
        obs, _, _, _ = env.step(a)
    can_end = get_can_pos(obs)
    z_gain = float(can_end[2] - can0[2])
    print(json.dumps({"event": "smoke_demo_replay_done",
                       "n_steps": n_steps,
                       "can_start_z": float(can0[2]),
                       "can_end_z": float(can_end[2]),
                       "z_gain_m": z_gain,
                       "lifted": z_gain >= LIFT_TARGET_Z_GAIN,
                       "place_success": pickplace_place_success(obs)}),
           flush=True)


if __name__ == "__main__":
    _smoke()
