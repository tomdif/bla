"""Phase D4 — NutAssemblySquare task primitives.

Mirrors phase_d3_pickplace.py for the Square task. Tests the
demo-prior doctrine on a precise-insertion task (vs PickPlaceCan's
grasp-and-place).

Provides:
- build_env_square: robosuite NutAssemblySquare
- get_nut_pos: extract SquareNut xy[z] from obs
- state_features_square: 10-dim engineered geo
- square_improvement: lift+place success metric
- find_nut_slot: Hungarian-style slot→nut matching
- rollout_demo_square_prior: replay robomimic square demo actions

Demos extracted on first use from
/workspace/robomimic_data/square_demo_v141.hdf5 and cached as
/workspace/robomimic_square_replay/ep_XXXXX.npz.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


# Lift-gain target. NutAssembly requires the nut be lifted ≥ ~6cm to
# clear the peg base and insert. A 5cm gate captures the grasp+lift
# phase, which is what demo_no_cem most clearly preserves.
LIFT_TARGET_Z_GAIN = 0.05


def build_env_square(image_size: int, horizon: int):
    """robosuite NutAssemblySquare with offscreen rendering at image_size."""
    import robosuite as rs
    return rs.make("NutAssemblySquare", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview", camera_heights=image_size,
                    camera_widths=image_size, horizon=horizon)


def get_nut_pos(obs: dict) -> np.ndarray:
    """robosuite NutAssemblySquare exposes the nut under 'SquareNut_pos'."""
    return np.asarray(obs["SquareNut_pos"], dtype=np.float32)


def get_eef_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def sample_square_goal(obs: dict, ep_id: int) -> tuple[np.ndarray, float]:
    """Returns (initial_nut_xy, lift_target_height) for compatibility
    with the value-head pipeline. The actual target for Square is the
    peg position, but we don't need it for the demo_no_cem mode."""
    nut_init_xy = get_nut_pos(obs)[:2].copy()
    return nut_init_xy, float(LIFT_TARGET_Z_GAIN)


def state_features_square(obs: dict, goal_xy: np.ndarray) -> np.ndarray:
    """10-dim engineered geometry for NutAssemblySquare (Lift/Can shape)."""
    nut_pos = get_nut_pos(obs)
    eef_pos = get_eef_pos(obs)
    nut_xy = nut_pos[:2].astype(np.float32)
    eef_xy = eef_pos[:2].astype(np.float32)
    eef_z = float(eef_pos[2])
    nut_z = float(nut_pos[2])
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    diff = (nut_xy - eef_xy).astype(np.float32)
    norm = float(np.linalg.norm(diff))
    push_dir = (diff / max(norm, 1e-9) if norm > 1e-9
                  else np.zeros(2, dtype=np.float32))
    return np.concatenate([
        nut_xy, eef_xy, [eef_z], [nut_z], goal_xy, push_dir
    ]).astype(np.float32)


def square_improvement(nut_max_z: float, nut_z_start: float,
                         target_z_gain: float = LIFT_TARGET_Z_GAIN) -> float:
    """Normalized maximum nut z-gain during the episode.

    Max-z, not final-z: Square's successful trajectory ends with the
    nut on the peg (potentially BELOW max-lift), so final-z under-
    counts successful inserts. Max-z captures the lift event.
    """
    gain = max(0.0, nut_max_z - nut_z_start)
    return float(min(1.0, gain / max(target_z_gain, 1e-9)))


def find_nut_slot(model, slot_state, nut_xy_norm: np.ndarray,
                    eef_xy_norm: np.ndarray) -> int:
    """Hungarian-match slot → nut (two-entity variant of find_cubeA_slot)."""
    from system1_jepa.identity_probe import hungarian_assign
    pred_pos = model.slot_to_pos_aux(slot_state.unsqueeze(0))[0
                                                                  ].detach().cpu().numpy()
    gt_pos = np.stack([nut_xy_norm, eef_xy_norm])
    rows, cols, _ = hungarian_assign(pred_pos, gt_pos)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if int(c) == 0:
            return int(r)
    return int(np.argmin(np.linalg.norm(pred_pos - nut_xy_norm, axis=1)))


# ---------- Demo extraction + replay ----------
_DEMO_CACHE: dict = {}
_DEMO_REPLAY_DIR = "/workspace/robomimic_square_replay"
_DEMO_SOURCE = "/workspace/robomimic_data/square_demo_v141.hdf5"


def extract_demos_to_cache(source: str = _DEMO_SOURCE,
                              out_dir: str = _DEMO_REPLAY_DIR,
                              n_demos: int = 20) -> list[str]:
    """Extract actions + states[0] from robomimic Square demos to npz."""
    import h5py
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
        return [str(out / f"ep_{i:05d}.npz") for i in range(n_demos)]

    f = h5py.File(source, "r")
    demos = sorted(f["data"].keys(),
                     key=lambda k: int(k.split("_")[1]))[:n_demos]
    paths = []
    for i, dname in enumerate(demos):
        actions = f[f"data/{dname}/actions"][:].astype(np.float32)
        init_state = f[f"data/{dname}/states"][0].astype(np.float64)
        path = out / f"ep_{i:05d}.npz"
        np.savez(path, actions=actions, init_state=init_state)
        paths.append(str(path))
    f.close()
    return paths


def reset_env_to_demo_init(env, demo_path: str) -> dict:
    """Reset env to a demo's recorded initial mujoco state."""
    env.reset()
    d = np.load(demo_path)
    init_state = d["init_state"]
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    obs = env._get_observations()
    return obs


def load_demo_actions(demo_dir: str = _DEMO_REPLAY_DIR,
                         demo_ids: tuple = tuple(range(20))
                         ) -> list[np.ndarray]:
    key = (demo_dir, demo_ids)
    if key in _DEMO_CACHE:
        return _DEMO_CACHE[key]
    if not Path(demo_dir).exists() or len(list(Path(demo_dir).glob(
            "ep_*.npz"))) < max(demo_ids) + 1:
        extract_demos_to_cache(out_dir=demo_dir, n_demos=max(demo_ids) + 1)
    actions_list = []
    for ep_id in demo_ids:
        path = f"{demo_dir}/ep_{ep_id:05d}.npz"
        d = np.load(path)
        actions_list.append(d["actions"].astype(np.float32))
    _DEMO_CACHE[key] = actions_list
    return actions_list


# ---------- Smoke test: screen working demos ----------
def _smoke():
    """Screen 20 demos for which ones lift the nut on state-matched reset."""
    import json
    extract_demos_to_cache(n_demos=20)
    demos = load_demo_actions(demo_ids=tuple(range(20)))
    env = build_env_square(image_size=128, horizon=400)
    working = []
    for ep_id in range(20):
        demo_path = f"{_DEMO_REPLAY_DIR}/ep_{ep_id:05d}.npz"
        obs = reset_env_to_demo_init(env, demo_path)
        nut_z0 = float(get_nut_pos(obs)[2])
        demo = demos[ep_id]
        max_z = nut_z0
        for a in demo[: min(len(demo), 130)]:
            obs, _, _, _ = env.step(a)
            max_z = max(max_z, float(get_nut_pos(obs)[2]))
        z_gain = max_z - nut_z0
        if z_gain >= 0.05:
            working.append((ep_id, float(z_gain)))
    print(json.dumps({"event": "smoke_screen_done",
                       "n_working_of_20": len(working),
                       "working": working}), flush=True)


if __name__ == "__main__":
    _smoke()
