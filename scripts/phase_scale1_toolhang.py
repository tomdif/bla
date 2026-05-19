"""Phase Scale-1 — ToolHang task primitives.

Mirrors phase_d4_square.py for the ToolHang task. ToolHang demos
are 5× longer than Lift/Can/Square (~600 steps).

Obs keys (verified):
  tool_pos      — the L-shaped tool that gets grasped
  frame_pos     — the stand/hook frame (target)
  tool_on_frame — boolean success metric
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


LIFT_TARGET_Z_GAIN = 0.05


def build_env_toolhang(image_size: int, horizon: int):
    import robosuite as rs
    return rs.make("ToolHang", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview", camera_heights=image_size,
                    camera_widths=image_size, horizon=horizon)


def get_tool_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs["tool_pos"], dtype=np.float32)


def get_eef_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def toolhang_improvement(tool_max_z: float, tool_z_start: float,
                            target_z_gain: float = LIFT_TARGET_Z_GAIN) -> float:
    gain = max(0.0, tool_max_z - tool_z_start)
    return float(min(1.0, gain / max(target_z_gain, 1e-9)))


_DEMO_CACHE: dict = {}
_DEMO_REPLAY_DIR = "/workspace/robomimic_toolhang_replay"
_DEMO_SOURCE = "/workspace/robomimic_data/tool_hang_demo_v141.hdf5"


def extract_demos_to_cache(source: str = _DEMO_SOURCE,
                              out_dir: str = _DEMO_REPLAY_DIR,
                              n_demos: int = 20) -> list[str]:
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
    env.reset()
    d = np.load(demo_path)
    env.sim.set_state_from_flattened(d["init_state"])
    env.sim.forward()
    return env._get_observations()


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


def _smoke():
    """Screen 20 demos for which lift the tool on state-matched reset.

    Uses up to 300 env steps (vs ToolHang's typical 400-680 demo length)
    since the LIFT event happens early in the demo.
    """
    import json
    extract_demos_to_cache(n_demos=20)
    demos = load_demo_actions(demo_ids=tuple(range(20)))
    env = build_env_toolhang(image_size=128, horizon=800)
    working = []
    for ep_id in range(20):
        demo_path = f"{_DEMO_REPLAY_DIR}/ep_{ep_id:05d}.npz"
        obs = reset_env_to_demo_init(env, demo_path)
        tool_z0 = float(get_tool_pos(obs)[2])
        demo = demos[ep_id]
        max_z = tool_z0
        for a in demo[: min(len(demo), 500)]:
            obs, _, _, _ = env.step(a)
            max_z = max(max_z, float(get_tool_pos(obs)[2]))
        z_gain = max_z - tool_z0
        if z_gain >= 0.05:
            working.append((ep_id, float(z_gain)))
    print(json.dumps({"event": "smoke_screen_done",
                       "n_working_of_20": len(working),
                       "working": working}), flush=True)


if __name__ == "__main__":
    _smoke()
