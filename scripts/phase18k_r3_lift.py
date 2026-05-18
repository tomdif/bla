"""Phase 18κ Regime 3 — Lift task primitives.

Shared module providing:
- build_env_lift: robosuite Lift task env
- sample_lift_goal: (cube_init_xy, target_lift_height) per episode
- state_features_lift: 10-dim engineered geo for Lift (same shape as Stack)
- lift_improvement: cube z-gain / target_lift_height (clipped to [0, 1])
- find_cube_slot_lift: Hungarian-style slot→cube matching for Lift
- scripted_lift_action: 3-stage finite-state machine
- rollout_scripted_lift_prior: env-clone rollout helper

Lift task differences from Stack:
- Single cube named "cube" (Stack has cubeA + cubeB)
- Goal is vertical, not horizontal displacement
- episode improvement = cube_z gain / 0.06 (target lift height)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


TARGET_LIFT_HEIGHT = 0.06  # 6 cm, robosuite default for Lift success


def build_env_lift(image_size: int, horizon: int):
    """Same configuration as build_env (Stack) but on Lift task."""
    import robosuite as rs
    return rs.make("Lift", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview", camera_heights=image_size,
                    camera_widths=image_size, horizon=horizon)


def sample_lift_goal(obs: dict, ep_id: int,
                       target_lift_height: float = TARGET_LIFT_HEIGHT
                       ) -> tuple[np.ndarray, float]:
    """For Lift, the 'goal_xy' is the initial cube xy position; the
    vertical target is encoded via lift_improvement at evaluation."""
    cube_init_xy = obs["cube_pos"][:2].astype(np.float32).copy()
    return cube_init_xy, float(target_lift_height)


def state_features_lift(obs: dict, goal_xy: np.ndarray) -> np.ndarray:
    """10-dim engineered geometry for Lift, matching state_features shape.

    Components:
      cube_xy (2): obs["cube_pos"][:2]
      eef_xy  (2): obs["robot0_eef_pos"][:2]
      eef_z   (1)
      cube_z  (1)
      goal_xy (2): cube initial xy (passed in)
      push_dir(2): direction eef→cube (not meaningful for Lift but
                    kept for architectural compatibility)
    """
    cube_xy = obs["cube_pos"][:2].astype(np.float32)
    eef_xy = obs["robot0_eef_pos"][:2].astype(np.float32)
    eef_z = float(obs["robot0_eef_pos"][2])
    cube_z = float(obs["cube_pos"][2])
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    # eef→cube direction, normalized
    diff = (cube_xy - eef_xy).astype(np.float32)
    norm = float(np.linalg.norm(diff))
    push_dir = diff / max(norm, 1e-9) if norm > 1e-9 else np.zeros(2, dtype=np.float32)
    return np.concatenate([
        cube_xy, eef_xy, [eef_z], [cube_z], goal_xy, push_dir
    ]).astype(np.float32)


def lift_improvement(cube_z_start: float, cube_z_end: float,
                       target_lift_height: float = TARGET_LIFT_HEIGHT) -> float:
    """Normalized cube z-gain. 0 = no lift, 1.0 = lifted ≥ target."""
    gain = max(0.0, cube_z_end - cube_z_start)
    return float(min(1.0, gain / max(target_lift_height, 1e-9)))


def find_cube_slot_lift(model, slot_state, cube_xy_norm,
                          eef_xy_norm) -> int:
    """Identity-aware slot→cube matching for Lift (1 cube + 1 eef).

    Reuses Stack's find_cubeA_slot pattern but with 2 entities
    (cube + eef) instead of 3 (cubeA + cubeB + eef).
    Returns the index of the slot whose match is to entity 0 (cube).
    """
    from system1_jepa.identity_probe import hungarian_assign

    pred_pos = model.slot_to_pos_aux(slot_state.unsqueeze(0))[0].detach().cpu().numpy()
    gt_pos = np.stack([cube_xy_norm, eef_xy_norm])  # [2, 2]
    rows, cols, _ = hungarian_assign(pred_pos, gt_pos)
    matches = list(zip(rows.tolist(), cols.tolist()))
    for r, c in matches:
        if c == 0:
            return int(r)
    # Fallback: closest slot to the cube
    return int(np.argmin(np.linalg.norm(pred_pos - cube_xy_norm, axis=1)))


# ---------- Scripted Lift prior (3-stage FSM) ----------
def _gripper_open(action_dim: int) -> np.ndarray:
    a = np.zeros(action_dim, dtype=np.float32)
    a[-1] = -1.0  # robosuite convention: -1 = open
    return a


def _gripper_close(action_dim: int) -> np.ndarray:
    a = np.zeros(action_dim, dtype=np.float32)
    a[-1] = 1.0  # +1 = close
    return a


def scripted_lift_action(env, t: int, obs: dict, ep_state: dict,
                          *, grasp_height_above_cube: float = 0.02,
                          grasp_steps: int = 10) -> np.ndarray:
    """Demo-pattern-matched scripted prior for Lift.

    Three FSM phases, calibrated against robomimic Lift demo 1 which
    successfully lifts (~0.067m gain):

    Phase 1 (reach):  closed-loop descent to (cube_xy, cube_z + 0.02m),
                      gripper open. The +0.02m offset puts the Panda
                      fingertips around cube center (cube is 0.04m tall;
                      eef_pos→fingertip offset for Panda is ~0.10m, so
                      eef_z = cube_z + 0.02 means fingertips at
                      cube_z - 0.08, below cube — but robosuite/MuJoCo's
                      compliant grasp allows fingers to wrap UP from
                      below as they close).

    Phase 2 (grasp):  hold position; close gripper. grasp_steps=10 gives
                      the parallel gripper enough time to fully close
                      (Panda gripper takes ~8 sim steps to close from
                      open to closed).

    Phase 3 (lift):   slow +z ramp matching demo pattern.
                      a[2] = 0.3 (gentle); cube follows.

    ep_state is per-episode dict; initialized to {} before first call.
    """
    action_dim = env.action_dim
    if "phase" not in ep_state:
        ep_state["phase"] = "reach"
        ep_state["grasp_counter"] = 0
        ep_state["lift_step"] = 0

    cube_xy = obs["cube_pos"][:2]
    cube_z = obs["cube_pos"][2]
    eef_xy = obs["robot0_eef_pos"][:2]
    eef_z = obs["robot0_eef_pos"][2]
    eef_to_cube_xy = cube_xy - eef_xy
    horiz_close = float(np.linalg.norm(eef_to_cube_xy)) < 0.015
    target_grasp_z = cube_z + grasp_height_above_cube

    if ep_state["phase"] == "reach":
        z_close = abs(eef_z - target_grasp_z) < 0.010
        if horiz_close and z_close:
            ep_state["phase"] = "grasp"
            return _gripper_close(action_dim)
        a = _gripper_open(action_dim)
        a[0] = np.clip(eef_to_cube_xy[0] * 10.0, -1, 1)
        a[1] = np.clip(eef_to_cube_xy[1] * 10.0, -1, 1)
        a[2] = np.clip((target_grasp_z - eef_z) * 8.0, -1, 1)
        return a

    if ep_state["phase"] == "grasp":
        # Hold position, close gripper
        a = _gripper_close(action_dim)
        a[0] = np.clip(eef_to_cube_xy[0] * 2.0, -0.2, 0.2)
        a[1] = np.clip(eef_to_cube_xy[1] * 2.0, -0.2, 0.2)
        a[2] = np.clip((target_grasp_z - eef_z) * 2.0, -0.2, 0.2)
        ep_state["grasp_counter"] += 1
        if ep_state["grasp_counter"] >= grasp_steps:
            ep_state["phase"] = "lift"
        return a

    # Lift phase: slow +z ramp, matching demo pattern
    a = _gripper_close(action_dim)
    ep_state["lift_step"] += 1
    # ramp from 0.2 to 0.5 over 5 steps, then hold at 0.5
    ramp = min(0.5, 0.2 + 0.06 * ep_state["lift_step"])
    a[2] = ramp
    return a


def rollout_scripted_lift_prior(env, obs: dict, goal_xy: np.ndarray,
                                  H: int, stride: int) -> np.ndarray:
    """Save env state, roll out scripted_lift for H actions, restore."""
    saved = env.sim.get_state()
    actions = []
    cur = obs
    ep_state = {}
    for _ in range(H):
        a = scripted_lift_action(env, 0, cur, ep_state)
        actions.append(a)
        for _ in range(stride):
            cur, _, _, _ = env.step(a)
    env.sim.set_state(saved); env.sim.forward()
    return np.stack(actions)


# ---------- Demo-replay prior (unblock for scripted_lift) ----------
_DEMO_ACTIONS_CACHE: dict = {}


def load_demo_actions(demo_dir: str = "/workspace/robomimic_lift_replay",
                         demo_ids: tuple = (1, 3)) -> list[np.ndarray]:
    """Load action sequences from successfully-lifting robomimic demos.

    Demos 1 and 3 were verified to achieve z_gain ~0.067 m on a fresh
    env reset (when their initial cube position roughly matches). Other
    demos fail due to cube-position mismatch. Use these as a fixed
    scripted prior — applied directly to fresh envs, they succeed
    when cube position matches and fail otherwise, generating useful
    label variance for training.
    """
    key = (demo_dir, demo_ids)
    if key in _DEMO_ACTIONS_CACHE:
        return _DEMO_ACTIONS_CACHE[key]
    actions_list = []
    for ep_id in demo_ids:
        path = f"{demo_dir}/ep_{ep_id:05d}.npz"
        d = np.load(path)
        actions_list.append(d["actions"].astype(np.float32))
    _DEMO_ACTIONS_CACHE[key] = actions_list
    return actions_list


def rollout_demo_lift_prior(env, obs: dict, goal_xy: np.ndarray,
                              H: int, stride: int,
                              demo_dir: str = "/workspace/robomimic_lift_replay",
                              demo_ids: tuple = (1, 3),
                              rng: np.random.RandomState | None = None) -> np.ndarray:
    """Demo-replay scripted prior: env-clone rollout using a random
    successful robomimic Lift demo's action sequence.

    For each call: pick a random demo from `demo_ids` (the verified
    working ones), apply its first H*stride actions to the env-cloned
    state, restore. Subsamples to H stride-steps to match plan_horizon.
    """
    demos = load_demo_actions(demo_dir, demo_ids)
    if rng is None:
        rng = np.random.RandomState()
    demo = demos[rng.randint(len(demos))]
    saved = env.sim.get_state()
    actions = []
    for t in range(H):
        # Use the demo's action for env-step t*stride (if available;
        # else repeat last action)
        demo_idx = min(t * stride, len(demo) - 1)
        a = demo[demo_idx]
        actions.append(a)
        for _ in range(stride):
            inner_idx = min(t * stride, len(demo) - 1)
            inner_a = demo[inner_idx]
            _ = env.step(inner_a)
    env.sim.set_state(saved); env.sim.forward()
    return np.stack(actions)


def closed_loop_gt_lift_step(env, obs: dict, goal_xy: np.ndarray) -> np.ndarray:
    """Oracle skyline for Lift: scripted_lift_action stateless (re-derives
    phase each step from current obs). Approximates closed-loop oracle."""
    cube_xy = obs["cube_pos"][:2]
    cube_z = obs["cube_pos"][2]
    eef_xy = obs["robot0_eef_pos"][:2]
    eef_z = obs["robot0_eef_pos"][2]
    eef_to_cube_xy = cube_xy - eef_xy
    horiz_close = float(np.linalg.norm(eef_to_cube_xy)) < 0.02
    has_lifted = cube_z > 0.85  # robosuite table is ~0.82; cube starts ~0.82
    action_dim = env.action_dim

    if not horiz_close:
        # Reach
        target_z = cube_z + 0.05
        a = _gripper_open(action_dim)
        a[0] = np.clip(eef_to_cube_xy[0] * 5.0, -1, 1)
        a[1] = np.clip(eef_to_cube_xy[1] * 5.0, -1, 1)
        a[2] = np.clip((target_z - eef_z) * 5.0, -1, 1)
        return a
    if not has_lifted:
        # Grasp + lift
        a = _gripper_close(action_dim)
        a[2] = 1.0  # up
        return a
    # Already lifted; maintain
    a = _gripper_close(action_dim)
    a[2] = 0.5
    return a
