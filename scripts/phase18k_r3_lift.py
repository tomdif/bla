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

    Reuses the Hungarian probe logic but with 2 entities instead of 3.
    Returns the index of the slot whose match is to entity 0 (cube).
    """
    import torch
    from system1_jepa.identity_probe import hungarian_assign

    # slot_state: [n_slots, slot_dim] tensor
    # Extract predicted 2D positions for each slot from its first 2 dims
    # of the state portion (matching the Stack convention used in
    # find_cubeA_slot).
    device = slot_state.device if hasattr(slot_state, "device") else "cpu"
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        id_dim = cfg.id_dim
    else:
        id_dim = slot_state.shape[-1] // 2
    # Slot's predicted 2D position is at indices [id_dim, id_dim+2)
    pred_pos = slot_state[:, id_dim:id_dim + 2].detach().cpu().numpy()
    n_slots = pred_pos.shape[0]

    # Ground-truth entity positions in normalized 2D
    gt = np.stack([cube_xy_norm, eef_xy_norm])  # [2, 2]
    # Cost matrix: distance from each gt entity to each slot
    cost = np.linalg.norm(gt[:, None, :] - pred_pos[None, :, :], axis=-1)
    # Hungarian assignment (2 entities, n_slots slots)
    assignment = hungarian_assign(cost)
    # assignment is a list of (row, col) pairs; find the slot assigned
    # to entity 0 (cube)
    for row, col in assignment:
        if row == 0:
            return int(col)
    return 0  # fallback


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
                          *, reach_height: float = 0.05,
                          grasp_steps: int = 3) -> np.ndarray:
    """3-stage FSM scripted prior for Lift.

    Phase 1 (reach):    move EEF to position above the cube, gripper open
    Phase 2 (grasp):    close gripper at the cube
    Phase 3 (lift):     move EEF straight up

    ep_state is a per-episode dict tracking phase + counters; must be
    initialized to {} before the first call in an episode.
    """
    action_dim = env.action_dim
    if "phase" not in ep_state:
        ep_state["phase"] = "reach"
        ep_state["grasp_counter"] = 0

    cube_xy = obs["cube_pos"][:2]
    cube_z = obs["cube_pos"][2]
    eef_xy = obs["robot0_eef_pos"][:2]
    eef_z = obs["robot0_eef_pos"][2]
    eef_to_cube_xy = cube_xy - eef_xy
    horiz_close = float(np.linalg.norm(eef_to_cube_xy)) < 0.02

    if ep_state["phase"] == "reach":
        # Aim above the cube at reach_height
        target_z = cube_z + reach_height
        z_close = abs(eef_z - target_z) < 0.02
        if horiz_close and z_close:
            ep_state["phase"] = "grasp"
            return _gripper_open(action_dim)
        a = _gripper_open(action_dim)
        # OSC_POSE: a[0:3] is dx,dy,dz target (small steps)
        a[0] = np.clip(eef_to_cube_xy[0] * 5.0, -1, 1)
        a[1] = np.clip(eef_to_cube_xy[1] * 5.0, -1, 1)
        a[2] = np.clip((target_z - eef_z) * 5.0, -1, 1)
        return a

    if ep_state["phase"] == "grasp":
        # Close gripper at the cube; descend a bit so fingers wrap cube
        a = _gripper_close(action_dim)
        target_z = cube_z + 0.005  # slightly above cube center
        a[2] = np.clip((target_z - eef_z) * 5.0, -1, 1)
        ep_state["grasp_counter"] += 1
        if ep_state["grasp_counter"] >= grasp_steps:
            ep_state["phase"] = "lift"
        return a

    # Lift phase: maintain grasp, +z motion
    a = _gripper_close(action_dim)
    a[2] = 1.0  # max up
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
