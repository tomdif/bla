"""Phase Scale-1 — task-agnostic 3-mode pilot/main eval.

Generic version of phase_d{3,4}_pilot.py. Accepts a --task arg
that selects which task primitives to import. For Scale-1 we only
add one new task (ToolHang) but this scaffolding is the start of
the bla/eval harness.

Modes (3): demo_no_cem, phase17_locked, naive_cem.
Protocol: state-matched demo init, n eps × seed × mode.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from system1_jepa.of_jepa import OFJEPAConfig
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


TASK_REGISTRY = {
    "toolhang": {
        "module": "scripts.phase_scale1_toolhang",
        "env_fn": "build_env_toolhang",
        "get_obj_pos": "get_tool_pos",
        "improvement_fn": "toolhang_improvement",
        "lift_target": "LIFT_TARGET_Z_GAIN",
        "demo_dir": "/workspace/robomimic_toolhang_replay",
        "working_demos": (1, 4, 12, 13, 14),
    },
}


encode_frame = None; predict_score_seq = None; cem_with_prior = None


def load_planning_deps():
    global encode_frame, predict_score_seq, cem_with_prior
    from scripts.phase15_planning import (
        encode_frame as _e, predict_score_seq as _p,
    )
    from scripts.phase16_policy_prior_mpc import cem_with_prior as _c
    encode_frame = _e; predict_score_seq = _p; cem_with_prior = _c


def load_action_model(args):
    cfg = OFJEPAConfig(
        n_files=args.n_slots, id_dim=args.slot_dim // 2,
        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim,
    )
    m = ActionConditionedOFJEPA(
        image_size=args.image_size, cfg=cfg,
        action_dim=args.action_dim, use_action=True,
    ).to(args.device)
    m.load_state_dict(torch.load(args.model_action, map_location=args.device))
    m.eval()
    return m


def norm_xy(p_xy):
    return np.clip((np.asarray(p_xy) + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def per_dim_sigma(scalar: float, gripper_sigma: float = 0.0,
                    action_dim: int = 7) -> np.ndarray:
    return np.array([scalar] * (action_dim - 1) + [gripper_sigma],
                       dtype=np.float32)


def get_demo_prior(demos_actions: list[np.ndarray], demo_id: int,
                     H: int, stride: int) -> np.ndarray:
    demo = demos_actions[demo_id]
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo) - 1)
        out.append(demo[idx])
    return np.stack(out, axis=0)


def run_episode(env, model, mode, args, demo_id, demos_actions,
                  task_mod, ep_id):
    demo_path = f"{TASK_REGISTRY[args.task]['demo_dir']}/ep_{demo_id:05d}.npz"
    obs = task_mod.reset_env_to_demo_init(env, demo_path)
    get_obj_pos = getattr(task_mod, TASK_REGISTRY[args.task]["get_obj_pos"])
    improvement_fn = getattr(task_mod, TASK_REGISTRY[args.task]["improvement_fn"])
    obj_z0 = float(get_obj_pos(obs)[2])
    obj_xy0 = get_obj_pos(obs)[:2].copy()
    max_obj_z = obj_z0

    init_slot = encode_frame(model, obs["agentview_image"])
    from system1_jepa.identity_probe import hungarian_assign
    pred_pos_t0 = model.slot_to_pos_aux(
        init_slot.unsqueeze(0))[0].detach().cpu().numpy()
    eef_xy0_norm = norm_xy(obs["robot0_eef_pos"][:2])
    obj_xy0_norm = norm_xy(obj_xy0)
    gt_pos = np.stack([obj_xy0_norm, eef_xy0_norm])
    rows, cols, _ = hungarian_assign(pred_pos_t0, gt_pos)
    obj_slot_idx = None
    for r, c in zip(rows.tolist(), cols.tolist()):
        if int(c) == 0:
            obj_slot_idx = int(r); break
    if obj_slot_idx is None:
        obj_slot_idx = int(np.argmin(
            np.linalg.norm(pred_pos_t0 - obj_xy0_norm, axis=1)))

    fake_goal_norm = norm_xy(obj_xy0 + np.array([0.0, 0.10], dtype=np.float32))
    def score_fn(action_seq_np):
        return predict_score_seq(
            model, init_slot, action_seq_np,
            obj_slot_idx, fake_goal_norm, use_action=True,
        )

    mu_demo = get_demo_prior(demos_actions, demo_id,
                               args.plan_horizon, args.jepa_stride)

    if mode == "demo_no_cem":
        actions_executed = mu_demo
    elif mode == "phase17_locked":
        sig = per_dim_sigma(args.eval_sigma, 0.0, args.action_dim)
        floor = per_dim_sigma(args.sigma_floor, 0.0, args.action_dim)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu_demo, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    elif mode == "naive_cem":
        mu0 = np.zeros((args.plan_horizon, args.action_dim), dtype=np.float32)
        sig = per_dim_sigma(args.naive_sigma, 0.0, args.action_dim)
        floor = per_dim_sigma(args.sigma_floor, 0.0, args.action_dim)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu0, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
            max_obj_z = max(max_obj_z, float(get_obj_pos(obs)[2]))
    obj_z_end = float(get_obj_pos(obs)[2])
    z_gain_max = max_obj_z - obj_z0
    lift_target = getattr(task_mod, TASK_REGISTRY[args.task]["lift_target"])
    imp = improvement_fn(max_obj_z, obj_z0, lift_target)
    success = z_gain_max >= 0.05
    return {"ep_id": ep_id, "demo_id": demo_id,
              "obj_z0": obj_z0, "obj_z_end": obj_z_end,
              "max_obj_z": max_obj_z,
              "z_gain_max_m": z_gain_max,
              "improvement": imp,
              "success": int(success), "mode": mode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_REGISTRY.keys()))
    p.add_argument("--model-action", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=100)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--eval-K", type=int, default=32)
    p.add_argument("--eval-cem-iters", type=int, default=1)
    p.add_argument("--eval-sigma", type=float, default=0.12)
    p.add_argument("--naive-sigma", type=float, default=0.5)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--modes", type=str,
                    default="demo_no_cem,phase17_locked,naive_cem")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    task_cfg = TASK_REGISTRY[args.task]
    task_mod = importlib.import_module(task_cfg["module"])
    print(json.dumps({"event": "args", "task": args.task,
                       "args": vars(args)}), flush=True)

    load_planning_deps()
    model = load_action_model(args)

    working = task_cfg["working_demos"]
    task_mod.extract_demos_to_cache(n_demos=max(working) + 1)
    demos_actions = task_mod.load_demo_actions(
        demo_ids=tuple(range(max(working) + 1)))

    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 200
    env_fn = getattr(task_mod, task_cfg["env_fn"])
    env = env_fn(args.image_size, big_horizon)

    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i in range(args.n_eval_episodes):
            demo_id = working[i % len(working)]
            t0 = time.time()
            r = run_episode(env, model, mode, args, demo_id, demos_actions,
                              task_mod, i)
            r["elapsed_s"] = time.time() - t0
            per_ep.append(r)
            print(json.dumps({"event": "ep", **r}), flush=True)
        imp_mean = float(np.mean([e["improvement"] for e in per_ep]))
        succ_mean = float(np.mean([e["success"] for e in per_ep]))
        z_mean = float(np.mean([e["z_gain_max_m"] for e in per_ep]))
        all_results[mode] = {"per_ep": per_ep, "improvement_mean": imp_mean,
                                "success_rate": succ_mean,
                                "z_gain_max_mean_m": z_mean}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "z_gain_max_mean_m": z_mean}), flush=True)

    summary = {"task": args.task, "args": vars(args), "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
