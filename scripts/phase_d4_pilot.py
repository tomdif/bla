"""Phase D4 — NutAssemblySquare pilot + main eval.

Mirrors phase_d3_pilot.py with Square task primitives. Tests the
demo-prior doctrine on a precise-insertion task.

Modes (3): demo_no_cem, phase17_locked, naive_cem.
Protocol: state-matched demo init, n eps × seed × mode.
Pilot: n=5, 1 seed. Main: n=30, 3 seeds.
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

from system1_jepa.of_jepa import OFJEPAConfig
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA
from scripts.phase_d4_square import (
    build_env_square, extract_demos_to_cache, load_demo_actions,
    reset_env_to_demo_init, get_nut_pos, LIFT_TARGET_Z_GAIN,
    square_improvement,
)


# First 5 demos that lift on state-matched reset (from screen: 17/20 work)
WORKING_DEMO_IDS = (1, 2, 3, 4, 5)


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


def square_sigma(scalar: float, gripper_sigma: float = 0.0) -> np.ndarray:
    return np.array([scalar] * 6 + [gripper_sigma], dtype=np.float32)


def get_demo_prior(demos_actions: list[np.ndarray], demo_id: int,
                     H: int, stride: int) -> np.ndarray:
    demo = demos_actions[demo_id]
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo) - 1)
        out.append(demo[idx])
    return np.stack(out, axis=0)


def run_episode(env, model, mode, args, demo_id, demos_actions, ep_id):
    demo_path = f"/workspace/robomimic_square_replay/ep_{demo_id:05d}.npz"
    obs = reset_env_to_demo_init(env, demo_path)
    nut_z0 = float(get_nut_pos(obs)[2])
    nut_xy0 = get_nut_pos(obs)[:2].copy()
    max_nut_z = nut_z0

    init_slot = encode_frame(model, obs["agentview_image"])
    from system1_jepa.identity_probe import hungarian_assign
    pred_pos_t0 = model.slot_to_pos_aux(
        init_slot.unsqueeze(0))[0].detach().cpu().numpy()
    eef_xy0_norm = norm_xy(obs["robot0_eef_pos"][:2])
    nut_xy0_norm = norm_xy(nut_xy0)
    gt_pos = np.stack([nut_xy0_norm, eef_xy0_norm])
    rows, cols, _ = hungarian_assign(pred_pos_t0, gt_pos)
    nut_slot_idx = None
    for r, c in zip(rows.tolist(), cols.tolist()):
        if int(c) == 0:
            nut_slot_idx = int(r); break
    if nut_slot_idx is None:
        nut_slot_idx = int(np.argmin(
            np.linalg.norm(pred_pos_t0 - nut_xy0_norm, axis=1)))

    fake_goal_norm = norm_xy(nut_xy0 + np.array([0.0, 0.10], dtype=np.float32))
    def score_fn(action_seq_np):
        return predict_score_seq(
            model, init_slot, action_seq_np,
            nut_slot_idx, fake_goal_norm, use_action=True,
        )

    mu_demo = get_demo_prior(demos_actions, demo_id,
                               args.plan_horizon, args.jepa_stride)

    if mode == "demo_no_cem":
        actions_executed = mu_demo
    elif mode == "phase17_locked":
        sig = square_sigma(args.eval_sigma, gripper_sigma=0.0)
        floor = square_sigma(args.sigma_floor, gripper_sigma=0.0)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu_demo, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    elif mode == "naive_cem":
        mu0 = np.zeros((args.plan_horizon, args.action_dim), dtype=np.float32)
        sig = square_sigma(args.naive_sigma, gripper_sigma=0.0)
        floor = square_sigma(args.sigma_floor, gripper_sigma=0.0)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu0, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Execute; track MAX nut z (since Square ends with drop-onto-peg)
    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
            max_nut_z = max(max_nut_z, float(get_nut_pos(obs)[2]))
    nut_z_end = float(get_nut_pos(obs)[2])
    z_gain_max = max_nut_z - nut_z0
    imp = square_improvement(max_nut_z, nut_z0, LIFT_TARGET_Z_GAIN)
    success = z_gain_max >= 0.05
    return {"ep_id": ep_id, "demo_id": demo_id,
              "nut_z0": nut_z0, "nut_z_end": nut_z_end,
              "max_nut_z": max_nut_z,
              "z_gain_max_m": z_gain_max,
              "improvement": imp,
              "success": int(success), "mode": mode}


def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--plan-horizon", type=int, default=30)
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

    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)
    load_planning_deps()
    model = load_action_model(args)

    extract_demos_to_cache(n_demos=max(WORKING_DEMO_IDS) + 1)
    demos_actions = load_demo_actions(
        demo_ids=tuple(range(max(WORKING_DEMO_IDS) + 1)))

    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 200
    env = build_env_square(args.image_size, horizon=big_horizon)

    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i in range(args.n_eval_episodes):
            demo_id = WORKING_DEMO_IDS[i % len(WORKING_DEMO_IDS)]
            t0 = time.time()
            r = run_episode(env, model, mode, args, demo_id, demos_actions, i)
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

    summary = {"args": vars(args), "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
