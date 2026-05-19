"""Phase D3 — PickPlaceCan pilot eval.

Tests the core doctrine claim on PickPlaceCan with 3 modes:
  - demo_no_cem (Recipe E)     — replay demo actions directly
  - phase17_locked              — CEM σ=0.12 around demo, Phase 17 predictor scoring
  - naive_cem                   — CEM σ=0.5 from zero mean (floor)

Protocol per episode:
  1. Sample a "working" demo (5/20 demos that lift can on state-matched reset).
  2. Reset env to that demo's recorded init state.
  3. Run the mode (replay / CEM-around-demo / pure CEM).
  4. Measure: cube z gain (improvement), success = z_gain >= 0.10 m.

This is a pilot at n=5 eps × 3 modes × 1 seed (~15 min wall).
If the doctrine holds, demo_no_cem dominates and CEM-around-demo
under-performs (matching Phase 18κ R3 Lift finding).
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
from scripts.phase_d3_pickplace import (
    build_env_pickplace, extract_demos_to_cache, load_demo_actions,
    reset_env_to_demo_init, get_can_pos, LIFT_TARGET_Z_GAIN,
    pickplace_improvement,
)


# Demos that lift the can on their own state-matched reset (screened from 0-19)
WORKING_DEMO_IDS = (5, 8, 10, 13, 16)


# Lazy imports
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


def pickplace_sigma(scalar: float, gripper_sigma: float = 0.0) -> np.ndarray:
    """Per-dim sigma: motion dims get `scalar`, gripper dim (6) gets `gripper_sigma`."""
    return np.array([scalar] * 6 + [gripper_sigma], dtype=np.float32)


def get_demo_prior(demos_actions: list[np.ndarray], demo_id: int,
                     H: int, stride: int) -> np.ndarray:
    """Return H stride-subsampled actions from a chosen demo."""
    demo = demos_actions[demo_id]
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo) - 1)
        out.append(demo[idx])
    return np.stack(out, axis=0)


def run_episode(env, model, mode, args, demo_id, demos_actions, ep_id):
    """One eval episode: state-matched reset to demo's init, run mode."""
    demo_path = f"/workspace/robomimic_can_replay/ep_{demo_id:05d}.npz"
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])
    can_xy0 = get_can_pos(obs)[:2].copy()

    # Build init slot for predictor scoring (used by phase17_locked)
    init_slot = encode_frame(model, obs["agentview_image"])
    # For PickPlaceCan, we use Hungarian against (can_xy0, eef_xy0) and pick the
    # slot closest to can_xy as our "target slot" — slot_to_pos_aux scoring uses
    # this to compute predictor score (distance to a goal). For phase17_locked,
    # the "goal" is the can's INITIAL xy (we want to lift, not move horizontally
    # — but the predictor's scoring is xy-distance-based; near-zero distance to
    # initial xy means cube stays grounded; we want to *escape* this distance).
    # Simpler approach: use the demo's actions as the prior, score by distance
    # to a +z "lifted" goal — but slot_to_pos_aux is 2D only.
    # Pragmatic: for the pilot, score the predictor's prediction against an
    # arbitrary FAR goal so CEM tries to move things. This is a deliberately
    # weak scoring signal — that's exactly what tests "does CEM hurt the demo?"
    from system1_jepa.identity_probe import hungarian_assign
    pred_pos_t0 = model.slot_to_pos_aux(
        init_slot.unsqueeze(0))[0].detach().cpu().numpy()
    eef_xy0_norm = norm_xy(obs["robot0_eef_pos"][:2])
    can_xy0_norm = norm_xy(can_xy0)
    gt_pos = np.stack([can_xy0_norm, eef_xy0_norm])
    rows, cols, _ = hungarian_assign(pred_pos_t0, gt_pos)
    can_slot_idx = None
    for r, c in zip(rows.tolist(), cols.tolist()):
        if int(c) == 0:
            can_slot_idx = int(r); break
    if can_slot_idx is None:
        can_slot_idx = int(np.argmin(
            np.linalg.norm(pred_pos_t0 - can_xy0_norm, axis=1)))

    # Score function for predictor-only CEM modes (negative dist to "moved" target)
    # Pick a +y target ~10cm away as an arbitrary moving goal
    fake_goal_norm = norm_xy(can_xy0 + np.array([0.0, 0.10], dtype=np.float32))
    def score_fn(action_seq_np):
        return predict_score_seq(
            model, init_slot, action_seq_np,
            can_slot_idx, fake_goal_norm, use_action=True,
        )

    # Get demo prior actions
    mu_demo = get_demo_prior(demos_actions, demo_id,
                               args.plan_horizon, args.jepa_stride)

    if mode == "demo_no_cem":
        actions_executed = mu_demo
    elif mode == "phase17_locked":
        # CEM around demo prior with per-dim sigma (gripper masked)
        sig = pickplace_sigma(args.eval_sigma, gripper_sigma=0.0)
        floor = pickplace_sigma(args.sigma_floor, gripper_sigma=0.0)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu_demo, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    elif mode == "naive_cem":
        # No prior (mu=0), high sigma
        mu0 = np.zeros((args.plan_horizon, args.action_dim), dtype=np.float32)
        sig = pickplace_sigma(args.naive_sigma, gripper_sigma=0.0)
        floor = pickplace_sigma(args.sigma_floor, gripper_sigma=0.0)
        best, _ = cem_with_prior(
            score_fn, mu_init=mu0, action_dim=args.action_dim,
            n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
            n_cand=args.eval_K, elite_frac=args.elite_frac,
            sigma=sig, sigma_floor=floor,
        )
        actions_executed = best
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Execute the chosen action sequence in env (no replanning for the pilot)
    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
    can_z_end = float(get_can_pos(obs)[2])
    z_gain = can_z_end - can_z0
    imp = pickplace_improvement(can_z0, can_z_end, LIFT_TARGET_Z_GAIN)
    success = z_gain >= 0.10
    return {"ep_id": ep_id, "demo_id": demo_id,
              "can_z0": can_z0, "can_z_end": can_z_end,
              "z_gain_m": z_gain, "improvement": imp,
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

    # Ensure demos cached + load
    extract_demos_to_cache(n_demos=max(WORKING_DEMO_IDS) + 1)
    demos_actions = load_demo_actions(demo_ids=tuple(range(max(WORKING_DEMO_IDS) + 1)))

    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 200
    env = build_env_pickplace(args.image_size, horizon=big_horizon)

    modes = args.modes.split(",")
    rng = np.random.RandomState(args.seed)
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
        z_mean = float(np.mean([e["z_gain_m"] for e in per_ep]))
        all_results[mode] = {"per_ep": per_ep, "improvement_mean": imp_mean,
                                "success_rate": succ_mean, "z_gain_mean_m": z_mean}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "z_gain_mean_m": z_mean}), flush=True)

    summary = {"args": vars(args), "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
