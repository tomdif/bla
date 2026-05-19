"""Phase DR1 — PickPlaceCan demo retrieval eval.

Per-episode: state-matched reset to a RANDOM demo from the 24-demo
working bank. Then test how each mode picks its action sequence.

Modes:
  demo_no_cem_cycle    — D3 baseline; cycle through first 5 working demos
  demo_no_cem_oracle   — ceiling; use the demo we reset to
  demo_retrieval_top1  — NN over 24-working bank by current (can+eef) pose
  demo_retrieval_top3_avg — avg of top-3 action sequences elementwise
  phase17_locked       — CEM σ=0.12 around demo_retrieval_top1
  naive_cem            — pure CEM (floor)
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

from bla.recipes import DemoState, DemoRetriever
from system1_jepa.of_jepa import OFJEPAConfig
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA
from scripts.phase_d3_pickplace import (
    build_env_pickplace, extract_demos_to_cache, load_demo_actions,
    reset_env_to_demo_init, get_can_pos, get_eef_pos,
    pickplace_improvement, LIFT_TARGET_Z_GAIN,
)


# 24 demos that lift the can on their own state-matched init (screened from 100)
WORKING_DEMO_IDS = (5, 8, 10, 13, 16, 23, 25, 28, 30, 41, 45, 46, 47, 58,
                       63, 66, 67, 69, 81, 82, 86, 90, 94, 96)
CYCLE_DEMOS = WORKING_DEMO_IDS[:5]   # D3 baseline cycling set


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


def read_mujoco_pose(env) -> np.ndarray:
    """Read can+eef poses DIRECTLY from mujoco sim data (not env obs).

    Robosuite's env._get_observations() returns positions cached from
    the most recent env.reset() randomization, NOT from the current
    mujoco qpos after a set_state_from_flattened call. So when we
    state-match-reset to a demo, the obs is stale (= env.reset's
    random can position) while the actual mujoco state matches the
    demo. Reading from sim.data bypasses the stale obs.

    Returns: 6-D [can_x, can_y, eef_x, eef_y, can_z, eef_z].
    """
    sim = env.sim
    can_xpos = sim.data.get_body_xpos("Can_main").copy()
    eef_xpos = sim.data.get_body_xpos("gripper0_right_eef").copy()
    return np.concatenate([
        can_xpos[:2], eef_xpos[:2], [can_xpos[2], eef_xpos[2]],
    ]).astype(np.float32)


def build_retriever(env, demos_actions, working_ids,
                      demo_dir="/workspace/robomimic_can_replay") -> DemoRetriever:
    """Build retrieval bank using TRUE mujoco poses (sim.data), not the
    stale env.obs after state-matched reset."""
    records = []
    for demo_id in working_ids:
        demo_path = f"{demo_dir}/ep_{demo_id:05d}.npz"
        _ = reset_env_to_demo_init(env, demo_path)
        key = read_mujoco_pose(env)
        d = np.load(demo_path)
        records.append(DemoState(
            key=key,
            action_seq=demos_actions[demo_id].astype(np.float32),
            init_state=d["init_state"],
            demo_id=int(demo_id),
        ))
    retriever = DemoRetriever()
    retriever.build_index(records)
    return retriever


def query_key_from_env(env) -> np.ndarray:
    return read_mujoco_pose(env)


def norm_xy(p_xy):
    return np.clip((np.asarray(p_xy) + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def per_dim_sigma(scalar: float, gripper_sigma: float = 0.0,
                    action_dim: int = 7) -> np.ndarray:
    return np.array([scalar] * (action_dim - 1) + [gripper_sigma],
                       dtype=np.float32)


def get_demo_prior_actions(demo_actions: np.ndarray, H: int,
                              stride: int) -> np.ndarray:
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo_actions) - 1)
        out.append(demo_actions[idx])
    return np.stack(out, axis=0)


def run_episode(env, model, retriever, mode, args, reset_demo_id,
                  demos_actions, ep_id):
    demo_path = f"/workspace/robomimic_can_replay/ep_{reset_demo_id:05d}.npz"
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])
    can_xy0 = get_can_pos(obs)[:2].copy()

    query_key = query_key_from_env(env)

    # ----- pick the action sequence by mode -----
    cycle_demo_id = CYCLE_DEMOS[ep_id % len(CYCLE_DEMOS)]
    retrieved_top1_id = None
    retrieved_top3_ids = None

    if mode == "demo_no_cem_oracle":
        actions_executed = get_demo_prior_actions(
            demos_actions[reset_demo_id], args.plan_horizon, args.jepa_stride)
        chose_id = reset_demo_id
    elif mode == "demo_no_cem_cycle":
        actions_executed = get_demo_prior_actions(
            demos_actions[cycle_demo_id], args.plan_horizon, args.jepa_stride)
        chose_id = cycle_demo_id
    elif mode == "demo_retrieval_top1":
        top1 = retriever.retrieve(query_key, k=1)[0]
        actions_executed = get_demo_prior_actions(
            top1.action_seq, args.plan_horizon, args.jepa_stride)
        retrieved_top1_id = top1.demo_id
        chose_id = retrieved_top1_id
    elif mode == "demo_retrieval_top3_avg":
        top3 = retriever.retrieve(query_key, k=3)
        retrieved_top3_ids = [d.demo_id for d in top3]
        # Average action sequences at stride-subsampled positions
        seqs = [get_demo_prior_actions(d.action_seq, args.plan_horizon,
                                          args.jepa_stride) for d in top3]
        actions_executed = np.stack(seqs, axis=0).mean(axis=0).astype(np.float32)
        chose_id = -1
    elif mode in ("phase17_locked", "naive_cem"):
        # For phase17_locked we need the retrieval top-1 as the prior
        top1 = retriever.retrieve(query_key, k=1)[0]
        mu_top1 = get_demo_prior_actions(
            top1.action_seq, args.plan_horizon, args.jepa_stride)
        retrieved_top1_id = top1.demo_id

        # Build scoring function (predictor-based; same as D3 protocol)
        init_slot = encode_frame(model, obs["agentview_image"])
        from system1_jepa.identity_probe import hungarian_assign
        pred_pos_t0 = model.slot_to_pos_aux(
            init_slot.unsqueeze(0))[0].detach().cpu().numpy()
        eef_xy0_norm = norm_xy(obs["robot0_eef_pos"][:2])
        can_xy0_norm = norm_xy(can_xy0)
        gt = np.stack([can_xy0_norm, eef_xy0_norm])
        rows, cols, _ = hungarian_assign(pred_pos_t0, gt)
        can_slot_idx = None
        for r, c in zip(rows.tolist(), cols.tolist()):
            if int(c) == 0:
                can_slot_idx = int(r); break
        if can_slot_idx is None:
            can_slot_idx = int(np.argmin(
                np.linalg.norm(pred_pos_t0 - can_xy0_norm, axis=1)))
        fake_goal_norm = norm_xy(can_xy0 + np.array([0.0, 0.10], dtype=np.float32))
        def score_fn(seq):
            return predict_score_seq(
                model, init_slot, seq, can_slot_idx, fake_goal_norm,
                use_action=True,
            )
        if mode == "phase17_locked":
            sig = per_dim_sigma(args.eval_sigma, 0.0, args.action_dim)
            floor = per_dim_sigma(args.sigma_floor, 0.0, args.action_dim)
            best, _ = cem_with_prior(
                score_fn, mu_init=mu_top1, action_dim=args.action_dim,
                n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
                n_cand=args.eval_K, elite_frac=args.elite_frac,
                sigma=sig, sigma_floor=floor,
            )
            actions_executed = best
        else:  # naive_cem
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
        chose_id = retrieved_top1_id if mode == "phase17_locked" else -1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ----- execute -----
    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
    can_z_end = float(get_can_pos(obs)[2])
    z_gain = can_z_end - can_z0
    imp = pickplace_improvement(can_z0, can_z_end, LIFT_TARGET_Z_GAIN)
    success = z_gain >= 0.10

    return {
        "ep_id": ep_id, "mode": mode,
        "reset_demo_id": int(reset_demo_id),
        "cycle_demo_id": int(cycle_demo_id) if mode == "demo_no_cem_cycle" else None,
        "retrieved_top1_id": int(retrieved_top1_id) if retrieved_top1_id is not None else None,
        "retrieved_top3_ids": (
            [int(x) for x in retrieved_top3_ids] if retrieved_top3_ids else None
        ),
        "chose_demo_id": int(chose_id),
        "matches_reset_target": bool(chose_id == int(reset_demo_id)),
        "can_z0": can_z0, "can_z_end": can_z_end,
        "z_gain_m": z_gain, "improvement": imp, "success": int(success),
    }


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
                    default="demo_no_cem_oracle,demo_no_cem_cycle,demo_retrieval_top1,demo_retrieval_top3_avg,phase17_locked,naive_cem")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    load_planning_deps()
    model = load_action_model(args)

    # Cache demos (need all 100 to support working bank + cycle subset)
    extract_demos_to_cache(n_demos=100)
    demos_actions = load_demo_actions(demo_ids=tuple(range(100)))

    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 200
    env = build_env_pickplace(args.image_size, horizon=big_horizon)

    retriever = build_retriever(env, demos_actions, WORKING_DEMO_IDS)
    print(json.dumps({"event": "retriever_built",
                       "n_demos_in_bank": len(retriever),
                       "key_dim": retriever.index_keys().shape[1]}), flush=True)

    # Pick reset targets — random sample from working bank, fixed by seed
    rng = np.random.RandomState(args.seed)
    reset_targets = list(rng.choice(WORKING_DEMO_IDS, size=args.n_eval_episodes,
                                       replace=True))

    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i in range(args.n_eval_episodes):
            t0 = time.time()
            r = run_episode(env, model, retriever, mode, args,
                              reset_targets[i], demos_actions, i)
            r["elapsed_s"] = time.time() - t0
            per_ep.append(r)
            print(json.dumps({"event": "ep", **r}), flush=True)
        imp_mean = float(np.mean([e["improvement"] for e in per_ep]))
        succ_mean = float(np.mean([e["success"] for e in per_ep]))
        match_mean = float(np.mean([e["matches_reset_target"]
                                         for e in per_ep]))
        all_results[mode] = {"per_ep": per_ep,
                                "improvement_mean": imp_mean,
                                "success_rate": succ_mean,
                                "retrieval_match_rate": match_mean}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "retrieval_match_rate": match_mean}), flush=True)

    summary = {"args": vars(args),
                  "reset_targets": [int(x) for x in reset_targets],
                  "working_demo_ids": list(WORKING_DEMO_IDS),
                  "cycle_demos": list(CYCLE_DEMOS),
                  "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
