"""Phase DR3 — bank-coverage diagnostic + constrained top-k rerank.

DR2 closed with geometry_top1 as the best retrieval mode (σ=0.088,
just above the 0.08 target). DR3 tests two hypotheses:

  H1 (coverage): residual variance is bank-coverage limited;
                 corr(NN_distance, -improvement) > 0
  H2 (rerank):   constrained top-k rerank (filter ≤ 1.25× NN dist,
                 rerank by outcome_score) improves over top-1 on
                 the subset where NN distance > 0

Protocol: reset env to a RANDOM demo from the FULL 100-demo cache
(24 working + 76 non-working). This creates NN distance > 0 for
the 76% non-working cases, making constrained rerank actually
trigger.
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
from scripts.phase_dr1_pickplace import read_mujoco_pose, CYCLE_DEMOS

# Bank: 24 working demos screened from DR1
WORKING_DEMO_IDS = (5, 8, 10, 13, 16, 23, 25, 28, 30, 41, 45, 46, 47, 58,
                       63, 66, 67, 69, 81, 82, 86, 90, 94, 96)
# Reset target pool: ALL 100 demos (working + non-working)
RESET_TARGET_POOL = tuple(range(100))


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


def measure_demo_outcome(env, demo_path, demo_actions,
                            n_eval_steps=120) -> float:
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])
    n = min(len(demo_actions), n_eval_steps)
    for a in demo_actions[:n]:
        obs, _, _, _ = env.step(a)
    return float(get_can_pos(obs)[2] - can_z0)


def build_retriever(env, demos_actions, working_ids,
                       demo_dir="/workspace/robomimic_can_replay") -> DemoRetriever:
    records = []
    for demo_id in working_ids:
        demo_path = f"{demo_dir}/ep_{demo_id:05d}.npz"
        _ = reset_env_to_demo_init(env, demo_path)
        key = read_mujoco_pose(env)
        outcome = measure_demo_outcome(env, demo_path,
                                          demos_actions[demo_id])
        d = np.load(demo_path)
        records.append(DemoState(
            key=key,
            action_seq=demos_actions[demo_id].astype(np.float32),
            init_state=d["init_state"],
            demo_id=int(demo_id),
            outcome_score=outcome,
        ))
        print(json.dumps({"event": "bank_demo", "demo_id": int(demo_id),
                           "outcome_z_gain": outcome,
                           "key": key.tolist()}), flush=True)
    r = DemoRetriever()
    r.build_index(records)
    return r


def get_demo_prior_actions(demo_actions, H, stride):
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo_actions) - 1)
        out.append(demo_actions[idx])
    return np.stack(out, axis=0)


def run_episode(env, retriever, mode, args, ep_id, reset_demo_id,
                  demos_actions):
    demo_path = f"/workspace/robomimic_can_replay/ep_{reset_demo_id:05d}.npz"
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])

    query_key = read_mujoco_pose(env)
    # NN distance (always for diagnostic)
    top1 = retriever.retrieve(query_key, k=1)[0]
    nn_dist = float(np.linalg.norm(top1.key - query_key))

    if mode == "demo_no_cem_oracle":
        actions_executed = get_demo_prior_actions(
            demos_actions[reset_demo_id], args.plan_horizon, args.jepa_stride)
        chose_id = reset_demo_id
    elif mode == "demo_no_cem_cycle":
        cycle_id = CYCLE_DEMOS[ep_id % len(CYCLE_DEMOS)]
        actions_executed = get_demo_prior_actions(
            demos_actions[cycle_id], args.plan_horizon, args.jepa_stride)
        chose_id = cycle_id
    elif mode == "geometry_top1":
        actions_executed = get_demo_prior_actions(
            top1.action_seq, args.plan_horizon, args.jepa_stride)
        chose_id = top1.demo_id
    elif mode == "geometry_constrained_rerank":
        chosen = retriever.retrieve_constrained_rerank(
            query_key, k=5, filter_ratio=args.filter_ratio)
        actions_executed = get_demo_prior_actions(
            chosen.action_seq, args.plan_horizon, args.jepa_stride)
        chose_id = chosen.demo_id
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
    can_z_end = float(get_can_pos(obs)[2])
    z_gain = can_z_end - can_z0
    imp = pickplace_improvement(can_z0, can_z_end, LIFT_TARGET_Z_GAIN)
    success = z_gain >= 0.10
    return {"ep_id": ep_id, "mode": mode,
              "reset_demo_id": int(reset_demo_id),
              "reset_target_is_working": bool(reset_demo_id in WORKING_DEMO_IDS),
              "chose_demo_id": int(chose_id),
              "top1_demo_id": int(top1.demo_id),
              "nn_dist": nn_dist,
              "z_gain_m": z_gain, "improvement": imp,
              "success": int(success)}


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
    p.add_argument("--filter-ratio", type=float, default=1.25)
    p.add_argument("--modes", type=str,
                    default="geometry_top1,geometry_constrained_rerank,"
                            "demo_no_cem_oracle,demo_no_cem_cycle")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    model = load_action_model(args)
    extract_demos_to_cache(n_demos=100)
    demos_actions = load_demo_actions(demo_ids=tuple(range(100)))
    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 400
    env = build_env_pickplace(args.image_size, horizon=big_horizon)

    retriever = build_retriever(env, demos_actions, WORKING_DEMO_IDS)
    print(json.dumps({"event": "retriever_built",
                       "n_demos": len(retriever)}), flush=True)

    rng = np.random.RandomState(args.seed)
    reset_targets = list(rng.choice(RESET_TARGET_POOL,
                                       size=args.n_eval_episodes,
                                       replace=True))
    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i in range(args.n_eval_episodes):
            t0 = time.time()
            r = run_episode(env, retriever, mode, args, i, int(reset_targets[i]),
                              demos_actions)
            r["elapsed_s"] = time.time() - t0
            per_ep.append(r)
            print(json.dumps({"event": "ep", **r}), flush=True)
        imp_mean = float(np.mean([e["improvement"] for e in per_ep]))
        succ_mean = float(np.mean([e["success"] for e in per_ep]))
        nn_mean = float(np.mean([e["nn_dist"] for e in per_ep]))
        # H1 diagnostic: correlation(nn_dist, -improvement) per mode
        imps = np.array([e["improvement"] for e in per_ep])
        dists = np.array([e["nn_dist"] for e in per_ep])
        if imps.std() > 0 and dists.std() > 0:
            from scipy.stats import spearmanr
            sp = float(spearmanr(dists, -imps).statistic)
        else:
            sp = float("nan")
        all_results[mode] = {"per_ep": per_ep,
                                "improvement_mean": imp_mean,
                                "success_rate": succ_mean,
                                "nn_dist_mean": nn_mean,
                                "spearman_nndist_vs_neg_imp": sp}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "nn_dist_mean": nn_mean,
                           "spearman_nndist_vs_neg_imp": sp}), flush=True)

    summary = {"args": vars(args),
                  "reset_targets": [int(x) for x in reset_targets],
                  "working_demo_ids": list(WORKING_DEMO_IDS),
                  "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
