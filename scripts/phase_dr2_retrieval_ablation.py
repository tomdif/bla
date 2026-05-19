"""Phase DR2 — PickPlaceCan retrieval-quality ablation.

DR1 showed NN demo retrieval scales Recipe E (top1 mean 0.346, σ 0.135).
The variance is high. DR2 ablates the retrieval distance metric to
find a mode that keeps top1's mean while reducing variance toward
top3_avg's level (σ 0.052).

Three retrievers built from the SAME 24-demo working bank, differing
only in key construction:

  geometry         — absolute (can_xy, eef_xy, can_z, eef_z), 6-D
  goal_relative    — relative (eef−can, can_to_table_center,
                     can_z, eef_z), 6-D
  slot_state       — OF-JEPA encoded slot state, n_slots × slot_dim D

Modes (9):
  geometry_top1                   — DR1 baseline
  goal_relative_top1              — relative key
  slot_state_top1                 — slot key
  geometry_top3_avg               — DR1 stable
  geometry_topk_outcome_rerank    — top-5 by geometry, max outcome_score
  demo_no_cem_oracle              — ceiling reference
  demo_no_cem_cycle               — broken baseline (D3 cycle)
  phase17_locked                  — CEM around geometry_top1
  naive_cem                       — floor

Outcome score recorded at bank build time: z_gain when replaying
the demo on its own state-matched reset.
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

WORKING_DEMO_IDS = (5, 8, 10, 13, 16, 23, 25, 28, 30, 41, 45, 46, 47, 58,
                       63, 66, 67, 69, 81, 82, 86, 90, 94, 96)

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


def get_demo_prior_actions(demo_actions: np.ndarray, H: int,
                              stride: int) -> np.ndarray:
    out = []
    for t in range(H):
        idx = min(t * stride, len(demo_actions) - 1)
        out.append(demo_actions[idx])
    return np.stack(out, axis=0)


# ---------- key construction (3 variants) ----------
def key_geometry(env) -> np.ndarray:
    """6-D absolute pose: [can_x, can_y, eef_x, eef_y, can_z, eef_z]."""
    return read_mujoco_pose(env)


def key_goal_relative(env) -> np.ndarray:
    """6-D relative pose: (eef−can, can−table_center, can_z, eef_z).
    table_center = (0, 0) by PickPlaceCan convention."""
    sim = env.sim
    can = sim.data.get_body_xpos("Can_main").copy()
    eef = sim.data.get_body_xpos("gripper0_right_eef").copy()
    return np.array([
        eef[0] - can[0], eef[1] - can[1],
        can[0] - 0.0, can[1] - 0.0,
        can[2], eef[2],
    ], dtype=np.float32)


@torch.no_grad()
def key_slot_state(model, obs) -> np.ndarray:
    """Slot-state key: flatten the encoded slot state."""
    slot = encode_frame(model, obs["agentview_image"])  # [S, slot_dim]
    return slot.detach().cpu().numpy().flatten().astype(np.float32)


# ---------- bank build (3 retrievers, outcome scoring) ----------
def measure_demo_outcome(env, demo_path, demo_actions,
                            n_eval_steps=120) -> float:
    """Replay demo briefly from its state-matched init; record z_gain."""
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])
    n = min(len(demo_actions), n_eval_steps)
    for a in demo_actions[:n]:
        obs, _, _, _ = env.step(a)
    return float(get_can_pos(obs)[2] - can_z0)


def build_dr2_retrievers(env, model, demos_actions, working_ids,
                            demo_dir="/workspace/robomimic_can_replay"):
    """Build all three retrievers + outcome scores in one bank pass."""
    geom_recs = []
    rel_recs = []
    slot_recs = []
    for demo_id in working_ids:
        demo_path = f"{demo_dir}/ep_{demo_id:05d}.npz"
        # Reset to demo's init
        obs = reset_env_to_demo_init(env, demo_path)
        # Extract keys (after set_state, read mujoco directly)
        kg = key_geometry(env)
        kr = key_goal_relative(env)
        ks = key_slot_state(model, obs)
        # Measure outcome by replaying the demo on its own init
        outcome = measure_demo_outcome(env, demo_path,
                                          demos_actions[demo_id])
        d = np.load(demo_path)
        common = dict(
            action_seq=demos_actions[demo_id].astype(np.float32),
            init_state=d["init_state"],
            demo_id=int(demo_id),
            outcome_score=outcome,
        )
        geom_recs.append(DemoState(key=kg, **common))
        rel_recs.append(DemoState(key=kr, **common))
        slot_recs.append(DemoState(key=ks, **common))
        print(json.dumps({"event": "bank_demo",
                           "demo_id": int(demo_id),
                           "outcome_z_gain": outcome}), flush=True)

    rs = {"geometry": DemoRetriever(),
           "goal_relative": DemoRetriever(),
           "slot_state": DemoRetriever()}
    rs["geometry"].build_index(geom_recs)
    rs["goal_relative"].build_index(rel_recs)
    rs["slot_state"].build_index(slot_recs)
    return rs


# ---------- mode dispatch ----------
def pick_actions_for_mode(env, model, retrievers, mode, args, ep_id,
                              reset_demo_id, demos_actions, obs_at_reset):
    """Return the action sequence the mode will execute."""
    H = args.plan_horizon
    stride = args.jepa_stride

    if mode == "demo_no_cem_oracle":
        return get_demo_prior_actions(
            demos_actions[reset_demo_id], H, stride), reset_demo_id

    if mode == "demo_no_cem_cycle":
        cycle_id = CYCLE_DEMOS[ep_id % len(CYCLE_DEMOS)]
        return get_demo_prior_actions(
            demos_actions[cycle_id], H, stride), cycle_id

    if mode == "geometry_top1":
        q = key_geometry(env)
        top1 = retrievers["geometry"].retrieve(q, k=1)[0]
        return get_demo_prior_actions(top1.action_seq, H, stride), top1.demo_id

    if mode == "goal_relative_top1":
        q = key_goal_relative(env)
        top1 = retrievers["goal_relative"].retrieve(q, k=1)[0]
        return get_demo_prior_actions(top1.action_seq, H, stride), top1.demo_id

    if mode == "slot_state_top1":
        q = key_slot_state(model, obs_at_reset)
        top1 = retrievers["slot_state"].retrieve(q, k=1)[0]
        return get_demo_prior_actions(top1.action_seq, H, stride), top1.demo_id

    if mode == "geometry_top3_avg":
        q = key_geometry(env)
        top3 = retrievers["geometry"].retrieve(q, k=3)
        seqs = [get_demo_prior_actions(d.action_seq, H, stride) for d in top3]
        return np.stack(seqs, 0).mean(axis=0).astype(np.float32), -1

    if mode == "geometry_topk_outcome_rerank":
        q = key_geometry(env)
        best = retrievers["geometry"].retrieve_rerank_by_outcome(q, k=5)
        return get_demo_prior_actions(best.action_seq, H, stride), best.demo_id

    raise ValueError(f"Unknown mode: {mode}")


def run_episode(env, model, retrievers, mode, args, ep_id, reset_demo_id,
                  demos_actions):
    demo_path = f"/workspace/robomimic_can_replay/ep_{reset_demo_id:05d}.npz"
    obs = reset_env_to_demo_init(env, demo_path)
    can_z0 = float(get_can_pos(obs)[2])
    can_xy0 = get_can_pos(obs)[:2].copy()

    # phase17_locked and naive_cem need score_fn + CEM
    if mode in ("phase17_locked", "naive_cem"):
        # Build scoring fn (predictor-based, same as DR1)
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
        fake_goal_norm = norm_xy(can_xy0 + np.array([0.0, 0.10],
                                                          dtype=np.float32))
        def score_fn(seq):
            return predict_score_seq(
                model, init_slot, seq, can_slot_idx, fake_goal_norm,
                use_action=True,
            )

        if mode == "phase17_locked":
            q = key_geometry(env)
            top1 = retrievers["geometry"].retrieve(q, k=1)[0]
            mu = get_demo_prior_actions(top1.action_seq, args.plan_horizon,
                                            args.jepa_stride)
            chose_id = top1.demo_id
            sig = per_dim_sigma(args.eval_sigma, 0.0, args.action_dim)
            floor = per_dim_sigma(args.sigma_floor, 0.0, args.action_dim)
            best, _ = cem_with_prior(
                score_fn, mu_init=mu, action_dim=args.action_dim,
                n_steps=args.plan_horizon, n_iters=args.eval_cem_iters,
                n_cand=args.eval_K, elite_frac=args.elite_frac,
                sigma=sig, sigma_floor=floor,
            )
            actions_executed = best
        else:
            mu0 = np.zeros((args.plan_horizon, args.action_dim),
                              dtype=np.float32)
            chose_id = -1
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
        actions_executed, chose_id = pick_actions_for_mode(
            env, model, retrievers, mode, args, ep_id, reset_demo_id,
            demos_actions, obs)

    for a in actions_executed:
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
    can_z_end = float(get_can_pos(obs)[2])
    z_gain = can_z_end - can_z0
    imp = pickplace_improvement(can_z0, can_z_end, LIFT_TARGET_Z_GAIN)
    success = z_gain >= 0.10
    return {"ep_id": ep_id, "mode": mode,
              "reset_demo_id": int(reset_demo_id),
              "chose_demo_id": int(chose_id),
              "matches_reset_target": bool(chose_id == int(reset_demo_id)),
              "z_gain_m": z_gain, "improvement": imp, "success": int(success)}


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
                    default="geometry_top1,goal_relative_top1,slot_state_top1,"
                            "geometry_top3_avg,geometry_topk_outcome_rerank,"
                            "demo_no_cem_oracle,demo_no_cem_cycle,"
                            "phase17_locked,naive_cem")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    load_planning_deps()
    model = load_action_model(args)

    extract_demos_to_cache(n_demos=100)
    demos_actions = load_demo_actions(demo_ids=tuple(range(100)))
    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 400
    env = build_env_pickplace(args.image_size, horizon=big_horizon)

    retrievers = build_dr2_retrievers(env, model, demos_actions,
                                          WORKING_DEMO_IDS)
    print(json.dumps({"event": "retrievers_built",
                       "n_demos": len(retrievers["geometry"]),
                       "outcome_scores": [
                           float(d.outcome_score)
                           for d in retrievers["geometry"]._bank
                       ]}), flush=True)

    rng = np.random.RandomState(args.seed)
    reset_targets = list(rng.choice(WORKING_DEMO_IDS, size=args.n_eval_episodes,
                                       replace=True))
    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i in range(args.n_eval_episodes):
            t0 = time.time()
            r = run_episode(env, model, retrievers, mode, args,
                              i, reset_targets[i], demos_actions)
            r["elapsed_s"] = time.time() - t0
            per_ep.append(r)
            print(json.dumps({"event": "ep", **r}), flush=True)
        imp_mean = float(np.mean([e["improvement"] for e in per_ep]))
        succ_mean = float(np.mean([e["success"] for e in per_ep]))
        match_mean = float(np.mean([e["matches_reset_target"]
                                         for e in per_ep]))
        all_results[mode] = {"per_ep": per_ep, "improvement_mean": imp_mean,
                                "success_rate": succ_mean,
                                "retrieval_match_rate": match_mean}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "retrieval_match_rate": match_mean}), flush=True)

    summary = {"args": vars(args),
                  "reset_targets": [int(x) for x in reset_targets],
                  "working_demo_ids": list(WORKING_DEMO_IDS),
                  "outcome_scores": [
                      float(d.outcome_score)
                      for d in retrievers["geometry"]._bank
                  ],
                  "results": all_results}
    with open(Path(out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
