"""Phase 18η — Goal-progress / multi-step value head.

Per Phase 18γ's three-variable decomposition, the BLA System-1
stack's binding constraint is episode-level goal compounding, not
local rank quality or candidate quality. This phase tests whether
adding a value head on top of OF-JEPA's existing one-step predictor
improves episode-level planning over the Phase 18β locked recipe.

Pipeline:
  1. COLLECT: run scripted_prior_light_cem for N_rollout episodes,
     log per-replan (geometric_features, goal_xy, refined_plan,
     full_episode_improvement) tuples.
  2. TRAIN: regress value head on (features, goal, actions) → label.
  3. EVAL: 6 modes × 30 ep × seed 0. Modes use value head as
     scoring function or combined with the OF-JEPA predictor.
  4. AUDIT (optional G3): mini 18γ-style audit on D2 distribution
     using value head as scorer; expect top_vs_bot_gap > +0.05.

Gates per PHASE_18H_VALUE_HEAD_PRECOMMIT.md.
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
from system1_jepa.value_head import (
    GoalProgressValueHead,
    build_value_head_from_config,
    value_head_config,
    train_value_head_supervised,
    normalize_score,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


# Lazy-loaded robosuite-dependent helpers
build_env = None
encode_frame = None
find_cubeA_slot = None
predict_score_seq = None
closed_loop_gt_step = None
state_features = None
rollout_scripted_prior = None
cem_with_prior = None


def load_planning_dependencies():
    global build_env, encode_frame, find_cubeA_slot, predict_score_seq
    global closed_loop_gt_step, state_features, rollout_scripted_prior
    global cem_with_prior
    from scripts.phase15_planning import (
        build_env as _b, encode_frame as _e, find_cubeA_slot as _f,
        predict_score_seq as _p, closed_loop_gt_step as _c,
    )
    from scripts.phase16_policy_prior_mpc import (
        state_features as _sf, rollout_scripted_prior as _rsp,
        cem_with_prior as _cwp,
    )
    build_env = _b; encode_frame = _e; find_cubeA_slot = _f
    predict_score_seq = _p; closed_loop_gt_step = _c
    state_features = _sf; rollout_scripted_prior = _rsp
    cem_with_prior = _cwp


def load_action_model(args):
    cfg = OFJEPAConfig(
        n_files=args.n_slots, id_dim=args.slot_dim // 2,
        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim,
    )
    model = ActionConditionedOFJEPA(
        image_size=args.image_size, cfg=cfg,
        action_dim=args.action_dim, use_action=True,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_action, map_location=args.device))
    model.eval()
    return model


def norm_xy(p_xy):
    return np.clip((p_xy + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def sample_goal(obs, ep_id, args):
    cube_init = obs["cubeA_pos"][:2].copy()
    rng = np.random.RandomState(ep_id + 1000)
    theta = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(args.goal_dist_min, args.goal_dist_max)
    goal_xy = cube_init + r * np.array([np.cos(theta), np.sin(theta)])
    start_dist = float(np.linalg.norm(cube_init - goal_xy))
    return goal_xy, start_dist


def step_and_track(env, obs, action, args, contact_ref):
    for _ in range(args.jepa_stride):
        obs, _, _, _ = env.step(action)
        if float(np.linalg.norm(obs["cubeA_pos"][:2]
                                  - obs["robot0_eef_pos"][:2])) < 0.04:
            contact_ref[0] = True
    return obs


def build_score_fn(env, model, obs, goal_xy_world):
    init_slot = encode_frame(model, obs["agentview_image"])
    cubeA_idx = find_cubeA_slot(
        model, init_slot,
        norm_xy(obs["cubeA_pos"][:2]),
        norm_xy(obs["cubeB_pos"][:2]),
        norm_xy(obs["robot0_eef_pos"][:2]),
    )
    goal_xy_norm = norm_xy(goal_xy_world)
    return lambda seq: predict_score_seq(
        model, init_slot, seq, cubeA_idx, goal_xy_norm, use_action=True,
    )


# ---------- Collection: locked recipe rollouts ----------
def collect_value_head_data(env, model, args):
    """Run scripted_prior_light_cem episodes; log per-replan-boundary tuples."""
    features = []
    goals = []
    plans = []
    labels = []
    t0 = time.time()
    for ep in range(args.rollout_episodes):
        obs = env.reset()
        cube_init = obs["cubeA_pos"][:2].copy()
        goal_xy_world, start_dist = sample_goal(obs, ep, args)
        actions_executed = 0
        per_replan_records = []
        while actions_executed < args.total_actions:
            feat = state_features(obs, goal_xy_world)
            score_fn = build_score_fn(env, model, obs, goal_xy_world)
            mu = rollout_scripted_prior(env, obs, goal_xy_world,
                                          args.plan_horizon, args.jepa_stride)
            plan, _ = cem_with_prior(
                score_fn, mu, env.action_dim,
                args.plan_horizon, args.train_cem_iters, args.train_K,
                args.elite_frac, sigma=args.train_sigma,
                sigma_floor=args.sigma_floor,
            )
            if plan is None:
                plan = mu
            per_replan_records.append((feat.astype(np.float32),
                                          np.array(goal_xy_world,
                                                     dtype=np.float32),
                                          plan.astype(np.float32)))
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            contact = [False]
            for action in plan[:n_exec]:
                obs = step_and_track(env, obs, action, args, contact)
            actions_executed += n_exec
        # Episode-level realized improvement
        end_xy = obs["cubeA_pos"][:2]
        end_dist = float(np.linalg.norm(end_xy - goal_xy_world))
        episode_imp = max(0.0, (start_dist - end_dist) / max(start_dist, 1e-9))
        for feat, goal_xy, plan in per_replan_records:
            features.append(feat)
            goals.append(goal_xy)
            plans.append(plan)
            labels.append(episode_imp)

        if (ep + 1) % max(args.rollout_log_every, 1) == 0:
            print(json.dumps({"event": "rollout_progress",
                               "episodes": ep + 1,
                               "samples": len(features),
                               "running_mean_imp": float(np.mean(labels)),
                               "elapsed_s": round(time.time() - t0, 1)}),
                   flush=True)

    return (np.stack(features), np.stack(goals),
             np.stack(plans), np.array(labels, dtype=np.float32))


def load_or_collect_rollouts(env, model, args, out: Path):
    cache = Path(args.rollout_cache) if args.rollout_cache \
            else out / "rollout_cache.npz"
    if cache.exists() and not args.rebuild_rollouts:
        print(json.dumps({"event": "loading_rollout_cache",
                           "path": str(cache)}), flush=True)
        d = np.load(cache)
        return d["features"], d["goals"], d["plans"], d["labels"], cache
    print(json.dumps({"event": "collecting_rollouts",
                       "n_episodes": args.rollout_episodes}), flush=True)
    f, g, p, l = collect_value_head_data(env, model, args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=f, goals=g, plans=p, labels=l)
    print(json.dumps({"event": "saved_rollout_cache", "path": str(cache),
                       "n_samples": len(f),
                       "mean_episode_imp": float(np.mean(l))}), flush=True)
    return f, g, p, l, cache


def train_or_load_value_head(features, goals, plans, labels, args, out: Path):
    ckpt = Path(args.value_head_ckpt) if args.value_head_ckpt \
            else out / "value_head.pt"
    state_dim = int(features.shape[1])
    horizon = int(plans.shape[1])
    action_dim = int(plans.shape[2])
    if ckpt.exists() and not args.retrain:
        print(json.dumps({"event": "loading_value_head",
                           "path": str(ckpt)}), flush=True)
        ck = torch.load(ckpt, map_location=args.device)
        head = build_value_head_from_config(ck["config"]).to(args.device)
        head.load_state_dict(ck["state_dict"])
        head.eval()
        return head, ckpt, {"loaded": True}

    head = GoalProgressValueHead(
        state_dim=state_dim, action_dim=action_dim, plan_horizon=horizon,
        hidden=args.hidden, n_hidden=args.n_hidden, dropout=args.dropout,
    ).to(args.device)
    stats = train_value_head_supervised(
        head,
        torch.from_numpy(features), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.train_steps, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.train_log_every,
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(),
                "config": value_head_config(head),
                "stats": stats.__dict__}, ckpt)
    print(json.dumps({"event": "saved_value_head", "path": str(ckpt),
                       **stats.__dict__}), flush=True)
    head.eval()
    return head, ckpt, stats.__dict__


# ---------- Scoring functions ----------
@torch.no_grad()
def value_head_score_seq(head, state_feat, goal_xy, action_seq, device):
    state = torch.from_numpy(state_feat).to(device).unsqueeze(0).float()
    goal = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    actions = torch.from_numpy(action_seq).to(device).unsqueeze(0).float()
    return float(head(state, goal, actions).cpu().item())


def build_combined_score_fn(env, model, head, obs, goal_xy_world, mode: str,
                              lam: float, args):
    """Returns score_fn(seq) using OF-JEPA predictor + value head per `mode`."""
    state_feat = state_features(obs, goal_xy_world)
    goal_xy = np.array(goal_xy_world, dtype=np.float32)
    pred_fn = build_score_fn(env, model, obs, goal_xy_world)

    def fn(seq):
        p = pred_fn(seq)
        if mode == "predictor_only":
            return p
        v = value_head_score_seq(head, state_feat, goal_xy,
                                    seq.astype(np.float32), args.device)
        if mode == "value_only":
            return v
        # For combined modes, normalization happens AT THE CEM ELITE-PICK
        # step, not per-candidate. But CEM scores candidates one-by-one,
        # so we can't z-normalize here. Use raw combination with a fixed
        # lambda; if scales differ wildly, switch to bandit-style picking.
        if mode == "sum":
            return lam * p + (1.0 - lam) * v
        if mode == "max":
            return max(p, v)
        raise ValueError(f"unknown mode {mode}")
    return fn


# ---------- Evaluation runner ----------
def run_episode(env, model, head, mode: str, args, ep_id: int):
    obs = env.reset()
    cube_init = obs["cubeA_pos"][:3].copy()
    goal_xy_world, start_dist = sample_goal(obs, ep_id, args)
    actions_executed = 0
    contact = [False]

    if mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            action = closed_loop_gt_step(env, obs, goal_xy_world)
            obs = step_and_track(env, obs, action, args, contact)
    elif mode == "naive_cem":
        while actions_executed < args.total_actions:
            score_fn = build_score_fn(env, model, obs, goal_xy_world)
            mu = np.zeros((args.plan_horizon, env.action_dim), dtype=np.float32)
            plan, _ = cem_with_prior(score_fn, mu, env.action_dim,
                                       args.plan_horizon, args.eval_cem_iters,
                                       args.eval_K, args.elite_frac,
                                       sigma=args.naive_sigma,
                                       sigma_floor=args.sigma_floor)
            if plan is None:
                plan = mu
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                obs = step_and_track(env, obs, a, args, contact)
            actions_executed += n_exec
    else:
        # phase17_locked, value_only, sum, max
        score_mode = {
            "phase17_locked": "predictor_only",
            "value_only": "value_only",
            "combined_sum": "sum",
            "combined_max": "max",
        }[mode]
        while actions_executed < args.total_actions:
            score_fn = build_combined_score_fn(
                env, model, head, obs, goal_xy_world, score_mode,
                args.combined_lambda, args,
            )
            mu = rollout_scripted_prior(env, obs, goal_xy_world,
                                          args.plan_horizon, args.jepa_stride)
            plan, _ = cem_with_prior(score_fn, mu, env.action_dim,
                                       args.plan_horizon, args.eval_cem_iters,
                                       args.eval_K, args.elite_frac,
                                       sigma=args.eval_sigma,
                                       sigma_floor=args.sigma_floor)
            if plan is None:
                plan = mu
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                obs = step_and_track(env, obs, a, args, contact)
            actions_executed += n_exec

    actual_final_xy = obs["cubeA_pos"][:2]
    actual_dist = float(np.linalg.norm(actual_final_xy - goal_xy_world))
    improvement = max(0.0, (start_dist - actual_dist) / max(start_dist, 1e-9))
    cube_disp = actual_final_xy - cube_init[:2]
    disp_n = float(np.linalg.norm(cube_disp))
    goal_dir = (goal_xy_world - cube_init[:2]) / max(
        np.linalg.norm(goal_xy_world - cube_init[:2]), 1e-9)
    dir_score = float(np.dot(cube_disp / max(disp_n, 1e-9), goal_dir)) \
                  if disp_n > 0.02 else 0.0
    return {
        "ep_id": ep_id, "mode": mode,
        "start_dist": start_dist, "actual_dist": actual_dist,
        "improvement": improvement, "dir_score": dir_score,
        "cube_displacement": disp_n, "contact": bool(contact[0]),
        "success": bool(actual_dist <= args.success_threshold),
    }


def summarize_mode(mode, per_ep):
    return {
        "mode": mode, "n_episodes": len(per_ep),
        "improvement": float(np.mean([r["improvement"] for r in per_ep])),
        "dir_score": float(np.mean([r["dir_score"] for r in per_ep])),
        "contact_rate": float(np.mean([r["contact"] for r in per_ep])),
        "mean_displacement": float(np.mean([r["cube_displacement"]
                                              for r in per_ep])),
        "success_rate": float(np.mean([r["success"] for r in per_ep])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout-cache", default=None)
    p.add_argument("--value-head-ckpt", default=None)
    p.add_argument("--rebuild-rollouts", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--total-actions", type=int, default=15)
    p.add_argument("--replan-every", type=int, default=5)
    p.add_argument("--rollout-episodes", type=int, default=300)
    p.add_argument("--rollout-log-every", type=int, default=20)
    p.add_argument("--train-K", type=int, default=32)
    p.add_argument("--train-cem-iters", type=int, default=1)
    p.add_argument("--train-sigma", type=float, default=0.12)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--train-log-every", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--eval-K", type=int, default=32)
    p.add_argument("--eval-cem-iters", type=int, default=1)
    p.add_argument("--eval-sigma", type=float, default=0.12)
    p.add_argument("--naive-sigma", type=float, default=0.5)
    p.add_argument("--combined-lambda", type=float, default=0.5,
                    help="weight for predictor in combined_sum mode")
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--modes", default="gt_closed_loop,phase17_locked,"
                    "value_only,combined_sum,combined_max,naive_cem")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = load_action_model(args)

    env = build_env(args.image_size,
                     horizon=args.total_actions * args.jepa_stride
                             + 3 * args.plan_horizon * args.jepa_stride + 60)

    # 1. Collect rollouts
    features, goals, plans, labels, rollout_cache = \
        load_or_collect_rollouts(env, model, args, out)

    # 2. Train value head
    head, head_ckpt, head_stats = \
        train_or_load_value_head(features, goals, plans, labels, args, out)

    # 3. Eval
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for m in modes:
        print(json.dumps({"event": "eval_mode", "mode": m}), flush=True)
        per_ep = []
        t0 = time.time()
        for ep in range(args.n_eval_episodes):
            per_ep.append(run_episode(env, model, head, m, args, ep))
            if (ep + 1) % 10 == 0:
                partial = summarize_mode(m, per_ep)
                print(json.dumps({"event": "progress", "mode": m,
                                   "done": ep + 1,
                                   "improvement":
                                       round(partial["improvement"], 3),
                                   "success":
                                       round(partial["success_rate"], 3),
                                   "elapsed_s":
                                       round(time.time() - t0, 1)}),
                       flush=True)
        with open(out / f"per_episode_{m}.jsonl", "w") as fh:
            for rec in per_ep:
                fh.write(json.dumps(rec) + "\n")
        summary = summarize_mode(m, per_ep)
        all_results.append(summary)
        print(json.dumps({"event": "mode_done", **summary}), flush=True)
    env.close()

    # 4. Gate evaluation
    by_mode = {r["mode"]: r for r in all_results}
    locked = by_mode.get("phase17_locked")
    vonly = by_mode.get("value_only")
    csum = by_mode.get("combined_sum")
    cmax = by_mode.get("combined_max")
    locked_imp = locked["improvement"] if locked else float("nan")

    gates = {"locked_improvement": locked_imp}
    if vonly:
        gates["g1_value_only_beats_locked_by_0_02"] = \
            vonly["improvement"] >= locked_imp + 0.02
        gates["value_only_improvement"] = vonly["improvement"]
    # G2: best combined beats locked + 0.04
    best_combined = None
    for c in [csum, cmax]:
        if c is None: continue
        if best_combined is None or c["improvement"] > best_combined["improvement"]:
            best_combined = c
    if best_combined:
        gates["g2_best_combined_beats_locked_by_0_04"] = \
            best_combined["improvement"] >= locked_imp + 0.04
        gates["best_combined_mode"] = best_combined["mode"]
        gates["best_combined_improvement"] = best_combined["improvement"]

    n_pass = int(gates.get("g1_value_only_beats_locked_by_0_02", False)) \
              + int(gates.get("g2_best_combined_beats_locked_by_0_04", False))
    gates["n_pass_main"] = n_pass
    gates["verdict"] = (
        "2/2 main gates pass — value head wins"
        if n_pass == 2 else
        "1/2 partial — value head adds in one mode"
        if n_pass == 1 else
        "0/2 — value head approach does not help"
    )

    report = {
        "args": vars(args),
        "rollout_cache": str(rollout_cache),
        "head_ckpt": str(head_ckpt),
        "head_stats": head_stats,
        "results": all_results,
        "gates": gates,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"event": "done", **gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
