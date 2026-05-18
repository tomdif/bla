"""Phase 18B — distill scripted-prior+CEM into a learned proposal policy.

Phase 18D locked the recipe:

    OF-JEPA predictor + competent prior + CEM refinement >= oracle

This script removes the hand-coded prior one step at a time:

1. Run the current teacher (`scripted_prior_cem`) and record the CEM-refined
   full action plan at each replan boundary.
2. Train a small proposal policy:

       state_features(obs, goal) -> [plan_horizon, action_dim]

3. Evaluate:
       learned_policy_only  — execute policy plans directly
       learned_policy_cem   — use policy plan as CEM mean with light compute
       scripted_prior_cem   — current heavy teacher baseline

The goal is not to change OF-JEPA. It tests whether the prior can become
learned while OF-JEPA remains the planner-facing world model.
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
from system1_jepa.planning_policy import (
    PlanProposalPolicy,
    build_policy_from_config,
    policy_config_dict,
    train_plan_policy_supervised,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


def load_planning_dependencies() -> None:
    """Lazy-load robosuite-dependent Phase 15/16 helpers.

    Keeping these imports out of module top-level lets `--help`,
    `py_compile`, and unit tests work on machines that do not have
    robosuite/MuJoCo installed. Real Phase 18B execution still requires
    the robosuite pod, just like Phases 15-17.
    """

    global build_env, closed_loop_gt_step, encode_frame, find_cubeA_slot, predict_score_seq
    global cem_with_prior, rollout_scripted_prior, state_features

    from scripts.phase15_planning import (
        build_env,
        closed_loop_gt_step,
        encode_frame,
        find_cubeA_slot,
        predict_score_seq,
    )
    from scripts.phase16_policy_prior_mpc import (
        cem_with_prior,
        rollout_scripted_prior,
        state_features,
    )


def norm_xy(p_xy: np.ndarray) -> np.ndarray:
    n = (p_xy + 0.3) / 0.6
    return np.clip(n, 0.0, 1.0).astype(np.float32)


def sample_goal(obs: dict, ep_id: int, args) -> tuple[np.ndarray, float]:
    cube_init = obs["cubeA_pos"][:2].copy()
    rng = np.random.RandomState(ep_id + 1000)
    theta = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(args.goal_dist_min, args.goal_dist_max)
    goal_xy = cube_init + r * np.array([np.cos(theta), np.sin(theta)])
    start_dist = float(np.linalg.norm(cube_init - goal_xy))
    return goal_xy, start_dist


def load_action_model(args):
    cfg = OFJEPAConfig(
        n_files=args.n_slots,
        id_dim=args.slot_dim // 2,
        state_dim=args.slot_dim // 2,
        proposal_dim=args.slot_dim,
    )
    model = ActionConditionedOFJEPA(
        image_size=args.image_size,
        cfg=cfg,
        action_dim=args.action_dim,
        use_action=True,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_action, map_location=args.device))
    model.eval()
    return model


def build_score_fn(env, model, obs, goal_xy_world):
    init_slot = encode_frame(model, obs["agentview_image"])
    cubeA_idx = find_cubeA_slot(
        model,
        init_slot,
        norm_xy(obs["cubeA_pos"][:2]),
        norm_xy(obs["cubeB_pos"][:2]),
        norm_xy(obs["robot0_eef_pos"][:2]),
    )
    goal_xy_norm = norm_xy(goal_xy_world)
    return lambda seq: predict_score_seq(
        model, init_slot, seq, cubeA_idx, goal_xy_norm, use_action=True,
    )


def step_and_track(env, obs, action, args, contact_ref):
    for _ in range(args.jepa_stride):
        obs, _, _, _ = env.step(action)
        if float(np.linalg.norm(obs["cubeA_pos"][:2] - obs["robot0_eef_pos"][:2])) < 0.04:
            contact_ref[0] = True
    return obs


def collect_teacher_dataset(env, model, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect (features, CEM-refined full plan) examples from the teacher."""

    features: list[np.ndarray] = []
    plans: list[np.ndarray] = []
    scores: list[float] = []
    t0 = time.time()
    for ep in range(args.teacher_episodes):
        obs = env.reset()
        goal_xy, _ = sample_goal(obs, ep, args)
        actions_executed = 0
        while actions_executed < args.total_actions:
            feat = state_features(obs, goal_xy)
            score_fn = build_score_fn(env, model, obs, goal_xy)
            mu = rollout_scripted_prior(
                env, obs, goal_xy, args.plan_horizon, args.jepa_stride,
            )
            plan, score = cem_with_prior(
                score_fn,
                mu,
                env.action_dim,
                args.plan_horizon,
                args.teacher_cem_iters,
                args.teacher_K,
                args.elite_frac,
                sigma=args.prior_sigma,
            )
            if plan is None:
                plan = mu
                score = float("nan")
            features.append(feat.astype(np.float32))
            plans.append(plan.astype(np.float32))
            scores.append(float(score) if np.isfinite(score) else float("nan"))

            n_exec = min(
                args.replan_every,
                args.total_actions - actions_executed,
                len(plan),
            )
            contact = [False]
            for action in plan[:n_exec]:
                obs = step_and_track(env, obs, action, args, contact)
            actions_executed += n_exec

        if (ep + 1) % max(args.teacher_log_every, 1) == 0:
            print(
                json.dumps({
                    "event": "teacher_progress",
                    "episodes": ep + 1,
                    "examples": len(features),
                    "elapsed_s": round(time.time() - t0, 1),
                }),
                flush=True,
            )

    meta = {
        "teacher_episodes": args.teacher_episodes,
        "n_examples": len(features),
        "plan_horizon": args.plan_horizon,
        "action_dim": int(env.action_dim),
        "teacher_K": args.teacher_K,
        "teacher_cem_iters": args.teacher_cem_iters,
        "mean_teacher_score": float(np.nanmean(scores)) if scores else None,
    }
    return np.stack(features), np.stack(plans), meta


def load_or_collect_teacher(env, model, args, out: Path):
    cache = Path(args.teacher_cache) if args.teacher_cache else out / "teacher_plans.npz"
    if cache.exists() and not args.rebuild_teacher:
        print(json.dumps({"event": "loading_teacher_cache", "path": str(cache)}), flush=True)
        data = np.load(cache, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        return data["features"], data["plans"], meta, cache

    print(json.dumps({"event": "collecting_teacher", "path": str(cache)}), flush=True)
    features, plans, meta = collect_teacher_dataset(env, model, args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        features=features,
        plans=plans,
        meta=json.dumps(meta),
    )
    print(json.dumps({"event": "saved_teacher_cache", "path": str(cache), **meta}), flush=True)
    return features, plans, meta, cache


def load_policy(path: Path, device: str) -> PlanProposalPolicy:
    ckpt = torch.load(path, map_location=device)
    policy = build_policy_from_config(ckpt["config"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def train_or_load_policy(features, plans, args, out: Path) -> tuple[PlanProposalPolicy, Path, dict]:
    policy_path = Path(args.policy_ckpt) if args.policy_ckpt else out / "plan_policy.pt"
    if policy_path.exists() and not args.retrain_policy:
        print(json.dumps({"event": "loading_policy", "path": str(policy_path)}), flush=True)
        return load_policy(policy_path, args.device), policy_path, {"loaded": True}

    policy = PlanProposalPolicy(
        in_dim=features.shape[1],
        hidden=args.policy_hidden,
        plan_horizon=plans.shape[1],
        action_dim=plans.shape[2],
        dropout=args.policy_dropout,
    ).to(args.device)
    stats = train_plan_policy_supervised(
        policy,
        torch.from_numpy(features),
        torch.from_numpy(plans),
        steps=args.policy_train_steps,
        batch_size=args.policy_batch_size,
        lr=args.policy_lr,
        weight_decay=args.policy_weight_decay,
        step_decay=args.policy_step_decay,
        seed=args.seed,
        log_every=args.policy_log_every,
    )
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "config": policy_config_dict(policy),
            "distill_stats": stats.__dict__,
        },
        policy_path,
    )
    print(json.dumps({"event": "saved_policy", "path": str(policy_path), **stats.__dict__}), flush=True)
    policy.eval()
    return policy, policy_path, stats.__dict__


@torch.no_grad()
def rollout_policy_prior(policy: PlanProposalPolicy, obs: dict, goal_xy_world: np.ndarray, device: str) -> np.ndarray:
    feat = state_features(obs, goal_xy_world)
    x = torch.from_numpy(feat).to(device)
    plan = policy.propose(x).cpu().numpy().astype(np.float32)
    return np.clip(plan, -1.0, 1.0)


def run_episode(env, model, policy, mode: str, args, ep_id: int) -> dict:
    obs = env.reset()
    cubeA_init = obs["cubeA_pos"][:3].copy()
    goal_xy_world, start_dist = sample_goal(obs, ep_id, args)
    actions_executed = 0
    contact = [False]
    pred_best_score_first = float("nan")
    n_total_candidates = 0

    if mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            action = closed_loop_gt_step(env, obs, goal_xy_world)
            obs = step_and_track(env, obs, action, args, contact)
    elif mode in (
        "naive_cem",
        "scripted_prior_cem",
        "scripted_prior_light_cem",
        "learned_policy_only",
        "learned_policy_cem",
    ):
        while actions_executed < args.total_actions:
            if mode == "learned_policy_only":
                plan = rollout_policy_prior(policy, obs, goal_xy_world, args.device)
                score = float("nan")
            else:
                score_fn = build_score_fn(env, model, obs, goal_xy_world)
                if mode == "naive_cem":
                    mu = np.zeros((args.plan_horizon, env.action_dim), dtype=np.float32)
                    sigma = args.naive_sigma
                    n_iters = args.cem_iters
                    n_cand = args.main_K
                elif mode == "scripted_prior_cem":
                    mu = rollout_scripted_prior(env, obs, goal_xy_world, args.plan_horizon, args.jepa_stride)
                    sigma = args.prior_sigma
                    n_iters = args.cem_iters
                    n_cand = args.main_K
                elif mode == "scripted_prior_light_cem":
                    mu = rollout_scripted_prior(env, obs, goal_xy_world, args.plan_horizon, args.jepa_stride)
                    sigma = args.policy_cem_sigma
                    n_iters = args.policy_cem_iters
                    n_cand = args.policy_K
                else:  # learned_policy_cem
                    mu = rollout_policy_prior(policy, obs, goal_xy_world, args.device)
                    sigma = args.policy_cem_sigma
                    n_iters = args.policy_cem_iters
                    n_cand = args.policy_K
                plan, score = cem_with_prior(
                    score_fn,
                    mu,
                    env.action_dim,
                    args.plan_horizon,
                    n_iters,
                    n_cand,
                    args.elite_frac,
                    sigma=sigma,
                    sigma_floor=args.sigma_floor,
                )
                if plan is None:
                    plan = mu
                n_total_candidates += n_iters * n_cand
            if actions_executed == 0 and np.isfinite(score):
                pred_best_score_first = float(score)

            n_exec = min(args.replan_every, args.total_actions - actions_executed, len(plan))
            for action in plan[:n_exec]:
                obs = step_and_track(env, obs, action, args, contact)
            actions_executed += n_exec
    elif mode == "random":
        for _ in range(args.total_actions):
            action = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            obs = step_and_track(env, obs, action, args, contact)
    else:
        raise ValueError(f"unknown mode: {mode}")

    actual_final_xy = obs["cubeA_pos"][:2]
    actual_dist = float(np.linalg.norm(actual_final_xy - goal_xy_world))
    improvement = max(0.0, (start_dist - actual_dist) / max(start_dist, 1e-9))
    cube_disp = actual_final_xy - cubeA_init[:2]
    disp_n = float(np.linalg.norm(cube_disp))
    goal_dir = (goal_xy_world - cubeA_init[:2]) / max(np.linalg.norm(goal_xy_world - cubeA_init[:2]), 1e-9)
    dir_score = float(np.dot(cube_disp / max(disp_n, 1e-9), goal_dir)) if disp_n > 0.02 else 0.0
    project = float(np.dot(cube_disp, goal_dir))
    goal_along = float(np.dot(goal_xy_world - cubeA_init[:2], goal_dir))
    return {
        "ep_id": ep_id,
        "mode": mode,
        "goal_xy_world": goal_xy_world.tolist(),
        "actual_final_xy_world": actual_final_xy.tolist(),
        "start_dist": start_dist,
        "actual_dist": actual_dist,
        "improvement": improvement,
        "dir_score": dir_score,
        "cube_displacement": disp_n,
        "contact": bool(contact[0]),
        "overshoot": bool(project > goal_along),
        "success": bool(actual_dist <= args.success_threshold),
        "pred_best_score": pred_best_score_first if np.isfinite(pred_best_score_first) else None,
        "n_total_candidates": n_total_candidates,
    }


def summarize_mode(mode: str, per_ep: list[dict]) -> dict:
    preds = [r["pred_best_score"] for r in per_ep if r["pred_best_score"] is not None]
    actuals = [r["actual_dist"] for r in per_ep if r["pred_best_score"] is not None]
    corr = float(np.corrcoef(-np.array(preds), np.array(actuals))[0, 1]) if len(preds) > 2 else float("nan")
    return {
        "mode": mode,
        "n_episodes": len(per_ep),
        "improvement": float(np.mean([r["improvement"] for r in per_ep])),
        "dir_score": float(np.mean([r["dir_score"] for r in per_ep])),
        "contact_rate": float(np.mean([r["contact"] for r in per_ep])),
        "overshoot_rate": float(np.mean([r["overshoot"] for r in per_ep])),
        "mean_displacement": float(np.mean([r["cube_displacement"] for r in per_ep])),
        "success_rate": float(np.mean([r["success"] for r in per_ep])),
        "mean_candidates": float(np.mean([r["n_total_candidates"] for r in per_ep])),
        "pred_actual_corr": corr,
    }


def gate_verdict(results: list[dict], args) -> dict:
    by_mode = {r["mode"]: r for r in results}
    teacher = by_mode.get("scripted_prior_cem")
    policy_only = by_mode.get("learned_policy_only")
    policy_cem = by_mode.get("learned_policy_cem")
    if not (teacher and policy_only and policy_cem):
        return {"n_pass": 0, "verdict": "missing required modes for gates"}

    g1_threshold = args.policy_only_ratio * teacher["improvement"]
    g1 = policy_only["improvement"] >= g1_threshold
    g2_threshold = teacher["improvement"] - args.policy_cem_margin
    g2 = policy_cem["improvement"] >= g2_threshold
    teacher_candidates = max(teacher["mean_candidates"], 1.0)
    compute_ratio = policy_cem["mean_candidates"] / teacher_candidates
    g3 = compute_ratio <= args.policy_cem_compute_ratio
    n_pass = int(g1) + int(g2) + int(g3)
    verdict = (
        "3/3 learned prior replaces scripted prior" if n_pass == 3 else
        "2/3 partial learned-prior success" if n_pass == 2 else
        "1/3 weak learned-prior signal" if n_pass == 1 else
        "0/3 keep scripted prior; revisit policy data or capacity"
    )
    return {
        "g1_policy_only_ratio_pass": g1,
        "g1_threshold": g1_threshold,
        "g2_policy_cem_matches_teacher_pass": g2,
        "g2_threshold": g2_threshold,
        "g3_compute_reduction_pass": g3,
        "policy_cem_compute_ratio": compute_ratio,
        "n_pass": n_pass,
        "verdict": verdict,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True, help="Phase 17 fine-tuned action model")
    p.add_argument("--out", required=True)
    p.add_argument("--policy-ckpt", default=None)
    p.add_argument("--teacher-cache", default=None)
    p.add_argument("--rebuild-teacher", action="store_true")
    p.add_argument("--retrain-policy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--total-actions", type=int, default=15)
    p.add_argument("--replan-every", type=int, default=5)
    p.add_argument("--teacher-episodes", type=int, default=120)
    p.add_argument("--teacher-K", type=int, default=128)
    p.add_argument("--teacher-cem-iters", type=int, default=3)
    p.add_argument("--teacher-log-every", type=int, default=10)
    p.add_argument("--policy-hidden", type=int, default=256)
    p.add_argument("--policy-dropout", type=float, default=0.0)
    p.add_argument("--policy-train-steps", type=int, default=2500)
    p.add_argument("--policy-batch-size", type=int, default=128)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--policy-weight-decay", type=float, default=1e-4)
    p.add_argument("--policy-step-decay", type=float, default=0.90)
    p.add_argument("--policy-log-every", type=int, default=250)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--modes", default="gt_closed_loop,scripted_prior_cem,learned_policy_only,learned_policy_cem,scripted_prior_light_cem,naive_cem")
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--main-K", type=int, default=128)
    p.add_argument("--policy-cem-iters", type=int, default=1)
    p.add_argument("--policy-K", type=int, default=32)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--prior-sigma", type=float, default=0.2)
    p.add_argument("--policy-cem-sigma", type=float, default=0.12)
    p.add_argument("--naive-sigma", type=float, default=0.5)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--oracle-sanity-n", type=int, default=30)
    p.add_argument("--oracle-min-improvement", type=float, default=0.10)
    p.add_argument("--oracle-min-contact", type=float, default=0.60)
    p.add_argument("--policy-only-ratio", type=float, default=0.75)
    p.add_argument("--policy-cem-margin", type=float, default=0.02)
    p.add_argument("--policy-cem-compute-ratio", type=float, default=0.35)
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_action_model(args)
    env = build_env(
        args.image_size,
        horizon=args.total_actions * args.jepa_stride
                + 3 * args.plan_horizon * args.jepa_stride + 60,
    )
    requested_policy_path = Path(args.policy_ckpt) if args.policy_ckpt else out / "plan_policy.pt"
    if requested_policy_path.exists() and not args.retrain_policy:
        print(json.dumps({"event": "loading_policy", "path": str(requested_policy_path)}), flush=True)
        policy = load_policy(requested_policy_path, args.device)
        policy_path = requested_policy_path
        policy_meta = {"loaded": True}
        teacher_meta = {"skipped": "existing policy checkpoint loaded"}
        teacher_cache = None
    else:
        features, plans, teacher_meta, teacher_cache = load_or_collect_teacher(env, model, args, out)
        policy, policy_path, policy_meta = train_or_load_policy(features, plans, args, out)

    print(json.dumps({"event": "oracle_sanity", "n": args.oracle_sanity_n}), flush=True)
    oracle_runs = [run_episode(env, model, policy, "gt_closed_loop", args, ep) for ep in range(args.oracle_sanity_n)]
    oracle = summarize_mode("gt_closed_loop", oracle_runs)
    sanity_pass = (
        oracle["improvement"] >= args.oracle_min_improvement
        and oracle["contact_rate"] >= args.oracle_min_contact
    )
    if not sanity_pass:
        with open(out / "summary.json", "w") as f:
            json.dump({
                "sanity_pass": False,
                "oracle": oracle,
                "teacher_cache": str(teacher_cache) if teacher_cache is not None else None,
                "policy_path": str(policy_path),
            }, f, indent=2)
        env.close()
        print(json.dumps({"event": "oracle_sanity_failed", **oracle}), flush=True)
        return

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for mode in modes:
        print(json.dumps({"event": "eval_mode", "mode": mode}), flush=True)
        if mode == "gt_closed_loop":
            per_ep = oracle_runs
        else:
            per_ep = []
            t0 = time.time()
            for ep in range(args.n_episodes):
                per_ep.append(run_episode(env, model, policy, mode, args, ep))
                if (ep + 1) % 10 == 0:
                    partial = summarize_mode(mode, per_ep)
                    print(json.dumps({
                        "event": "progress",
                        "mode": mode,
                        "done": ep + 1,
                        "improvement": round(partial["improvement"], 3),
                        "success": round(partial["success_rate"], 3),
                        "elapsed_s": round(time.time() - t0, 1),
                    }), flush=True)
        with open(out / f"per_episode_{mode}.jsonl", "w") as fh:
            for rec in per_ep:
                fh.write(json.dumps(rec) + "\n")
        summary = summarize_mode(mode, per_ep)
        all_results.append(summary)
        print(json.dumps({"event": "mode_done", **summary}), flush=True)
    env.close()

    gates = gate_verdict(all_results, args)
    report = {
        "sanity_pass": True,
        "teacher_cache": str(teacher_cache) if teacher_cache is not None else None,
        "teacher_meta": teacher_meta,
        "policy_path": str(policy_path),
        "policy_meta": policy_meta,
        "results": all_results,
        "gates": gates,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"event": "done", **gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
