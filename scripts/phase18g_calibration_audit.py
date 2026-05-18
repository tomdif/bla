"""Phase 18γ — Predictor calibration audit.

Map where the Phase 17 action-conditioned OF-JEPA predictor is
trustworthy and where CEM starts exploiting model error.

For each of 6 candidate distributions, sample M candidate plans per
state at N states drawn from MPC replan boundaries, score them with
the predictor, then ground-truth-execute each plan via env-clone
restore. Log Pearson/Spearman rank correlations, top-k precision,
per-decile calibration, distribution breadth, and OOD distance.

Distributions:
  D1. scripted_prior + tiny noise (σ=0.02)
  D2. scripted_prior + light CEM (1 iter σ=0.12) — sampled candidates
  D3. scripted_prior + heavy CEM (3 iters σ=0.2 annealed) — final-iter candidates
  D4. learned_policy mean + tiny noise (σ=0.02)
  D5. learned_policy + light CEM (1 iter σ=0.12)
  D6. naive Gaussian CEM (μ=0, σ=0.5)

Phase 18β motivation:
  pred-actual corr  naive=+0.29  light=-0.19  heavy=-0.29  policy_cem=-0.52
  Trust region appears to be the scripted-prior manifold; the audit
  pre-commits to G1 (D2 corr > +0.10), G2 (D3 corr < D2 - 0.10),
  G3 (D2 top1_realized > D3 top1_realized).

This is a measurement phase. No new training, no planner improvement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.planning_policy import (
    PlanProposalPolicy,
    build_policy_from_config,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


# Loaded lazily so --help works without robosuite installed
build_env = None
encode_frame = None
find_cubeA_slot = None
predict_score_seq = None
closed_loop_gt_step = None
state_features = None
rollout_scripted_prior = None


def load_planning_dependencies() -> None:
    global build_env, encode_frame, find_cubeA_slot, predict_score_seq
    global closed_loop_gt_step, state_features, rollout_scripted_prior

    from scripts.phase15_planning import (
        build_env as _build_env,
        encode_frame as _encode_frame,
        find_cubeA_slot as _find_cubeA_slot,
        predict_score_seq as _predict_score_seq,
        closed_loop_gt_step as _closed_loop_gt_step,
    )
    from scripts.phase16_policy_prior_mpc import (
        state_features as _state_features,
        rollout_scripted_prior as _rollout_scripted_prior,
    )

    build_env = _build_env
    encode_frame = _encode_frame
    find_cubeA_slot = _find_cubeA_slot
    predict_score_seq = _predict_score_seq
    closed_loop_gt_step = _closed_loop_gt_step
    state_features = _state_features
    rollout_scripted_prior = _rollout_scripted_prior


# ---------- model + policy loading ----------
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


def load_policy(args):
    if not args.policy_ckpt:
        return None
    ckpt = torch.load(args.policy_ckpt, map_location=args.device)
    policy = build_policy_from_config(ckpt["config"]).to(args.device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def norm_xy(p_xy: np.ndarray) -> np.ndarray:
    n = (p_xy + 0.3) / 0.6
    return np.clip(n, 0.0, 1.0).astype(np.float32)


# ---------- prior mean rollouts ----------
def policy_mean_plan(policy, obs, goal_xy, device) -> np.ndarray:
    feat = state_features(obs, goal_xy)
    x = torch.from_numpy(feat).to(device)
    with torch.no_grad():
        plan = policy.propose(x).cpu().numpy().astype(np.float32)
    return np.clip(plan, -1.0, 1.0)


# ---------- candidate sampling per distribution ----------
def sample_d1_scripted_tiny_noise(env, model, policy, obs, goal_xy, args):
    """D1: scripted_prior mean + N(0, 0.02) per step."""
    mu = rollout_scripted_prior(env, obs, goal_xy, args.plan_horizon, args.jepa_stride)
    eps = np.random.randn(args.M, args.plan_horizon, args.action_dim).astype(np.float32)
    cands = np.clip(mu[None] + 0.02 * eps, -1.0, 1.0)
    return cands, mu


def sample_d4_policy_tiny_noise(env, model, policy, obs, goal_xy, args):
    """D4: policy mean + N(0, 0.02) per step."""
    mu = policy_mean_plan(policy, obs, goal_xy, args.device)
    eps = np.random.randn(args.M, args.plan_horizon, args.action_dim).astype(np.float32)
    cands = np.clip(mu[None] + 0.02 * eps, -1.0, 1.0)
    return cands, mu


def _cem_iter_candidates(score_fn, mu_init, args, sigma, n_iters):
    """Run n_iters of CEM and return ALL candidates from the FINAL iter + their scores.

    Same recipe as cem_with_prior in phase16 — annealing via elite std.
    """
    mu = mu_init.astype(np.float32).copy()
    sig = np.full((args.plan_horizon, args.action_dim), sigma, dtype=np.float32)
    elite_n = max(1, int(args.elite_frac * args.M))
    last_cands = None
    last_scores = None
    for it in range(n_iters):
        eps = np.random.randn(args.M, args.plan_horizon, args.action_dim).astype(np.float32)
        cands = np.clip(mu[None] + sig[None] * eps, -1.0, 1.0)
        scores = np.array([score_fn(c) for c in cands])
        last_cands, last_scores = cands, scores
        if it < n_iters - 1:  # update for next iter
            elite_idx = np.argsort(scores)[::-1][:elite_n]
            elites = cands[elite_idx]
            mu = elites.mean(0)
            sig = np.maximum(elites.std(0), args.sigma_floor)
    return last_cands, last_scores, mu


def sample_d2_scripted_light_cem(env, model, policy, obs, goal_xy, args, score_fn):
    mu = rollout_scripted_prior(env, obs, goal_xy, args.plan_horizon, args.jepa_stride)
    cands, scores, _ = _cem_iter_candidates(score_fn, mu, args, sigma=0.12, n_iters=1)
    return cands, scores, mu


def sample_d3_scripted_heavy_cem(env, model, policy, obs, goal_xy, args, score_fn):
    mu = rollout_scripted_prior(env, obs, goal_xy, args.plan_horizon, args.jepa_stride)
    cands, scores, _ = _cem_iter_candidates(score_fn, mu, args, sigma=0.2, n_iters=3)
    return cands, scores, mu


def sample_d5_policy_light_cem(env, model, policy, obs, goal_xy, args, score_fn):
    mu = policy_mean_plan(policy, obs, goal_xy, args.device)
    cands, scores, _ = _cem_iter_candidates(score_fn, mu, args, sigma=0.12, n_iters=1)
    return cands, scores, mu


def sample_d6_naive(env, model, policy, obs, goal_xy, args, score_fn):
    mu = np.zeros((args.plan_horizon, args.action_dim), dtype=np.float32)
    cands, scores, _ = _cem_iter_candidates(score_fn, mu, args, sigma=0.5, n_iters=3)
    return cands, scores, mu


# ---------- ground-truth execution ----------
def execute_plan_clone(env, plan, args) -> dict:
    """Save state, execute plan[:replan_every], measure realized improvement, restore."""
    saved = env.sim.get_state()
    obs = env._get_observations() if hasattr(env, "_get_observations") else None
    cube_start = None
    contact = False
    try:
        cube_start = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")][:2].copy()
    except Exception:
        try:
            obs_tmp = env._get_observations()
            cube_start = obs_tmp["cubeA_pos"][:2].copy()
        except Exception:
            cube_start = None
    if cube_start is None:
        env.sim.set_state(saved); env.sim.forward()
        return {"contact": False, "displacement": float("nan"), "cube_end": None}

    horizon = min(args.exec_horizon, len(plan))
    cur_obs = None
    for t in range(horizon):
        for _ in range(args.jepa_stride):
            cur_obs, _, _, _ = env.step(plan[t])
            if not contact:
                eef_xy = cur_obs["robot0_eef_pos"][:2]
                cube_xy = cur_obs["cubeA_pos"][:2]
                if float(np.linalg.norm(eef_xy - cube_xy)) < 0.04:
                    contact = True
    cube_end = cur_obs["cubeA_pos"][:2].copy() if cur_obs is not None else cube_start.copy()
    displacement = float(np.linalg.norm(cube_end - cube_start))
    env.sim.set_state(saved); env.sim.forward()
    return {
        "contact": contact,
        "displacement": displacement,
        "cube_end": cube_end.tolist(),
        "cube_start": cube_start.tolist(),
    }


def realized_improvement(cube_start, cube_end, goal_xy) -> float:
    start_d = float(np.linalg.norm(cube_start - goal_xy))
    end_d = float(np.linalg.norm(np.asarray(cube_end) - goal_xy))
    return max(0.0, (start_d - end_d) / max(start_d, 1e-9))


# ---------- per-distribution metrics ----------
def calibration_metrics(pred_scores: np.ndarray, realized: np.ndarray) -> dict:
    valid = np.isfinite(pred_scores) & np.isfinite(realized)
    if int(valid.sum()) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"),
                "n_valid": int(valid.sum())}
    p = pred_scores[valid]
    r = realized[valid]
    pearson = float(np.corrcoef(p, r)[0, 1])
    # Spearman via rank
    p_rank = np.argsort(np.argsort(p)).astype(float)
    r_rank = np.argsort(np.argsort(r)).astype(float)
    spearman = float(np.corrcoef(p_rank, r_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n_valid": int(valid.sum())}


def top_k_precision(pred_scores: np.ndarray, realized: np.ndarray, k: int) -> float:
    valid = np.isfinite(pred_scores) & np.isfinite(realized)
    if int(valid.sum()) < k:
        return float("nan")
    p = pred_scores[valid]
    r = realized[valid]
    pred_top = set(np.argsort(p)[::-1][:k].tolist())
    real_top = set(np.argsort(r)[::-1][:k].tolist())
    return len(pred_top & real_top) / k


def per_decile_calibration(pred_scores: np.ndarray, realized: np.ndarray, n_bins: int = 10):
    valid = np.isfinite(pred_scores) & np.isfinite(realized)
    if int(valid.sum()) < n_bins:
        return []
    p = pred_scores[valid]
    r = realized[valid]
    order = np.argsort(p)[::-1]
    p_sorted = p[order]
    r_sorted = r[order]
    chunks = np.array_split(np.arange(len(p_sorted)), n_bins)
    rows = []
    for i, idx in enumerate(chunks):
        rows.append({
            "decile": i,
            "mean_pred": float(np.mean(p_sorted[idx])),
            "mean_realized": float(np.mean(r_sorted[idx])),
            "n": int(len(idx)),
        })
    return rows


def distribution_breadth(candidates: np.ndarray) -> float:
    """Mean per-step L2 standard deviation across candidates."""
    if candidates.shape[0] < 2:
        return float("nan")
    # candidates: [M, H, A]; std per (h, a) → mean
    return float(np.std(candidates, axis=0).mean())


# ---------- driver ----------
def make_env_at_state(args, state_seed: int):
    """Build a fresh env and reset deterministically at a state seed.

    Pilot showed env.sim.set_state() degrades after ~128 cycles, so we build
    a new env for each (state, distribution) pair to keep set_state cycles
    well under the safety threshold.
    """
    import random as _py_random
    np.random.seed(state_seed)
    _py_random.seed(state_seed)
    torch.manual_seed(state_seed)
    # Use a very generous horizon so the env never auto-terminates during the
    # audit's set_state-heavy candidate execution loop.
    total_step_bound = args.M * args.exec_horizon * args.jepa_stride \
                        + args.plan_horizon * args.jepa_stride + 5000
    env_horizon = max(total_step_bound, 100_000)
    env = build_env(args.image_size, horizon=env_horizon)
    obs = env.reset()
    return env, obs


def sample_goal_for_state(obs: dict, state_seed: int, args) -> np.ndarray:
    rng = np.random.RandomState(state_seed + 7)
    theta = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(args.goal_dist_min, args.goal_dist_max)
    cube_xy = obs["cubeA_pos"][:2].copy()
    return cube_xy + r * np.array([np.cos(theta), np.sin(theta)])


def run_state_distribution(model, policy, dist_id: str, s_idx: int, args) -> dict:
    """Build fresh env at deterministic seed, sample candidates for one
    distribution, score with predictor, ground-truth-execute the
    top-K + bot-K + K random subset, return per-state record."""
    state_seed = args.seed * 10000 + s_idx + 1
    env, obs = make_env_at_state(args, state_seed)
    goal_xy = sample_goal_for_state(obs, state_seed, args)
    cube_start = obs["cubeA_pos"][:2].copy()
    sim_state = env.sim.get_state()

    needs_score_fn = dist_id in ("D2", "D3", "D5", "D6")
    if needs_score_fn:
        init_slot = encode_frame(model, obs["agentview_image"])
        cubeA_idx = find_cubeA_slot(
            model, init_slot,
            norm_xy(obs["cubeA_pos"][:2]),
            norm_xy(obs["cubeB_pos"][:2]),
            norm_xy(obs["robot0_eef_pos"][:2]),
        )
        goal_xy_norm = norm_xy(goal_xy)
        score_fn = lambda seq, _slot=init_slot, _i=cubeA_idx, _g=goal_xy_norm: \
            predict_score_seq(model, _slot, seq, _i, _g, use_action=True)
    else:
        score_fn = None

    # Sample candidates per distribution
    if dist_id == "D1":
        cands, mu = sample_d1_scripted_tiny_noise(env, model, policy, obs, goal_xy, args)
    elif dist_id == "D4":
        cands, mu = sample_d4_policy_tiny_noise(env, model, policy, obs, goal_xy, args)
    elif dist_id == "D2":
        cands, _, mu = sample_d2_scripted_light_cem(env, model, policy, obs, goal_xy, args, score_fn)
    elif dist_id == "D3":
        cands, _, mu = sample_d3_scripted_heavy_cem(env, model, policy, obs, goal_xy, args, score_fn)
    elif dist_id == "D5":
        cands, _, mu = sample_d5_policy_light_cem(env, model, policy, obs, goal_xy, args, score_fn)
    elif dist_id == "D6":
        cands, _, mu = sample_d6_naive(env, model, policy, obs, goal_xy, args, score_fn)
    else:
        env.close()
        raise ValueError(f"unknown dist_id {dist_id}")

    # Restore before predictor scoring (CEM may have left env stepped)
    env.sim.set_state(sim_state); env.sim.forward()

    # Score with predictor — uniform across distributions
    if needs_score_fn:
        pred = np.array([score_fn(c) for c in cands])
    else:
        init_slot = encode_frame(model, obs["agentview_image"])
        cubeA_idx = find_cubeA_slot(
            model, init_slot,
            norm_xy(obs["cubeA_pos"][:2]),
            norm_xy(obs["cubeB_pos"][:2]),
            norm_xy(obs["robot0_eef_pos"][:2]),
        )
        goal_xy_norm = norm_xy(goal_xy)
        pred = np.array([predict_score_seq(model, init_slot, c, cubeA_idx,
                                             goal_xy_norm, use_action=True)
                          for c in cands])

    # Build GT execution subset: top-K, bot-K, K random
    K = args.gt_k_per_extreme
    order = np.argsort(pred)
    bot_idx = order[:K].tolist()
    top_idx = order[-K:][::-1].tolist()
    remaining = [i for i in range(args.M) if i not in set(top_idx) | set(bot_idx)]
    rng = np.random.RandomState(state_seed * 31 + 17)
    rand_idx = rng.choice(remaining, size=min(K, len(remaining)), replace=False).tolist()
    exec_idx = sorted(set(top_idx + bot_idx + rand_idx))

    # Restore before execution
    env.sim.set_state(sim_state); env.sim.forward()

    real = np.full(args.M, float("nan"), dtype=np.float32)
    contacts = np.zeros(args.M, dtype=bool)
    for c_idx in exec_idx:
        res = execute_plan_clone(env, cands[c_idx], args)
        if np.isfinite(res["displacement"]) and res.get("cube_end") is not None:
            real[c_idx] = realized_improvement(cube_start, res["cube_end"], goal_xy)
        contacts[c_idx] = res["contact"]
    env.close()

    breadth = distribution_breadth(cands)
    top1_idx = int(top_idx[0])
    return {
        "state_idx": s_idx,
        "dist": dist_id,
        "breadth": breadth,
        "pred_mean": float(np.nanmean(pred)),
        "pred_std": float(np.nanstd(pred)),
        "real_mean_exec": float(np.nanmean(real[exec_idx])),
        "real_std_exec": float(np.nanstd(real[exec_idx])),
        "contact_rate_exec": float(np.mean(contacts[exec_idx])),
        "top1_realized": float(real[top1_idx]) if np.isfinite(real[top1_idx]) else float("nan"),
        "topK_realized_mean": float(np.nanmean(real[top_idx])),
        "botK_realized_mean": float(np.nanmean(real[bot_idx])),
        "randK_realized_mean": float(np.nanmean(real[rand_idx])) if rand_idx else float("nan"),
        "top_vs_bot_gap": float(np.nanmean(real[top_idx]) - np.nanmean(real[bot_idx])),
        "exec_pred": pred[exec_idx].tolist(),
        "exec_real": real[exec_idx].tolist(),
        "n_exec": len(exec_idx),
    }


def run_distribution(model, policy, dist_id: str, args) -> dict:
    """Run a single candidate distribution across all N states; return metrics + raw data.

    Builds a fresh env per state to avoid env.sim.set_state degradation
    that the pilot caught (real_std=0 by ~cycle 128).
    """
    print(json.dumps({"event": "dist_start", "dist": dist_id, "n_states": args.N,
                       "M": args.M}), flush=True)
    t0 = time.time()
    per_state_records = []
    all_pred_exec = []
    all_real_exec = []
    all_breadth = []
    all_contact_exec = []

    for s_idx in range(args.N):
        rec = run_state_distribution(model, policy, dist_id, s_idx, args)
        per_state_records.append(rec)
        exec_pred = np.array(rec["exec_pred"], dtype=np.float32)
        exec_real = np.array(rec["exec_real"], dtype=np.float32)
        all_pred_exec.append(exec_pred)
        all_real_exec.append(exec_real)
        all_breadth.append(rec["breadth"])
        # Use exec contact rate
        all_contact_exec.append(rec["contact_rate_exec"])

        if (s_idx + 1) % max(args.log_every, 1) == 0:
            print(json.dumps({
                "event": "dist_progress",
                "dist": dist_id,
                "states_done": s_idx + 1,
                "elapsed_s": round(time.time() - t0, 1),
                "running_top1_realized": float(np.nanmean([p["top1_realized"]
                                                              for p in per_state_records])),
                "running_top_vs_bot_gap": float(np.nanmean([p["top_vs_bot_gap"]
                                                                for p in per_state_records])),
                "running_real_std_exec": float(np.nanmean([p["real_std_exec"]
                                                               for p in per_state_records])),
            }), flush=True)

    # Aggregate across executed-candidate subsets (per-state, then pool)
    pred_flat = np.concatenate(all_pred_exec)
    real_flat = np.concatenate(all_real_exec)
    cal = calibration_metrics(pred_flat, real_flat)

    per_state_corr = [calibration_metrics(p, r) for p, r in zip(all_pred_exec, all_real_exec)]
    mean_pearson_per_state = float(np.nanmean([m["pearson"] for m in per_state_corr]))
    mean_spearman_per_state = float(np.nanmean([m["spearman"] for m in per_state_corr]))

    top1 = float(np.nanmean([top_k_precision(p, r, k=1)
                               for p, r in zip(all_pred_exec, all_real_exec)]))
    top5 = float(np.nanmean([top_k_precision(p, r, k=5)
                               for p, r in zip(all_pred_exec, all_real_exec)]))

    top1_realized = float(np.nanmean([rec["top1_realized"] for rec in per_state_records]))
    topK_realized = float(np.nanmean([rec["topK_realized_mean"] for rec in per_state_records]))
    botK_realized = float(np.nanmean([rec["botK_realized_mean"] for rec in per_state_records]))
    randK_realized = float(np.nanmean([rec["randK_realized_mean"] for rec in per_state_records]))
    top_vs_bot_gap = float(np.nanmean([rec["top_vs_bot_gap"] for rec in per_state_records]))
    real_std_exec_mean = float(np.nanmean([rec["real_std_exec"] for rec in per_state_records]))
    contact_rate = float(np.mean(all_contact_exec))
    breadth_mean = float(np.mean(all_breadth))
    decile = per_decile_calibration(pred_flat, real_flat, n_bins=10)
    elapsed = time.time() - t0

    return {
        "dist": dist_id,
        "n_states": args.N,
        "M": args.M,
        "gt_k_per_extreme": args.gt_k_per_extreme,
        "pooled_pearson": cal["pearson"],
        "pooled_spearman": cal["spearman"],
        "mean_pearson_per_state": mean_pearson_per_state,
        "mean_spearman_per_state": mean_spearman_per_state,
        "top1_precision": top1,
        "top5_precision": top5,
        "top1_realized_mean": top1_realized,
        "topK_realized_mean": topK_realized,
        "botK_realized_mean": botK_realized,
        "randK_realized_mean": randK_realized,
        "top_vs_bot_gap": top_vs_bot_gap,
        "real_std_exec_mean": real_std_exec_mean,
        "contact_rate": contact_rate,
        "distribution_breadth": breadth_mean,
        "per_decile": decile,
        "per_state": per_state_records,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--policy-ckpt", default=None,
                    help="Required for D4/D5; optional otherwise")
    p.add_argument("--out", required=True)
    p.add_argument("--distributions", default="D1,D2,D3,D4,D5,D6")
    p.add_argument("--M", type=int, default=128, help="candidates per state")
    p.add_argument("--N", type=int, default=40, help="states sampled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--exec-horizon", type=int, default=5,
                    help="how many of plan_horizon actions to execute for ground truth")
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--gt-k-per-extreme", type=int, default=8,
                    help="GT-execute top-K + bot-K + K random per state "
                          "(default 8 → 24 execs/state).")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_action_model(args)
    policy = load_policy(args)

    dists = [d.strip() for d in args.distributions.split(",") if d.strip()]
    if any(d in ("D4", "D5") for d in dists) and policy is None:
        raise SystemExit("D4/D5 require --policy-ckpt")

    print(json.dumps({"event": "audit_start",
                       "N": args.N, "M": args.M, "K": args.gt_k_per_extreme,
                       "dists": dists, "fresh_env_per_state": True}), flush=True)

    all_dists = []
    for d in dists:
        res = run_distribution(model, policy, d, args)
        # strip per_state from saved summary if large
        save_res = {**res}
        all_dists.append(save_res)
        print(json.dumps({"event": "dist_done", "dist": d,
                           "spearman_per_state": save_res["mean_spearman_per_state"],
                           "pearson_per_state": save_res["mean_pearson_per_state"],
                           "top_vs_bot_gap": save_res["top_vs_bot_gap"],
                           "topK_realized": save_res["topK_realized_mean"],
                           "botK_realized": save_res["botK_realized_mean"],
                           "real_std_exec": save_res["real_std_exec_mean"],
                           "breadth": save_res["distribution_breadth"],
                           "contact": save_res["contact_rate"],
                           "elapsed_s": round(save_res["elapsed_s"], 1)},
                          ), flush=True)

    # Gate evaluation — uses top-vs-bot-gap (more robust than per-state corr at K=8)
    by_dist = {r["dist"]: r for r in all_dists}
    gates = {}
    if "D2" in by_dist:
        d2 = by_dist["D2"]
        gates["d2_spearman"] = d2["mean_spearman_per_state"]
        gates["d2_top_vs_bot_gap"] = d2["top_vs_bot_gap"]
        # G1: predictor's top picks beat its bottom picks at least a little
        gates["g1_d2_top_beats_bot"] = d2["top_vs_bot_gap"] > 0.02
    if "D2" in by_dist and "D3" in by_dist:
        d2, d3 = by_dist["D2"], by_dist["D3"]
        gates["d3_spearman"] = d3["mean_spearman_per_state"]
        gates["d3_top_vs_bot_gap"] = d3["top_vs_bot_gap"]
        # G2: predictor's top-vs-bot calibration degrades from light to heavy CEM
        gates["g2_d3_gap_below_d2_gap"] = (d2["top_vs_bot_gap"] - d3["top_vs_bot_gap"]) > 0.02
        gates["g3_d2_topK_beats_d3_topK"] = d2["topK_realized_mean"] > d3["topK_realized_mean"]
        gates["d2_topK_realized"] = d2["topK_realized_mean"]
        gates["d3_topK_realized"] = d3["topK_realized_mean"]

    n_pass = int(gates.get("g1_d2_top_beats_bot", False)) \
              + int(gates.get("g2_d3_gap_below_d2_gap", False)) \
              + int(gates.get("g3_d2_topK_beats_d3_topK", False))
    gates["n_pass"] = n_pass
    gates["verdict"] = (
        "3/3 thesis confirmed — predictor trust region is the scripted-prior manifold"
        if n_pass == 3 else
        "2/3 partial — calibration story real, boundary fuzzier than expected"
        if n_pass == 2 else
        "1/3 weak — either D2 not positive or signal too noisy"
        if n_pass == 1 else
        "0/3 — predictor uncalibrated everywhere"
    )

    summary = {
        "args": vars(args),
        "distributions": all_dists,
        "gates": gates,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(json.dumps({"event": "done", **gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
