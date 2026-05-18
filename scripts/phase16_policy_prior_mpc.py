"""Phase 16 — Policy-prior MPC: does a contact-aware action prior unlock OF-JEPA planning?

Phase 15b proved OF-JEPA's predictor is calibrated (corr=0.53) but
naïve Gaussian-around-zero CEM rarely samples cube-engaging action
sequences. Phase 16 tests whether providing a competent action prior
(scripted contact or learned BC) lets the same predictor refine into
a planner.

Methods (same OF-JEPA predictor for all CEM modes):
  naive_cem            — μ=0, σ=0.5 (15b baseline reproduction)
  scripted_prior_cem   — μ from closed_loop_gt_step env-clone rollout, σ=0.2
  bc_prior_cem         — μ from BC policy env-clone rollout, σ=0.2
  gt_closed_loop       — oracle skyline (no CEM)

BC policy: tiny MLP on raw geometric features (cube_xy + eef_xyz + cube_z
+ goal_xy + push_dir = 10 dims) → 7-dim action. Trained inline on
scripted_push trajectories with random goals (~5 min).

Pre-committed gates per PHASE_16_POLICY_PRIOR_PRECOMMIT.md:
  G1. contact_rate(bc_prior_cem) >= 0.50
  G2. improvement(bc_prior_cem) >= 0.10
  G3. improvement(bc_prior_cem) - improvement(naive_cem) >= 0.05
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite as rs
from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.identity_probe import hungarian_assign
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA
from scripts.phase15_planning import (
    build_env, encode_frame, find_cubeA_slot, predict_score_seq,
    cem_plan, closed_loop_gt_step,
)


# ---------- BC policy ----------
class BCPolicy(nn.Module):
    def __init__(self, in_dim=10, hidden=128, out_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def state_features(obs, goal_xy):
    cube_xy = obs["cubeA_pos"][:2]
    eef_xy = obs["robot0_eef_pos"][:2]
    eef_z = obs["robot0_eef_pos"][2:3]
    cube_z = obs["cubeA_pos"][2:3]
    push = goal_xy - cube_xy
    d = float(np.linalg.norm(push))
    push_dir = push / max(d, 1e-9)
    return np.concatenate([cube_xy, eef_xy, eef_z, cube_z, goal_xy, push_dir]).astype(np.float32)


def collect_bc_data(env, n_episodes, n_steps_per_ep, stride, rng_seed=42):
    """Run closed_loop_gt_step with random goals; collect (state, action) pairs."""
    rng = np.random.RandomState(rng_seed)
    feats, actions = [], []
    for ep in range(n_episodes):
        obs = env.reset()
        cube_init = obs["cubeA_pos"][:2].copy()
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.05, 0.10)
        goal_xy = cube_init + r * np.array([np.cos(theta), np.sin(theta)])
        for _ in range(n_steps_per_ep):
            feats.append(state_features(obs, goal_xy))
            a = closed_loop_gt_step(env, obs, goal_xy)
            actions.append(a)
            for _ in range(stride):
                obs, _, _, _ = env.step(a)
    return np.stack(feats), np.stack(actions)


def train_bc(env, args, device):
    """Inline BC training: collect data from oracle, fit MLP."""
    print(f"\n=== BC data collection (n={args.bc_episodes} eps × {args.total_actions} actions) ===", flush=True)
    t0 = time.time()
    feats, acts = collect_bc_data(env, args.bc_episodes, args.total_actions,
                                   args.jepa_stride, rng_seed=42)
    print(f"  Collected {len(feats)} (state, action) pairs in {time.time()-t0:.0f}s", flush=True)
    bc = BCPolicy(in_dim=feats.shape[1], out_dim=acts.shape[1]).to(device)
    opt = torch.optim.Adam(bc.parameters(), lr=3e-4)
    feats_t = torch.from_numpy(feats).float().to(device)
    acts_t = torch.from_numpy(acts).float().to(device)
    n = len(feats)
    bs = 64
    for step in range(args.bc_train_steps):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = bc(feats_t[idx])
        loss = F.mse_loss(pred, acts_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{args.bc_train_steps}  loss={float(loss):.4f}", flush=True)
    bc.eval()
    return bc


# ---------- prior trajectory generators (env-clone rollouts) ----------
def rollout_scripted_prior(env, obs, goal_xy, H, stride):
    """Save env state, roll out closed_loop_gt_step for H actions, restore state."""
    saved = env.sim.get_state()
    actions = []
    cur = obs
    for _ in range(H):
        a = closed_loop_gt_step(env, cur, goal_xy)
        actions.append(a)
        for _ in range(stride):
            cur, _, _, _ = env.step(a)
    env.sim.set_state(saved); env.sim.forward()
    return np.stack(actions)


def rollout_bc_prior(env, obs, goal_xy, H, stride, bc_policy, device):
    """Save env state, roll out BC policy for H actions, restore state."""
    saved = env.sim.get_state()
    actions = []
    cur = obs
    bc_policy.eval()
    with torch.no_grad():
        for _ in range(H):
            f = state_features(cur, goal_xy)
            ft = torch.from_numpy(f).unsqueeze(0).to(device)
            a = bc_policy(ft)[0].cpu().numpy()
            actions.append(a.astype(np.float32))
            for _ in range(stride):
                cur, _, _, _ = env.step(a)
    env.sim.set_state(saved); env.sim.forward()
    return np.stack(actions)


# ---------- prior-aware CEM ----------
def cem_with_prior(score_fn, mu_init, action_dim, n_steps, n_iters, n_cand,
                    elite_frac, sigma=0.2, sigma_floor=0.05):
    """CEM where μ is initialized from a prior trajectory instead of zeros."""
    mu = mu_init.astype(np.float32).copy()
    sig = np.full((n_steps, action_dim), sigma, dtype=np.float32)
    best_score = -np.inf; best_seq = None
    elite_n = max(1, int(elite_frac * n_cand))
    for it in range(n_iters):
        eps = np.random.randn(n_cand, n_steps, action_dim).astype(np.float32)
        cands = np.clip(mu[None] + sig[None] * eps, -1, 1)
        scores = np.array([score_fn(c) for c in cands])
        elite_idx = np.argsort(scores)[::-1][:elite_n]
        elites = cands[elite_idx]
        mu = elites.mean(0)
        sig = np.maximum(elites.std(0), sigma_floor)
        if scores[elite_idx[0]] > best_score:
            best_score = float(scores[elite_idx[0]])
            best_seq = cands[elite_idx[0]].copy()
    return best_seq, best_score


# ---------- episode runner ----------
def run_episode(env, model_a, bc_policy, mode, args, ep_id, device):
    obs = env.reset()
    cubeA_init = obs["cubeA_pos"][:3].copy()
    rng = np.random.RandomState(ep_id + 1000)
    theta = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(args.goal_dist_min, args.goal_dist_max)
    goal_xy_world = cubeA_init[:2] + r * np.array([np.cos(theta), np.sin(theta)])

    def norm(p_xy):
        n = (p_xy + 0.3) / 0.6
        return np.clip(n, 0.0, 1.0).astype(np.float32)
    goal_xy_norm = norm(goal_xy_world)
    start_dist = float(np.linalg.norm(cubeA_init[:2] - goal_xy_world))

    K = args.main_K
    contact = False
    actions_executed = 0
    pred_best_score_first = float("nan")
    n_total_candidates = 0

    def step_and_track(a):
        nonlocal contact, obs
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
            if float(np.linalg.norm(obs["cubeA_pos"][:2] - obs["robot0_eef_pos"][:2])) < 0.04:
                contact = True

    if mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            a = closed_loop_gt_step(env, obs, goal_xy_world)
            step_and_track(a)
    elif mode in ("naive_cem", "scripted_prior_cem", "bc_prior_cem"):
        while actions_executed < args.total_actions:
            init_slot = encode_frame(model_a, obs["agentview_image"])
            cubeA_idx = find_cubeA_slot(
                model_a, init_slot,
                norm(obs["cubeA_pos"][:2]),
                norm(obs["cubeB_pos"][:2]),
                norm(obs["robot0_eef_pos"][:2]),
            )
            sf = lambda seq: predict_score_seq(model_a, init_slot, seq,
                                                 cubeA_idx, goal_xy_norm, use_action=True)
            if mode == "naive_cem":
                mu_init = np.zeros((args.plan_horizon, env.action_dim), dtype=np.float32)
                sigma = 0.5
            elif mode == "scripted_prior_cem":
                mu_init = rollout_scripted_prior(env, obs, goal_xy_world,
                                                   args.plan_horizon, args.jepa_stride)
                sigma = args.prior_sigma
            else:  # bc_prior_cem
                mu_init = rollout_bc_prior(env, obs, goal_xy_world,
                                             args.plan_horizon, args.jepa_stride,
                                             bc_policy, device)
                sigma = args.prior_sigma
            plan, score = cem_with_prior(sf, mu_init, env.action_dim, args.plan_horizon,
                                           args.cem_iters, K, args.elite_frac, sigma=sigma)
            if actions_executed == 0:
                pred_best_score_first = float(score) if np.isfinite(score) else float("nan")
            n_total_candidates += args.cem_iters * K
            n_exec = min(args.replan_every, args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                step_and_track(a)
            actions_executed += n_exec
    elif mode == "random":
        for _ in range(args.total_actions):
            a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            step_and_track(a)
    elif mode == "bc_only":
        # Run BC policy directly, no CEM
        for _ in range(args.total_actions):
            f = state_features(obs, goal_xy_world)
            ft = torch.from_numpy(f).unsqueeze(0).to(device)
            with torch.no_grad():
                a = bc_policy(ft)[0].cpu().numpy().astype(np.float32)
            step_and_track(a)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    actual_final_xy = obs["cubeA_pos"][:2]
    actual_dist = float(np.linalg.norm(actual_final_xy - goal_xy_world))
    improvement = max(0.0, (start_dist - actual_dist) / max(start_dist, 1e-9))
    cube_disp = actual_final_xy - cubeA_init[:2]
    disp_n = float(np.linalg.norm(cube_disp))
    goal_dir = (goal_xy_world - cubeA_init[:2]) / max(np.linalg.norm(goal_xy_world - cubeA_init[:2]), 1e-9)
    dir_score = float(np.dot(cube_disp / max(disp_n, 1e-9), goal_dir)) if disp_n > 0.02 else 0.0
    project = float(np.dot(cube_disp, goal_dir))
    goal_along = float(np.dot(goal_xy_world - cubeA_init[:2], goal_dir))
    overshoot = bool(project > goal_along)
    return {
        "ep_id": ep_id, "mode": mode, "K": K,
        "goal_xy_world": goal_xy_world.tolist(),
        "actual_final_xy_world": actual_final_xy.tolist(),
        "start_dist": start_dist, "actual_dist": actual_dist,
        "improvement": improvement, "dir_score": dir_score,
        "cube_displacement": disp_n, "contact": bool(contact),
        "overshoot": overshoot, "success": bool(actual_dist <= args.success_threshold),
        "pred_best_score": pred_best_score_first if not np.isnan(pred_best_score_first) else None,
        "n_total_candidates": n_total_candidates,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True, help="Path to model_action.pt from Phase 15")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--total-actions", type=int, default=15)
    p.add_argument("--replan-every", type=int, default=5)
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--main-K", type=int, default=128)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--prior-sigma", type=float, default=0.2)
    p.add_argument("--modes",
        default="gt_closed_loop,naive_cem,scripted_prior_cem,bc_prior_cem,bc_only,random")
    p.add_argument("--bc-episodes", type=int, default=40)
    p.add_argument("--bc-train-steps", type=int, default=1200)
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--oracle-sanity-n", type=int, default=30)
    p.add_argument("--oracle-min-improvement", type=float, default=0.10)
    p.add_argument("--oracle-min-dir-score", type=float, default=0.0)
    p.add_argument("--oracle-min-contact", type=float, default=0.60)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load OF-JEPA action-conditioned model.
    print(f"=== Loading model_action.pt from {args.model_action} ===", flush=True)
    cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
    model_a = ActionConditionedOFJEPA(image_size=args.image_size, cfg=cfg,
                                        action_dim=7, use_action=True).to(args.device)
    model_a.load_state_dict(torch.load(args.model_action, map_location=args.device))
    model_a.eval()

    # Train BC only if needed by selected modes. Generous horizon: per-episode
    # we use total_actions*stride=60 real steps PLUS up to 3 replans ×
    # plan_horizon*stride=120 simulated steps for prior rollouts (env-clone).
    env = build_env(args.image_size,
                    horizon=args.total_actions * args.jepa_stride
                            + 3 * args.plan_horizon * args.jepa_stride + 60)
    needs_bc = any(m in args.modes for m in ("bc_prior_cem", "bc_only"))
    if needs_bc and args.bc_episodes > 0:
        bc_policy = train_bc(env, args, args.device)
        torch.save(bc_policy.state_dict(), out / "bc_policy.pt")
    else:
        bc_policy = None
        print("Skipping BC training (no BC-using modes selected or bc_episodes=0).", flush=True)

    # Oracle sanity gate
    print(f"\n=== Oracle sanity gate (n={args.oracle_sanity_n}) ===", flush=True)
    oracle_runs = [run_episode(env, model_a, bc_policy, "gt_closed_loop", args, ep, args.device)
                    for ep in range(args.oracle_sanity_n)]
    o_imp = float(np.mean([r["improvement"] for r in oracle_runs]))
    o_dir = float(np.mean([r["dir_score"] for r in oracle_runs]))
    o_contact = float(np.mean([r["contact"] for r in oracle_runs]))
    o_succ = float(np.mean([r["success"] for r in oracle_runs]))
    print(f"  Oracle: imp={o_imp:.3f}  dir={o_dir:.3f}  contact={o_contact:.2f}  succ={o_succ:.3f}", flush=True)
    if not (o_imp >= args.oracle_min_improvement and o_dir > args.oracle_min_dir_score
            and o_contact >= args.oracle_min_contact):
        print(f"ORACLE SANITY GATE FAILED. Abort.")
        with open(out / "summary.json", "w") as f:
            json.dump({"sanity_pass": False,
                       "oracle": {"improvement": o_imp, "dir_score": o_dir,
                                   "contact_rate": o_contact, "success_rate": o_succ,
                                   "n_episodes": args.oracle_sanity_n}}, f, indent=2)
        env.close(); return
    print(f"  Oracle sanity gate PASSED.", flush=True)

    # Modes
    modes = [m.strip() for m in args.modes.split(",")]
    all_results = []
    for mode in modes:
        if mode == "gt_closed_loop":
            per_ep = oracle_runs
        else:
            print(f"\n=== {mode} ===", flush=True)
            t0 = time.time()
            per_ep = []
            for ep_id in range(args.n_episodes):
                r = run_episode(env, model_a, bc_policy, mode, args, ep_id, args.device)
                per_ep.append(r)
                if (ep_id + 1) % 10 == 0:
                    suc = np.mean([x["success"] for x in per_ep])
                    imp = np.mean([x["improvement"] for x in per_ep])
                    print(f"  ep {ep_id+1}/{args.n_episodes}  imp={imp:.3f}  succ={suc:.3f}  "
                          f"t={time.time()-t0:.0f}s", flush=True)
        with open(out / f"per_episode_{mode}.jsonl", "w") as f:
            for r in per_ep: f.write(json.dumps(r) + "\n")
        succ = float(np.mean([r["success"] for r in per_ep]))
        imp = float(np.mean([r["improvement"] for r in per_ep]))
        dir_s = float(np.mean([r["dir_score"] for r in per_ep]))
        contact_r = float(np.mean([r["contact"] for r in per_ep]))
        over_r = float(np.mean([r["overshoot"] for r in per_ep]))
        mean_disp = float(np.mean([r["cube_displacement"] for r in per_ep]))
        preds = [r["pred_best_score"] for r in per_ep if r["pred_best_score"] is not None]
        actuals = [r["actual_dist"] for r in per_ep if r["pred_best_score"] is not None]
        corr = float(np.corrcoef(-np.array(preds), np.array(actuals))[0, 1]) if len(preds) > 2 else float("nan")
        summary = {"mode": mode, "n_episodes": len(per_ep),
                    "improvement": imp, "dir_score": dir_s,
                    "contact_rate": contact_r, "overshoot_rate": over_r,
                    "mean_displacement": mean_disp, "success_rate": succ,
                    "pred_actual_corr": corr}
        all_results.append(summary)
        print(f"  RESULT: imp={imp:.3f}  dir={dir_s:.3f}  contact={contact_r:.2f}  "
              f"succ={succ:.3f}  disp={mean_disp:.4f}m  over={over_r:.2f}  corr={corr:.3f}", flush=True)
    env.close()

    # Gates
    print("\n=== Phase 16 Gate Verdicts ===", flush=True)
    bc_cem = next((r for r in all_results if r["mode"] == "bc_prior_cem"), None)
    naive = next((r for r in all_results if r["mode"] == "naive_cem"), None)
    g1 = g2 = g3 = False
    if bc_cem and naive:
        g1 = bc_cem["contact_rate"] >= 0.50
        g2 = bc_cem["improvement"] >= 0.10
        g3_diff = bc_cem["improvement"] - naive["improvement"]
        g3 = g3_diff >= 0.05
        print(f"  G1 contact_rate(bc_prior): {bc_cem['contact_rate']:.3f}  "
              f"(>=0.50 {'PASS' if g1 else 'FAIL'})")
        print(f"  G2 improvement(bc_prior): {bc_cem['improvement']:.3f}  "
              f"(>=0.10 {'PASS' if g2 else 'FAIL'})")
        print(f"  G3 gap_over_naive: {g3_diff:+.3f}  (>=0.05 {'PASS' if g3 else 'FAIL'})")
    n_pass = int(g1) + int(g2) + int(g3)
    verdict = (
        "3/3 supports planning with prior" if n_pass == 3 else
        "2/3 partial" if n_pass == 2 else
        "1/3 marginal" if n_pass == 1 else
        "0/3 prior alone insufficient"
    )
    print(f"\nOVERALL: {verdict}  ({n_pass}/3)")
    with open(out / "summary.json", "w") as f:
        json.dump({"sanity_pass": True, "results": all_results,
                   "g1_pass": g1, "g2_pass": g2, "g3_pass": g3,
                   "n_pass": n_pass, "verdict": verdict,
                   "g3_diff": g3_diff if bc_cem and naive else None}, f, indent=2)


if __name__ == "__main__":
    main()
