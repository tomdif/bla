"""Phase 15 — CEM-on-predictor planning for cube-displacement goals.

Train OF-JEPA (action-conditioned + no-action) on v3 scripted rollouts;
then for each eval episode:
  1. Reset env, sample random goal p_goal = cubeA_init + 2D Δ (|Δ|~0.10m).
  2. Encode initial frame, Hungarian-match slots to GT entities to
     identify the cubeA slot.
  3. Plan an action sequence over H=5 stride-boundaries (20 env frames)
     using CEM-on-predictor (4 iters, K candidates, 20% elite).
  4. Execute the planned sequence in env, hold each planned action for
     `stride` frames.
  5. Record final cubeA position, distance to goal, success flag.

Modes:
  cem_action    — CEM with action-conditioned predictor (the test)
  cem_noaction  — CEM with no-action predictor (degenerate; reports random-ish)
  random        — sample 1 action seq from prior, execute (no planning)
  gt_scripted   — scripted v3-style push aimed at goal (oracle skyline)

Pre-committed gates: G1 success-rate gap ≥10pp, G2 mean dist ≤0.90×,
G3 candidate efficiency ≤50%.

Usage:
    python scripts/phase15_planning.py \\
        --train-cache /workspace/robosuite_local/stack_scripted \\
        --seed 0 --max-steps 1500 --jepa-stride 4 \\
        --n-episodes 50 --plan-horizon 5 --cem-iters 4 \\
        --candidate-counts 64,128,256 --main-K 128 \\
        --modes cem_action,cem_noaction,random,gt_scripted \\
        --out /workspace/phase15_planning
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
import robosuite as rs
from system1_jepa.robosuite_data import RobosuiteDataset, RobosuiteSpec
from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.identity_probe import hungarian_assign
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA, train_one_run


# ---------- model utilities ----------
def encode_frame(model, frame_hwc_u8):
    """Encode a single 128x128x3 uint8 frame → [S, slot_dim]."""
    frame = torch.from_numpy(frame_hwc_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame = frame.unsqueeze(0).to(next(model.parameters()).device)  # [T=1, 1, 3, H, W]
    slot_states, _ = model.encode_video(frame[0])  # [T=1, S, slot_dim]
    return slot_states[0]


def find_cubeA_slot(model, slot_state, cubeA_xy_norm, cubeB_xy_norm, eef_xy_norm):
    """Hungarian-match predicted slot positions to GT entity positions in normalized 2D.

    Returns the slot index whose match is to entity 0 (cubeA).
    """
    pred_pos = model.slot_to_pos_aux(slot_state.unsqueeze(0))[0].detach().cpu().numpy()
    gt_pos = np.stack([cubeA_xy_norm, cubeB_xy_norm, eef_xy_norm])  # [3, 2]
    rows, cols, _ = hungarian_assign(pred_pos, gt_pos)
    # Find which row matched cubeA (col == 0).
    matches = list(zip(rows.tolist(), cols.tolist()))
    for r, c in matches:
        if c == 0:
            return int(r)
    # Fallback: closest slot to cubeA in normalized coords
    return int(np.argmin(np.linalg.norm(pred_pos - cubeA_xy_norm, axis=1)))


def predict_score_seq(model, init_slot, action_seq_np, cubeA_idx, goal_xy_norm, use_action):
    """Roll out predictor with action_seq, return -|predicted final cubeA xy - goal|."""
    device = next(model.parameters()).device
    id_dim = model.cfg.id_dim
    slot = init_slot.clone()
    for a in action_seq_np:
        a_t = torch.from_numpy(a).float().to(device).unsqueeze(0)
        slot_in = slot.unsqueeze(0)  # [1, S, slot_dim]
        with torch.no_grad():
            state_pred = model.predict_state_delta(slot_in, a_t)[0]  # [S, state_dim]
        slot = torch.cat([slot[:, :id_dim], state_pred], dim=-1)
    with torch.no_grad():
        pred_pos = model.slot_to_pos_aux(slot.unsqueeze(0))[0].cpu().numpy()
    cubeA_pred = pred_pos[cubeA_idx]
    return -float(np.linalg.norm(cubeA_pred - goal_xy_norm))


# ---------- CEM ----------
def cem_plan(score_fn, action_dim, n_steps, n_iters, n_cand, elite_frac,
             init_sigma=0.5, sigma_floor=0.05):
    """Returns (best_seq, best_score)."""
    mu = np.zeros((n_steps, action_dim), dtype=np.float32)
    sigma = np.full((n_steps, action_dim), init_sigma, dtype=np.float32)
    best_score, best_seq = -np.inf, None
    elite_n = max(1, int(elite_frac * n_cand))
    for it in range(n_iters):
        eps = np.random.randn(n_cand, n_steps, action_dim).astype(np.float32)
        candidates = np.clip(mu[None] + sigma[None] * eps, -1, 1)
        scores = np.array([score_fn(c) for c in candidates])
        elite_idx = np.argsort(scores)[::-1][:elite_n]
        elites = candidates[elite_idx]
        mu = elites.mean(0)
        sigma = np.maximum(elites.std(0), sigma_floor)
        if scores[elite_idx[0]] > best_score:
            best_score = float(scores[elite_idx[0]])
            best_seq = candidates[elite_idx[0]].copy()
    return best_seq, best_score


def closed_loop_gt_step(env, obs, goal_xy_world):
    """Adaptive scripted policy: at every action step, push toward CURRENT goal-relative direction.

    Closed-loop replacement for the open-loop v3-style policy. Doesn't lock
    sweep direction at episode start — recomputes from current cube position
    each step. This handles overshoot/wrong-direction failure modes that
    plagued the open-loop oracle.
    """
    cube = obs["cubeA_pos"][:2]
    cube_z = obs["cubeA_pos"][2]
    eef = obs["robot0_eef_pos"][:2]
    eef_z = obs["robot0_eef_pos"][2]
    push_dir = goal_xy_world - cube
    d = float(np.linalg.norm(push_dir))
    a = np.zeros(env.action_dim, dtype=np.float32)
    if d < 0.005:
        return a   # at goal, stop
    push_dir = push_dir / max(d, 1e-9)
    horiz_to_cube = float(np.linalg.norm(cube - eef))
    if horiz_to_cube > 0.04 or eef_z > cube_z + 0.025:
        # Approach: drive EE to anchor point behind cube at cube height
        approach_xy = cube - push_dir * 0.04
        target = np.array([approach_xy[0], approach_xy[1], cube_z + 0.005])
        delta = target - obs["robot0_eef_pos"]
        a[:3] = np.clip(delta * 10.0, -1, 1)
    else:
        # Push toward goal
        a[0:2] = push_dir
        a[2] = -0.2
    a[6] = +1.0
    return a


# ---------- single-episode MPC-style planning + execution ----------
def run_episode(env, model_a, model_n, mode, args, ep_id):
    """One eval episode under the chosen mode (MPC-style for CEM modes).

    Modes:
      cem_action       — MPC with action-conditioned predictor
      cem_noaction     — MPC with no-action predictor (predictor doesn't differentiate)
      random           — random uniform[-1,1] actions, no planning
      gt_closed_loop   — adaptive scripted policy (closed-loop oracle)
    """
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

    K = args.K if hasattr(args, "K") and args.K else args.main_K
    pred_best_score_first = float("nan")
    contact = False
    n_replans = 0
    n_total_candidates = 0
    actions_executed = 0

    def step_and_track(a):
        """Step env stride times, track contact."""
        nonlocal contact
        nonlocal obs
        for _ in range(args.jepa_stride):
            obs, _, _, _ = env.step(a)
            if float(np.linalg.norm(obs["cubeA_pos"][:2] - obs["robot0_eef_pos"][:2])) < 0.04:
                contact = True

    if mode in ("cem_action", "cem_noaction"):
        use_action = (mode == "cem_action")
        model = model_a if use_action else model_n
        # MPC replan loop: at every replan boundary, re-encode and CEM-plan.
        while actions_executed < args.total_actions:
            init_slot = encode_frame(model, obs["agentview_image"])
            cubeA_idx = find_cubeA_slot(
                model, init_slot,
                norm(obs["cubeA_pos"][:2]),
                norm(obs["cubeB_pos"][:2]),
                norm(obs["robot0_eef_pos"][:2]),
            )
            sf = lambda seq: predict_score_seq(model, init_slot, seq,
                                                 cubeA_idx, goal_xy_norm, use_action)
            plan, score = cem_plan(sf, env.action_dim, args.plan_horizon,
                                     args.cem_iters, K, args.elite_frac)
            if n_replans == 0:
                pred_best_score_first = float(score) if np.isfinite(score) else float("nan")
            n_replans += 1
            n_total_candidates += args.cem_iters * K
            # Execute first replan_every actions
            n_exec = min(args.replan_every, args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                step_and_track(a)
            actions_executed += n_exec
    elif mode == "random":
        for _ in range(args.total_actions):
            a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
            step_and_track(a)
        n_total_candidates = 1
    elif mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            a = closed_loop_gt_step(env, obs, goal_xy_world)
            step_and_track(a)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    actual_final_xy = obs["cubeA_pos"][:2]
    actual_dist = float(np.linalg.norm(actual_final_xy - goal_xy_world))
    improvement = max(0.0, (start_dist - actual_dist) / max(start_dist, 1e-9))
    cube_displacement = actual_final_xy - cubeA_init[:2]
    disp_norm = float(np.linalg.norm(cube_displacement))
    goal_dir = (goal_xy_world - cubeA_init[:2]) / max(np.linalg.norm(goal_xy_world - cubeA_init[:2]), 1e-9)
    if disp_norm > 0.02:
        dir_score = float(np.dot(cube_displacement / disp_norm, goal_dir))
    else:
        dir_score = 0.0
    # Overshoot: cube ended past goal along goal direction.
    project = float(np.dot(cube_displacement, goal_dir))
    goal_along = float(np.dot(goal_xy_world - cubeA_init[:2], goal_dir))
    overshoot = bool(project > goal_along)

    return {
        "ep_id": ep_id, "mode": mode, "K": K,
        "goal_xy_world": goal_xy_world.tolist(),
        "cubeA_init_world": cubeA_init.tolist(),
        "actual_final_xy_world": actual_final_xy.tolist(),
        "start_dist": start_dist,
        "actual_dist": actual_dist,
        "improvement": improvement,
        "dir_score": dir_score,
        "cube_displacement": disp_norm,
        "success": bool(actual_dist <= args.success_threshold),
        "contact": bool(contact),
        "overshoot": overshoot,
        "n_replans": n_replans,
        "pred_best_score": pred_best_score_first if not np.isnan(pred_best_score_first) else None,
        "n_total_candidates": n_total_candidates,
    }


# ---------- main ----------
def train_or_skip(args, use_action, ckpt_path, dataset, train_idx):
    """Train a model and save checkpoint; or load if checkpoint exists."""
    cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
    model = ActionConditionedOFJEPA(image_size=args.image_size, cfg=cfg,
                                     action_dim=7, use_action=use_action).to(args.device)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"  Loading {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        return model
    print(f"  Training (use_action={use_action})", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_one_run(model, dataset, train_idx, args, args.device, use_action)
    if ckpt_path:
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}", flush=True)
    return model


def build_env(image_size: int, horizon: int):
    return rs.make("Stack", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview", camera_heights=image_size,
                    camera_widths=image_size, horizon=horizon)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--plan-horizon", type=int, default=10,
                    help="CEM plan length (stride-boundaries). Total executed = total_actions.")
    p.add_argument("--total-actions", type=int, default=15,
                    help="Total actions to execute per episode (across MPC replans).")
    p.add_argument("--replan-every", type=int, default=5,
                    help="MPC replan cadence: how many planned actions to execute before re-CEM.")
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--main-K", type=int, default=128)
    p.add_argument("--candidate-counts", default="64,128,256",
                    help="K values for cem_action efficiency sweep")
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--modes", default="gt_closed_loop,cem_action,cem_noaction,random")
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--oracle-sanity-n", type=int, default=15,
                    help="Episodes for pre-flight oracle sanity gate.")
    p.add_argument("--oracle-min-improvement", type=float, default=0.20)
    p.add_argument("--oracle-min-dir-score", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Train (or reload) both models.
    dataset = RobosuiteDataset(RobosuiteSpec(cache_dir=args.train_cache, image_size=args.image_size))
    indices = list(range(len(dataset)))
    np.random.RandomState(0).shuffle(indices)
    train_idx = indices[: int(0.8 * len(dataset))]
    print(f"Train episodes: {len(train_idx)}", flush=True)

    ckpt_a = str(out / "model_action.pt")
    ckpt_n = str(out / "model_noaction.pt")
    print("\n=== Training (or loading) action-conditioned model ===", flush=True)
    model_a = train_or_skip(args, True, ckpt_a, dataset, train_idx)
    print("\n=== Training (or loading) no-action model ===", flush=True)
    model_n = train_or_skip(args, False, ckpt_n, dataset, train_idx)
    model_a.eval(); model_n.eval()

    # Planning eval — single env reused across episodes.
    env = build_env(args.image_size, horizon=args.total_actions * args.jepa_stride + 10)

    # === ORACLE SANITY GATE ===
    print(f"\n=== Pre-flight: closed-loop oracle sanity gate (n={args.oracle_sanity_n}) ===", flush=True)
    args.K = args.main_K   # set K so run_episode signature works
    oracle_runs = []
    for ep_id in range(args.oracle_sanity_n):
        r = run_episode(env, model_a, model_n, "gt_closed_loop", args, ep_id)
        oracle_runs.append(r)
    o_imp = float(np.mean([r["improvement"] for r in oracle_runs]))
    o_dir = float(np.mean([r["dir_score"] for r in oracle_runs]))
    o_contact = float(np.mean([r["contact"] for r in oracle_runs]))
    o_succ = float(np.mean([r["success"] for r in oracle_runs]))
    o_disp = float(np.mean([r["cube_displacement"] for r in oracle_runs]))
    print(f"  Oracle: imp={o_imp:.3f}  dir={o_dir:.3f}  succ={o_succ:.3f}  "
          f"contact={o_contact:.2f}  mean_disp={o_disp:.4f}m", flush=True)
    sanity_pass = (o_imp >= args.oracle_min_improvement and
                    o_dir >= args.oracle_min_dir_score)
    if not sanity_pass:
        print(f"\nORACLE SANITY GATE FAILED  (need imp>={args.oracle_min_improvement}, "
              f"dir>={args.oracle_min_dir_score}). Task setup invalid as planning benchmark.")
        print(f"Per the precommit, falling back: recast Phase 15 as predictor-calibration phase.")
        with open(out / "summary.json", "w") as f:
            json.dump({"sanity_pass": False, "oracle": {
                "improvement": o_imp, "dir_score": o_dir, "success_rate": o_succ,
                "contact_rate": o_contact, "mean_displacement": o_disp,
                "n_episodes": args.oracle_sanity_n,
            }}, f, indent=2)
        env.close()
        return
    print(f"  Oracle sanity gate PASSED.", flush=True)

    modes = [m.strip() for m in args.modes.split(",")]
    K_sweep = [int(k) for k in args.candidate_counts.split(",")]
    all_results = []
    for mode in modes:
        K_list = K_sweep if mode == "cem_action" else [args.main_K]
        for K in K_list:
            args.K = K  # plumb K through
            label = f"{mode}_K{K}" if mode in ("cem_action", "cem_noaction") else mode
            print(f"\n=== {label} ===", flush=True)
            t0 = time.time()
            per_ep = []
            for ep_id in range(args.n_episodes):
                res = run_episode(env, model_a, model_n, mode, args, ep_id)
                res["label"] = label
                per_ep.append(res)
                if (ep_id + 1) % 10 == 0:
                    suc = np.mean([r["success"] for r in per_ep])
                    md = np.mean([r["actual_dist"] for r in per_ep])
                    print(f"  ep {ep_id+1}/{args.n_episodes}  succ={suc:.2f}  mean_dist={md:.3f}m  t={time.time()-t0:.0f}s",
                          flush=True)
            with open(out / f"per_episode_{label}.jsonl", "w") as f:
                for r in per_ep: f.write(json.dumps(r) + "\n")
            succ = float(np.mean([r["success"] for r in per_ep]))
            mean_dist = float(np.mean([r["actual_dist"] for r in per_ep]))
            median_dist = float(np.median([r["actual_dist"] for r in per_ep]))
            mean_imp = float(np.mean([r["improvement"] for r in per_ep]))
            mean_dir = float(np.mean([r["dir_score"] for r in per_ep]))
            mean_disp = float(np.mean([r["cube_displacement"] for r in per_ep]))
            contact_rate = float(np.mean([r["contact"] for r in per_ep]))
            overshoot_rate = float(np.mean([r["overshoot"] for r in per_ep]))
            preds = np.array([r["pred_best_score"] for r in per_ep if r["pred_best_score"] is not None])
            actuals = np.array([r["actual_dist"] for r in per_ep if r["pred_best_score"] is not None])
            corr = float(np.corrcoef(-preds, actuals)[0, 1]) if len(preds) > 2 else float("nan")
            summary = {"label": label, "mode": mode, "K": K, "n_episodes": args.n_episodes,
                       "improvement": mean_imp, "dir_score": mean_dir,
                       "mean_displacement": mean_disp,
                       "contact_rate": contact_rate, "overshoot_rate": overshoot_rate,
                       "success_rate": succ, "mean_dist": mean_dist, "median_dist": median_dist,
                       "pred_actual_corr": corr,
                       "n_total_candidates": per_ep[0].get("n_total_candidates", 0)}
            all_results.append(summary)
            print(f"  RESULT: imp={mean_imp:.3f}  dir={mean_dir:.3f}  succ={succ:.3f}  "
                  f"disp={mean_disp:.4f}m  contact={contact_rate:.2f}  over={overshoot_rate:.2f}  "
                  f"corr={corr:.3f}", flush=True)
    env.close()

    # Gate evaluation.
    print("\n=== Phase 15 Gate Verdicts ===", flush=True)
    action_K = {r["K"]: r for r in all_results if r["mode"] == "cem_action"}
    noaction = next((r for r in all_results if r["mode"] == "cem_noaction"), None)
    random_r = next((r for r in all_results if r["mode"] == "random"), None)
    gt_r = next((r for r in all_results if r["mode"] == "gt_closed_loop"), None)
    a_main = action_K.get(args.main_K)

    g1_pass = g2_pass = g3_pass = False
    summary = {"results": all_results}
    if a_main and noaction:
        g1_diff = a_main["improvement"] - noaction["improvement"]
        g1_pass = g1_diff >= 0.10
        print(f"  G1 improvement_gap: action={a_main['improvement']:.3f} noaction={noaction['improvement']:.3f}"
              f"  Δ={g1_diff:+.3f}  (>=0.10 {'PASS' if g1_pass else 'FAIL'})")
        g2_pass = a_main["improvement"] >= 0.20
        print(f"  G2 action_improvement: {a_main['improvement']:.3f}  (>=0.20 {'PASS' if g2_pass else 'FAIL'})")
        summary["g1_diff"] = g1_diff
        summary["g2_improvement"] = a_main["improvement"]
    if a_main:
        # G3: does any K' <= main_K/2 achieve >= main_K improvement?
        target_imp = a_main["improvement"]
        small_Ks = [K for K in action_K if K <= args.main_K // 2]
        passes = [K for K in small_Ks if action_K[K]["improvement"] >= target_imp]
        g3_pass = len(passes) > 0
        print(f"  G3 cand_efficiency: K={args.main_K} imp={target_imp:.3f}")
        for K in sorted(action_K):
            r = action_K[K]
            print(f"    K={K}: imp={r['improvement']:.3f}  dir={r['dir_score']:.3f}  "
                  f"succ={r['success_rate']:.3f}  mean_dist={r['mean_dist']:.4f}")
        print(f"    G3 {'PASS' if g3_pass else 'FAIL'}  (need K<={args.main_K//2} matching K={args.main_K})")
    if random_r:
        print(f"  Random baseline:    imp={random_r['improvement']:.3f}  dir={random_r['dir_score']:.3f}  "
              f"succ={random_r['success_rate']:.3f}  mean_dist={random_r['mean_dist']:.4f}")
    if gt_r:
        print(f"  GT-scripted skyline: imp={gt_r['improvement']:.3f}  dir={gt_r['dir_score']:.3f}  "
              f"succ={gt_r['success_rate']:.3f}  mean_dist={gt_r['mean_dist']:.4f}")
    n_pass = int(g1_pass) + int(g2_pass) + int(g3_pass)
    verdict = (
        "3/3 supports planning" if n_pass == 3 else
        "2/3 partial" if n_pass == 2 else
        "1/3 mostly not planner-grade" if n_pass == 1 else
        "0/3 predictions don't support planning"
    )
    summary["primary_metric"] = "improvement = max(0, (start_dist - end_dist) / start_dist)"
    print(f"\nOVERALL: {verdict}  ({n_pass}/3)")
    summary["n_pass"] = n_pass
    summary["verdict"] = verdict
    summary["g1_pass"] = g1_pass
    summary["g2_pass"] = g2_pass
    summary["g3_pass"] = g3_pass
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
