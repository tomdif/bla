"""Phase 18θ — Slot-feature value head.

Phase 18η-multi locked combined_sum (predictor + value head) as the
new BLA System-1 planning recipe, with the value head reading 10-dim
hand-engineered geometric features. Phase 18θ tests whether OF-JEPA's
learned slot/object-file features carry richer episode-level signal:
swap the value head's state input from geo → slot → geo+slot, and
compare combined_sum results.

Pipeline:
  1. COLLECT: scripted_prior_light_cem rollouts; save BOTH geo and
     slot features per replan boundary.
  2. TRAIN: three value heads on the same 720 train / 180 val split.
     - value_head_geo:     state = 10-dim BC features
     - value_head_slot:    state = OF-JEPA slot features (n_slots*slot_dim)
     - value_head_geoslot: state = concat(geo, slot)
  3. EVAL: 6 modes × n_eval_episodes seed-0:
     - gt_closed_loop          (oracle reference)
     - phase17_locked          (predictor only — Phase 18η baseline)
     - combined_sum_geo        (Phase 18η recipe — within-seed reference)
     - combined_sum_slot       (slot value head + predictor)
     - combined_sum_geoslot    (geo+slot value head + predictor)
     - naive_cem               (floor)

Gates per PHASE_18T_SLOT_VALUE_HEAD_PRECOMMIT.md.
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
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


# Lazy imports
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


def slot_features_at_obs(model, obs) -> np.ndarray:
    """Encode current frame -> flattened slot features [n_slots * slot_dim]."""
    slot = encode_frame(model, obs["agentview_image"])
    return slot.flatten().detach().cpu().numpy().astype(np.float32)


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


# ---------- Collection ----------
def collect_dataset(env, model, args):
    """Run scripted_prior_light_cem; record geo + slot features per replan."""
    geo_features = []
    slot_features = []
    goals = []
    plans = []
    labels = []
    t0 = time.time()
    for ep in range(args.rollout_episodes):
        obs = env.reset()
        goal_xy_world, start_dist = sample_goal(obs, ep, args)
        actions_executed = 0
        per_replan_records = []
        while actions_executed < args.total_actions:
            geo_feat = state_features(obs, goal_xy_world)
            slot_feat = slot_features_at_obs(model, obs)
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
            per_replan_records.append((geo_feat.astype(np.float32),
                                          slot_feat,
                                          np.array(goal_xy_world,
                                                     dtype=np.float32),
                                          plan.astype(np.float32)))
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            contact = [False]
            for action in plan[:n_exec]:
                obs = step_and_track(env, obs, action, args, contact)
            actions_executed += n_exec

        end_xy = obs["cubeA_pos"][:2]
        end_dist = float(np.linalg.norm(end_xy - goal_xy_world))
        episode_imp = max(0.0, (start_dist - end_dist) / max(start_dist, 1e-9))
        for geo, slot, gxy, p in per_replan_records:
            geo_features.append(geo)
            slot_features.append(slot)
            goals.append(gxy)
            plans.append(p)
            labels.append(episode_imp)

        if (ep + 1) % max(args.rollout_log_every, 1) == 0:
            print(json.dumps({"event": "rollout_progress",
                               "episodes": ep + 1,
                               "samples": len(geo_features),
                               "running_mean_imp": float(np.mean(labels)),
                               "elapsed_s": round(time.time() - t0, 1)}),
                   flush=True)

    return (np.stack(geo_features), np.stack(slot_features),
             np.stack(goals), np.stack(plans),
             np.array(labels, dtype=np.float32))


def load_or_collect(env, model, args, out: Path):
    cache = Path(args.rollout_cache) if args.rollout_cache \
            else out / "rollout_cache.npz"
    if cache.exists() and not args.rebuild_rollouts:
        print(json.dumps({"event": "loading_rollout_cache",
                           "path": str(cache)}), flush=True)
        d = np.load(cache)
        return (d["geo_features"], d["slot_features"], d["goals"],
                 d["plans"], d["labels"], cache)
    print(json.dumps({"event": "collecting_rollouts",
                       "n_episodes": args.rollout_episodes}), flush=True)
    geo, slot, g, p, l = collect_dataset(env, model, args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, geo_features=geo, slot_features=slot,
                          goals=g, plans=p, labels=l)
    print(json.dumps({"event": "saved_rollout_cache", "path": str(cache),
                       "n_samples": len(geo),
                       "mean_episode_imp": float(np.mean(l)),
                       "geo_dim": int(geo.shape[1]),
                       "slot_dim": int(slot.shape[1])}), flush=True)
    return geo, slot, g, p, l, cache


# ---------- Training: three heads ----------
def train_one_head(state_train: np.ndarray, goals: np.ndarray,
                    plans: np.ndarray, labels: np.ndarray,
                    args, label: str, out: Path):
    state_dim = state_train.shape[1]
    head = GoalProgressValueHead(
        state_dim=state_dim, action_dim=plans.shape[2],
        plan_horizon=plans.shape[1], hidden=args.hidden,
        n_hidden=args.n_hidden, dropout=args.dropout,
    ).to(args.device)
    stats = train_value_head_supervised(
        head,
        torch.from_numpy(state_train), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.train_steps, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.train_log_every,
    )
    ckpt_path = out / f"value_head_{label}.pt"
    torch.save({"state_dict": head.state_dict(),
                "config": value_head_config(head),
                "stats": stats.__dict__, "label": label}, ckpt_path)
    print(json.dumps({"event": "saved_value_head", "label": label,
                       "path": str(ckpt_path), **stats.__dict__}),
           flush=True)
    head.eval()
    return head, ckpt_path, stats.__dict__


def train_three_heads(geo, slot, goals, plans, labels, args, out: Path):
    """Train geo, slot, geo+slot heads. Returns dict by label."""
    print(json.dumps({"event": "training_heads",
                       "geo_dim": int(geo.shape[1]),
                       "slot_dim": int(slot.shape[1])}), flush=True)
    heads = {}
    geoslot = np.concatenate([geo, slot], axis=1)
    for label, state_train in [("geo", geo), ("slot", slot),
                                  ("geoslot", geoslot)]:
        head, ck, stats = train_one_head(state_train, goals, plans, labels,
                                           args, label, out)
        heads[label] = {"head": head, "ckpt": ck, "stats": stats}
    return heads


# ---------- Scoring ----------
@torch.no_grad()
def value_score(head, state, goal_xy, action_seq, device):
    s = torch.from_numpy(state).to(device).unsqueeze(0).float()
    g = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    a = torch.from_numpy(action_seq).to(device).unsqueeze(0).float()
    return float(head(s, g, a).cpu().item())


def build_combined_score_fn(env, model, head, head_input: str,
                              obs, goal_xy_world, lam, args):
    """head_input ∈ {geo, slot, geoslot, none}. Returns score_fn(seq)."""
    geo_state = state_features(obs, goal_xy_world)
    slot_state = slot_features_at_obs(model, obs) if head_input != "geo" else None
    if head_input == "geo":
        head_state = geo_state
    elif head_input == "slot":
        head_state = slot_state
    elif head_input == "geoslot":
        head_state = np.concatenate([geo_state, slot_state])
    elif head_input == "none":
        head_state = None
    else:
        raise ValueError(head_input)
    goal_xy = np.array(goal_xy_world, dtype=np.float32)
    pred_fn = build_score_fn(env, model, obs, goal_xy_world)

    def fn(seq):
        p = pred_fn(seq)
        if head_input == "none" or head is None:
            return p
        v = value_score(head, head_state, goal_xy,
                         seq.astype(np.float32), args.device)
        return lam * p + (1.0 - lam) * v
    return fn


def build_value_only_fn(model, head, head_input: str, obs, goal_xy_world, args):
    """Value-head-only score; no predictor."""
    geo_state = state_features(obs, goal_xy_world)
    slot_state = slot_features_at_obs(model, obs) if head_input != "geo" else None
    if head_input == "geo":
        head_state = geo_state
    elif head_input == "slot":
        head_state = slot_state
    elif head_input == "geoslot":
        head_state = np.concatenate([geo_state, slot_state])
    else:
        raise ValueError(head_input)
    goal_xy = np.array(goal_xy_world, dtype=np.float32)

    def fn(seq):
        return value_score(head, head_state, goal_xy,
                            seq.astype(np.float32), args.device)
    return fn


# ---------- Episode runner ----------
def run_episode(env, model, heads, mode: str, args, ep_id: int):
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
        # phase17_locked, combined_sum_{geo,slot,geoslot},
        # value_only_{slot,geoslot}
        mode_cfg = {
            "phase17_locked":           ("none",     None),
            "combined_sum_geo":         ("geo",      "geo"),
            "combined_sum_slot":        ("slot",     "slot"),
            "combined_sum_geoslot":     ("geoslot",  "geoslot"),
            "value_only_slot":          ("value_only", "slot"),
            "value_only_geoslot":       ("value_only", "geoslot"),
        }
        scoring, head_label = mode_cfg[mode]
        head = heads[head_label]["head"] if head_label else None
        while actions_executed < args.total_actions:
            if scoring == "value_only":
                score_fn = build_value_only_fn(model, head, head_label,
                                                  obs, goal_xy_world, args)
            else:
                score_fn = build_combined_score_fn(env, model, head, scoring,
                                                      obs, goal_xy_world,
                                                      args.combined_lambda, args)
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


# ---------- Decile diagnostic for each trained head ----------
def compute_decile_diagnostic(head, state_eval, goals, plans, labels,
                                 args, label: str):
    n = len(state_eval)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(args.val_split * n))
    val_idx = perm[:n_val]
    head.eval()
    device = next(head.parameters()).device
    with torch.no_grad():
        pred = head(
            torch.from_numpy(state_eval[val_idx]).to(device).float(),
            torch.from_numpy(goals[val_idx]).to(device).float(),
            torch.from_numpy(plans[val_idx]).to(device).float(),
        ).cpu().numpy()
    actual = labels[val_idx]
    pearson = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-9 else float("nan")
    prank = np.argsort(np.argsort(pred)).astype(float)
    arank = np.argsort(np.argsort(actual)).astype(float)
    spearman = float(np.corrcoef(prank, arank)[0, 1])
    order = np.argsort(pred)[::-1]
    chunks = np.array_split(order, 10)
    deciles = [{"decile": i, "mean_pred": float(pred[c].mean()),
                  "mean_actual": float(actual[c].mean()), "n": int(len(c))}
                 for i, c in enumerate(chunks)]
    return {"label": label, "pearson": pearson, "spearman": spearman,
             "n_val": int(len(val_idx)), "deciles": deciles,
             "top_vs_bot_gap": deciles[0]["mean_actual"]
                                 - deciles[-1]["mean_actual"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout-cache", default=None)
    p.add_argument("--rebuild-rollouts", action="store_true")
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
    p.add_argument("--combined-lambda", type=float, default=0.5)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--modes", default="gt_closed_loop,phase17_locked,"
                    "combined_sum_geo,combined_sum_slot,"
                    "combined_sum_geoslot,naive_cem")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = load_action_model(args)

    env = build_env(args.image_size,
                     horizon=args.total_actions * args.jepa_stride
                             + 3 * args.plan_horizon * args.jepa_stride + 60)

    # 1. Collect rollouts
    geo, slot, goals, plans, labels, rollout_cache = \
        load_or_collect(env, model, args, out)

    # 2. Train three heads
    heads = train_three_heads(geo, slot, goals, plans, labels, args, out)

    # 3. Decile diagnostics
    geoslot = np.concatenate([geo, slot], axis=1)
    deciles = {
        "geo": compute_decile_diagnostic(heads["geo"]["head"], geo, goals,
                                            plans, labels, args, "geo"),
        "slot": compute_decile_diagnostic(heads["slot"]["head"], slot, goals,
                                             plans, labels, args, "slot"),
        "geoslot": compute_decile_diagnostic(heads["geoslot"]["head"], geoslot,
                                                goals, plans, labels, args,
                                                "geoslot"),
    }
    for lbl, d in deciles.items():
        print(json.dumps({"event": "decile_diagnostic", "label": lbl,
                           "pearson": d["pearson"], "spearman": d["spearman"],
                           "top_decile_actual": d["deciles"][0]["mean_actual"],
                           "bot_decile_actual": d["deciles"][-1]["mean_actual"],
                           "top_vs_bot_gap": d["top_vs_bot_gap"]}),
               flush=True)

    # 4. Eval
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for m in modes:
        print(json.dumps({"event": "eval_mode", "mode": m}), flush=True)
        per_ep = []
        t0 = time.time()
        for ep in range(args.n_eval_episodes):
            per_ep.append(run_episode(env, model, heads, m, args, ep))
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

    # 5. Gates
    by_mode = {r["mode"]: r for r in all_results}
    locked = by_mode.get("phase17_locked", {}).get("improvement", float("nan"))
    cs_geo = by_mode.get("combined_sum_geo", {}).get("improvement", float("nan"))
    cs_slot = by_mode.get("combined_sum_slot", {}).get("improvement", float("nan"))
    cs_geoslot = by_mode.get("combined_sum_geoslot", {}).get("improvement", float("nan"))
    slot_spearman = deciles["slot"]["spearman"]
    slot_top = deciles["slot"]["deciles"][0]["mean_actual"]
    slot_bot = deciles["slot"]["deciles"][-1]["mean_actual"]

    gates = {
        "locked_improvement": locked,
        "cs_geo_improvement": cs_geo,
        "cs_slot_improvement": cs_slot,
        "cs_geoslot_improvement": cs_geoslot,
        "g1_slot_beats_locked_by_0_02": cs_slot >= locked + 0.02,
        "g2_geoslot_beats_geo": cs_geoslot >= cs_geo,
        "g3_slot_spearman_geq_0_20": slot_spearman >= 0.20,
        "g4_slot_top_geq_2x_bot": slot_top >= 2.0 * max(slot_bot, 1e-9),
        "slot_spearman": slot_spearman,
        "slot_top_decile_actual": slot_top,
        "slot_bot_decile_actual": slot_bot,
    }
    n_pass = sum([gates["g1_slot_beats_locked_by_0_02"],
                   gates["g2_geoslot_beats_geo"],
                   gates["g3_slot_spearman_geq_0_20"],
                   gates["g4_slot_top_geq_2x_bot"]])
    gates["n_pass"] = n_pass
    gates["verdict"] = (
        "4/4 — slot features are directly planner-valuable AND additive"
        if n_pass == 4 else
        "3/4 — slot features are planner-valuable; some weakness elsewhere"
        if n_pass == 3 else
        "1-2/4 — partial signal; investigate"
        if n_pass >= 1 else
        "0/4 — slot-feature value head does not help"
    )

    report = {"args": vars(args),
               "rollout_cache": str(rollout_cache),
               "head_paths": {k: str(v["ckpt"]) for k, v in heads.items()},
               "head_stats": {k: v["stats"] for k, v in heads.items()},
               "deciles": deciles,
               "results": all_results, "gates": gates}
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"event": "done", **gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
