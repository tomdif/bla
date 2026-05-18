"""Phase 18κ Regime 3 — Lift fine-tune full pipeline.

Per seed:
  1. Auto-collect ~200 Lift rollouts with scripted_lift + light CEM
  2. Train 4 value heads on Lift cache:
       geo, supervised, end2end, pretrain+ft
  3. Decile diagnostic per head on Lift held-out (including z-feature
     recovery for G4)
  4. Eval 5 modes (locked, geo, supervised, end2end, pretrain+ft) +
     oracle (closed_loop_gt_lift_step) + naive_cem, 30 ep each

Modes are evaluated on Lift task (cube_z gain / target).

Reuses Phase 18ν training routines (geo head, supervised adapter+VH,
end2end, pretrain+ft) with task-swapped collection.
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
    GoalProgressValueHead, value_head_config,
    train_value_head_supervised,
)
from system1_jepa.geometry_adapter import (
    ObjectFileGeometryAdapter, train_adapter_supervised,
    adapter_config, End2EndAdapterValue,
    train_end2end_supervised, train_end2end_with_aux_schedule,
    end2end_config,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA
from scripts.phase18k_r3_lift import (
    build_env_lift, sample_lift_goal, state_features_lift,
    lift_improvement, find_cube_slot_lift, scripted_lift_action,
    rollout_scripted_lift_prior, closed_loop_gt_lift_step,
    TARGET_LIFT_HEIGHT,
)


# Lazy
encode_frame = None; predict_score_seq = None; cem_with_prior = None


def load_planning_deps():
    global encode_frame, predict_score_seq, cem_with_prior
    from scripts.phase15_planning import encode_frame as _e, predict_score_seq as _p
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
    return np.clip((p_xy + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def slot_features_at_obs(model, obs) -> np.ndarray:
    slot = encode_frame(model, obs["agentview_image"])
    return slot.flatten().detach().cpu().numpy().astype(np.float32)


def step_and_track(env, obs, action, args):
    for _ in range(args.jepa_stride):
        obs, _, _, _ = env.step(action)
    return obs


def build_score_fn_lift(env, model, obs, goal_xy_world):
    init_slot = encode_frame(model, obs["agentview_image"])
    cube_idx = find_cube_slot_lift(
        model, init_slot,
        norm_xy(obs["cube_pos"][:2]),
        norm_xy(obs["robot0_eef_pos"][:2]),
    )
    goal_xy_norm = norm_xy(goal_xy_world)
    return lambda seq: predict_score_seq(
        model, init_slot, seq, cube_idx, goal_xy_norm, use_action=True,
    )


# ---------- Collection ----------
def collect_lift_dataset(env, model, args):
    """Run scripted_lift_prior + light CEM episodes; record per-replan
    (geo, slot, goal, plan, episode_imp) tuples on Lift task."""
    geos = []; slots = []; goals = []; plans = []; labels = []
    t0 = time.time()
    for ep in range(args.rollout_episodes):
        obs = env.reset()
        goal_xy, target_h = sample_lift_goal(obs, ep)
        cube_z_start = float(obs["cube_pos"][2])
        actions_executed = 0
        per_replan = []
        while actions_executed < args.total_actions:
            geo_feat = state_features_lift(obs, goal_xy)
            slot_feat = slot_features_at_obs(model, obs)
            score_fn = build_score_fn_lift(env, model, obs, goal_xy)
            mu = rollout_scripted_lift_prior(env, obs, goal_xy,
                                               args.plan_horizon, args.jepa_stride)
            plan, _ = cem_with_prior(
                score_fn, mu, env.action_dim,
                args.plan_horizon, args.train_cem_iters, args.train_K,
                args.elite_frac, sigma=args.train_sigma,
                sigma_floor=args.sigma_floor,
            )
            if plan is None: plan = mu
            per_replan.append((geo_feat, slot_feat, goal_xy.astype(np.float32),
                                  plan.astype(np.float32)))
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for action in plan[:n_exec]:
                obs = step_and_track(env, obs, action, args)
            actions_executed += n_exec
        cube_z_end = float(obs["cube_pos"][2])
        ep_imp = lift_improvement(cube_z_start, cube_z_end, target_h)
        for g, s, gxy, p in per_replan:
            geos.append(g); slots.append(s); goals.append(gxy)
            plans.append(p); labels.append(ep_imp)
        if (ep + 1) % args.rollout_log_every == 0:
            print(json.dumps({"event": "rollout_progress",
                               "episodes": ep + 1,
                               "samples": len(geos),
                               "running_mean_imp": float(np.mean(labels)),
                               "elapsed_s": round(time.time() - t0, 1)}),
                   flush=True)
    return (np.stack(geos), np.stack(slots), np.stack(goals),
             np.stack(plans), np.array(labels, dtype=np.float32))


# ---------- Training routines (same as phase18nu) ----------
def train_geo_head(geo, goals, plans, labels, args):
    head = GoalProgressValueHead(
        state_dim=int(geo.shape[1]), action_dim=plans.shape[2],
        plan_horizon=plans.shape[1], hidden=args.vh_hidden,
        n_hidden=args.vh_n_hidden, dropout=args.vh_dropout,
    ).to(args.device)
    stats = train_value_head_supervised(
        head, torch.from_numpy(geo), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.vh_train_steps, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.vh_log_every,
    )
    head.eval()
    return head, stats


def train_supervised_adapter_and_head(slots, geo, goals, plans, labels, args):
    adapter = ObjectFileGeometryAdapter(
        slot_dim=int(slots.shape[1]), goal_dim=int(goals.shape[1]),
        out_dim=int(geo.shape[1]),
        hidden=args.adapter_hidden, n_hidden=args.adapter_n_hidden,
    ).to(args.device)
    ad_stats = train_adapter_supervised(
        adapter, torch.from_numpy(slots), torch.from_numpy(goals),
        torch.from_numpy(geo),
        steps=args.adapter_train_steps, batch_size=args.adapter_batch_size,
        lr=args.adapter_lr, weight_decay=args.adapter_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.adapter_log_every,
    )
    adapter.eval()
    device = next(adapter.parameters()).device
    with torch.no_grad():
        adapter_geo = adapter(torch.from_numpy(slots).to(device).float(),
                                torch.from_numpy(goals).to(device).float()
                                ).cpu().numpy()
    head = GoalProgressValueHead(
        state_dim=int(adapter_geo.shape[1]),
        action_dim=plans.shape[2], plan_horizon=plans.shape[1],
        hidden=args.vh_hidden, n_hidden=args.vh_n_hidden,
    ).to(args.device)
    vh_stats = train_value_head_supervised(
        head, torch.from_numpy(adapter_geo), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.vh_train_steps, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.vh_log_every,
    )
    head.eval()
    return adapter, head, ad_stats, vh_stats, adapter_geo


def make_end2end(slots, plans, args):
    return End2EndAdapterValue(
        slot_dim=int(slots.shape[1]), goal_dim=2,
        latent_dim=args.end2end_latent_dim,
        action_dim=plans.shape[2], plan_horizon=plans.shape[1],
        adapter_hidden=args.adapter_hidden,
        adapter_n_hidden=args.adapter_n_hidden,
        value_hidden=args.vh_hidden, value_n_hidden=args.vh_n_hidden,
    ).to(args.device)


def train_end2end_pure(slots, goals, plans, labels, args):
    model = make_end2end(slots, plans, args)
    stats = train_end2end_supervised(
        model, torch.from_numpy(slots), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.end2end_train_steps, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        val_split=args.val_split, seed=args.seed,
    )
    model.eval()
    return model, stats


def train_pretrain_ft(slots, geo, goals, plans, labels, args):
    model = make_end2end(slots, plans, args)
    stats = train_end2end_with_aux_schedule(
        model, torch.from_numpy(slots), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        torch.from_numpy(geo),
        schedule_kind="pretrain_ft",
        steps=args.end2end_train_steps,
        pretrain_steps=args.pretrain_steps,
        aux_residual_weight=args.aux_residual_weight,
        val_split=args.val_split, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        seed=args.seed,
    )
    model.eval()
    return model, stats


# ---------- Scoring ----------
@torch.no_grad()
def value_score_head(head, state, goal_xy, action_seq, device):
    s = torch.from_numpy(state).to(device).unsqueeze(0).float()
    g = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    a = torch.from_numpy(action_seq).to(device).unsqueeze(0).float()
    return float(head(s, g, a).cpu().item())


@torch.no_grad()
def value_score_end2end(model, slot, goal_xy, action_seq, device):
    s = torch.from_numpy(slot).to(device).unsqueeze(0).float()
    g = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    a = torch.from_numpy(action_seq).to(device).unsqueeze(0).float()
    return float(model(s, g, a).cpu().item())


def build_combined_score_fn(env, model_action, head_or_model, head_kind: str,
                              adapter, obs, goal_xy_world, lam, args):
    goal_xy = np.array(goal_xy_world, dtype=np.float32)
    pred_fn = build_score_fn_lift(env, model_action, obs, goal_xy_world)
    if head_kind == "none":
        return lambda seq: pred_fn(seq)
    if head_kind == "geo":
        head_state = state_features_lift(obs, goal_xy_world)
        def fn(seq):
            p = pred_fn(seq)
            v = value_score_head(head_or_model, head_state, goal_xy,
                                    seq.astype(np.float32), args.device)
            return lam * p + (1.0 - lam) * v
        return fn
    if head_kind == "supervised_adapter":
        slot_feat = slot_features_at_obs(model_action, obs)
        device = args.device
        with torch.no_grad():
            head_state = adapter(
                torch.from_numpy(slot_feat).to(device).unsqueeze(0).float(),
                torch.from_numpy(goal_xy).to(device).unsqueeze(0).float(),
            )[0].cpu().numpy().astype(np.float32)
        def fn(seq):
            p = pred_fn(seq)
            v = value_score_head(head_or_model, head_state, goal_xy,
                                    seq.astype(np.float32), args.device)
            return lam * p + (1.0 - lam) * v
        return fn
    if head_kind in ("end2end", "pretrain_ft"):
        slot_feat = slot_features_at_obs(model_action, obs)
        def fn(seq):
            p = pred_fn(seq)
            v = value_score_end2end(head_or_model, slot_feat, goal_xy,
                                       seq.astype(np.float32), args.device)
            return lam * p + (1.0 - lam) * v
        return fn
    raise ValueError(head_kind)


# ---------- Episode runner ----------
def run_episode(env, model_action, heads, adapters, mode, args, ep_id):
    obs = env.reset()
    cube_z_start = float(obs["cube_pos"][2])
    goal_xy_world, target_h = sample_lift_goal(obs, ep_id)

    if mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            action = closed_loop_gt_lift_step(env, obs, goal_xy_world)
            obs = step_and_track(env, obs, action, args)
    elif mode == "naive_cem":
        actions_executed = 0
        while actions_executed < args.total_actions:
            score_fn = build_score_fn_lift(env, model_action, obs, goal_xy_world)
            mu = np.zeros((args.plan_horizon, env.action_dim), dtype=np.float32)
            plan, _ = cem_with_prior(score_fn, mu, env.action_dim,
                                       args.plan_horizon, args.eval_cem_iters,
                                       args.eval_K, args.elite_frac,
                                       sigma=args.naive_sigma,
                                       sigma_floor=args.sigma_floor)
            if plan is None: plan = mu
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                obs = step_and_track(env, obs, a, args)
            actions_executed += n_exec
    else:
        mode_cfg = {
            "phase17_locked":            ("none",                None,           None),
            "combined_sum_geo":          ("geo",                 "geo",          None),
            "combined_sum_supervised":   ("supervised_adapter",  "supervised",   "supervised"),
            "combined_sum_end2end":      ("end2end",             "end2end",      None),
            "combined_sum_pretrain_ft":  ("pretrain_ft",         "pretrain_ft",  None),
        }
        head_kind, head_label, adapter_label = mode_cfg[mode]
        head_or_model = heads[head_label] if head_label else None
        adapter = adapters.get(adapter_label) if adapter_label else None
        actions_executed = 0
        while actions_executed < args.total_actions:
            score_fn = build_combined_score_fn(env, model_action, head_or_model,
                                                  head_kind, adapter,
                                                  obs, goal_xy_world,
                                                  args.combined_lambda, args)
            mu = rollout_scripted_lift_prior(env, obs, goal_xy_world,
                                                args.plan_horizon, args.jepa_stride)
            plan, _ = cem_with_prior(score_fn, mu, env.action_dim,
                                       args.plan_horizon, args.eval_cem_iters,
                                       args.eval_K, args.elite_frac,
                                       sigma=args.eval_sigma,
                                       sigma_floor=args.sigma_floor)
            if plan is None: plan = mu
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                obs = step_and_track(env, obs, a, args)
            actions_executed += n_exec

    cube_z_end = float(obs["cube_pos"][2])
    improvement = lift_improvement(cube_z_start, cube_z_end, target_h)
    success = cube_z_end - cube_z_start >= target_h
    return {"ep_id": ep_id, "mode": mode,
             "cube_z_start": cube_z_start, "cube_z_end": cube_z_end,
             "z_gain": float(cube_z_end - cube_z_start),
             "improvement": improvement, "success": bool(success)}


def summarize_mode(mode, per_ep):
    return {"mode": mode, "n_episodes": len(per_ep),
             "improvement": float(np.mean([r["improvement"] for r in per_ep])),
             "z_gain": float(np.mean([r["z_gain"] for r in per_ep])),
             "success_rate": float(np.mean([r["success"] for r in per_ep]))}


# ---------- Decile + height-Spearman diagnostics ----------
def decile_on_state(head, state, goals, plans, labels, args):
    n = len(state)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(args.val_split * n))
    val_idx = perm[:n_val]
    device = next(head.parameters()).device
    with torch.no_grad():
        pred = head(
            torch.from_numpy(state[val_idx]).to(device).float(),
            torch.from_numpy(goals[val_idx]).to(device).float(),
            torch.from_numpy(plans[val_idx]).to(device).float(),
        ).cpu().numpy()
    actual = labels[val_idx]
    if pred.std() < 1e-9:
        return {"spearman": float("nan"), "top_vs_bot_gap": float("nan")}
    prk = np.argsort(np.argsort(pred)).astype(float)
    ark = np.argsort(np.argsort(actual)).astype(float)
    spearman = float(np.corrcoef(prk, ark)[0, 1])
    order = np.argsort(pred)[::-1]
    chunks = np.array_split(order, 10)
    top = float(actual[chunks[0]].mean())
    bot = float(actual[chunks[-1]].mean())
    return {"spearman": spearman, "top_vs_bot_gap": top - bot}


def decile_for_end2end(model, slots, goals, plans, labels, args):
    n = len(slots)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(args.val_split * n))
    val_idx = perm[:n_val]
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(
            torch.from_numpy(slots[val_idx]).to(device).float(),
            torch.from_numpy(goals[val_idx]).to(device).float(),
            torch.from_numpy(plans[val_idx]).to(device).float(),
        ).cpu().numpy()
    actual = labels[val_idx]
    if pred.std() < 1e-9:
        return {"spearman": float("nan"), "top_vs_bot_gap": float("nan")}
    prk = np.argsort(np.argsort(pred)).astype(float)
    ark = np.argsort(np.argsort(actual)).astype(float)
    spearman = float(np.corrcoef(prk, ark)[0, 1])
    order = np.argsort(pred)[::-1]
    chunks = np.array_split(order, 10)
    top = float(actual[chunks[0]].mean())
    bot = float(actual[chunks[-1]].mean())
    return {"spearman": spearman, "top_vs_bot_gap": top - bot}


def height_feature_spearman(adapter, slots, goals, geo, args):
    """G4 diagnostic: how well does the supervised adapter recover
    cube_z (feat 5) and eef_z (feat 4) on held-out Lift data?"""
    n = len(slots)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(args.val_split * n))
    val_idx = perm[:n_val]
    device = next(adapter.parameters()).device
    with torch.no_grad():
        pred_geo = adapter(
            torch.from_numpy(slots[val_idx]).to(device).float(),
            torch.from_numpy(goals[val_idx]).to(device).float(),
        ).cpu().numpy()
    true_geo = geo[val_idx]
    # Feature 4 = eef_z; feature 5 = cube_z
    def feature_spearman(j):
        p = pred_geo[:, j]; t = true_geo[:, j]
        if p.std() < 1e-9 or t.std() < 1e-9:
            return float("nan")
        prk = np.argsort(np.argsort(p)).astype(float)
        trk = np.argsort(np.argsort(t)).astype(float)
        return float(np.corrcoef(prk, trk)[0, 1])
    return {"eef_z_spearman": feature_spearman(4),
             "cube_z_spearman": feature_spearman(5)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout-cache", default=None,
                    help="Optional pre-collected cache; else auto-collect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=10)
    p.add_argument("--total-actions", type=int, default=20)  # Lift needs more steps
    p.add_argument("--replan-every", type=int, default=5)
    p.add_argument("--rollout-episodes", type=int, default=200)
    p.add_argument("--rollout-log-every", type=int, default=20)
    p.add_argument("--train-K", type=int, default=32)
    p.add_argument("--train-cem-iters", type=int, default=1)
    p.add_argument("--train-sigma", type=float, default=0.12)
    # Adapter
    p.add_argument("--adapter-hidden", type=int, default=256)
    p.add_argument("--adapter-n-hidden", type=int, default=3)
    p.add_argument("--adapter-train-steps", type=int, default=2000)
    p.add_argument("--adapter-batch-size", type=int, default=64)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-log-every", type=int, default=500)
    # VH + end2end
    p.add_argument("--vh-hidden", type=int, default=256)
    p.add_argument("--vh-n-hidden", type=int, default=3)
    p.add_argument("--vh-dropout", type=float, default=0.0)
    p.add_argument("--vh-train-steps", type=int, default=2000)
    p.add_argument("--vh-batch-size", type=int, default=64)
    p.add_argument("--vh-lr", type=float, default=3e-4)
    p.add_argument("--vh-weight-decay", type=float, default=1e-4)
    p.add_argument("--vh-log-every", type=int, default=500)
    p.add_argument("--end2end-train-steps", type=int, default=2000)
    p.add_argument("--end2end-latent-dim", type=int, default=10)
    p.add_argument("--pretrain-steps", type=int, default=1000)
    p.add_argument("--aux-residual-weight", type=float, default=0.05)
    p.add_argument("--val-split", type=float, default=0.2)
    # Eval
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--eval-K", type=int, default=32)
    p.add_argument("--eval-cem-iters", type=int, default=1)
    p.add_argument("--eval-sigma", type=float, default=0.12)
    p.add_argument("--naive-sigma", type=float, default=0.5)
    p.add_argument("--combined-lambda", type=float, default=0.5)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--modes", default="gt_closed_loop,phase17_locked,"
                    "combined_sum_geo,combined_sum_supervised,"
                    "combined_sum_end2end,combined_sum_pretrain_ft,naive_cem")
    args = p.parse_args()

    load_planning_deps()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. Load model + env, COLLECT or load cache
    model = load_action_model(args)
    env_horizon = (args.total_actions + 3 * args.plan_horizon) * args.jepa_stride + 60
    env = build_env_lift(args.image_size, env_horizon)

    cache_path = Path(args.rollout_cache) if args.rollout_cache else (out / "rollout_cache_lift.npz")
    if cache_path.exists():
        print(json.dumps({"event": "loading_cache", "path": str(cache_path)}), flush=True)
        d = np.load(cache_path)
        geo = d["geo_features"]; slot = d["slot_features"]
        goals = d["goals"]; plans = d["plans"]; labels = d["labels"]
    else:
        print(json.dumps({"event": "collecting_lift_rollouts",
                           "n_episodes": args.rollout_episodes}), flush=True)
        geo, slot, goals, plans, labels = collect_lift_dataset(env, model, args)
        np.savez_compressed(cache_path, geo_features=geo, slot_features=slot,
                              goals=goals, plans=plans, labels=labels)
        print(json.dumps({"event": "saved_lift_cache",
                           "path": str(cache_path),
                           "n_samples": int(len(geo)),
                           "mean_episode_imp": float(labels.mean())}),
               flush=True)
    print(json.dumps({"event": "cache_loaded",
                       "n_samples": int(len(geo)),
                       "geo_dim": int(geo.shape[1]),
                       "slot_dim": int(slot.shape[1]),
                       "mean_episode_imp": float(labels.mean())}), flush=True)

    # 2. Train heads
    print(json.dumps({"event": "train_geo_start"}), flush=True)
    geo_head, geo_stats = train_geo_head(geo, goals, plans, labels, args)
    print(json.dumps({"event": "saved_head", "label": "geo",
                       "final_val_loss": geo_stats.final_val_loss}), flush=True)

    print(json.dumps({"event": "train_supervised_start"}), flush=True)
    sup_adapter, sup_head, sup_ad_stats, sup_vh_stats, sup_ad_geo = \
        train_supervised_adapter_and_head(slot, geo, goals, plans, labels, args)
    print(json.dumps({"event": "saved_head", "label": "supervised",
                       "adapter_spearman": sup_ad_stats.mean_val_spearman,
                       "vh_val_loss": sup_vh_stats.final_val_loss}), flush=True)

    print(json.dumps({"event": "train_end2end_start"}), flush=True)
    e2e_model, e2e_stats = train_end2end_pure(slot, goals, plans, labels, args)
    print(json.dumps({"event": "saved_head", "label": "end2end",
                       "final_val_loss": e2e_stats.final_val_loss}), flush=True)

    print(json.dumps({"event": "train_pretrain_ft_start"}), flush=True)
    pft_model, pft_stats = train_pretrain_ft(slot, geo, goals, plans, labels, args)
    print(json.dumps({"event": "saved_head", "label": "pretrain_ft",
                       "final_val_loss": pft_stats.final_val_value_loss}),
           flush=True)

    # 3. Decile + G4 height diagnostics
    height_diag = height_feature_spearman(sup_adapter, slot, goals, geo, args)
    print(json.dumps({"event": "height_spearman", **height_diag}), flush=True)

    deciles = {
        "geo": decile_on_state(geo_head, geo, goals, plans, labels, args),
        "supervised": decile_on_state(sup_head, sup_ad_geo, goals, plans, labels, args),
        "end2end": decile_for_end2end(e2e_model, slot, goals, plans, labels, args),
        "pretrain_ft": decile_for_end2end(pft_model, slot, goals, plans, labels, args),
    }
    for lbl, dd in deciles.items():
        print(json.dumps({"event": "decile_diagnostic", "label": lbl, **dd}),
               flush=True)

    # 4. Eval
    heads = {"geo": geo_head, "supervised": sup_head,
              "end2end": e2e_model, "pretrain_ft": pft_model}
    adapters = {"supervised": sup_adapter}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for m in modes:
        per_ep = []
        t0 = time.time()
        for ep in range(args.n_eval_episodes):
            per_ep.append(run_episode(env, model, heads, adapters, m, args, ep))
        summary = summarize_mode(m, per_ep)
        all_results.append(summary)
        print(json.dumps({"event": "mode_done", **summary,
                           "elapsed_s": round(time.time() - t0, 1)}), flush=True)
        with open(out / f"per_episode_{m}.jsonl", "w") as fh:
            for rec in per_ep:
                fh.write(json.dumps(rec) + "\n")
    env.close()

    report = {
        "args": vars(args),
        "rollout_cache": str(cache_path),
        "head_stats": {
            "geo": geo_stats.__dict__,
            "supervised_adapter": sup_ad_stats.__dict__,
            "supervised_vh": sup_vh_stats.__dict__,
            "end2end": e2e_stats.__dict__,
            "pretrain_ft": pft_stats.__dict__,
        },
        "height_spearman": height_diag,
        "deciles": deciles,
        "results": all_results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2,
                    default=lambda o: float(o) if isinstance(o, np.floating)
                    else bool(o) if isinstance(o, np.bool_)
                    else str(o))
    print(json.dumps({"event": "done"}), flush=True)


if __name__ == "__main__":
    main()
