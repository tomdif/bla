"""Phase 18ν — Scheduled aux loss.

Phase 18κ Regime 2 revealed that:
  - supervised (geo-MSE aux) wins in-distribution
  - end2end (no aux)         wins out-of-distribution

Phase 18ν tests whether a scheduled / pretrain+finetune training
procedure captures BOTH:
  - in-distribution capacity from the geo aux (early training)
  - OOD generalization from value-only training (late / fine-tune)

Trains 4 heads per seed (A=supervised, B=end2end, C=annealed,
D=pretrain+ft) and evaluates each at BOTH in-distribution and OOD
goal-distance.

Reuses Phase 18λ-multi / 18λ-v2 caches.
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


# Lazy globals
build_env = None; encode_frame = None; find_cubeA_slot = None
predict_score_seq = None; closed_loop_gt_step = None
state_features = None; rollout_scripted_prior = None; cem_with_prior = None


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
    m = ActionConditionedOFJEPA(
        image_size=args.image_size, cfg=cfg,
        action_dim=args.action_dim, use_action=True,
    ).to(args.device)
    m.load_state_dict(torch.load(args.model_action, map_location=args.device))
    m.eval()
    return m


def norm_xy(p_xy):
    return np.clip((p_xy + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def step_and_track(env, obs, action, args, contact_ref):
    for _ in range(args.jepa_stride):
        obs, _, _, _ = env.step(action)
        if float(np.linalg.norm(obs["cubeA_pos"][:2]
                                  - obs["robot0_eef_pos"][:2])) < 0.04:
            contact_ref[0] = True
    return obs


def slot_features_at_obs(model, obs) -> np.ndarray:
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


# ---------- Training routines ----------
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
        dropout=args.adapter_dropout,
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
        dropout=args.vh_dropout,
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
        dropout=args.vh_dropout,
    ).to(args.device)


def train_end2end_pure(slots, goals, plans, labels, args):
    model = make_end2end(slots, plans, args)
    stats = train_end2end_supervised(
        model, torch.from_numpy(slots), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.end2end_train_steps, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.vh_log_every,
    )
    model.eval()
    return model, stats


def train_scheduled(slots, geo, goals, plans, labels, args,
                     schedule_kind: str):
    model = make_end2end(slots, plans, args)
    stats = train_end2end_with_aux_schedule(
        model, torch.from_numpy(slots), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        torch.from_numpy(geo),
        schedule_kind=schedule_kind,
        steps=args.end2end_train_steps,
        pretrain_steps=args.pretrain_steps,
        aux_weight_init=1.0, aux_weight_final=0.0,
        aux_residual_weight=args.aux_residual_weight,
        val_split=args.val_split, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        seed=args.seed, log_every=args.vh_log_every,
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
    pred_fn = build_score_fn(env, model_action, obs, goal_xy_world)
    if head_kind == "none":
        return lambda seq: pred_fn(seq)
    if head_kind == "geo":
        head_state = state_features(obs, goal_xy_world)
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
    if head_kind in ("end2end", "annealed", "pretrain_ft"):
        slot_feat = slot_features_at_obs(model_action, obs)
        def fn(seq):
            p = pred_fn(seq)
            v = value_score_end2end(head_or_model, slot_feat, goal_xy,
                                       seq.astype(np.float32), args.device)
            return lam * p + (1.0 - lam) * v
        return fn
    raise ValueError(head_kind)


# ---------- Episode runner ----------
def sample_goal(obs, ep_id, goal_min, goal_max):
    cube_init = obs["cubeA_pos"][:2].copy()
    rng = np.random.RandomState(ep_id + 1000)
    theta = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(goal_min, goal_max)
    goal_xy = cube_init + r * np.array([np.cos(theta), np.sin(theta)])
    start_dist = float(np.linalg.norm(cube_init - goal_xy))
    return goal_xy, start_dist


def run_episode(env, model_action, heads, adapters, mode, args, ep_id,
                  goal_min, goal_max):
    obs = env.reset()
    cube_init = obs["cubeA_pos"][:3].copy()
    goal_xy_world, start_dist = sample_goal(obs, ep_id, goal_min, goal_max)
    actions_executed = 0
    contact = [False]

    if mode == "gt_closed_loop":
        for _ in range(args.total_actions):
            action = closed_loop_gt_step(env, obs, goal_xy_world)
            obs = step_and_track(env, obs, action, args, contact)
    elif mode == "naive_cem":
        while actions_executed < args.total_actions:
            score_fn = build_score_fn(env, model_action, obs, goal_xy_world)
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
                obs = step_and_track(env, obs, a, args, contact)
            actions_executed += n_exec
    else:
        mode_cfg = {
            "phase17_locked":            ("none",                None,           None),
            "combined_sum_geo":          ("geo",                 "geo",          None),
            "combined_sum_supervised":   ("supervised_adapter",  "supervised",   "supervised"),
            "combined_sum_end2end":      ("end2end",             "end2end",      None),
            "combined_sum_annealed":     ("annealed",            "annealed",     None),
            "combined_sum_pretrain_ft":  ("pretrain_ft",         "pretrain_ft",  None),
        }
        head_kind, head_label, adapter_label = mode_cfg[mode]
        head_or_model = heads[head_label] if head_label else None
        adapter = adapters.get(adapter_label) if adapter_label else None
        while actions_executed < args.total_actions:
            score_fn = build_combined_score_fn(env, model_action, head_or_model,
                                                  head_kind, adapter,
                                                  obs, goal_xy_world,
                                                  args.combined_lambda, args)
            mu = rollout_scripted_prior(env, obs, goal_xy_world,
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
                obs = step_and_track(env, obs, a, args, contact)
            actions_executed += n_exec

    actual_final_xy = obs["cubeA_pos"][:2]
    actual_dist = float(np.linalg.norm(actual_final_xy - goal_xy_world))
    improvement = max(0.0, (start_dist - actual_dist) / max(start_dist, 1e-9))
    return {"ep_id": ep_id, "mode": mode,
             "start_dist": start_dist, "actual_dist": actual_dist,
             "improvement": improvement, "contact": bool(contact[0]),
             "success": bool(actual_dist <= args.success_threshold)}


def summarize_mode(mode, per_ep):
    return {"mode": mode, "n_episodes": len(per_ep),
             "improvement": float(np.mean([r["improvement"] for r in per_ep])),
             "contact_rate": float(np.mean([r["contact"] for r in per_ep])),
             "success_rate": float(np.mean([r["success"] for r in per_ep]))}


# ---------- Decile diagnostic for end2end-style heads ----------
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
        return {"pearson": float("nan"), "spearman": float("nan"),
                 "deciles": [], "top_vs_bot_gap": float("nan")}
    pearson = float(np.corrcoef(pred, actual)[0, 1])
    prk = np.argsort(np.argsort(pred)).astype(float)
    ark = np.argsort(np.argsort(actual)).astype(float)
    spearman = float(np.corrcoef(prk, ark)[0, 1])
    order = np.argsort(pred)[::-1]
    chunks = np.array_split(order, 10)
    deciles = [{"decile": i, "mean_pred": float(pred[c].mean()),
                  "mean_actual": float(actual[c].mean()), "n": int(len(c))}
                 for i, c in enumerate(chunks)]
    return {"pearson": pearson, "spearman": spearman, "deciles": deciles,
             "top_vs_bot_gap": deciles[0]["mean_actual"]
                                - deciles[-1]["mean_actual"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--rollout-cache", required=True)
    p.add_argument("--out", required=True)
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
    # Adapter
    p.add_argument("--adapter-hidden", type=int, default=256)
    p.add_argument("--adapter-n-hidden", type=int, default=3)
    p.add_argument("--adapter-dropout", type=float, default=0.0)
    p.add_argument("--adapter-train-steps", type=int, default=2000)
    p.add_argument("--adapter-batch-size", type=int, default=64)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-log-every", type=int, default=500)
    # Value head + end2end
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
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--id-goal-min", type=float, default=0.05)
    p.add_argument("--id-goal-max", type=float, default=0.08)
    p.add_argument("--ood-goal-min", type=float, default=0.10)
    p.add_argument("--ood-goal-max", type=float, default=0.15)
    p.add_argument("--modes", default="phase17_locked,combined_sum_geo,"
                    "combined_sum_supervised,combined_sum_end2end,"
                    "combined_sum_annealed,combined_sum_pretrain_ft,naive_cem")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load cache
    d = np.load(args.rollout_cache)
    if "geo_features" not in d.files:
        raise SystemExit(f"Cache missing slot/geo features")
    geo = d["geo_features"].astype(np.float32)
    slot = d["slot_features"].astype(np.float32)
    goals = d["goals"].astype(np.float32)
    plans = d["plans"].astype(np.float32)
    labels = d["labels"].astype(np.float32)
    print(json.dumps({"event": "loaded_cache",
                       "n_samples": int(len(geo)),
                       "geo_dim": int(geo.shape[1]),
                       "slot_dim": int(slot.shape[1])}), flush=True)

    # Train heads
    print(json.dumps({"event": "train_geo_start"}), flush=True)
    geo_head, geo_stats = train_geo_head(geo, goals, plans, labels, args)
    print(json.dumps({"event": "saved_value_head", "label": "geo",
                       "val_loss": geo_stats.final_val_loss}), flush=True)

    print(json.dumps({"event": "train_supervised_start"}), flush=True)
    sup_adapter, sup_head, sup_ad_stats, sup_vh_stats, sup_adapter_geo = \
        train_supervised_adapter_and_head(slot, geo, goals, plans, labels, args)
    print(json.dumps({"event": "saved_value_head", "label": "supervised",
                       "val_loss": sup_vh_stats.final_val_loss,
                       "adapter_spearman": sup_ad_stats.mean_val_spearman}),
           flush=True)

    print(json.dumps({"event": "train_end2end_start"}), flush=True)
    e2e_model, e2e_stats = train_end2end_pure(slot, goals, plans, labels, args)
    print(json.dumps({"event": "saved_value_head", "label": "end2end",
                       "val_loss": e2e_stats.final_val_loss}), flush=True)

    print(json.dumps({"event": "train_annealed_start"}), flush=True)
    ann_model, ann_stats = train_scheduled(slot, geo, goals, plans, labels,
                                              args, "linear_anneal")
    print(json.dumps({"event": "saved_value_head", "label": "annealed",
                       "val_loss": ann_stats.final_val_value_loss,
                       "final_geo_mse": ann_stats.final_geo_mse}), flush=True)

    print(json.dumps({"event": "train_pretrain_ft_start"}), flush=True)
    pft_model, pft_stats = train_scheduled(slot, geo, goals, plans, labels,
                                              args, "pretrain_ft")
    print(json.dumps({"event": "saved_value_head", "label": "pretrain_ft",
                       "val_loss": pft_stats.final_val_value_loss,
                       "final_geo_mse": pft_stats.final_geo_mse}), flush=True)

    # Decile diagnostics for end2end-style heads
    deciles = {
        "supervised": {"spearman": float("nan"), "top_vs_bot_gap": float("nan")},  # reuse 18l2 result
        "end2end": decile_for_end2end(e2e_model, slot, goals, plans, labels, args),
        "annealed": decile_for_end2end(ann_model, slot, goals, plans, labels, args),
        "pretrain_ft": decile_for_end2end(pft_model, slot, goals, plans, labels, args),
    }
    for lbl in ["end2end", "annealed", "pretrain_ft"]:
        dd = deciles[lbl]
        print(json.dumps({"event": "decile_diagnostic", "label": lbl,
                           "pearson": dd["pearson"], "spearman": dd["spearman"],
                           "top_vs_bot_gap": dd["top_vs_bot_gap"]}), flush=True)

    # Eval
    model_action = load_action_model(args)
    env = build_env(args.image_size,
                     horizon=args.total_actions * args.jepa_stride
                             + 3 * args.plan_horizon * args.jepa_stride + 60)
    heads = {"geo": geo_head, "supervised": sup_head,
              "end2end": e2e_model, "annealed": ann_model,
              "pretrain_ft": pft_model}
    adapters = {"supervised": sup_adapter}

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = {"in_dist": [], "ood": []}
    for dist_label, gmin, gmax in [("in_dist", args.id_goal_min, args.id_goal_max),
                                       ("ood", args.ood_goal_min, args.ood_goal_max)]:
        print(json.dumps({"event": "eval_dist_start", "dist": dist_label,
                           "goal_range": [gmin, gmax]}), flush=True)
        for m in modes:
            per_ep = []
            t0 = time.time()
            for ep in range(args.n_eval_episodes):
                per_ep.append(run_episode(env, model_action, heads, adapters,
                                              m, args, ep, gmin, gmax))
            summary = summarize_mode(m, per_ep)
            summary["dist"] = dist_label
            all_results[dist_label].append(summary)
            print(json.dumps({"event": "mode_done", "dist": dist_label,
                               **summary, "elapsed_s": round(time.time()-t0, 1)}),
                   flush=True)
    env.close()

    # Save
    report = {
        "args": vars(args),
        "head_stats": {
            "geo": geo_stats.__dict__,
            "supervised_adapter": sup_ad_stats.__dict__,
            "supervised_vh": sup_vh_stats.__dict__,
            "end2end": e2e_stats.__dict__,
            "annealed": ann_stats.__dict__,
            "pretrain_ft": pft_stats.__dict__,
        },
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
