"""Phase 18λ-v2 — End-to-end adapter + value head.

Phase 18λ-multi showed:
  - Supervised adapter (slot→engineered_geo MSE) recovers value-
    relevant subspace robustly (mean Spearman 0.506 / 3 seeds).
  - But the value head built on adapter output planned at only 85%
    of the engineered-geo recipe — bottleneck is integration, not
    representation.

Phase 18λ-v2 trains the adapter+value-head jointly with MSE on
episode_imp DIRECTLY. The adapter is free to find a 10-dim latent
that maximizes value prediction, not features that match engineered
geometry.

Three heads trained per seed on the same cached rollouts:
  vh_geo:        engineered 10-dim geo → MSE on episode_imp
  vh_supervised: adapter(slot, goal) → 10-dim geo (MSE) THEN
                  frozen adapter + VH (MSE on episode_imp)
  vh_end2end:    adapter(slot, goal) → 10-dim latent + VH jointly
                  (MSE on episode_imp), no intermediate geometry

Eval modes (6):
  gt_closed_loop
  phase17_locked
  combined_sum_geo
  combined_sum_supervised
  combined_sum_end2end
  naive_cem

Gates per PHASE_18L2_END2END_PRECOMMIT.md.
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
    GoalProgressValueHead, build_value_head_from_config,
    value_head_config, train_value_head_supervised,
)
from system1_jepa.geometry_adapter import (
    ObjectFileGeometryAdapter, train_adapter_supervised,
    adapter_config, build_adapter_from_config,
    End2EndAdapterValue, train_end2end_supervised,
    end2end_config, build_end2end_from_config,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


# Lazy globals (robosuite-dependent)
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


# ---------- Per-head training ----------
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
    """Phase 18λ supervised path: adapter trained slot→geo, then frozen for VH."""
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
    # Frozen adapter — derive adapter_geo for value-head training
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


def train_end2end_model(slots, goals, plans, labels, args):
    model = End2EndAdapterValue(
        slot_dim=int(slots.shape[1]), goal_dim=int(goals.shape[1]),
        latent_dim=args.end2end_latent_dim,
        action_dim=plans.shape[2], plan_horizon=plans.shape[1],
        adapter_hidden=args.adapter_hidden,
        adapter_n_hidden=args.adapter_n_hidden,
        value_hidden=args.vh_hidden, value_n_hidden=args.vh_n_hidden,
        dropout=args.vh_dropout,
    ).to(args.device)
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


# ---------- Decile diagnostic ----------
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


def decile_for_end2end(model, slots, goals, plans, labels, args):
    """Same as decile_on_state but for the end2end model (takes slot directly)."""
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


# ---------- Scoring in env ----------
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
    """head_kind ∈ {none, geo, supervised_adapter, end2end}"""
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
    if head_kind == "end2end":
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
            "phase17_locked":         ("none",                None,  None),
            "combined_sum_geo":       ("geo",                 "geo", None),
            "combined_sum_supervised":("supervised_adapter",  "supervised",
                                         "supervised"),
            "combined_sum_end2end":   ("end2end",             "end2end", None),
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
    cube_disp = actual_final_xy - cube_init[:2]
    disp_n = float(np.linalg.norm(cube_disp))
    goal_dir = (goal_xy_world - cube_init[:2]) / max(
        np.linalg.norm(goal_xy_world - cube_init[:2]), 1e-9)
    dir_score = float(np.dot(cube_disp / max(disp_n, 1e-9), goal_dir)) \
                  if disp_n > 0.02 else 0.0
    return {"ep_id": ep_id, "mode": mode,
             "start_dist": start_dist, "actual_dist": actual_dist,
             "improvement": improvement, "dir_score": dir_score,
             "cube_displacement": disp_n, "contact": bool(contact[0]),
             "success": bool(actual_dist <= args.success_threshold)}


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
    p.add_argument("--rollout-cache", required=True,
                    help="Cache with geo_features + slot_features (from 18θ/18λ)")
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
    # Adapter (supervised)
    p.add_argument("--adapter-hidden", type=int, default=256)
    p.add_argument("--adapter-n-hidden", type=int, default=3)
    p.add_argument("--adapter-dropout", type=float, default=0.0)
    p.add_argument("--adapter-train-steps", type=int, default=2000)
    p.add_argument("--adapter-batch-size", type=int, default=64)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-log-every", type=int, default=200)
    # Value head (and end2end VH part)
    p.add_argument("--vh-hidden", type=int, default=256)
    p.add_argument("--vh-n-hidden", type=int, default=3)
    p.add_argument("--vh-dropout", type=float, default=0.0)
    p.add_argument("--vh-train-steps", type=int, default=2000)
    p.add_argument("--vh-batch-size", type=int, default=64)
    p.add_argument("--vh-lr", type=float, default=3e-4)
    p.add_argument("--vh-weight-decay", type=float, default=1e-4)
    p.add_argument("--vh-log-every", type=int, default=200)
    # End2end-specific
    p.add_argument("--end2end-train-steps", type=int, default=2000)
    p.add_argument("--end2end-latent-dim", type=int, default=10)
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
    p.add_argument("--goal-dist-min", type=float, default=0.05)
    p.add_argument("--goal-dist-max", type=float, default=0.08)
    p.add_argument("--success-threshold", type=float, default=0.04)
    p.add_argument("--modes", default="gt_closed_loop,phase17_locked,"
                    "combined_sum_geo,combined_sum_supervised,"
                    "combined_sum_end2end,naive_cem")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load cache
    d = np.load(args.rollout_cache)
    if "geo_features" not in d.files:
        raise SystemExit(f"Cache {args.rollout_cache} missing slot/geo features.")
    geo = d["geo_features"].astype(np.float32)
    slot = d["slot_features"].astype(np.float32)
    goals = d["goals"].astype(np.float32)
    plans = d["plans"].astype(np.float32)
    labels = d["labels"].astype(np.float32)
    print(json.dumps({"event": "loaded_cache",
                       "path": args.rollout_cache,
                       "n_samples": int(len(geo)),
                       "geo_dim": int(geo.shape[1]),
                       "slot_dim": int(slot.shape[1]),
                       "mean_episode_imp": float(labels.mean())}),
           flush=True)

    # 1. Train geo head
    print(json.dumps({"event": "train_geo_start"}), flush=True)
    geo_head, geo_stats = train_geo_head(geo, goals, plans, labels, args)
    torch.save({"state_dict": geo_head.state_dict(),
                "config": value_head_config(geo_head),
                "stats": geo_stats.__dict__, "label": "geo"},
                out / "value_head_geo.pt")
    print(json.dumps({"event": "saved_value_head", "label": "geo",
                       **geo_stats.__dict__}), flush=True)

    # 2. Train supervised adapter + head
    print(json.dumps({"event": "train_supervised_start"}), flush=True)
    sup_adapter, sup_head, sup_ad_stats, sup_vh_stats, sup_adapter_geo = \
        train_supervised_adapter_and_head(slot, geo, goals, plans, labels, args)
    torch.save({"state_dict": sup_adapter.state_dict(),
                "config": adapter_config(sup_adapter),
                "stats": sup_ad_stats.__dict__},
                out / "supervised_adapter.pt")
    torch.save({"state_dict": sup_head.state_dict(),
                "config": value_head_config(sup_head),
                "stats": sup_vh_stats.__dict__, "label": "supervised"},
                out / "value_head_supervised.pt")
    print(json.dumps({"event": "saved_supervised_adapter",
                       **sup_ad_stats.__dict__}), flush=True)
    print(json.dumps({"event": "saved_value_head", "label": "supervised",
                       **sup_vh_stats.__dict__}), flush=True)

    # 3. Train end2end
    print(json.dumps({"event": "train_end2end_start"}), flush=True)
    e2e_model, e2e_stats = train_end2end_model(slot, goals, plans, labels, args)
    torch.save({"state_dict": e2e_model.state_dict(),
                "config": end2end_config(e2e_model),
                "stats": e2e_stats.__dict__, "label": "end2end"},
                out / "value_head_end2end.pt")
    print(json.dumps({"event": "saved_end2end", **e2e_stats.__dict__}),
           flush=True)

    # 4. Decile diagnostics
    deciles = {
        "geo": decile_on_state(geo_head, geo, goals, plans, labels, args),
        "supervised": decile_on_state(sup_head, sup_adapter_geo, goals,
                                          plans, labels, args),
        "end2end": decile_for_end2end(e2e_model, slot, goals, plans, labels, args),
    }
    for lbl, dd in deciles.items():
        print(json.dumps({"event": "decile_diagnostic", "label": lbl,
                           "pearson": dd["pearson"], "spearman": dd["spearman"],
                           "top_decile_actual": dd["deciles"][0]["mean_actual"]
                              if dd["deciles"] else float("nan"),
                           "bot_decile_actual": dd["deciles"][-1]["mean_actual"]
                              if dd["deciles"] else float("nan"),
                           "top_vs_bot_gap": dd["top_vs_bot_gap"]}),
               flush=True)

    # 5. Eval
    model_action = load_action_model(args)
    env = build_env(args.image_size,
                     horizon=args.total_actions * args.jepa_stride
                             + 3 * args.plan_horizon * args.jepa_stride + 60)
    heads = {"geo": geo_head, "supervised": sup_head, "end2end": e2e_model}
    adapters = {"supervised": sup_adapter}

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for m in modes:
        print(json.dumps({"event": "eval_mode", "mode": m}), flush=True)
        per_ep = []
        t0 = time.time()
        for ep in range(args.n_eval_episodes):
            per_ep.append(run_episode(env, model_action, heads, adapters,
                                          m, args, ep))
            if (ep + 1) % 10 == 0:
                pp = summarize_mode(m, per_ep)
                print(json.dumps({"event": "progress", "mode": m,
                                   "done": ep + 1,
                                   "improvement": round(pp["improvement"], 3),
                                   "success": round(pp["success_rate"], 3),
                                   "elapsed_s": round(time.time() - t0, 1)}),
                       flush=True)
        with open(out / f"per_episode_{m}.jsonl", "w") as fh:
            for rec in per_ep:
                fh.write(json.dumps(rec) + "\n")
        summary = summarize_mode(m, per_ep)
        all_results.append(summary)
        print(json.dumps({"event": "mode_done", **summary}), flush=True)
    env.close()

    # 6. Gates
    by_mode = {r["mode"]: r for r in all_results}
    locked = by_mode.get("phase17_locked", {}).get("improvement", float("nan"))
    cs_geo = by_mode.get("combined_sum_geo", {}).get("improvement", float("nan"))
    cs_sup = by_mode.get("combined_sum_supervised", {}).get("improvement", float("nan"))
    cs_e2e = by_mode.get("combined_sum_end2end", {}).get("improvement", float("nan"))
    e2e_d = deciles["end2end"]
    e2e_top = e2e_d["deciles"][0]["mean_actual"] if e2e_d["deciles"] else 0.0
    e2e_bot = e2e_d["deciles"][-1]["mean_actual"] if e2e_d["deciles"] else 1e-9

    g1 = bool(e2e_d["spearman"] >= 0.25) if e2e_d["spearman"] == e2e_d["spearman"] else False
    g2 = bool(e2e_top >= 2.0 * max(e2e_bot, 1e-9))
    g3 = bool(cs_e2e >= 0.90 * cs_geo) if cs_geo == cs_geo else False
    g4 = bool(cs_e2e >= cs_sup) if cs_sup == cs_sup else False
    g5 = bool(cs_e2e >= locked + 0.02) if locked == locked else False

    gates = {
        "g1_end2end_vh_spearman_geq_0_25": g1,
        "g2_end2end_top_geq_2x_bot": g2,
        "g3_end2end_combined_geq_0_90_x_geo": g3,
        "g4_end2end_combined_geq_supervised": g4,
        "g5_stretch_end2end_beats_locked": g5,
        "end2end_spearman": e2e_d["spearman"],
        "end2end_top_decile": e2e_top,
        "end2end_bot_decile": e2e_bot,
        "combined_sum_end2end": cs_e2e,
        "combined_sum_supervised": cs_sup,
        "combined_sum_geo": cs_geo,
        "phase17_locked": locked,
        "n_pass_main": int(g1) + int(g2) + int(g3) + int(g4),
    }
    gates["verdict"] = (
        "4/4 main + G5 — end2end matches geo and beats locked"
        if gates["n_pass_main"] == 4 and g5 else
        "4/4 main — end2end matches engineered-geo planning"
        if gates["n_pass_main"] == 4 else
        "G4 only — end2end beats supervised but still below geo"
        if g4 and gates["n_pass_main"] < 4 else
        "0-3/4 — partial; investigate"
    )

    report = {
        "args": vars(args),
        "rollout_cache": args.rollout_cache,
        "head_stats": {"geo": geo_stats.__dict__,
                        "supervised_adapter": sup_ad_stats.__dict__,
                        "supervised_vh": sup_vh_stats.__dict__,
                        "end2end": e2e_stats.__dict__},
        "deciles": deciles,
        "results": all_results,
        "gates": gates,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2,
                    default=lambda o: float(o) if isinstance(o, np.floating)
                    else bool(o) if isinstance(o, np.bool_)
                    else str(o))
    print(json.dumps({"event": "done", **{k: v for k, v in gates.items()
                                              if not isinstance(v, dict)}},
                      indent=2), flush=True)


if __name__ == "__main__":
    main()
