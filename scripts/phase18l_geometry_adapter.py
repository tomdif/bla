"""Phase 18λ — Object-file geometry adapter.

Per Phase 18θ, raw frozen OF-JEPA slot features do not directly
support goal-relative value prediction. Phase 18λ tests whether a
small structured adapter can extract goal-relative geometry from
slots, then feed those derived features to the same Phase 18η-style
value head.

Pipeline (reuses Phase 18θ rollout cache; no new collection):
  1. Load cache: (slot[768], geo[10], goal[2], plan[10x7], episode_imp).
  2. TRAIN ADAPTER: regress adapter(slot, goal) -> geo with MSE.
     Report per-feature Pearson/Spearman on held-out 20%.
  3. TRAIN VALUE_HEAD_ADAPTER: predict episode_imp from
     (adapter(slot, goal), goal, plan).
  4. EVAL 5 modes × n_eval_episodes seed-0:
     - gt_closed_loop
     - phase17_locked
     - combined_sum_geo          (Phase 18η-multi locked reference)
     - combined_sum_adapter      (NEW; adapter-derived geo + predictor)
     - naive_cem

Gates per PHASE_18L_GEOMETRY_ADAPTER_PRECOMMIT.md.
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
from system1_jepa.geometry_adapter import (
    ObjectFileGeometryAdapter,
    train_adapter_supervised,
    adapter_config,
    build_adapter_from_config,
)
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


# Lazy
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


# ---------- Adapter-derived geometry at runtime ----------
@torch.no_grad()
def derive_geometry(adapter, slot_feat: np.ndarray, goal_xy: np.ndarray,
                     device: str) -> np.ndarray:
    s = torch.from_numpy(slot_feat).to(device).unsqueeze(0).float()
    g = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    return adapter(s, g)[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def value_score(head, state, goal_xy, action_seq, device):
    s = torch.from_numpy(state).to(device).unsqueeze(0).float()
    g = torch.from_numpy(goal_xy).to(device).unsqueeze(0).float()
    a = torch.from_numpy(action_seq).to(device).unsqueeze(0).float()
    return float(head(s, g, a).cpu().item())


def build_combined_score_fn(env, model, head, adapter, head_input: str,
                              obs, goal_xy_world, lam, args):
    """head_input ∈ {geo, adapter, none}."""
    goal_xy = np.array(goal_xy_world, dtype=np.float32)
    pred_fn = build_score_fn(env, model, obs, goal_xy_world)
    if head_input == "geo":
        head_state = state_features(obs, goal_xy_world)
    elif head_input == "adapter":
        slot_feat = slot_features_at_obs(model, obs)
        head_state = derive_geometry(adapter, slot_feat, goal_xy, args.device)
    elif head_input == "none":
        head_state = None
    else:
        raise ValueError(head_input)

    def fn(seq):
        p = pred_fn(seq)
        if head_input == "none" or head is None:
            return p
        v = value_score(head, head_state, goal_xy,
                         seq.astype(np.float32), args.device)
        return lam * p + (1.0 - lam) * v
    return fn


# ---------- Episode runner ----------
def run_episode(env, model, heads, adapter, mode, args, ep_id):
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
            if plan is None: plan = mu
            n_exec = min(args.replan_every,
                          args.total_actions - actions_executed, len(plan))
            for a in plan[:n_exec]:
                obs = step_and_track(env, obs, a, args, contact)
            actions_executed += n_exec
    else:
        mode_cfg = {
            "phase17_locked":          ("none", None),
            "combined_sum_geo":        ("geo",  "geo"),
            "combined_sum_adapter":    ("adapter", "adapter"),
        }
        scoring, head_label = mode_cfg[mode]
        head = heads[head_label]["head"] if head_label else None
        while actions_executed < args.total_actions:
            score_fn = build_combined_score_fn(env, model, head, adapter,
                                                  scoring, obs, goal_xy_world,
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


# ---------- Value-head training on adapter-derived state ----------
def train_value_head_on_adapter(adapter, slots, goals, plans, labels, args):
    """Run adapter on every sample, then train value head on (adapter_geo, goal, plan)."""
    device = next(adapter.parameters()).device
    adapter.eval()
    with torch.no_grad():
        adapter_geo = adapter(torch.from_numpy(slots).to(device).float(),
                                torch.from_numpy(goals).to(device).float()).cpu().numpy()

    head = GoalProgressValueHead(
        state_dim=int(adapter_geo.shape[1]),
        action_dim=plans.shape[2], plan_horizon=plans.shape[1],
        hidden=args.vh_hidden, n_hidden=args.vh_n_hidden,
        dropout=args.vh_dropout,
    ).to(args.device)
    stats = train_value_head_supervised(
        head,
        torch.from_numpy(adapter_geo), torch.from_numpy(goals),
        torch.from_numpy(plans), torch.from_numpy(labels),
        steps=args.vh_train_steps, batch_size=args.vh_batch_size,
        lr=args.vh_lr, weight_decay=args.vh_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.vh_log_every,
    )
    head.eval()
    return head, stats, adapter_geo


def train_value_head_on_geo(geo, goals, plans, labels, args):
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


def compute_value_decile(head, state_eval, goals, plans, labels, args):
    n = len(state_eval)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(args.val_split * n))
    val_idx = perm[:n_val]
    device = next(head.parameters()).device
    with torch.no_grad():
        pred = head(
            torch.from_numpy(state_eval[val_idx]).to(device).float(),
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
    p.add_argument("--rollout-cache", default=None,
                    help="Reuse Phase 18θ-style cache (must contain geo+slot). "
                          "If absent, collect a fresh one at --seed.")
    p.add_argument("--auto-collect-episodes", type=int, default=300,
                    help="If no cache provided, collect this many episodes.")
    p.add_argument("--auto-collect-log-every", type=int, default=20)
    p.add_argument("--out", required=True)
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
    # Adapter
    p.add_argument("--adapter-hidden", type=int, default=256)
    p.add_argument("--adapter-n-hidden", type=int, default=3)
    p.add_argument("--adapter-dropout", type=float, default=0.0)
    p.add_argument("--adapter-train-steps", type=int, default=2000)
    p.add_argument("--adapter-batch-size", type=int, default=64)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--adapter-weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-log-every", type=int, default=200)
    # Value head
    p.add_argument("--vh-hidden", type=int, default=256)
    p.add_argument("--vh-n-hidden", type=int, default=3)
    p.add_argument("--vh-dropout", type=float, default=0.0)
    p.add_argument("--vh-train-steps", type=int, default=2000)
    p.add_argument("--vh-batch-size", type=int, default=64)
    p.add_argument("--vh-lr", type=float, default=3e-4)
    p.add_argument("--vh-weight-decay", type=float, default=1e-4)
    p.add_argument("--vh-log-every", type=int, default=200)
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
                    "combined_sum_geo,combined_sum_adapter,naive_cem")
    args = p.parse_args()

    load_planning_dependencies()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.rollout_cache and Path(args.rollout_cache).exists():
        d = np.load(args.rollout_cache)
        if "geo_features" not in d.files:
            raise SystemExit(f"Cache {args.rollout_cache} missing slot_features.")
        geo = d["geo_features"].astype(np.float32)
        slot = d["slot_features"].astype(np.float32)
        goals = d["goals"].astype(np.float32)
        plans = d["plans"].astype(np.float32)
        labels = d["labels"].astype(np.float32)
        cache_source = args.rollout_cache
        print(json.dumps({"event": "loaded_cache", "path": cache_source,
                           "n_samples": int(len(geo)),
                           "geo_dim": int(geo.shape[1]),
                           "slot_dim": int(slot.shape[1]),
                           "mean_episode_imp": float(labels.mean())}),
               flush=True)
    else:
        # Auto-collect at this seed (no compatible cache provided)
        print(json.dumps({"event": "auto_collect_start", "seed": args.seed,
                           "n_episodes": args.auto_collect_episodes}),
               flush=True)
        # phase18t has its own module-level globals for planning helpers
        # (encode_frame, state_features, etc.); initialize them too.
        from scripts.phase18t_slot_value_head import (
            collect_dataset as _collect,
            load_planning_dependencies as _phase18t_load,
        )
        _phase18t_load()
        # Build env + model for collection (matches phase18t configuration)
        model_for_collect = load_action_model(args)
        env = build_env(args.image_size,
                         horizon=args.total_actions * args.jepa_stride
                                 + 3 * args.plan_horizon * args.jepa_stride
                                 + 60)
        # phase18t's collect_dataset reads several args; map ours over
        class CollectArgs:
            pass
        ca = CollectArgs()
        for k, v in vars(args).items():
            setattr(ca, k, v)
        # phase18t names
        ca.rollout_episodes = args.auto_collect_episodes
        ca.rollout_log_every = args.auto_collect_log_every
        ca.train_K = 32
        ca.train_cem_iters = 1
        ca.train_sigma = 0.12
        geo, slot, goals, plans, labels = _collect(env, model_for_collect, ca)
        env.close()
        cache_source = str(out / "rollout_cache.npz")
        np.savez_compressed(cache_source, geo_features=geo,
                              slot_features=slot, goals=goals, plans=plans,
                              labels=labels)
        print(json.dumps({"event": "auto_collect_done", "path": cache_source,
                           "n_samples": int(len(geo)),
                           "geo_dim": int(geo.shape[1]),
                           "slot_dim": int(slot.shape[1]),
                           "mean_episode_imp": float(labels.mean())}),
               flush=True)
        del model_for_collect

    # 1. Train adapter (slot -> geo)
    adapter = ObjectFileGeometryAdapter(
        slot_dim=int(slot.shape[1]), goal_dim=int(goals.shape[1]),
        out_dim=int(geo.shape[1]),
        hidden=args.adapter_hidden, n_hidden=args.adapter_n_hidden,
        dropout=args.adapter_dropout,
    ).to(args.device)
    adapter_stats = train_adapter_supervised(
        adapter, torch.from_numpy(slot), torch.from_numpy(goals),
        torch.from_numpy(geo),
        steps=args.adapter_train_steps, batch_size=args.adapter_batch_size,
        lr=args.adapter_lr, weight_decay=args.adapter_weight_decay,
        val_split=args.val_split, seed=args.seed,
        log_every=args.adapter_log_every,
    )
    adapter_ckpt = out / "geometry_adapter.pt"
    torch.save({"state_dict": adapter.state_dict(),
                "config": adapter_config(adapter),
                "stats": adapter_stats.__dict__}, adapter_ckpt)
    print(json.dumps({"event": "saved_adapter", "path": str(adapter_ckpt),
                       "initial_val_mse": adapter_stats.initial_val_mse,
                       "final_val_mse": adapter_stats.final_val_mse,
                       "mean_val_pearson": adapter_stats.mean_val_pearson,
                       "mean_val_spearman": adapter_stats.mean_val_spearman}),
           flush=True)

    # 2. Train value heads (geo reference + adapter)
    geo_head, geo_stats = train_value_head_on_geo(geo, goals, plans, labels, args)
    geo_head_ckpt = out / "value_head_geo.pt"
    torch.save({"state_dict": geo_head.state_dict(),
                "config": value_head_config(geo_head),
                "stats": geo_stats.__dict__, "label": "geo"}, geo_head_ckpt)
    print(json.dumps({"event": "saved_value_head", "label": "geo",
                       **geo_stats.__dict__}), flush=True)

    adapter_head, ad_vh_stats, adapter_geo = train_value_head_on_adapter(
        adapter, slot, goals, plans, labels, args)
    ad_head_ckpt = out / "value_head_adapter.pt"
    torch.save({"state_dict": adapter_head.state_dict(),
                "config": value_head_config(adapter_head),
                "stats": ad_vh_stats.__dict__, "label": "adapter"},
                ad_head_ckpt)
    print(json.dumps({"event": "saved_value_head", "label": "adapter",
                       **ad_vh_stats.__dict__}), flush=True)

    # 3. Decile diagnostics
    deciles = {
        "geo": compute_value_decile(geo_head, geo, goals, plans, labels, args),
        "adapter": compute_value_decile(adapter_head, adapter_geo, goals,
                                            plans, labels, args),
    }
    for lbl, d in deciles.items():
        print(json.dumps({"event": "decile_diagnostic", "label": lbl,
                           "pearson": d["pearson"], "spearman": d["spearman"],
                           "top_decile_actual": d["deciles"][0]["mean_actual"]
                              if d["deciles"] else float("nan"),
                           "bot_decile_actual": d["deciles"][-1]["mean_actual"]
                              if d["deciles"] else float("nan"),
                           "top_vs_bot_gap": d["top_vs_bot_gap"]}),
               flush=True)

    # 4. Eval on env
    model = load_action_model(args)
    env = build_env(args.image_size,
                     horizon=args.total_actions * args.jepa_stride
                             + 3 * args.plan_horizon * args.jepa_stride + 60)

    heads = {"geo": {"head": geo_head, "ckpt": geo_head_ckpt},
              "adapter": {"head": adapter_head, "ckpt": ad_head_ckpt}}

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for m in modes:
        print(json.dumps({"event": "eval_mode", "mode": m}), flush=True)
        per_ep = []
        t0 = time.time()
        for ep in range(args.n_eval_episodes):
            per_ep.append(run_episode(env, model, heads, adapter, m, args, ep))
            if (ep + 1) % 10 == 0:
                partial = summarize_mode(m, per_ep)
                print(json.dumps({"event": "progress", "mode": m,
                                   "done": ep + 1,
                                   "improvement": round(partial["improvement"], 3),
                                   "success": round(partial["success_rate"], 3),
                                   "elapsed_s": round(time.time() - t0, 1)}),
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
    cs_adapter = by_mode.get("combined_sum_adapter", {}).get("improvement",
                                                                  float("nan"))
    adapter_mean_spearman = adapter_stats.mean_val_spearman
    vh_adapter = deciles["adapter"]
    vh_geo = deciles["geo"]

    g1 = adapter_mean_spearman >= 0.50
    g2 = (vh_adapter["spearman"] >= 0.25)
    g3_top = vh_adapter["deciles"][0]["mean_actual"] if vh_adapter["deciles"] else 0
    g3_bot = vh_adapter["deciles"][-1]["mean_actual"] if vh_adapter["deciles"] else 1e-9
    g3 = g3_top >= 2.0 * max(g3_bot, 1e-9)
    g4 = cs_adapter >= locked + 0.02
    g5 = cs_adapter >= 0.90 * cs_geo if cs_geo == cs_geo else False

    gates = {
        "g1_adapter_geo_recovery_mean_spearman_geq_0_50":
            g1,
        "adapter_mean_val_spearman": adapter_mean_spearman,
        "g2_value_head_adapter_spearman_geq_0_25":
            g2,
        "value_head_adapter_spearman": vh_adapter["spearman"],
        "g3_value_head_adapter_top_geq_2x_bot": g3,
        "value_head_adapter_top": g3_top,
        "value_head_adapter_bot": g3_bot,
        "g4_combined_sum_adapter_beats_locked_by_0_02": g4,
        "combined_sum_adapter_improvement": cs_adapter,
        "phase17_locked_improvement": locked,
        "g5_combined_sum_adapter_within_10pct_of_geo": g5,
        "combined_sum_geo_improvement": cs_geo,
    }
    n_pass_main = sum([g1, g2, g3, g4])
    gates["n_pass_main"] = n_pass_main
    gates["g5_stretch_pass"] = g5
    gates["verdict"] = (
        "4/4 + G5 — adapter recovers geometry from slots; matches geo recipe within 10%"
        if n_pass_main == 4 and g5 else
        "4/4 — adapter recovers geometry AND plans well; BLA no longer needs simulator geo"
        if n_pass_main == 4 else
        "G1 + G2/G3 pass, G4 fail — adapter works but planning fails (env-variance or value head issue)"
        if g1 and (g2 or g3) and not g4 else
        "G1 pass only — slots contain geometry but downstream chain fails"
        if g1 and not g2 and not g3 else
        "G1 fail — slots do NOT contain goal-relative geometry; need encoder retraining"
        if not g1 else
        "mixed: investigate"
    )

    report = {
        "args": vars(args),
        "rollout_cache": args.rollout_cache,
        "adapter": {"ckpt": str(adapter_ckpt),
                     "stats": adapter_stats.__dict__},
        "value_heads": {"geo": {"ckpt": str(geo_head_ckpt),
                                  "stats": geo_stats.__dict__},
                         "adapter": {"ckpt": str(ad_head_ckpt),
                                      "stats": ad_vh_stats.__dict__}},
        "deciles": deciles,
        "results": all_results,
        "gates": gates,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"event": "done", **gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
