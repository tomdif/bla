#!/usr/bin/env python3
"""LEVER 1 -- cheap goal-agnostic exploration is the world model's UNFAIR (but honest) advantage.

Setup recap (R1 v2): imitation (bc_goal) collapses on SHIFTED goals; the FULL-data WM holds because it
trained on rich RANDOM-exploration that covers (state x action) over the whole workspace. The honest
challenge was the FAIR-data control: train the WM on the SAME narrow-goal expert demos BC sees -- it
FAILED (0.05), because expert demos are a thin goal-biased slice of dynamics and CEM (which samples
off-tube actions) cannot plan inside that slice.

THE LEVER (this file): the world model can CONSUME cheap goal-agnostic exploration that imitation
STRUCTURALLY cannot (BC has no labels for random actions; bc_goal has no goal for goal-free data). So we
do not hand the WM the *full* exploration buffer -- we sweep a small budget E of it ON TOP of the SAME
narrow expert demos BC trains on, and ask: how little cheap exploration unlocks the shifted-goal win?

  WM(E) trains on  transitions_from_demos(narrow demos)  ++  E subsampled exploration transitions
  bc / bc_goal     train on  narrow demos only            (the fixed imitation baseline, unchanged)

The "moat" framing: covering shifted goals with EXPERT demos costs infinity (narrow demos never see the
shifted region by construction). Covering them with cheap random exploration costs E* transitions. If a
small E* flips the verdict, that gap IS the world model's economic moat.

CEM precision fix (so the strict @6px can pass): the R1 CEM reached the target VICINITY (~8.9px) but not
pixel-precision. We add a thin tuned wrapper around cem_plan -- longer horizon, more iters, and a
TERMINAL-weighted cost (heavier weight on the last predicted step's fingertip-to-target distance, so the
plan optimizes where the finger ENDS, not just the path-integral). Everything else mirrors r1 exactly.

Pre-registered gate (set BEFORE running; printed OK/XX + VERDICT, written to json):
  (C) FAIR-DATA CONTROL FAILS:  at E=0 the WM does NOT beat imitation on shift
        wm(E=0)@6 - bc_goal@6 < 0.20  (the architecture alone does not win -- data is doing the work)
  (G) THE MOAT EXISTS:  some E* unlocks the win on shifted goals --
        EXISTS E* with  wm(E*)@6 - bc_goal@6 >= 0.30   OR   wm(E*)@12 - bc_goal@12 >= 0.50
  (M) MONOTONE LEVER:  success-vs-E is (weakly) non-decreasing over the sweep (cheap data only helps)
        report the curve; pass if best-E >= E=0 by the gate margin (the lever, not noise)
  Falsification: if even full-E WM with tuned CEM does not beat bc_goal @6px (and @12px), the precision
  fix FAILED -- report it honestly (escalate to a different planner/cost, do not bury it).

Reuses system1_motion.r1_imitation_fails wholesale (do NOT reimplement): make_env, render, gen_expert_demos,
transitions_from_demos, load_transitions, train_world_model, train_bc, cem_plan, eval_method patterns,
finger_world/target_world/to_px/region_of, EXTENT.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch

from system1_motion.r1_imitation_fails import (
    make_env, render, finger_world, target_world, to_px, region_of,
    gen_expert_demos, transitions_from_demos, load_transitions,
    train_world_model, train_bc, cem_plan, EXTENT,
)


# ----------------------------- data mixing: narrow demos ++ E cheap exploration -----------------------------
def subsample_exploration(expl, E, rng):
    """Subsample E *transitions* (consecutive same-episode pairs) from an exploration buffer.
    expl = (frames, actions, pos, tgt, idx); idx = valid pair-start indices. We pick E starts and return
    a COMPACT buffer holding only the touched (start, start+1) frames, with a fresh idx into it. Keeping
    every pair self-contained (start & start+1 adjacent in the new buffer) means the recomputed idx never
    bridges unrelated frames."""
    frames, actions, pos, tgt, idx = expl
    if E <= 0:
        return None
    E = min(int(E), len(idx))
    starts = rng.choice(idx, size=E, replace=False)
    # gather each pair as two adjacent rows -> new buffer length 2E, pair-starts are the even rows
    fr = np.empty((2 * E,) + frames.shape[1:], frames.dtype)
    ac = np.empty((2 * E,) + actions.shape[1:], actions.dtype)
    po = np.empty((2 * E,) + pos.shape[1:], pos.dtype)
    tg = np.empty((2 * E,) + tgt.shape[1:], tgt.dtype)
    for k, s in enumerate(starts):
        fr[2 * k] = frames[s];     fr[2 * k + 1] = frames[s + 1]
        ac[2 * k] = actions[s];    ac[2 * k + 1] = actions[s + 1]   # action at the pair start is what matters
        po[2 * k] = pos[s];        po[2 * k + 1] = pos[s + 1]
        tg[2 * k] = tgt[s];        tg[2 * k + 1] = tgt[s + 1]
    new_idx = np.arange(0, 2 * E, 2, dtype=np.int64)                # even rows are valid pair-starts
    return fr, ac, po, tg, new_idx


def mix_transitions(demo_tr, expl_sub):
    """Concatenate (frames,actions,pos,tgt,idx) buffers, OFFSETTING and concatenating idx so no pair
    bridges the seam between the two sources."""
    if expl_sub is None:
        return demo_tr
    f0, a0, p0, t0, i0 = demo_tr
    f1, a1, p1, t1, i1 = expl_sub
    fr = np.concatenate([f0, f1], 0)
    ac = np.concatenate([a0, a1], 0).astype(np.float32)
    po = np.concatenate([p0, p1], 0).astype(np.float32)
    tg = np.concatenate([t0, t1], 0).astype(np.float32)
    idx = np.concatenate([i0, i1 + len(f0)]).astype(np.int64)
    return fr, ac, po, tg, idx


# ----------------------------- CEM precision fix: tuned, terminal-weighted -----------------------------
@torch.no_grad()
def cem_plan_tuned(wm, z0, target_px, aspec, device, horizon=8, iters=5, pop=256, elite=24, terminal_w=4.0):
    """Thin tuned wrapper around the r1 cem_plan recipe (mirrors its structure exactly), adding:
      - longer horizon + more iters + larger population (precision)
      - TERMINAL-weighted cost: the final predicted step's fingertip-to-target distance counts terminal_w
        times more, so the plan optimizes where the finger ENDS (pixel-precision) not just the path sum.
    Returns the first action (MPC), identical interface to cem_plan."""
    adim = wm["adim"]
    lo = torch.tensor(aspec.minimum, device=device, dtype=torch.float32)
    hi = torch.tensor(aspec.maximum, device=device, dtype=torch.float32)
    mu = torch.zeros(horizon, adim, device=device); sigma = torch.ones(horizon, adim, device=device) * 0.5
    tpx = torch.tensor(target_px, device=device).float()
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, adim, device=device)).clamp(lo, hi)
        z = z0.expand(pop, -1).clone()
        cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h])
            d = (wm["dec_arm"](z) - tpx[None]).pow(2).sum(-1).sqrt()      # predicted fingertip-to-target px
            w = terminal_w if h == horizon - 1 else 1.0                  # heavier weight on the final step
            cost = cost + w * d
        elite_idx = cost.topk(elite, largest=False).indices
        e = seqs[elite_idx]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    return mu[0].cpu().numpy()


# ----------------------------- evaluation (local copy of r1 eval_method, WM-only, takes a planner fn) -----------------------------
@torch.no_grad()
def eval_wm_cem(wm, region, n_eps, seed0, image_size, ep_len, action_repeat, device,
                planner=cem_plan, thresholds=(6.0, 12.0)):
    """Mirror of r1_imitation_fails.eval_method's wm_cem branch, parameterized by the CEM planner fn so we
    can swap in cem_plan_tuned. Same env-reset-into-region loop, same per-step encode->plan->step, same
    normalized-target convention (tpx/image_size), same success metric."""
    dists = []
    for e in range(n_eps):
        env, renderer = make_env(seed0 + 1000 + e, image_size); env.reset()
        for _ in range(200):
            if region_of(target_world(env)) == region: break
            env.reset()
        aspec = env.action_spec(); tpx = to_px(target_world(env), image_size)
        for t in range(ep_len):
            x = torch.from_numpy(render(env, renderer).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
            z0 = wm["enc"](x)
            a = planner(wm, z0, tpx / image_size, aspec, device)         # dec_arm outputs normalized [0,1]
            a = np.clip(a, aspec.minimum, aspec.maximum)
            for _ in range(action_repeat): env.physics.set_control(a); env.physics.step()
        dists.append(float(np.linalg.norm(to_px(finger_world(env), image_size) - tpx)))
    dists = np.array(dists)
    return {"mean_px": float(dists.mean()), "succ": {t: float((dists < t).mean()) for t in thresholds}}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--out", default="runs/lever1_result.json")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--demos", type=int, default=120)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--E-sweep", default="0,2000,8000,30000,full",
                    help="comma list of exploration budgets; 'full' = entire exploration buffer")
    ap.add_argument("--cem-horizon", type=int, default=8)
    ap.add_argument("--cem-iters", type=int, default=5)
    ap.add_argument("--cem-pop", type=int, default=256)
    ap.add_argument("--cem-elite", type=int, default=24)
    ap.add_argument("--terminal-w", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.bc_steps, args.demos, args.eval_eps, args.ep_len = 80, 60, 8, 4, 12
        args.E_sweep = "0,500,full"
        args.cem_horizon, args.cem_iters, args.cem_pop, args.cem_elite = 5, 3, 64, 12
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    THRESH = (6.0, 12.0)                                              # 6px = target radius; 12px = vicinity
    print(f"=== LEVER 1: cheap-exploration moat (goal-shift Reacher) | device={dev} smoke={args.smoke} ===", flush=True)

    # ---- shared baseline data: narrow expert demos (the SAME data BC and the fair-data WM both consume) ----
    print("[1/5] generating narrow-goal (TRAIN region) expert demos...", flush=True)
    demos, _ = gen_expert_demos(args.demos, "train", args.seed, args.image_size,
                                args.ep_len, args.action_repeat, rng)
    print(f"      {len(demos)} expert demos; mean final finger-target = "
          f"{np.mean([np.linalg.norm(d['final_px']-d['target_px']) for d in demos]):.1f}px", flush=True)
    demo_tr = transitions_from_demos(demos)

    # ---- imitation baseline (fixed; trained on narrow demos only) ----
    print("[2/5] training bc + bc_goal on narrow demos (fixed imitation baseline)...", flush=True)
    img = args.image_size; adim = demos[0]["actions"].shape[1]
    bc = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=False)
    bc_goal = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=True)

    # bc_goal on TRAIN (competence control) and TEST (shift baseline) -- reuse r1 eval_method
    from system1_motion.r1_imitation_fails import eval_method
    bcg_train = eval_method("bc_goal", {"bc_goal": bc_goal}, demos, "train", args.eval_eps, args.seed,
                            img, args.ep_len, args.action_repeat, dev, THRESH)
    bcg_test = eval_method("bc_goal", {"bc_goal": bc_goal}, demos, "test", args.eval_eps, args.seed,
                           img, args.ep_len, args.action_repeat, dev, THRESH)
    print(f"      bc_goal TRAIN succ@6={bcg_train['succ'][6.0]:.2f} @12={bcg_train['succ'][12.0]:.2f} | "
          f"TEST succ@6={bcg_test['succ'][6.0]:.2f} @12={bcg_test['succ'][12.0]:.2f}", flush=True)

    # ---- cheap exploration buffer ----
    print("[3/5] loading exploration buffer...", flush=True)
    expl = load_transitions(args.data)
    n_pairs = len(expl[4])
    print(f"      exploration buffer: {n_pairs} usable transition pairs", flush=True)

    # ---- parse E sweep (resolve 'full', clamp, de-dup, sort) ----
    Es = []
    for tok in args.E_sweep.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        Es.append(n_pairs if tok == "full" else min(int(tok), n_pairs))
    Es = sorted(set(Es))
    print(f"[4/5] sweeping E in {Es} (training a WM per E on demos++E_exploration)...", flush=True)

    def tuned(wm, z0, tpx, aspec, device):
        return cem_plan_tuned(wm, z0, tpx, aspec, device, horizon=args.cem_horizon, iters=args.cem_iters,
                              pop=args.cem_pop, elite=args.cem_elite, terminal_w=args.terminal_w)

    curve = []   # list of dicts per E
    for E in Es:
        sub = subsample_exploration(expl, E, np.random.RandomState(args.seed + 1))
        mixed = mix_transitions(demo_tr, sub)
        print(f"  -- E={E}: WM trains on {len(mixed[4])} pairs "
              f"({len(demo_tr[4])} demo + {0 if sub is None else len(sub[4])} expl) --", flush=True)
        wm = train_world_model(mixed, args.wm_steps, dev, tag=f"_E{E}", rollout_eval=expl)  # OOD rollout on diverse exploration
        print(f"  -- E={E}: rollout held-out={wm['rollout_px']:.1f}px  OOD(exploration)={wm['rollout_ood_px']:.1f}px "
              f"(OOD reveals off-tube dynamics) --", flush=True)
        # TEST-shift eval with BOTH the r1 default planner and the tuned (precision-fix) planner
        r_base = eval_wm_cem(wm, "test", args.eval_eps, args.seed, img, args.ep_len, args.action_repeat,
                             dev, planner=cem_plan, thresholds=THRESH)
        r_tuned = eval_wm_cem(wm, "test", args.eval_eps, args.seed, img, args.ep_len, args.action_repeat,
                              dev, planner=tuned, thresholds=THRESH)
        rec = {"E": E,
               "base":  {"succ6": r_base["succ"][6.0],  "succ12": r_base["succ"][12.0],  "mean_px": r_base["mean_px"]},
               "tuned": {"succ6": r_tuned["succ"][6.0], "succ12": r_tuned["succ"][12.0], "mean_px": r_tuned["mean_px"]}}
        curve.append(rec)
        print(f"     E={E:>7}  base @6={rec['base']['succ6']:.2f} @12={rec['base']['succ12']:.2f} "
              f"mean={rec['base']['mean_px']:.1f}  |  tuned @6={rec['tuned']['succ6']:.2f} "
              f"@12={rec['tuned']['succ12']:.2f} mean={rec['tuned']['mean_px']:.1f}", flush=True)

    # ---- pre-registered gate ----
    print("[5/5] computing pre-registered gate...", flush=True)
    bcg6, bcg12 = bcg_test["succ"][6.0], bcg_test["succ"][12.0]
    by_E = {r["E"]: r for r in curve}
    E0 = min(by_E)                                                   # the E=0 (fair-data) control row

    # use the tuned planner (precision fix) for the WM throughout the gate
    def wm6(E):  return by_E[E]["tuned"]["succ6"]
    def wm12(E): return by_E[E]["tuned"]["succ12"]

    gap6_at_E0 = wm6(E0) - bcg6
    best_E6 = max(by_E, key=wm6); best_E12 = max(by_E, key=wm12)
    best_gap6 = wm6(best_E6) - bcg6
    best_gap12 = wm12(best_E12) - bcg12
    full_E = max(by_E)
    full_gap6 = wm6(full_E) - bcg6
    full_gap12 = wm12(full_E) - bcg12

    # (M) monotone: best-E beats the E=0 control by the gate margin (cheap data is the lever, not noise)
    monotone_lever = (wm6(best_E6) - wm6(E0) >= 0.20) or (wm12(best_E12) - wm12(E0) >= 0.30)

    checks = {
        "(C) FAIR-DATA CONTROL FAILS: wm(E=0)@6 - bc_goal@6 < 0.20 (architecture alone does not win)":
            gap6_at_E0 < 0.20,
        "(G) THE MOAT EXISTS: some E* with wm@6-bcg@6 >= 0.30 OR wm@12-bcg@12 >= 0.50":
            best_gap6 >= 0.30 or best_gap12 >= 0.50,
        "(M) MONOTONE LEVER: best-E beats E=0 (wm@6 by >=0.20 or wm@12 by >=0.30) -- cheap data unlocks it":
            monotone_lever,
    }
    print("\n=== LEVER 1 PRE-REGISTERED GATE ===")
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\n  bc_goal (shift) @6={bcg6:.2f} @12={bcg12:.2f}")
    print(f"  E=0 (fair-data) gap @6 = {gap6_at_E0:+.2f}")
    print(f"  best-E @6 = E{best_E6} -> gap {best_gap6:+.2f} | best-E @12 = E{best_E12} -> gap {best_gap12:+.2f}")
    print(f"  full-E({full_E}) gap @6 = {full_gap6:+.2f}  @12 = {full_gap12:+.2f}")
    print("  curve (E -> tuned succ@6 / succ@12):")
    for r in curve:
        print(f"    E={r['E']:>7}  @6={r['tuned']['succ6']:.2f}  @12={r['tuned']['succ12']:.2f}")

    # falsification: even full-E tuned WM doesn't clear @6 NOR @12 vs imitation -> precision fix failed
    precision_fix_failed = (full_gap6 < 0.30) and (full_gap12 < 0.50)
    if precision_fix_failed and not checks["(G) THE MOAT EXISTS: some E* with wm@6-bcg@6 >= 0.30 OR wm@12-bcg@12 >= 0.50"]:
        verdict = ("FALSIFIED -- precision fix FAILED: even full-E tuned WM does not beat imitation "
                   "@6px (>=0.30) nor @12px (>=0.50). Escalate planner/cost.")
    elif all(checks.values()):
        verdict = ("MOAT CONFIRMED -- cheap goal-agnostic exploration unlocks the shifted-goal win that "
                   "imitation structurally cannot buy (fair-data control fails, E* exists, lever monotone).")
    elif checks["(G) THE MOAT EXISTS: some E* with wm@6-bcg@6 >= 0.30 OR wm@12-bcg@12 >= 0.50"]:
        verdict = ("MOAT EXISTS but not clean -- WM beats imitation at some E* yet either the E=0 control "
                   "already wins (architecture, not data) or the curve is not a clean lever. Inspect curve.")
    else:
        verdict = "INCONCLUSIVE -- no E* cleared the moat gate; not a clean falsification either."

    print(f"\n  LEVER 1 VERDICT: {verdict}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({
        "curve": curve,
        "bc_goal_test": {"succ6": bcg6, "succ12": bcg12, "mean_px": bcg_test["mean_px"]},
        "bc_goal_train": {"succ6": bcg_train["succ"][6.0], "succ12": bcg_train["succ"][12.0]},
        "gap6_at_E0": gap6_at_E0, "best_E6": int(best_E6), "best_gap6": best_gap6,
        "best_E12": int(best_E12), "best_gap12": best_gap12,
        "full_E": int(full_E), "full_gap6": full_gap6, "full_gap12": full_gap12,
        "checks": {k: bool(v) for k, v in checks.items()},
        "verdict": verdict, "args": vars(args),
    }, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
