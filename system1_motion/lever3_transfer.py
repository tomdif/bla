#!/usr/bin/env python3
"""Lever-3 -- BICAMERAL ZERO-SHOT GOAL TRANSFER: quantifying the adaptation-cost asymmetry.

The moat. R1 established that a frozen world model (System-1) reaches BOTH train and shifted goals via
CEM-MPC over learned dynamics, while goal-conditioned imitation collapses on shifted goals. This script
turns that qualitative gap into a COST CURVE: how much NEW supervision does imitation have to buy to do
what the world model does for free?

Setup (reuses system1_motion.r1_imitation_fails -- do NOT reimplement):
  System-1 (WM): trained ONCE on exploration (train region) dynamics; adapts to NEW goals (test region)
                 by re-planning. N=0 new demos, 0 retraining. -> eval_method("wm_cem", region="test").
  Imitation (bc_goal): trained on TRAIN-region demos; to adapt to test goals it must COLLECT N new
                 test-region expert demos and RETRAIN. We SWEEP N in {0,10,30,100,300}. For each N:
                 generate N test-region demos, (re)train bc_goal on (train demos + N test demos),
                 eval on test goals -> success(N).

The "moat number" N* = the smallest N at which retrained bc_goal MATCHES the frozen WM's zero-shot test
success. The WM pays N=0 + 0 retrains; imitation pays N* demos + a retrain. If imitation never matches
even at the largest N, that is a STRONGER moat (report it).

Pre-registered gate (set BEFORE running):
  (W) ZERO-SHOT WM: frozen WM reaches test succ@12 >= 0.50 at N=0 (no adaptation at all)
  (C) NO-FREE-TRANSFER sanity: at N=0, bc_goal test succ@12 is LOW (<= 0.30) -- it is the thing being adapted
  (A) ASYMMETRY/MOAT: imitation needs N* >= 30 new test-region demos + a retrain to MATCH the WM's
      zero-shot test success; if it cannot match even at the largest N, the moat is unbounded (stronger)
  falsification: if bc_goal matches the WM at N ~ 0 (transfers for free), there is NO asymmetry/moat.

Reuses the exact WM anti-collapse recipe and CEM planner from r1_imitation_fails (imported, untouched).
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch

from system1_motion.r1_imitation_fails import (
    make_env, render, finger_world, target_world, to_px, region_of,
    gen_expert_demos, transitions_from_demos, load_transitions,
    train_world_model, BCNet, train_bc, cem_plan, eval_method, EXTENT,
)


# ----------------------------- BC retrain on (train demos + N test demos) -----------------------------
def retrain_bc_goal(train_demos, test_demos, img, adim, device, steps, seed, log=print):
    """(Re)train goal-conditioned BC FROM SCRATCH on the union of train demos + N adapted test demos.
    From-scratch (not finetune) is the conservative / charitable-to-imitation choice: it gives imitation
    the *full* combined dataset with no stale-init handicap, so N* is a clean lower bound on what
    imitation must pay. Reuses train_bc unchanged (goal_cond=True)."""
    demos = list(train_demos) + list(test_demos)
    # train_bc seeds its sampler from RandomState(0) internally; pass a fresh net each call (train_bc builds one).
    return train_bc(demos, img, adim, device, steps, goal_cond=True, log=log)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--init-enc", default="")
    ap.add_argument("--out", default="runs/lever3_transfer.json")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--train-demos", type=int, default=120)
    ap.add_argument("--n-sweep", default="0,10,30,100,300", help="comma-sep N of NEW test-region demos for BC adaptation")
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--match-thresh", type=float, default=12.0, help="px threshold the WM-match is judged at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.bc_steps, args.train_demos = 80, 60, 8
        args.eval_eps, args.ep_len, args.n_sweep = 4, 12, "0,4"
    N_SWEEP = [int(x) for x in args.n_sweep.split(",") if x.strip() != ""]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    THRESH = (6.0, 12.0)
    MT = args.match_thresh
    print(f"=== Lever-3: bicameral zero-shot goal transfer (adaptation-cost asymmetry) | device={dev} smoke={args.smoke} ===", flush=True)
    print(f"    N sweep (NEW test-region demos for imitation) = {N_SWEEP} | match threshold = {MT:.0f}px", flush=True)

    # ----- System-1: train WM ONCE on exploration; train baseline bc_goal on TRAIN-region demos -----
    print("[1/5] generating TRAIN-region expert demos (imitation's home turf)...", flush=True)
    train_demos, _ = gen_expert_demos(args.train_demos, "train", args.seed, args.image_size, args.ep_len, args.action_repeat, rng)
    print(f"      {len(train_demos)} train demos; mean final finger-target = "
          f"{np.mean([np.linalg.norm(d['final_px']-d['target_px']) for d in train_demos]):.1f}px", flush=True)

    print("[2/5] System-1 WM trained ONCE on exploration dynamics (goal-invariant)...", flush=True)
    wm = train_world_model(load_transitions(args.data), args.wm_steps, dev, init_enc=args.init_enc or None, tag="_s1")
    img, adim = wm["img"], wm["adim"]
    models = {"wm_cem": wm}

    # ----- WM zero-shot adaptation = re-plan on TEST goals. N=0 new demos, 0 retraining. -----
    print("[3/5] WM ZERO-SHOT adaptation to NEW (test) goals: just re-plan, N=0 new demos, 0 retrain...", flush=True)
    wm_test = eval_method("wm_cem", models, train_demos, "test", args.eval_eps, args.seed,
                          args.image_size, args.ep_len, args.action_repeat, dev, THRESH)
    wm_train = eval_method("wm_cem", models, train_demos, "train", args.eval_eps, args.seed,
                           args.image_size, args.ep_len, args.action_repeat, dev, THRESH)
    print(f"      [WM zero-shot] TRAIN succ@6={wm_train['succ'][6.0]:.2f} succ@12={wm_train['succ'][12.0]:.2f} | "
          f"TEST succ@6={wm_test['succ'][6.0]:.2f} succ@12={wm_test['succ'][12.0]:.2f}", flush=True)
    wm_test_match = wm_test["succ"][MT]   # the bar imitation must clear

    # ----- Imitation adaptation cost curve: sweep N = new test-region demos, retrain, eval on test -----
    print(f"[4/5] IMITATION adaptation: sweep N test-region demos, retrain bc_goal, eval on TEST goals...", flush=True)
    # pre-generate the largest pool ONCE, then take prefixes -> N=10 demos are a subset of N=30, etc. (nested, fair).
    max_n = max(N_SWEEP) if N_SWEEP else 0
    pool = []
    if max_n > 0:
        pool, _ = gen_expert_demos(max_n, "test", args.seed + 7777, args.image_size, args.ep_len, args.action_repeat, rng)
        if len(pool) < max_n:
            print(f"      WARNING: requested {max_n} test demos but only generated {len(pool)} (rejection-sampling cap)", flush=True)
    curve = []
    for N in N_SWEEP:
        n_avail = min(N, len(pool))
        test_demos = pool[:n_avail]
        bc_goal = retrain_bc_goal(train_demos, test_demos, img, adim, dev, args.bc_steps, args.seed, log=print)
        models_bc = {"bc_goal": bc_goal}
        r_test = eval_method("bc_goal", models_bc, train_demos + test_demos, "test", args.eval_eps, args.seed,
                             args.image_size, args.ep_len, args.action_repeat, dev, THRESH)
        r_train = eval_method("bc_goal", models_bc, train_demos + test_demos, "train", args.eval_eps, args.seed,
                              args.image_size, args.ep_len, args.action_repeat, dev, THRESH)
        row = {"N": N, "n_avail": n_avail,
               "test_succ6": r_test["succ"][6.0], "test_succ12": r_test["succ"][12.0], "test_mean_px": r_test["mean_px"],
               "train_succ6": r_train["succ"][6.0], "train_succ12": r_train["succ"][12.0]}
        curve.append(row)
        print(f"      [bc_goal N={N:4d} (avail {n_avail})] TEST succ@6={r_test['succ'][6.0]:.2f} "
              f"succ@12={r_test['succ'][12.0]:.2f} mean_px={r_test['mean_px']:.1f} | match-bar({MT:.0f}px)={wm_test_match:.2f}", flush=True)

    # ----- moat number N*: smallest N where retrained bc_goal matches WM zero-shot test success @ MT -----
    bc_n0 = next((c[f"test_succ{int(MT)}"] for c in curve if c["N"] == 0), None)
    N_star = None
    for c in curve:
        if c[f"test_succ{int(MT)}"] >= wm_test_match:
            N_star = c["N"]; break
    matched = N_star is not None
    best_bc = max((c[f"test_succ{int(MT)}"] for c in curve), default=0.0)

    # ----- pre-registered checks -----
    checks = {
        "(W) ZERO-SHOT WM reaches test succ@12 >= 0.50 at N=0 (no adaptation)": wm_test["succ"][12.0] >= 0.50,
        "(C) NO-FREE-TRANSFER: bc_goal test succ@12 at N=0 is LOW (<= 0.30)": (bc_n0 is not None and bc_n0 <= 0.30),
        "(A) MOAT: imitation needs N* >= 30 new test demos + retrain to match WM (or cannot match at all)":
            (matched and N_star >= 30) or (not matched),
    }
    if not matched:
        moat_str = (f"UNBOUNDED -- imitation NEVER matched WM zero-shot ({wm_test_match:.2f}@{MT:.0f}px) "
                    f"even at N={max_n} (best bc_goal={best_bc:.2f}); the moat is the FULL curve plus a retrain.")
        verdict = "MOAT (UNBOUNDED) -- imitation cannot buy parity within the sweep"
    elif N_star is None or N_star < 10:
        moat_str = f"~0 -- imitation matched WM at N*={N_star}; NO meaningful asymmetry."
        verdict = "FALSIFIED -- imitation transfers ~for free; no asymmetry/moat"
    else:
        moat_str = (f"N* = {N_star} NEW test-region expert demos + 1 retrain to match WM zero-shot "
                    f"({wm_test_match:.2f}@{MT:.0f}px); WM paid 0 demos + 0 retrains.")
        verdict = "MOAT -- bicameral WM adapts for free; imitation pays N* demos + retrain"
    if checks["(W) ZERO-SHOT WM reaches test succ@12 >= 0.50 at N=0 (no adaptation)"] is False:
        verdict = "INCONCLUSIVE -- WM did not clear the zero-shot floor (CEM precision-bound); re-examine"

    print("\n=== Lever-3 PRE-REGISTERED GATE ===")
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\n  WM zero-shot test ({MT:.0f}px) = {wm_test_match:.2f} | bc_goal@N=0 = "
          f"{bc_n0 if bc_n0 is not None else float('nan'):.2f} | best bc_goal over sweep = {best_bc:.2f}")
    print(f"  MOAT NUMBER: {moat_str}")
    print(f"  Lever-3 VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({
        "wm_zero_shot": {"train": wm_train, "test": wm_test},
        "wm_test_match_succ": wm_test_match, "match_thresh_px": MT,
        "bc_curve": curve, "N_sweep": N_SWEEP,
        "bc_test_succ_at_N0": bc_n0, "best_bc_test_succ": best_bc,
        "N_star": N_star, "matched": matched, "moat": moat_str,
        "checks": {k: bool(v) for k, v in checks.items()}, "verdict": verdict, "args": vars(args),
    }, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
