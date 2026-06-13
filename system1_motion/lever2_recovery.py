#!/usr/bin/env python3
"""LEVER-2 / R4 -- the PURE-ARCHITECTURE win: planning RECOVERS, reactive imitation does NOT.

Mechanism under test (isolated from the R1 goal-shift result):
  A REACTIVE policy (BC) maps the current observation to an action; its demos only ever traversed the
  on-demo manifold (expert trajectories from reset -> target). A PLANNER (world-model + CEM-MPC) re-plans
  from WHATEVER state it is in, every step, using goal-INVARIANT learned dynamics. So if we KNOCK the arm
  to an off-demo state mid-episode, the planner can re-plan a recovery path to the (fixed) target, while the
  reactive policy has no demonstrated behaviour for that region of state space and collapses.

  This is deliberately NOT a goal-shift experiment. Goals stay in the TRAIN region for ALL methods so the
  only thing being tested is RECOVERY / LOOKAHEAD -- the structural advantage of re-planning over reacting.

Methods (reuse R1 harness verbatim; do NOT reimplement WM/CEM/BC):
  bc        plain behavioral cloning: image -> action           (reactive)
  bc_goal   goal-conditioned BC: image + PERCEIVED target -> a   (reactive, the real baseline)
  wm_cem    JEPA world model (enc + LatentDynamics) + CEM-MPC    (re-plans every step from current state)

Perturbation (identical, seeded, applied to ALL methods):
  At a random mid-episode step t_p (seeded per eval-episode), the arm joints (qpos[:2]) are SET to random
  angles drawn UNIFORMLY over the full joint range, and the corresponding joint velocities (qvel[:2]) are
  zeroed, then env.physics.forward() recomputes geometry. The TARGET (slider qpos/qvel) is left untouched, so
  the goal does not move. Because the angles are sampled uniformly over the WHOLE configuration space -- not
  along an expert trajectory -- the post-perturbation state is overwhelmingly OFF the demo manifold (BC never
  saw it). The same RNG seed => the same t_p and the same knocked-to angles for every method on a given
  episode, so the comparison is fair: all methods face the identical disturbance and must recover from it.

Pre-registered gate (set BEFORE running):
  (C) CONTROL -- WITHOUT perturbation BC is competent in-region (so any collapse is perturbation-specific,
      not a generic BC weakness): bc_goal_noperturb succ@12 >= 0.30
  (R) RECOVERY -- UNDER perturbation the planner recovers while the reactive policy collapses:
      wm_cem_perturb succ@12 >= 0.50  AND  bc_goal_perturb succ@12 <= 0.20
  (G) STRUCTURAL EDGE IS IN RECOVERY -- the WM's advantage UNDER perturbation exceeds its advantage WITHOUT
      perturbation: gap_perturb > gap_noperturb   (gap = wm_cem - bc_goal, @12px)
  Falsification: if BC recovers as well as the WM under perturbation (bc_goal_perturb succ@12 >= 0.50 or
      gap_perturb <= gap_noperturb), planning has NO structural recovery advantage here -- report it.

Reuses system1_motion.r1_imitation_fails (make_env/render/finger_world/target_world/to_px/region_of/
gen_expert_demos/load_transitions/train_world_model/train_bc/cem_plan). Mirrors eval_method exactly,
adding only the perturbation injection.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch

from system1_motion.r1_imitation_fails import (
    make_env, render, finger_world, target_world, to_px, region_of,
    gen_expert_demos, load_transitions, train_world_model, train_bc, cem_plan,
)


# ----------------------------- perturbation (off-demo, seeded, method-agnostic) -----------------------------
def apply_perturbation(env, rng):
    """Knock the ARM (and only the arm) to a random off-demo configuration; leave the target fixed.

    dm_control reacher state = [qpos | qvel]. qpos = [arm_joint0, arm_joint1, target_x, target_y] (the target
    is two slide joints), qvel mirrors that ordering. We randomize qpos[:2] over the full joint range and zero
    qvel[:2], so the FINGERTIP jumps to an arbitrary pose while the GOAL is untouched. Sampling uniformly over
    the configuration space (not along an expert path) puts the state off the BC demo manifold."""
    s = env.physics.get_state().copy()
    nq = env.physics.model.nq                       # number of position coords
    # joint limits for the two arm hinges; fall back to [-pi, pi] if ranges are not set.
    jr = np.asarray(env.physics.model.jnt_range)    # [njnt, 2]
    def lim(i):
        lo, hi = float(jr[i, 0]), float(jr[i, 1])
        return (lo, hi) if hi > lo else (-np.pi, np.pi)
    lo0, hi0 = lim(0); lo1, hi1 = lim(1)
    s[0] = rng.uniform(lo0, hi0)                     # arm joint 0
    s[1] = rng.uniform(lo1, hi1)                     # arm joint 1
    s[nq + 0] = 0.0                                  # zero arm joint 0 velocity
    s[nq + 1] = 0.0                                  # zero arm joint 1 velocity
    env.physics.set_state(s)
    env.physics.forward()                           # recompute geom_xpos for the new pose


# ----------------------------- evaluation in the live env (mirror eval_method + perturbation) -----------------------------
@torch.no_grad()
def eval_method_perturb(method, models, region, n_eps, seed0, image_size, ep_len, action_repeat,
                        device, perturb, thresholds=(6.0, 12.0)):
    """Identical to r1_imitation_fails.eval_method (same env stepping, same final-distance success metric)
    EXCEPT: when perturb=True, at a seeded mid-episode step t_p the arm is knocked off-demo. The perturbation
    RNG is derived from (seed0, e) ONLY -- not from the method -- so t_p and the knocked-to pose are identical
    across methods for the same episode."""
    dists = []
    for e in range(n_eps):
        env, renderer = make_env(seed0 + 1000 + e, image_size); env.reset()
        for _ in range(200):                                                  # reset until target in eval region
            if region_of(target_world(env)) == region: break
            env.reset()
        aspec = env.action_spec(); tpx = to_px(target_world(env), image_size)
        # seeded, method-INDEPENDENT perturbation schedule for this episode
        prng = np.random.RandomState(seed0 * 100003 + e)
        # knock somewhere in the middle third so the agent has time before AND after to (try to) recover
        t_p = int(prng.randint(max(1, ep_len // 4), max(2, (3 * ep_len) // 4))) if perturb else -1
        for t in range(ep_len):
            if perturb and t == t_p:
                apply_perturbation(env, prng)                                 # off-demo knock (all methods)
            x = torch.from_numpy(render(env, renderer).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
            if method.startswith("wm_cem"):
                wm = models[method]; z0 = wm["enc"](x)
                a = cem_plan(wm, z0, tpx / image_size, aspec, device)         # re-plans from CURRENT (perturbed) state
            elif method == "bc":
                a = models["bc"](x).cpu().numpy()[0]
            elif method == "bc_goal":
                g = torch.tensor(tpx / image_size, device=device).float()[None]
                a = models["bc_goal"](x, g).cpu().numpy()[0]
            a = np.clip(a, aspec.minimum, aspec.maximum)
            for _ in range(action_repeat): env.physics.set_control(a); env.physics.step()
        dists.append(float(np.linalg.norm(to_px(finger_world(env), image_size) - tpx)))
    dists = np.array(dists)
    return {"mean_px": float(dists.mean()), "succ": {t: float((dists < t).mean()) for t in thresholds}}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--out", default="runs/lever2_recovery.json")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--demos", type=int, default=120)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.bc_steps, args.demos, args.eval_eps, args.ep_len = 80, 60, 8, 4, 12
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    THRESH = (6.0, 12.0)                                                       # 6px = target radius; 12px = vicinity
    REGION = "train"                                                           # TRAIN-region goals ONLY (no goal-shift)
    print(f"=== LEVER-2 (R4): recovery / lookahead -- planning vs reactive | device={dev} smoke={args.smoke} ===", flush=True)

    print("[1/4] generating TRAIN-region expert demos (BC + goal-BC train set)...", flush=True)
    demos, _ = gen_expert_demos(args.demos, REGION, args.seed, args.image_size, args.ep_len, args.action_repeat, rng)
    print(f"      {len(demos)} expert demos; mean final finger-target = "
          f"{np.mean([np.linalg.norm(d['final_px']-d['target_px']) for d in demos]):.1f}px", flush=True)

    print("[2/4] WM (rich exploration -> goal-invariant dynamics; full data is fine -- claim is planning-vs-reactive)...", flush=True)
    wm = train_world_model(load_transitions(args.data), args.wm_steps, dev, tag="_wm")
    img, adim = wm["img"], wm["adim"]

    print("[3/4] training BC and goal-conditioned BC on TRAIN-region demos...", flush=True)
    bc = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=False)
    bc_goal = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=True)
    models = {"wm_cem": wm, "bc": bc, "bc_goal": bc_goal}

    print("[4/4] evaluating WITHOUT perturbation (control) and UNDER perturbation (recovery), TRAIN goals...", flush=True)
    R = {}
    for cond, perturb in (("noperturb", False), ("perturb", True)):
        R[cond] = {}
        for m in ("bc", "bc_goal", "wm_cem"):
            r = eval_method_perturb(m, models, REGION, args.eval_eps, args.seed, args.image_size,
                                    args.ep_len, args.action_repeat, dev, perturb, THRESH)
            R[cond][m] = r
            print(f"      [{cond:9}] {m:8} succ@6={r['succ'][6.0]:.2f} succ@12={r['succ'][12.0]:.2f} mean_px={r['mean_px']:.1f}", flush=True)

    def s(cond, m, t): return R[cond][m]["succ"][t]
    gap_noperturb = s("noperturb", "wm_cem", 12.0) - s("noperturb", "bc_goal", 12.0)
    gap_perturb   = s("perturb",   "wm_cem", 12.0) - s("perturb",   "bc_goal", 12.0)
    checks = {
        "(C) CONTROL: BC competent in-region w/o perturbation (bc_goal noperturb succ@12 >= 0.30)":
            s("noperturb", "bc_goal", 12.0) >= 0.30,
        "(R) RECOVERY: under perturbation WM recovers (wm_cem succ@12 >= 0.50) AND BC collapses (bc_goal succ@12 <= 0.20)":
            s("perturb", "wm_cem", 12.0) >= 0.50 and s("perturb", "bc_goal", 12.0) <= 0.20,
        "(G) STRUCTURAL: recovery gap exceeds no-perturbation gap (gap_perturb > gap_noperturb)":
            gap_perturb > gap_noperturb,
    }
    print("\n=== LEVER-2 (R4) PRE-REGISTERED GATE ===")
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    bc_recovers = s("perturb", "bc_goal", 12.0) >= 0.50 or gap_perturb <= gap_noperturb
    if bc_recovers:
        verdict = "FALSIFIED (reactive BC recovers as well as the planner -- no structural recovery advantage)"
    elif all(checks.values()):
        verdict = "RECOVERY IS A PLANNING WIN -- WM re-plans out of off-demo states, reactive BC cannot"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n  gap@12px: noperturb wm_cem-bc_goal = {gap_noperturb:+.2f} | perturb wm_cem-bc_goal = {gap_perturb:+.2f}")
    print(f"  LEVER-2 VERDICT: {verdict}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"results": R, "gap_noperturb": gap_noperturb, "gap_perturb": gap_perturb,
               "checks": {k: bool(v) for k, v in checks.items()}, "verdict": verdict, "args": vars(args)},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
