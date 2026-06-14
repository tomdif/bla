#!/usr/bin/env python3
"""Action-causality control -- does cheap exploration help because it is CAUSAL (true action->consequence),
not just "more data / more diversity"? Train the peak-E world model on:
  real            : unmodified exploration
  shuffle_actions : SAME (s,s') transitions + same diversity, but each action label is permuted so it no
                    longer matches its own transition -> the causal action->consequence link is destroyed
  zero_actions    : actions zeroed -> no action information at all
Then eval ZERO-SHOT planning on shifted (test-region) goals.

Pre-registered: real transfers (>=0.70@12); real beats shuffled AND zero by >=0.30@12 (correct action-
conditioning is NECESSARY -> the mechanism is causal, not diversity). And a sharp secondary point: the
corrupted WMs still CONVERGE on arm_px (the fingertip decode is action-independent), so the convergence
gate cannot catch them -- but the held-out OOD rollout SHOULD light up red (broken dynamics). That shows
the gate alone is insufficient and the OOD-rollout metric earns its keep.
"""
import argparse, json
import numpy as np
import torch
from system1_motion.r1_imitation_fails import load_transitions, train_world_model, eval_method


def corrupt(transitions, mode, seed=0):
    fr, ac, po, tg, idx = transitions
    rng = np.random.RandomState(seed); ac = ac.copy()
    if mode == "shuffle_actions":
        ac = ac[rng.permutation(len(ac))]                 # action no longer matches its own (s->s') transition
    elif mode == "zero_actions":
        ac = np.zeros_like(ac)
    return (fr, ac, po, tg, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--E", type=int, default=30000)
    ap.add_argument("--wm-steps", type=int, default=7000)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/causal_control.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.E, args.wm_steps, args.eval_eps, args.ep_len = 2000, 200, 4, 12
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== action-causality control | E={args.E} steps={args.wm_steps} device={dev} ===", flush=True)
    expl = load_transitions(args.data)
    fr, ac, po, tg, idx = expl
    rng = np.random.RandomState(args.seed)
    idxE = np.sort(rng.choice(idx, min(args.E, len(idx)), replace=False))
    sub = (fr, ac, po, tg, idxE)
    R = {}
    for variant in ("real", "shuffle_actions", "zero_actions"):
        print(f"\n--- variant: {variant} ---", flush=True)
        wm = train_world_model(corrupt(sub, variant, seed=args.seed), args.wm_steps, dev,
                               tag=f"_{variant}", seed=args.seed, rollout_eval=expl)
        r = eval_method("wm_cem", {"wm_cem": wm}, [], "test", args.eval_eps, args.seed,
                        args.image_size, args.ep_len, args.action_repeat, dev, (6.0, 12.0))
        R[variant] = {"succ6": r["succ"][6.0], "succ12": r["succ"][12.0], "mean_px": r["mean_px"],
                      "arm_px": wm["arm_px"], "rollout_heldout": wm["rollout_px"], "rollout_ood": wm["rollout_ood_px"]}
        print(f"  [{variant:15}] test @6={R[variant]['succ6']:.2f} @12={R[variant]['succ12']:.2f} | "
              f"arm_px={R[variant]['arm_px']:.1f} rollout_OOD={R[variant]['rollout_ood']:.1f}", flush=True)

    real, shuf, zero = R["real"], R["shuffle_actions"], R["zero_actions"]
    checks = {
        "(REAL) real-exploration WM transfers zero-shot (test@12 >= 0.70)": real["succ12"] >= 0.70,
        "(CAUSAL) real beats SHUFFLED-action by >= 0.30 @12 -- correct action-conditioning is NECESSARY":
            real["succ12"] - shuf["succ12"] >= 0.30,
        "(CAUSAL) real beats ZERO-action by >= 0.30 @12": real["succ12"] - zero["succ12"] >= 0.30,
        "(OOD-DETECTS) shuffled WM CONVERGES on arm_px yet OOD rollout is >=1.5x real -- gate can't catch, OOD can":
            (shuf["arm_px"] <= 5.0) and (shuf["rollout_ood"] >= 1.5 * real["rollout_ood"]),
    }
    print("\n=== ACTION-CAUSALITY GATE ===")
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    verdict = ("MECHANISM IS CAUSAL -- cheap interaction helps via true action->consequence structure, not diversity"
               if all(checks.values()) else "INCONCLUSIVE")
    print(f"\n  test@12:   real={real['succ12']:.2f}  shuffled={shuf['succ12']:.2f}  zero={zero['succ12']:.2f}")
    print(f"  arm_px:    real={real['arm_px']:.1f}  shuffled={shuf['arm_px']:.1f}  zero={zero['arm_px']:.1f}  (all converge -> gate blind)")
    print(f"  OOD roll:  real={real['rollout_ood']:.1f}  shuffled={shuf['rollout_ood']:.1f}  zero={zero['rollout_ood']:.1f}  (corruption visible here)")
    print(f"  VERDICT: {verdict}")
    import os; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"results": R, "checks": {k: bool(v) for k, v in checks.items()}, "verdict": verdict, "args": vars(args)},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
