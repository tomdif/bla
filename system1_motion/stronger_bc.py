#!/usr/bin/env python3
"""STRONGER imitation baselines vs the gated world model on SHIFTED (held-out) Reacher goals.

The moat claim: a GATED world model reaches test@12 ~0.97 zero-shot on shifted goals while plain BC
stays ~0.15-0.28. The reviewer risk: "your BC baseline is weak." This script builds the strongest
standard imitation baselines and tests whether the WM still wins -- or HONESTLY reports if one matches
(a matching baseline would WEAKEN the moat, which we want to know).

Stronger imitation baselines (all evaluated on TEST goals via the SAME eval_method as r1):
  bc_goal       goal-conditioned BC (the existing strong baseline)            -- demos=args.demos
  HER_bc        HINDSIGHT goal relabeling: relabel each demo step's goal with an ACHIEVED fingertip
                position from LATER in that same trajectory, so goal-conditioned BC learns to reach a
                RANGE of fingertip goals (the standard way to make BC generalize to new goals).
  wmrep_bc      goal-conditioned BC head on top of the FROZEN gated-WM encoder (wm["enc"]); tests
                whether the WM's REPRESENTATION (not its planner) is the advantage.
  bc_goal@300   goal-conditioned BC with MORE expert data (300 demos).
  bc_goal@1000  goal-conditioned BC with even MORE expert data (1000 demos).
  wm_cem        the gated WM (enc+dyn+decode) + CEM-MPC -- the comparison point.

Pre-registered gate (set BEFORE running):
  (W) gated WM reaches test@12 >= 0.70.
  (M) the WM beats the STRONGEST imitation baseline -- max over {HER_bc, wmrep_bc, bc_goal@1000} --
      by >= 0.30 @12 on shifted goals  => the moat SURVIVES stronger baselines.
  falsification: if ANY strong BC comes within 0.10 of the WM @12 on shift -> MOAT WEAKER THAN CLAIMED
      (honest negative; the important result if it happens).

Reuses system1_motion.r1_imitation_fails building blocks (no modification to that file).
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_motion.r1_imitation_fails import (
    make_env, render, finger_world, target_world, to_px, region_of,
    gen_expert_demos, transitions_from_demos, load_transitions,
    train_world_model, BCNet, train_bc, cem_plan, eval_method, EXTENT,
)


# ----------------------------- HER (hindsight) goal relabeling -----------------------------
def relabel_demos_her(demos, k_future=4, seed=0):
    """HINDSIGHT EXPERIENCE REPLAY relabeling for imitation.

    For each demo and each timestep t, we replace the demo's fixed `target_px` with an ACHIEVED
    fingertip position finger_px[t'] sampled from a FUTURE step t' > t of the SAME trajectory. The
    action a_t the expert took at t is, by construction, a step that moves the fingertip toward where
    it actually ended up later -- so (frame_t, achieved_goal=finger_px[t'], action_t) is a valid
    (state, goal, action) demonstration of reaching a DIFFERENT goal than the demo's nominal target.

    This is the standard HER 'future' strategy adapted to BC: it manufactures supervision for a RANGE
    of fingertip goals from the same motions, so goal-conditioned BC can generalize off the narrow
    nominal-target distribution. We emit one relabeled demo per original demo with the SAME schema
    train_bc expects (frames[T,3,H,W], actions[T,adim], target_px[2]) but a PER-STEP goal carried in a
    new 'goal_px' field; train_bc only reads frames/actions/target_px, so we additionally build a
    flat (frame, goal, action) training set and a tiny custom trainer below (her relabel is per-STEP,
    not per-EPISODE, so the goal cannot be a single target_px).
    """
    rng = np.random.RandomState(seed)
    X, A, G = [], [], []
    for d in demos:
        T = len(d["actions"])
        fpx = d["finger_px"]                                    # [T,2] achieved fingertip per step
        for t in range(T):
            hi = T - 1
            if hi <= t:                                         # last step: relabel with its own achieved pos
                tp = min(t + 1, T - 1)
            else:
                tp = int(rng.randint(t + 1, hi + 1))            # a FUTURE achieved step (HER 'future')
            X.append(d["frames"][t]); A.append(d["actions"][t]); G.append(fpx[min(tp, len(fpx) - 1)])
    X = np.asarray(X, np.float32) / 255.0
    A = np.asarray(A, np.float32)
    G = np.asarray(G, np.float32)
    return X, A, G


def train_bc_goal_flat(X, A, G, img, adim, device, steps, lr=3e-4, batch=128, log=print, seed=0):
    """Train a goal-conditioned BCNet on a FLAT (frame, goal_px, action) set (used for HER relabeling,
    whose goal varies per step). Mirrors train_bc's optimizer/seed/normalization conventions; goals are
    normalized by img exactly like train_bc and eval_method (which divides tpx by image_size)."""
    torch.manual_seed(seed)
    net = BCNet(img, adim, goal_cond=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X); At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    rng = np.random.RandomState(0)
    for step in range(steps):
        b = rng.choice(len(A), batch)
        x = Xt[b].to(device); a = At[b].to(device); g = Gt[b].to(device) / img
        loss = F.mse_loss(net(x, g), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"[HER_bc step {step}/{steps}] loss={loss.item():.4f}")
    return net.eval()


# ----------------------------- frozen-WM-encoder BC -----------------------------
class FrozenEncBC(nn.Module):
    """Goal-conditioned BC head on top of a FROZEN gated-WM encoder. Exposes the SAME forward(x, g)
    signature as BCNet so eval_method's bc_goal branch drives it unchanged. The encoder weights are the
    WM's wm['enc']; only the small MLP head trains -> isolates whether the REPRESENTATION is the edge."""
    def __init__(self, frozen_enc, d_z, adim):
        super().__init__()
        self.enc = frozen_enc                                  # already .eval(); frozen below
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(d_z + 2, 256), nn.ReLU(), nn.Linear(256, adim))

    def forward(self, x, g=None):
        with torch.no_grad():
            z = self.enc(x)
        return self.head(torch.cat([z, g], -1))


def train_wmrep_bc(wm, demos, img, adim, device, steps, lr=3e-4, batch=128, log=print, seed=0):
    """Train FrozenEncBC (frozen WM encoder + goal-conditioned MLP head) on the same narrow-goal demos
    and the same per-EPISODE target_px goals as bc_goal -> a like-for-like representation ablation."""
    torch.manual_seed(seed)
    X = np.concatenate([d["frames"] for d in demos], 0).astype(np.float32) / 255.0
    A = np.concatenate([d["actions"] for d in demos], 0).astype(np.float32)
    G = np.concatenate([np.tile(d["target_px"], (len(d["actions"]), 1)) for d in demos], 0).astype(np.float32)
    X, A, G = X[:len(A)], A, G[:len(A)]
    d_z = getattr(wm["enc"], "d_z", 384)
    net = FrozenEncBC(wm["enc"], d_z, adim).to(device)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    Xt = torch.from_numpy(X); At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    rng = np.random.RandomState(0)
    for step in range(steps):
        b = rng.choice(len(A), batch)
        x = Xt[b].to(device); a = At[b].to(device); g = Gt[b].to(device) / img
        loss = F.mse_loss(net(x, g), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"[wmrep_bc step {step}/{steps}] loss={loss.item():.4f}")
    return net.eval()


# ----------------------------- eval shim -----------------------------
def eval_bc_goal_model(net, demos, region, args, device, thresh):
    """Run eval_method's goal-conditioned-BC path on an arbitrary BCNet-compatible (x,g)->a model by
    registering it under the 'bc_goal' key. Works for bc_goal / HER_bc / wmrep_bc / bc_goal@N alike."""
    return eval_method("bc_goal", {"bc_goal": net}, demos, region, args.eval_eps, args.seed,
                       args.image_size, args.ep_len, args.action_repeat, device, thresh)


def eval_wm(wm, demos, region, args, device, thresh, perceived_goal=False):
    return eval_method("wm_cem", {"wm_cem": wm}, demos, region, args.eval_eps, args.seed,
                       args.image_size, args.ep_len, args.action_repeat, device, thresh, perceived_goal=perceived_goal)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--init-enc", default="")
    ap.add_argument("--out", default="runs/stronger_bc_result.json")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--demos", type=int, default=120)
    ap.add_argument("--demos-more", type=int, nargs="+", default=[300, 1000])  # extra-data bc_goal points
    ap.add_argument("--her-k", type=int, default=4)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # smoke: tiny everything; the WM can't reach arm_px<=5 in ~200 steps, so OPEN the gate (arm_gate/early
    # =99, max_attempts=1) so train_world_model returns instead of RAISING -- we only need it to run end-to-end.
    smoke_wm_kwargs = {}
    if args.smoke:
        args.wm_steps, args.bc_steps, args.demos, args.eval_eps, args.ep_len = 80, 60, 8, 4, 12
        args.demos_more = [10, 12]
        smoke_wm_kwargs = dict(arm_gate_px=99.0, early_px=99.0, max_attempts=1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    THRESH = (6.0, 12.0)
    print(f"=== stronger_bc: strong imitation baselines vs gated WM on SHIFTED goals | device={dev} smoke={args.smoke} ===", flush=True)

    # ---- demos at the base size (for bc_goal, HER_bc, wmrep_bc) ----
    print(f"[1/6] generating narrow-goal (TRAIN region) expert demos: base={args.demos} + extra {args.demos_more}...", flush=True)
    demos, _ = gen_expert_demos(args.demos, "train", args.seed, args.image_size, args.ep_len, args.action_repeat, rng)
    print(f"      {len(demos)} base expert demos; mean final finger-target = "
          f"{np.mean([np.linalg.norm(d['final_px']-d['target_px']) for d in demos]):.1f}px", flush=True)

    # ---- gated WM (the comparison point), trained ONCE on exploration data ----
    print("[2/6] training the GATED world model on exploration data (load_transitions)...", flush=True)
    wm = train_world_model(load_transitions(args.data), args.wm_steps, dev,
                           init_enc=args.init_enc or None, tag="_gated", seed=args.seed, **smoke_wm_kwargs)
    img, adim = wm["img"], wm["adim"]
    print(f"      WM: converged={wm.get('converged', False)} arm_px={wm['arm_px']:.1f} "
          f"rollout_px={wm['rollout_px']:.1f} rollout_ood_px={wm['rollout_ood_px']:.1f}", flush=True)

    # ---- strong imitation baselines ----
    print("[3/6] bc_goal (existing strong baseline) on base demos...", flush=True)
    bc_goal = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=True, seed=args.seed)

    print(f"[4/6] HER_bc (hindsight goal relabeling, k_future={args.her_k}) + wmrep_bc (frozen-WM-encoder head)...", flush=True)
    Xh, Ah, Gh = relabel_demos_her(demos, k_future=args.her_k, seed=args.seed)
    print(f"      HER relabeled set: {len(Ah)} (frame, achieved_goal, action) tuples", flush=True)
    her_bc = train_bc_goal_flat(Xh, Ah, Gh, img, adim, dev, args.bc_steps, seed=args.seed)
    wmrep_bc = train_wmrep_bc(wm, demos, img, adim, dev, args.bc_steps, seed=args.seed)

    print(f"[5/6] bc_goal at MORE demos {args.demos_more} (extra expert data)...", flush=True)
    bc_goal_more = {}
    for nd in args.demos_more:
        dm, _ = gen_expert_demos(nd, "train", args.seed, args.image_size, args.ep_len, args.action_repeat,
                                 np.random.RandomState(args.seed + 7))
        print(f"      trained {len(dm)} demos for bc_goal@{nd}", flush=True)
        bc_goal_more[nd] = train_bc(dm, img, adim, dev, args.bc_steps, goal_cond=True, seed=args.seed)

    # ---- evaluate ALL methods on TEST (shifted) goals ----
    print("[6/6] evaluating on TEST (shifted) goals...", flush=True)
    R = {}
    R["wm_cem"] = eval_wm(wm, demos, "test", args, dev, THRESH)
    R["bc_goal"] = eval_bc_goal_model(bc_goal, demos, "test", args, dev, THRESH)
    R["HER_bc"] = eval_bc_goal_model(her_bc, demos, "test", args, dev, THRESH)
    R["wmrep_bc"] = eval_bc_goal_model(wmrep_bc, demos, "test", args, dev, THRESH)
    for nd in args.demos_more:
        R[f"bc_goal@{nd}"] = eval_bc_goal_model(bc_goal_more[nd], demos, "test", args, dev, THRESH)
    for m, r in R.items():
        print(f"      [test] {m:14} succ@6={r['succ'][6.0]:.2f} succ@12={r['succ'][12.0]:.2f} mean_px={r['mean_px']:.1f}", flush=True)

    # ---- pre-registered gate ----
    def s12(m): return R[m]["succ"][12.0]
    wm12 = s12("wm_cem")
    # strongest strong-imitation baseline @12 on shift. include the largest extra-data bc_goal point.
    largest_more = max(args.demos_more) if args.demos_more else args.demos
    strong_set = {"HER_bc": s12("HER_bc"), "wmrep_bc": s12("wmrep_bc"),
                  f"bc_goal@{largest_more}": s12(f"bc_goal@{largest_more}")}
    strongest_name = max(strong_set, key=strong_set.get)
    strongest = strong_set[strongest_name]
    margin = wm12 - strongest

    checks = {
        "(W) gated WM reaches test@12 >= 0.70": wm12 >= 0.70,
        "(M) WM beats STRONGEST imitation (max{HER_bc,wmrep_bc,bc_goal@%d}) by >= 0.30 @12 on shift" % largest_more:
            margin >= 0.30,
    }
    # falsification: any strong BC within 0.10 of the WM @12 on shift
    within = {k: v for k, v in strong_set.items() if (wm12 - v) <= 0.10}
    moat_weaker = len(within) > 0

    print("\n=== stronger_bc PRE-REGISTERED GATE ===")
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\n  WM test@12 = {wm12:.2f} | strongest imitation = {strongest_name} @ {strongest:.2f} | margin = {margin:+.2f}")
    print(f"  strong-baseline @12 table: " + "  ".join(f"{k}={v:.2f}" for k, v in strong_set.items()))

    if moat_weaker:
        verdict = ("MOAT WEAKER THAN CLAIMED -- strong BC within 0.10 of WM @12 on shift: "
                   + ", ".join(f"{k}={strong_set[k]:.2f}" for k in within) + f" (WM={wm12:.2f})")
    elif all(checks.values()):
        verdict = f"MOAT SURVIVES STRONGER BASELINES -- WM beats strongest imitation ({strongest_name}) by {margin:+.2f} @12 on shift"
    elif not checks["(W) gated WM reaches test@12 >= 0.70"]:
        verdict = f"INCONCLUSIVE -- WM did not clear test@12>=0.70 (got {wm12:.2f})"
    else:
        verdict = f"INCONCLUSIVE -- WM clears (W) but margin {margin:+.2f} < 0.30 (no strong BC within 0.10 either)"
    print(f"\n  stronger_bc VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "results": R,
        "wm_test12": wm12,
        "strong_set_test12": strong_set,
        "strongest_name": strongest_name,
        "strongest_test12": strongest,
        "margin_test12": margin,
        "moat_weaker": moat_weaker,
        "within_0p10": within,
        "wm_info": {"converged": bool(wm.get("converged", False)), "arm_px": wm["arm_px"],
                    "rollout_px": wm["rollout_px"], "rollout_ood_px": wm["rollout_ood_px"],
                    "attempts": wm.get("attempts")},
        "checks": {k: bool(v) for k, v in checks.items()},
        "verdict": verdict,
        "args": vars(args),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
