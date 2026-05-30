#!/usr/bin/env python3
"""GC-IDM on the REAL pretrained OF-JEPA substrate (Navigate domain).

First runnable, non-synthetic test of the GC-IDM head (system1_jepa/gc_idm.py)
on the actual frozen OF-JEPA checkpoint (runs/of_jepa_navigate_pretrained.pt).
Navigation is the paper's FAVORABLE regime (smooth, low-contact) — the
contact-rich Robosuite test is out of reach here (no checkpoint/demo data).

Protocol:
  * frozen OF-JEPA substrate encodes 128x128 navigate frames -> object files
  * expert (greedy oracle) generates demos; tuples are (z_t, z_goal, h, a_t=dxy)
    with z_goal = encoding of the agent-on-target frame, h = steps remaining
  * train GC-IDM in flatten (paper-faithful) and perfile (OF-JEPA-native:
    identity-aligned files + Sinkhorn-confidence pooling) modes
  * closed-loop eval: success rate vs expert oracle, a no-goal control
    (z_goal := z_t, to confirm goal-conditioning is load-bearing), and random.

This tests the paper's central claim in our substrate: is the OF-JEPA latent
geometry amortization-friendly enough that a one-forward inverse map reaches
expert-level control? It does NOT include a CEM-over-world-model baseline (needs
an action-conditioned latent predictor wired separately) — so it answers "does
amortization work here", not yet "does it match search at 100x less cost".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from system1_jepa.navigate_env import NavigateEnv, NavigateSpec
from system1_jepa.of_jepa.interfaces import OFJEPAObjectFiles
from system1_jepa.gc_idm import GCInverseDynamics, train_gc_idm_supervised


def make_spec():
    # image_size/patch must match the checkpoint's visual scale; move_max/max_steps
    # sized so the task is solvable in a finite horizon at 128px.
    return NavigateSpec(image_size=128, patch_size=8, max_steps=20,
                        success_radius=6.0, move_max=10.0)


@torch.no_grad()
def goal_latents(sub, env, spec, device):
    """Encode the agent-on-target frame (one per env) as z_goal."""
    x0, y0 = env.x.clone(), env.y.clone()
    env.x, env.y = env.tx.clone(), env.ty.clone()
    gframe = env.observe()
    env.x, env.y = x0, y0
    sub.reset_episode(batch_size=gframe.shape[0])
    ofb = sub.observe(gframe.to(device))
    return ofb


def flat(ofb):
    return ofb.full_slot.flatten(start_dim=1)          # [B, N*(id+state)]


@torch.no_grad()
def collect_demos(sub, spec, device, n_rollouts, batch, seed):
    """Batched expert rollouts -> (z_t, z_goal, h, a) tuples for both modes."""
    Sf, Gf, Hh, Aa = [], [], [], []          # flatten state, goal, horizon, action
    Sp, Gp, Cp = [], [], []                  # perfile slots, goal slots, conf
    for r in range(n_rollouts):
        env = NavigateEnv(spec, batch_size=batch, device=device, seed=seed + r)
        sub.reset_episode(batch_size=batch)
        gofb = goal_latents(sub, env, spec, device)
        gflat, gslot, gconf = flat(gofb), gofb.full_slot, gofb.confidences
        sub.reset_episode(batch_size=batch)
        obs = env.observe().to(device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(spec.max_steps):
            ofb = sub.observe(obs)
            a = env.expert_action()                      # [B,2] oracle
            h = torch.full((batch,), float(spec.max_steps - t), device=device)
            keep = ~done
            if keep.any():
                Sf.append(flat(ofb)[keep]); Gf.append(gflat[keep])
                Hh.append(h[keep]); Aa.append(a[keep])
                Sp.append(ofb.full_slot[keep]); Gp.append(gslot[keep]); Cp.append(gofb.confidences[keep])
            obs, _, d = env.step(a)
            obs = obs.to(device)
            done = done | d
            if done.all():
                break
    return (torch.cat(Sf), torch.cat(Gf), torch.cat(Hh), torch.cat(Aa),
            torch.cat(Sp), torch.cat(Gp), torch.cat(Cp))


@torch.no_grad()
def eval_policy(sub, head, spec, device, n_episodes, batch, seed, mode, no_goal=False, random=False):
    succ = 0; total = 0; steps_to_succ = []
    n_roll = (n_episodes + batch - 1) // batch
    for r in range(n_roll):
        env = NavigateEnv(spec, batch_size=batch, device=device, seed=seed + 9999 + r)
        gofb = goal_latents(sub, env, spec, device)
        sub.reset_episode(batch_size=batch)
        obs = env.observe().to(device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        reached = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(spec.max_steps):
            ofb = sub.observe(obs)
            if random:
                a = (torch.rand(batch, 2, device=device) * 2 - 1) * spec.move_max
            else:
                if mode == "perfile":
                    st, gl, cf = ofb.full_slot, (ofb.full_slot if no_goal else gofb.full_slot), gofb.confidences
                else:
                    st, gl, cf = flat(ofb), (flat(ofb) if no_goal else flat(gofb)), None
                h = torch.full((batch,), float(spec.max_steps - t), device=device)
                a = head(st, gl, h, cf)                  # [B,2] one forward
            obs, _, d = env.step(a); obs = obs.to(device)
            newly = d & (~done)
            dist = ((env.x - env.tx) ** 2 + (env.y - env.ty) ** 2).sqrt()
            reached = reached | (dist < spec.success_radius)
            done = done | d
            if done.all():
                break
        succ += int(reached.sum()); total += batch
    return succ / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/of_jepa_navigate_pretrained.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--demo-rollouts", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-episodes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--out", default="artifacts/gc_idm_navigate.json")
    args = ap.parse_args()
    dev = args.device
    spec = make_spec()
    sub = OFJEPAObjectFiles(image_size=128, n_files=5, slot_dim=64, version="v0",
                            checkpoint_path=args.ckpt, device=dev)
    t0 = time.time()
    Sf, Gf, Hh, Aa, Sp, Gp, Cp = collect_demos(sub, spec, dev, args.demo_rollouts, args.batch, seed=0)
    print(f"[gc-idm] demos: {len(Sf)} tuples in {time.time()-t0:.0f}s | "
          f"flat_dim={Sf.shape[1]} perfile={tuple(Sp.shape[1:])}", flush=True)

    res = {"n_tuples": int(len(Sf)), "spec": vars(spec)}

    # expert oracle (run the greedy policy directly) + random floor
    def expert_eval():
        succ = total = 0; n_roll = (args.eval_episodes + args.batch - 1)//args.batch
        for r in range(n_roll):
            env = NavigateEnv(spec, batch_size=args.batch, device=dev, seed=9999 + r)
            done = torch.zeros(args.batch, dtype=torch.bool, device=dev)
            reached = torch.zeros(args.batch, dtype=torch.bool, device=dev)
            for t in range(spec.max_steps):
                a = env.expert_action(); _, _, d = env.step(a)
                dist = ((env.x-env.tx)**2+(env.y-env.ty)**2).sqrt()
                reached = reached | (dist < spec.success_radius); done = done | d
                if done.all(): break
            succ += int(reached.sum()); total += args.batch
        return succ/total
    res["expert_success"] = expert_eval()
    res["random_success"] = eval_policy(sub, None, spec, dev, args.eval_episodes, args.batch, 0, "flatten", random=True)

    for mode in ("flatten", "perfile"):
        if mode == "flatten":
            head = GCInverseDynamics(state_dim=Sf.shape[1], action_dim=2, hidden=256, mode="flatten", tanh_output=False).to(dev)
            st = train_gc_idm_supervised(head, Sf, Gf, Hh, Aa, steps=args.steps)
        else:
            head = GCInverseDynamics(state_dim=Sp.shape[2], action_dim=2, hidden=256, mode="perfile", tanh_output=False).to(dev)
            st = train_gc_idm_supervised(head, Sp, Gp, Hh, Aa, confs=Cp, steps=args.steps)
        succ = eval_policy(sub, head, spec, dev, args.eval_episodes, args.batch, 0, mode)
        succ_ng = eval_policy(sub, head, spec, dev, args.eval_episodes, args.batch, 0, mode, no_goal=True)
        res[mode] = {"val_mse": round(st.final_val_loss, 5), "success": round(succ, 4),
                     "success_no_goal_control": round(succ_ng, 4)}
        print(f"[gc-idm] {mode}: val_mse={st.final_val_loss:.4f} success={succ:.3f} "
              f"(no-goal control {succ_ng:.3f})", flush=True)

    print(json.dumps(res, indent=2))
    import os; os.makedirs("artifacts", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[gc-idm] wrote {args.out}")


if __name__ == "__main__":
    main()
