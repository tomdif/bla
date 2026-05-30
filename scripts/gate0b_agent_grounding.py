#!/usr/bin/env python3
"""Gate 0b — can OF-JEPA be made to expose IDENTITY-ADDRESSABLE agent position?

Gate 0 failed even after a proper pretrain: agent position read off the slots is
at chance (~50px). Diagnosis: the pretrain's position grounding uses HUNGARIAN
matching, which only forces the SET of 5 predicted positions to cover the SET of
GT positions — it never binds a specific entity to a readable slot. So the slots
encode "positions exist," not "the agent is HERE." Control needs the latter.

This test fine-tunes the substrate (warm-started from the trained checkpoint)
with the AGENT'S position as a PRIMARY, FIXED regression target read off the
slots (no Hungarian — the agent is entity 0, always visible). JEPA loss kept to
preserve the object-file structure. Then re-gate on HELD-OUT agent decodability.

PRECOMMIT: held-out agent-position rmse < 5px  => the slots CAN carry
identity-addressable geometry (Gate 0 green for the control-relevant quantity).
If it stays ~25-50px, the OF-JEPA slot bottleneck cannot localize the agent even
with a direct objective => architectural ceiling for grounded planning.
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_jepa import OccludedMultiTargetNavigateEnv, OccludedNavigateSpec
from system1_jepa.of_jepa.interfaces import OFJEPAObjectFiles

ID_DIM = 32  # slot_dim//2


def rollout(env, T, dev):
    frames, agent = [], []
    env.reset()
    for _ in range(T):
        frames.append(env.observe()[0])
        agent.append(torch.stack([env.x[0], env.y[0]]) / env.spec.image_size)
        dxy = (torch.rand(1, 2, device=dev) * 2 - 1) * env.spec.move_max
        env.step(env.encode_action(dxy) if hasattr(env, "encode_action") else dxy)
    return torch.stack(frames).to(dev), torch.stack(agent).to(dev)   # [T,3,H,W], [T,2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/of_jepa_navigate_v0_long.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--jepa-stride", type=int, default=2)
    ap.add_argument("--jepa-weight", type=float, default=1.0)
    ap.add_argument("--agent-weight", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="runs/of_jepa_navigate_agentgrounded.pt")
    args = ap.parse_args()
    dev = args.device
    spec = OccludedNavigateSpec(image_size=128, n_targets=3)
    sub = OFJEPAObjectFiles(image_size=128, n_files=5, slot_dim=64, version="v0",
                            checkpoint_path=args.init, device=dev)
    model = sub.of_jepa
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    slot_flat = 5 * 64
    agent_head = nn.Sequential(nn.Linear(slot_flat, 256), nn.GELU(),
                               nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(agent_head.parameters()),
                            lr=args.lr, weight_decay=1e-4)

    @torch.no_grad()
    def held_out_rmse(n=40, seed0=10000):
        model.eval(); errs = []
        for e in range(n):
            env = OccludedMultiTargetNavigateEnv(spec, batch_size=1, device=dev, seed=seed0 + e)
            video, agent = rollout(env, args.T, dev)
            slots, _ = model.encode_video(video)
            pred = agent_head(slots.flatten(1))
            errs.append(float(((pred - agent) ** 2).sum(-1).mean().sqrt()) * spec.image_size)
        model.train()
        return sum(errs) / len(errs)

    print(f"[gate0b] held-out agent rmse @init: {held_out_rmse():.1f}px", flush=True)
    for step in range(args.steps):
        env = OccludedMultiTargetNavigateEnv(spec, batch_size=1, device=dev, seed=step)
        video, agent = rollout(env, args.T, dev)
        slots, _ = model.encode_video_grad(video)              # [T,N,D] with grad
        state = slots[..., ID_DIM:]
        jepa = sum(F.mse_loss(state[t], state[t + args.jepa_stride].detach())
                   for t in range(args.T - args.jepa_stride)) / max(args.T - args.jepa_stride, 1)
        pred = agent_head(slots.flatten(1))                    # [T,2]
        agent_loss = F.mse_loss(pred, agent)
        loss = args.jepa_weight * jepa + args.agent_weight * agent_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(agent_head.parameters()), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0:
            print(f"[gate0b] step {step+1}/{args.steps} jepa={float(jepa):.4g} "
                  f"agent_mse={float(agent_loss):.4f} held_out_rmse={held_out_rmse():.1f}px", flush=True)
    final = held_out_rmse(n=120)
    torch.save({"model": model.state_dict(), "agent_head": agent_head.state_dict()}, args.out)
    res = {"final_heldout_agent_rmse_px": round(final, 2), "precommit_px": 5.0,
           "gate0b_green": bool(final < 5.0)}
    print(json.dumps(res), flush=True)
    os.makedirs("artifacts", exist_ok=True)
    json.dump(res, open("artifacts/gate0b_agent_grounding.json", "w"), indent=2)


if __name__ == "__main__":
    main()
