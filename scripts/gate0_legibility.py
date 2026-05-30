#!/usr/bin/env python3
"""Gate 0 — legible grounded state.

The GC-IDM run failed upstream: position was undecodable from the substrate's
per-frame `observe()` slots (R2~0.05; even slot_to_pos_aux read at ~35px). Before
any planning can be grounded in System-1's physics, System-1 must EXPOSE that
physics. This probe tests whether it can, and attributes any failure:

  ladder (each -> agent-position rmse in px @128, held-out where trained):
    A. canonical: encode_video(T) slots -> slot_to_pos_aux -> Hungarian vs visible GT
    B. control:   observe() per-frame slots -> slot_to_pos_aux -> Hungarian   (the failing path)
    C. adapter:   trained MLP on encode_video slots -> AGENT xy (assignment-free)

PRECOMMIT: best held-out agent-position rmse < 5px  => Gate 0 GREEN.
Hypothesis (from banked finding "legibility requires the temporal window"):
A << B, i.e. the per-frame path was the bug and encode_video is legible.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from system1_jepa import OccludedMultiTargetNavigateEnv, OccludedNavigateSpec
from system1_jepa.of_jepa.interfaces import OFJEPAObjectFiles
from system1_jepa.identity_probe import hungarian_assign


def rollout(env, T):
    """Random-action rollout -> (video[T,3,H,W], gt_pos[T,n_ent,2] in [0,1],
    gt_vis[T,n_ent]). Entities = [agent, target_0..]. Mirrors pretrain."""
    frames, positions, vis = [], [], []
    env.reset()
    for _ in range(T):
        obs = env.observe()
        frames.append(obs[0])
        agent_xy = torch.stack([env.x[0], env.y[0]])
        tgt = torch.stack([env.tx[0], env.ty[0]], dim=-1)
        ent = torch.cat([agent_xy.unsqueeze(0), tgt], dim=0)
        is_hidden = env._is_hidden_step()
        v = torch.ones(ent.shape[0], dtype=torch.bool, device=env.device)
        if is_hidden:
            v[1:] = False
        positions.append(ent / env.spec.image_size)
        vis.append(v)
        dxy = (torch.rand(1, 2, device=env.device) * 2 - 1) * env.spec.move_max
        env.step(env.encode_action(dxy) if hasattr(env, "encode_action") else dxy)
    return torch.stack(frames), torch.stack(positions), torch.stack(vis)


def aux_rmse(model, slots, gt_pos, gt_vis, img):
    """slot_to_pos_aux -> Hungarian-match to visible GT -> rmse px. slots [T,N,D]."""
    aux = getattr(model, "slot_to_pos_aux", None) or model.memory.slot_to_pos_aux
    with torch.no_grad():
        pred = aux(slots)                          # [T,N,2] normalized
    errs = []
    for t in range(slots.shape[0]):
        m = gt_vis[t]
        if not m.any():
            continue
        gt = gt_pos[t][m].cpu().numpy()
        pp = pred[t].cpu().numpy()
        rows, cols, _ = hungarian_assign(pp, gt)
        if len(rows):
            d = ((pred[t][torch.as_tensor(rows)] - gt_pos[t][m][torch.as_tensor(cols)]) ** 2).sum(-1)
            errs.append(float(d.mean().sqrt()) * img)
    return float(np.mean(errs)) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/of_jepa_navigate_pretrained.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="artifacts/gate0_legibility.json")
    args = ap.parse_args()
    dev = args.device
    spec = OccludedNavigateSpec(image_size=128, n_targets=4)
    sub = OFJEPAObjectFiles(image_size=128, n_files=5, slot_dim=64, version="v0",
                            checkpoint_path=args.ckpt, device=dev)
    model = sub.of_jepa.eval()
    img = spec.image_size

    EV_slots, OBS_slots, GTpos, GTvis, AgentXY = [], [], [], [], []
    for e in range(args.episodes):
        env = OccludedMultiTargetNavigateEnv(spec, batch_size=1, device=dev, seed=e)
        video, gt_pos, gt_vis = rollout(env, args.T)
        video = video.to(dev)
        with torch.no_grad():
            ev, _ = model.encode_video(video)                  # [T,N,D] canonical path
        # observe() per-frame path (the failing one), same frames
        sub.reset_episode(batch_size=1)
        obs_slots = []
        with torch.no_grad():
            for t in range(video.shape[0]):
                ofb = sub.observe(video[t:t+1])
                obs_slots.append(ofb.full_slot[0])
        EV_slots.append(ev.cpu()); OBS_slots.append(torch.stack(obs_slots).cpu())
        GTpos.append(gt_pos.cpu()); GTvis.append(gt_vis.cpu())
        AgentXY.append(gt_pos[:, 0, :].cpu())                  # entity 0 = agent

    # ---- A: canonical aux on encode_video slots ----
    a = np.mean([aux_rmse(model, EV_slots[i].to(dev), GTpos[i].to(dev), GTvis[i].to(dev), img)
                 for i in range(len(EV_slots))])
    # ---- B: aux on observe() slots (the failing path) ----
    b = np.mean([aux_rmse(model, OBS_slots[i].to(dev), GTpos[i].to(dev), GTvis[i].to(dev), img)
                 for i in range(len(OBS_slots))])

    # ---- C: trained adapter, encode_video slots -> AGENT xy (assignment-free) ----
    X = torch.cat([s.flatten(1) for s in EV_slots]).to(dev)    # [sumT, N*D]
    Y = torch.cat(AgentXY).to(dev)                             # [sumT, 2] normalized
    n = len(X); idx = torch.randperm(n); ntr = int(0.8 * n)
    tr, va = idx[:ntr].to(dev), idx[ntr:].to(dev)
    adapter = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(),
                            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2)).to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(0)
    for s in range(args.steps):
        sel = tr[torch.randint(0, len(tr), (256,), generator=g, device=dev)]
        loss = ((adapter(X[sel]) - Y[sel]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        c_rmse = float(((adapter(X[va]) - Y[va]) ** 2).sum(-1).mean().sqrt()) * img

    res = {"episodes": args.episodes, "T": args.T, "img": img,
           "A_encode_video_aux_rmse_px": round(float(a), 2),
           "B_observe_aux_rmse_px": round(float(b), 2),
           "C_trained_adapter_agent_rmse_px": round(c_rmse, 2),
           "precommit_px": 5.0,
           "gate0_green": bool(min(a, c_rmse) < 5.0)}
    print(json.dumps(res, indent=2))
    os.makedirs("artifacts", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
