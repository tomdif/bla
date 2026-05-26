"""Pretrain OF-JEPA v0 on visible-only navigate env, save checkpoint.

Two-loss training, mirroring slot_jepa_movi_train.py but on navigate-env
rollouts instead of MOVi:

  1. JEPA self-prediction on state_value: state[t] -> state[t+stride] MSE
  2. Position grounding via slot_to_pos_aux + Hungarian-matched env GT
     (agent + visible targets)

The point is NOT to compete with MOVi-trained OF-JEPA, just to give the
Sinkhorn assignment enough training signal that per-file confidence
becomes meaningfully discriminative — i.e. confidence delta between
visible and hidden frames grows from the ~23-26% random-init level
toward something tighter, which raises the ceiling for the per-slot
SlotExistenceHead path downstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import OccludedMultiTargetNavigateEnv, OccludedNavigateSpec
from system1_jepa.identity_probe import hungarian_assign
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OF-JEPA pretrainer on navigate env.")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--episode-len", type=int, default=10,
                   help="frames per training episode (single-batch encode_video)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--n-files", type=int, default=5)
    p.add_argument("--slot-dim", type=int, default=64)
    p.add_argument("--jepa-stride", type=int, default=2)
    p.add_argument("--jepa-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/of_jepa_navigate_v0.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


def rollout_episode(env: OccludedMultiTargetNavigateEnv, T: int):
    """Random-action rollout. Returns (video [T, 3, H, W], positions [T, n_entities, 2],
    visibility [T, n_entities] bool). Entities = [agent, target_0, ..., target_{n-1}]."""
    frames, positions, visibilities = [], [], []
    env.reset()
    for _ in range(T):
        obs = env.observe()                                  # [B=1, 3, H, W]
        frames.append(obs[0])
        # Build per-entity GT for this frame BEFORE stepping.
        agent_xy = torch.stack([env.x[0], env.y[0]])         # [2]
        target_xys = torch.stack([env.tx[0], env.ty[0]], dim=-1)  # [n_t, 2]
        ent_xy = torch.cat([agent_xy.unsqueeze(0), target_xys], dim=0)  # [1+n_t, 2]
        # Visibility: agent always visible. Targets visible only outside hidden window.
        is_hidden = env._is_hidden_step()
        n_targets = target_xys.shape[0]
        vis = torch.ones(1 + n_targets, dtype=torch.bool, device=env.device)
        if is_hidden:
            vis[1:] = False
        positions.append(ent_xy / env.spec.image_size)        # normalize to [0, 1]
        visibilities.append(vis)
        # Random step.
        dxy = (torch.rand(1, 2, device=env.device) * 2 - 1) * env.spec.move_max
        env.step(dxy)
    video = torch.stack(frames, dim=0)                       # [T, 3, H, W]
    pos = torch.stack(positions, dim=0)                      # [T, n_ent, 2]
    visb = torch.stack(visibilities, dim=0)                  # [T, n_ent]
    return video, pos, visb


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Visible-only env: full cycle visible.
    spec = OccludedNavigateSpec(
        image_size=args.image_size, n_targets=args.n_targets,
        visible_steps=args.episode_len * 2, hidden_steps=0,
    )
    env = OccludedMultiTargetNavigateEnv(
        spec=spec, batch_size=1, device=device, seed=args.seed,
    )

    cfg = OFJEPAConfig(
        n_files=args.n_files,
        id_dim=args.slot_dim // 2, state_dim=args.slot_dim // 2,
        proposal_dim=args.slot_dim,
    )
    model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(args.steps):
        video, gt_pos, gt_vis = rollout_episode(env, args.episode_len)
        T = video.shape[0]

        opt.zero_grad(set_to_none=True)
        slot_states, _ = model.encode_video_grad(video)       # [T, n_files, slot_dim]

        # JEPA self-prediction on state_value only (id_key updates via EMA).
        state = slot_states[..., cfg.id_dim:]                # [T, n_files, state_dim]
        stride = args.jepa_stride
        jepa_loss = torch.zeros((), device=device)
        n_pairs = max(T - stride, 1)
        for t in range(T - stride):
            jepa_loss = jepa_loss + F.mse_loss(state[t], state[t + stride].detach())
        jepa_loss = jepa_loss / n_pairs

        # Position grounding via aux head + Hungarian matching to GT entities.
        pred_pos = model.slot_to_pos_aux(slot_states)         # [T, n_files, 2]
        pos_loss = torch.zeros((), device=device)
        pos_count = 0
        for t in range(T):
            vis_mask = gt_vis[t]
            if not vis_mask.any():
                continue
            gt_vis_t = gt_pos[t][vis_mask]                    # [n_visible, 2]
            pp = pred_pos[t].detach().cpu().numpy()
            gt = gt_vis_t.detach().cpu().numpy()
            rows, cols, _ = hungarian_assign(pp, gt)
            if len(rows) > 0:
                rs = torch.from_numpy(rows).to(device)
                cs = torch.from_numpy(cols).to(device)
                pos_loss = pos_loss + F.mse_loss(pred_pos[t][rs], gt_vis_t[cs])
                pos_count += 1
        pos_loss = pos_loss / max(pos_count, 1)

        loss = args.jepa_weight * jepa_loss + args.pos_weight * pos_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.print_every == 0:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach()),
                "jepa": float(jepa_loss.detach()),
                "pos": float(pos_loss.detach()),
            }))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(json.dumps({"event": "saved", "path": args.out, "n_params":
                       sum(p.numel() for p in model.parameters())}))


if __name__ == "__main__":
    main()
