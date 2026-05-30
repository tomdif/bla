#!/usr/bin/env python3
"""Dump (substrate embeddings Z, agent positions P in pixels) to an npz for the
Gate 0 precommit harness. Adapter for the OF-JEPA navigate checkpoint (the
KNOWN-FAILING substrate) — used to validate that the harness returns FAIL.

For the new system1_motion substrate, write the analogous adapter that embeds
frames with f_theta; the harness itself (gate0_precommit) is substrate-agnostic.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from system1_jepa import OccludedMultiTargetNavigateEnv, OccludedNavigateSpec
from system1_jepa.of_jepa.interfaces import OFJEPAObjectFiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/of_jepa_navigate_v0_long.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--out", default="gates/navigate_embeddings.npz")
    args = ap.parse_args()
    dev = args.device
    spec = OccludedNavigateSpec(image_size=128, n_targets=3)
    sub = OFJEPAObjectFiles(image_size=128, n_files=5, slot_dim=64, version="v0",
                            checkpoint_path=args.ckpt, device=dev).of_jepa.eval()
    Z, P = [], []
    for e in range(args.episodes):
        env = OccludedMultiTargetNavigateEnv(spec, batch_size=1, device=dev, seed=e)
        env.reset()
        frames, agent = [], []
        for _ in range(args.T):
            frames.append(env.observe()[0])
            agent.append([float(env.x[0]), float(env.y[0])])     # PIXELS
            dxy = (torch.rand(1, 2, device=dev) * 2 - 1) * spec.move_max
            env.step(env.encode_action(dxy) if hasattr(env, "encode_action") else dxy)
        video = torch.stack(frames).to(dev)
        with torch.no_grad():
            slots, _ = sub.encode_video(video)                    # [T,N,D]
        Z.append(slots.flatten(1).cpu().numpy())
        P.append(np.array(agent, dtype=np.float32))
    Z = np.concatenate(Z); P = np.concatenate(P)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, Z=Z.astype(np.float32), P=P.astype(np.float32), img_px=128)
    print(f"wrote {args.out}: Z{Z.shape} P{P.shape} img_px=128 "
          f"(agent px range {P.min():.0f}-{P.max():.0f})")


if __name__ == "__main__":
    main()
