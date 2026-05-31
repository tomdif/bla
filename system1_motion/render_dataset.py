#!/usr/bin/env python3
"""Render a cached Reacher transition dataset for substrate training.

Collects random-action rollouts, renders each frame, records action + the
fingertip pixel position (Gate-0 GT) + episode ids, saves a .npz consumed by
data.TransitionDataset.

REQUIRES headless rendering. Set MUJOCO_GL=egl (GPU, fast) or osmesa (software).
This is the ONE place rendering is needed — the preflight guards against it.

CALIBRATION NOTE: fingertip world (x,y) is mapped to pixels by a linear arena-
extent scale (--extent). The Gate-0 '5px' threshold is in these image-scaled
units; verify the mapping once rendering works (overlay decoded vs true on a
rendered frame) before trusting the absolute px number.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dm_control import suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reacher")
    ap.add_argument("--task", default="easy")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--episode-len", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--extent", type=float, default=0.27, help="arena half-extent (world units) for px mapping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/reacher_transitions.npz")
    args = ap.parse_args()
    if not os.environ.get("MUJOCO_GL"):
        print("WARN: MUJOCO_GL unset; rendering may fail. Set egl or osmesa.", file=sys.stderr)

    H = W = args.image_size
    frames, actions, pos, ep_id = [], [], [], []
    for e in range(args.episodes):
        env = suite.load(args.env, args.task, task_kwargs={"random": args.seed + e})
        aspec = env.action_spec()
        rng = np.random.RandomState(args.seed + e)
        ts = env.reset()
        for t in range(args.episode_len):
            img = env.physics.render(height=H, width=W, camera_id=args.camera_id)  # [H,W,3] uint8
            xy = np.asarray(env.physics.named.data.geom_xpos["finger"][:2])
            px = (xy + args.extent) / (2 * args.extent) * args.image_size           # -> [0,img]
            frames.append(img.transpose(2, 0, 1))                                   # [3,H,W]
            pos.append(px.astype(np.float32))
            ep_id.append(e)
            a = rng.uniform(aspec.minimum, aspec.maximum, aspec.shape).astype(np.float32)
            actions.append(a)
            ts = env.step(a)
            if ts.last():
                break
        if (e + 1) % 50 == 0:
            print(f"[render] {e+1}/{args.episodes} episodes, {len(frames)} frames", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        frames=np.asarray(frames, dtype=np.uint8),
                        actions=np.asarray(actions, dtype=np.float32),
                        pos=np.asarray(pos, dtype=np.float32),
                        ep_id=np.asarray(ep_id, dtype=np.int64),
                        img_px=args.image_size)
    print(f"[render] wrote {args.out}: {len(frames)} frames @ {args.image_size}px "
          f"(pos px range {np.min(pos):.1f}-{np.max(pos):.1f})")


if __name__ == "__main__":
    main()
