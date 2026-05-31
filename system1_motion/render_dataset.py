#!/usr/bin/env python3
"""Render a cached Reacher transition dataset for substrate training.

Collects random-action rollouts, renders each frame with MuJoCo's NATIVE
offscreen renderer (mujoco.Renderer, 3.0+), records action + fingertip pixel
position (Gate-0 GT) + episode ids, saves a .npz consumed by data.TransitionDataset.

REQUIRES MUJOCO_GL=egl (GPU, fast; needs libEGL.so.1 + nvidia EGL ICD) — set it
BEFORE launching. We use mujoco.Renderer, NOT dm_control's physics.render(): the
latter routes through PyOpenGL's EGL loader which fails on headless GLVND setups,
while the native renderer uses MuJoCo's own EGL context and works.

STARTUP SELF-CHECK: renders one frame and asserts pixel variance > MIN_VAR, so a
"renders-but-all-black / broken-context" failure dies loud in seconds, not after
an hour of caching garbage.

CALIBRATION: fingertip world (x,y) -> pixels via linear arena-extent scale
(--extent); the Gate-0 '5px' threshold is in these image-scaled units. Verify the
mapping by overlaying decoded vs true on a rendered frame before trusting absolute px.
"""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
from dm_control import suite

MIN_VAR = 10.0   # a real Reacher frame has var ~1000s; <10 means broken context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reacher")
    ap.add_argument("--task", default="easy")
    ap.add_argument("--episodes", type=int, default=350)
    ap.add_argument("--episode-len", type=int, default=120, help="max RECORDED frames/episode")
    ap.add_argument("--action-repeat", type=int, default=8,
                    help="apply each action K control steps before recording the next frame; "
                         "per-step fingertip motion is ~0.46px at K=1 (sub-pixel) — frame-skip "
                         "makes per-transition motion visible. Canonical for pixel DMControl.")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--extent", type=float, default=0.27, help="arena half-extent (world units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/reacher_transitions.npz")
    args = ap.parse_args()
    if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
        raise SystemExit("set MUJOCO_GL=egl (or osmesa) before launching the renderer.")

    H = W = args.image_size
    env = suite.load(args.env, args.task, task_kwargs={"random": args.seed})
    renderer = mujoco.Renderer(env.physics.model.ptr, height=H, width=W)
    aspec = env.action_spec()
    rng = np.random.RandomState(args.seed)

    def render_frame():
        renderer.update_scene(env.physics.data.ptr, camera=args.camera_id)
        return renderer.render()                                  # [H,W,3] uint8

    # --- startup self-check (the all-black guard) ---
    env.reset()
    for _ in range(5):
        env.step(rng.uniform(aspec.minimum, aspec.maximum, aspec.shape))
    test = render_frame()
    v = float(np.var(test))
    if v < MIN_VAR:
        raise SystemExit(f"render self-check FAILED: frame variance {v:.2f} < {MIN_VAR} "
                         "(broken GL context / all-black frames).")
    print(f"[render] self-check OK: frame {test.shape} var={v:.1f} mean={test.mean():.1f}", flush=True)

    frames, actions, pos, ep_id = [], [], [], []
    for e in range(args.episodes):
        env.reset()
        for t in range(args.episode_len):
            frames.append(render_frame().transpose(2, 0, 1))      # [3,H,W]
            xy = np.asarray(env.physics.named.data.geom_xpos["finger"][:2])
            pos.append(((xy + args.extent) / (2 * args.extent) * args.image_size).astype(np.float32))
            a = rng.uniform(aspec.minimum, aspec.maximum, aspec.shape).astype(np.float32)
            actions.append(a); ep_id.append(e)
            last = False                                          # apply action K control steps
            for _ in range(args.action_repeat):
                if env.step(a).last():
                    last = True; break
            if last:
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
          f"(pos px range {np.min(pos):.1f}-{np.max(pos):.1f})", flush=True)


if __name__ == "__main__":
    main()
