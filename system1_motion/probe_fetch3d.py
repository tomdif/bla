#!/usr/bin/env python3
"""Probe the FetchReach 3D env so the real pipeline is written against facts, not guesses:
observation keys/shapes, action dim, render shape, the 3D goal bounds (for normalization + region split),
and that a scripted Cartesian expert actually reaches. Run: MUJOCO_GL=egl python3 -m system1_motion.probe_fetch3d
"""
import numpy as np
import gymnasium as gym
try:
    import gymnasium_robotics
    try: gym.register_envs(gymnasium_robotics)
    except Exception: pass
except Exception as e:
    print("gymnasium_robotics import failed:", e); raise

ENVID = None
for cand in ("FetchReach-v4", "FetchReach-v3", "FetchReach-v2", "FetchReach-v1"):
    try:
        env = gym.make(cand, render_mode="rgb_array"); ENVID = cand; break
    except Exception as e:
        print("  (could not make", cand, "->", repr(e)[:80], ")")
print("ENV:", ENVID)

obs, info = env.reset(seed=0)
print("obs keys/shapes:", {k: np.asarray(v).shape for k, v in obs.items()})
print("action_space:", env.action_space, "dim", env.action_space.shape)
print("achieved_goal:", np.round(obs["achieved_goal"], 3), " desired_goal:", np.round(obs["desired_goal"], 3))
img = np.asarray(env.render()); print("render shape/dtype:", img.shape, img.dtype)

# goal + initial-gripper bounds over many resets (for normalization + train/test split axis)
dg, ag = [], []
for s in range(300):
    o, _ = env.reset(seed=s); dg.append(o["desired_goal"]); ag.append(o["achieved_goal"])
dg, ag = np.array(dg), np.array(ag)
print("desired_goal  min:", np.round(dg.min(0), 3), "max:", np.round(dg.max(0), 3), "span:", np.round(dg.max(0) - dg.min(0), 3))
print("init gripper  min:", np.round(ag.min(0), 3), "max:", np.round(ag.max(0), 3))
# which axis has the widest goal spread -> good split axis for shifted-goal moat
print("widest goal axis (0=x,1=y,2=z):", int(np.argmax(dg.max(0) - dg.min(0))))

# scripted Cartesian expert: action[:3] = clip(gain*(desired-achieved)); does it reach?
o, _ = env.reset(seed=0); adim = env.action_space.shape[0]
for t in range(40):
    a = np.zeros(adim, np.float32); a[:3] = np.clip(10.0 * (o["desired_goal"] - o["achieved_goal"]), -1, 1)
    o, r, term, trunc, info = env.step(a)
    d = float(np.linalg.norm(o["desired_goal"] - o["achieved_goal"]))
    if t % 8 == 0 or d < 0.02: print(f"  t={t:2d} gripper={np.round(o['achieved_goal'],3)} dist={d*100:.1f}cm done={d<0.05}")
print("DONE")
