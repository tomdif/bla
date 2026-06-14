#!/usr/bin/env python3
"""M1b -- control-STATE-space dynamics from pixels (the fix M1a forced). M1a showed that with PERFECT object
states (pos+VEL) the contact/torque dynamics is faithful (cosine 0.66), but the image WM ran dynamics in the
entangled 384-d reconstruction latent and plateaued at 0.42. Key subtlety: a SINGLE frame has no velocity, but
M1a's oracle state did -> so the fix is dynamics in a clean state space WITH velocity, which needs a 2-FRAME input.

M1b: encode a 2-frame stack -> decode clean state [ee_pos(3), ee_vel(3)] -> dynamics in STATE space (MLP on the
decoded state + action), NOT in the latent. Measure action-cosine from PIXELS. If it jumps from 0.42 toward ~0.66,
the M1a fix is confirmed from pixels. Baseline = the latent-dynamics number (0.42). On the torque arm (reuse all).
Run: MUJOCO_GL=egl python3 -m system1_motion.m1b_state_dynamics --run
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import mujoco
from system1_motion.models import ViTEncoder, DecodeHead
from system1_motion.r3_torque import Arm, norm3, SPAN, IMG, ADIM
import system1_motion.r3_torque as R

VEL_S = 2.0                                                      # ee velocity normalization (m/s)
span_t = None


def ee_vel(arm):
    jacp = np.zeros((3, arm.m.nv)); mujoco.mj_jacSite(arm.m, arm.d, jacp, None, arm.eid)
    return (jacp @ arm.d.qvel) / VEL_S                          # normalized 3D ee cartesian velocity

def state_of(arm):
    return np.concatenate([norm3(arm.ee()), ee_vel(arm)]).astype(np.float32)   # [pos3, vel3]


def collect(n_steps, seed=0, ar=12, rand_frac=0.6, log=print):
    """torque-arm exploration; record 2-frame stacks + clean states (pos+vel). stack_t = [frame_{t-1}, frame_t]."""
    arm = Arm(seed); rng = np.random.RandomState(seed)
    ST, A, S, S1 = [], [], [], []; arm.reset(); prev = arm.render(); on = 0; mode = int(rng.rand() < rand_frac)
    t0 = time.time()
    for i in range(n_steps):
        cur = arm.render(); s = state_of(arm)
        a = arm.rng.uniform(-1, 1, ADIM).astype(np.float32) if mode else \
            np.clip(arm.pd_reach() + arm.rng.normal(0, 0.35, ADIM), -1, 1).astype(np.float32)
        arm.step(a); s1 = state_of(arm); nxt = arm.render()
        ST.append(np.concatenate([prev, cur], 0)); A.append(a); S.append(s); S1.append(s1)   # [6,H,W]
        prev = cur; on += 1
        if (i + 1) % 26 == 0: arm.reset(); prev = arm.render(); mode = int(rng.rand() < rand_frac); on = 0
        if (i + 1) % 5000 == 0: log(f"  [collect] {i+1}/{n_steps} ({time.time()-t0:.0f}s)", flush=True)
    arm.close()
    return (np.array(ST, np.uint8), np.array(A, np.float32), np.array(S, np.float32), np.array(S1, np.float32))


class StateWM(nn.Module):
    """2-frame encoder -> clean state [pos,vel]; dynamics MLP operates in STATE space (not the latent)."""
    def __init__(self, d_z=256):
        super().__init__()
        self.enc = ViTEncoder(IMG, 8, 6, d_z, 4)               # 6 channels = 2 stacked frames
        self.dec_state = DecodeHead(d_z, out_dim=6)            # [pos3, vel3]
        self.dyn = nn.Sequential(nn.Linear(6 + ADIM, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 6))
    def perceive(self, stack): return self.dec_state(self.enc(stack))      # pixels -> state
    def step_state(self, s, a): return s + self.dyn(torch.cat([s, a], -1)) # clean state-space dynamics (delta)


def train(model, data, device, steps=8000, lr=3e-4, batch=128, log=print):
    ST, A, S, S1 = data
    st = torch.from_numpy(ST); a_t = torch.from_numpy(A).to(device)
    s_t = torch.from_numpy(S).to(device); s1_t = torch.from_numpy(S1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); rng = np.random.RandomState(0); t0 = time.time()
    for step in range(steps):
        b = rng.randint(0, len(ST), batch)
        x = st[b].float().to(device) / 255.0
        s_pred = model.perceive(x)                              # perception: pixels -> state
        dec = F.mse_loss(s_pred, s_t[b])                        # decode grounding (pos+vel)
        dyn = F.mse_loss(model.step_state(s_t[b], a_t[b]), s1_t[b])   # CLEAN state-space dynamics (M1a-style)
        loss = dec + dyn
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            with torch.no_grad():
                pos_cm = ((s_pred[:, :3] - s_t[b][:, :3]) * span_t).pow(2).sum(-1).sqrt().mean().item() * 100
            log(f"[m1b {step}/{steps}] dec={dec.item():.4f} dyn={dyn.item():.4f} pos_cm={pos_cm:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def action_cosine(model, device, n=800, seed=0, ar=12):
    """from PIXELS: perceive state, then state-dynamics predicts ee-pos delta under a vs 0; cosine vs real sim."""
    arm = Arm(seed + 99); rng = np.random.RandomState(seed); arm.reset(); prev = arm.render(); on = 0
    cos = []; real_auth = []; model_auth = []
    for i in range(n):
        cur = arm.render(); stack = torch.from_numpy(np.concatenate([prev, cur], 0).astype(np.float32) / 255.0)[None].to(device)
        s = model.perceive(stack)
        a = np.clip(arm.pd_reach() + rng.normal(0, 0.4, ADIM), -1, 1).astype(np.float32) if i % 2 else rng.uniform(-1, 1, ADIM).astype(np.float32)
        at = torch.tensor(a, device=device)[None]
        ee_a = (model.step_state(s, at)[0, :3].cpu().numpy()) * SPAN
        ee_0 = (model.step_state(s, torch.zeros(1, ADIM, device=device))[0, :3].cpu().numpy()) * SPAN
        qp, qv = arm.d.qpos.copy(), arm.d.qvel.copy()
        arm.step(a); ra = arm.ee().copy()
        arm.d.qpos[:] = qp; arm.d.qvel[:] = qv; mujoco.mj_forward(arm.m, arm.d); arm.step(np.zeros(ADIM, np.float32)); r0 = arm.ee().copy()
        dm, dr = ee_a - ee_0, ra - r0
        if np.linalg.norm(dm) > 1e-5 and np.linalg.norm(dr) > 1e-5:
            cos.append(float(dm @ dr / (np.linalg.norm(dm) * np.linalg.norm(dr))))
            real_auth.append(np.linalg.norm(dr) * 100); model_auth.append(np.linalg.norm(dm) * 100)
        arm.d.qpos[:] = qp; arm.d.qvel[:] = qv; mujoco.mj_forward(arm.m, arm.d); prev = cur; arm.step(a); on += 1
        if (i + 1) % 26 == 0: arm.reset(); prev = arm.render(); on = 0
    arm.close()
    return float(np.mean(cos)), float(np.mean(real_auth)), float(np.mean(model_auth))


def main():
    global span_t
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect", type=int, default=24000); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ar", type=int, default=12); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R.AR = args.ar; span_t = torch.tensor(SPAN, device=dev).float()
    if args.smoke: args.collect, args.steps = 3000, 400
    print(f"[m1b] STATE-space dynamics from pixels (2-frame) on {dev} | collect={args.collect} steps={args.steps} ar={args.ar}", flush=True)
    data = collect(args.collect, ar=args.ar)
    print(f"[m1b] collected {len(data[0])} 2-frame transitions", flush=True)
    model = train(StateWM().to(dev), data, dev, args.steps)
    c, ra, ma = action_cosine(model, dev, ar=args.ar)
    print("\n[m1b] ===== ACTION-COSINE from PIXELS (state-space dynamics, pos+vel) =====", flush=True)
    print(f"  cosine={c:.2f}  real_auth={ra:.2f}cm  model_auth={ma:.2f}cm   (latent-dynamics baseline was 0.42)", flush=True)
    print(f"  VERDICT: {'STATE-SPACE FIX CONFIRMED from pixels (broke 0.42 -> M1a mechanism holds)' if c > 0.55 else 'partial (>0.42 but <0.55)' if c > 0.45 else 'state-space alone did NOT break 0.42 -- velocity/perception still limiting'}", flush=True)


if __name__ == "__main__":
    main()
