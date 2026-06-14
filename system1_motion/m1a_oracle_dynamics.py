#!/usr/bin/env python3
"""M1a -- ORACLE-slot dynamics: the true core test, isolated from perception and slot binding.

Given PERFECT object states (sim positions+velocities of pusher + 3 pucks), can a per-object RELATIONAL / contact
dynamics model predict the controlled-object delta faithfully (action-cosine > 0.42), where a MONOLITHIC MLP cannot?
State-based (no rendering) so it's cheap. The decisive decomposition:
  - relational beats monolithic on contacted-puck cosine -> object-relational dynamics IS the lever; the 0.42 image
    ceiling was perception (the reconstruction latent) -> fix = control-trained latent feeding relational dynamics.
  - even oracle-state relational dynamics ~0.42 -> the contact dynamics/objective is the bottleneck, not perception.

Models compared: MONOLITHIC (concat all states + action -> MLP -> next) vs RELATIONAL (interaction net: action enters
ONLY the pusher node; contact messages propagate to pucks). Run: MUJOCO_GL=egl python3 -m system1_motion.m1a_oracle_dynamics --run
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import mujoco
from system1_motion.probe_push import PushEnv

NOBJ, SDIM, ADIM = 4, 4, 2          # 4 dynamic objects (pusher,puck0,1,2) x [x,y,vx,vy]; action = pusher vel cmd
POS_S, VEL_S = 0.34, 1.0            # normalization scales


def get_state(env):
    q, v = env.d.qpos.copy(), env.d.qvel.copy()                # order: px,py,p0x,p0y,p1x,p1y,p2x,p2y
    s = np.zeros(NOBJ * SDIM, np.float32)
    for k in range(NOBJ):
        s[k*SDIM:k*SDIM+2] = q[2*k:2*k+2] / POS_S
        s[k*SDIM+2:k*SDIM+4] = v[2*k:2*k+2] / VEL_S
    return s

def contacted_puck(env):
    pg = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_GEOM, "pusher")
    pk = {mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_GEOM, f"puck{i}"): i for i in range(3)}
    for i in range(env.d.ncon):
        c = env.d.contact[i]; g1, g2 = c.geom1, c.geom2
        if g1 == pg and g2 in pk: return pk[g2]
        if g2 == pg and g1 in pk: return pk[g1]
    return -1


def collect(n_steps, seed=0, ar=8, log=print):
    """contact-rich: drive the pusher toward a random puck (+noise) so it bumps/pushes pucks; mix random walks."""
    env = PushEnv(seed, ar=ar); rng = np.random.RandomState(seed)
    S, A, S1, C = [], [], [], []; env.reset(); mode = 0; tgt = 0; t0 = time.time()
    for i in range(n_steps):
        s = get_state(env)
        if mode == 0:                                          # push toward a random puck (generates contact)
            a = env.move_toward(env.pos(f"puck{tgt}"), gain=9) + rng.normal(0, 0.3, ADIM)
        else:                                                 # random walk (state coverage)
            a = rng.uniform(-1, 1, ADIM)
        a = np.clip(a, -1, 1).astype(np.float32); env.step(a)
        S.append(s); A.append(a); S1.append(get_state(env)); C.append(contacted_puck(env))
        if (i + 1) % 30 == 0:
            env.reset() if (i + 1) % 300 == 0 else None
            mode = rng.randint(2); tgt = rng.randint(3)
        if (i + 1) % 5000 == 0: log(f"  [collect] {i+1}/{n_steps} ({time.time()-t0:.0f}s)", flush=True)
    return (np.array(S, np.float32), np.array(A, np.float32), np.array(S1, np.float32), np.array(C, np.int64))


# ----------------------------- dynamics models -----------------------------
class MonolithicDyn(nn.Module):
    def __init__(self, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NOBJ * SDIM + ADIM, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, NOBJ * SDIM))
    def forward(self, s, a): return s + self.net(torch.cat([s, a], -1))   # predict DELTA


class RelationalDyn(nn.Module):
    """interaction net: action enters ONLY the pusher node (idx 0); pairwise contact/proximity messages propagate."""
    def __init__(self, hid=128):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)                     # 0=pusher, 1=puck (NOT self.type -> shadows Module.type)
        self.edge = nn.Sequential(nn.Linear(2 * (SDIM + 8), hid), nn.SiLU(), nn.Linear(hid, hid))
        self.node = nn.Sequential(nn.Linear(SDIM + 8 + ADIM + hid, hid), nn.SiLU(), nn.Linear(hid, SDIM))
    def forward(self, s, a):
        B = s.shape[0]; nodes = s.view(B, NOBJ, SDIM)
        tids = torch.zeros(B, NOBJ, dtype=torch.long, device=s.device); tids[:, 1:] = 1
        h = torch.cat([nodes, self.type_emb(tids)], -1)        # [B,N,SDIM+8]
        hi = h[:, :, None].expand(B, NOBJ, NOBJ, h.shape[-1])
        hj = h[:, None, :].expand(B, NOBJ, NOBJ, h.shape[-1])
        msg = self.edge(torch.cat([hi, hj], -1))               # [B,N,N,hid]
        eye = torch.eye(NOBJ, device=s.device).bool()[None, :, :, None]
        msg = msg.masked_fill(eye, 0.0).sum(2)                 # sum over j != i -> [B,N,hid]
        anode = torch.zeros(B, NOBJ, ADIM, device=s.device); anode[:, 0] = a   # action only to pusher node
        delta = self.node(torch.cat([h, anode, msg], -1))      # [B,N,SDIM]
        return s + delta.reshape(B, -1)


def train_dyn(model, data, device, steps=8000, lr=3e-4, batch=256, log=print, tag=""):
    S, A, S1, C = data; S, A, S1 = (torch.tensor(x, device=device) for x in (S, A, S1))
    opt = torch.optim.AdamW(model.parameters(), lr=lr); rng = np.random.RandomState(0); t0 = time.time()
    for step in range(steps):
        b = rng.randint(0, len(S), batch)
        loss = F.mse_loss(model(S[b], A[b]), S1[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"[dyn{tag} {step}/{steps}] mse={loss.item():.5f} ({time.time()-t0:.0f}s)", flush=True)
    return model.eval()


# ----------------------------- action-cosine (per object, + contacted puck) -----------------------------
@torch.no_grad()
def action_cosine(model, device, n=1500, seed=0, ar=8):
    """counterfactual: from real states, model & sim predict each object's pos-delta under action a vs 0; compare
    direction. Report overall (all objects) and CONTACTED-PUCK cosine (the object-dynamics faithfulness)."""
    env = PushEnv(seed + 99, ar=ar); rng = np.random.RandomState(seed); env.reset(); mode = 0; tgt = 0
    cos_all, cos_contact = [], []
    for i in range(n):
        s = get_state(env); qp, qv = env.d.qpos.copy(), env.d.qvel.copy()
        a = (env.move_toward(env.pos(f"puck{tgt}"), 9) + rng.normal(0, 0.3, ADIM)) if mode == 0 else rng.uniform(-1, 1, ADIM)
        a = np.clip(a, -1, 1).astype(np.float32)
        st = torch.tensor(s, device=device)[None]; at = torch.tensor(a, device=device)[None]
        m_a = model(st, at)[0].cpu().numpy(); m_0 = model(st, torch.zeros(1, ADIM, device=device))[0].cpu().numpy()
        ck = contacted_puck(env)
        # real counterfactual
        env.step(a); ra = get_state(env)
        env.d.qpos[:] = qp; env.d.qvel[:] = qv; mujoco.mj_forward(env.m, env.d); env.step(np.zeros(ADIM, np.float32)); r0 = get_state(env)
        for k in range(NOBJ):
            dm = m_a[k*SDIM:k*SDIM+2] - m_0[k*SDIM:k*SDIM+2]
            dr = ra[k*SDIM:k*SDIM+2] - r0[k*SDIM:k*SDIM+2]
            if np.linalg.norm(dm) > 1e-5 and np.linalg.norm(dr) > 1e-5:
                c = float(dm @ dr / (np.linalg.norm(dm) * np.linalg.norm(dr))); cos_all.append(c)
                if ck >= 0 and k == ck + 1: cos_contact.append(c)        # object index of contacted puck = ck+1 (pusher is 0)
        # advance from the real-a branch
        env.d.qpos[:] = qp; env.d.qvel[:] = qv; mujoco.mj_forward(env.m, env.d); env.step(a)
        if (i + 1) % 30 == 0: mode = rng.randint(2); tgt = rng.randint(3)
        if (i + 1) % 300 == 0: env.reset()
    return float(np.mean(cos_all)), float(np.mean(cos_contact) if cos_contact else 0.0), len(cos_contact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--collect", type=int, default=40000)
    ap.add_argument("--ar", type=int, default=8)                # env action_repeat (per-step contact window vs noise)
    args = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke: args.steps, args.collect = 300, 3000
    print(f"[m1a] ORACLE-slot dynamics test on {dev} | collect={args.collect} steps={args.steps} ar={args.ar}", flush=True)
    data = collect(args.collect, ar=args.ar)
    print(f"[m1a] collected {len(data[0])} transitions, {int((data[3]>=0).sum())} in contact", flush=True)
    mono = train_dyn(MonolithicDyn().to(dev), data, dev, args.steps, tag="_mono")
    rel = train_dyn(RelationalDyn().to(dev), data, dev, args.steps, tag="_rel")
    cm, cmc, nm = action_cosine(mono, dev, ar=args.ar); cr, crc, nr = action_cosine(rel, dev, ar=args.ar)
    print("\n[m1a] ===== ACTION-COSINE (oracle slots) =====", flush=True)
    print(f"  MONOLITHIC : all={cm:.2f}  contacted-puck={cmc:.2f}  (n_contact={nm})", flush=True)
    print(f"  RELATIONAL : all={cr:.2f}  contacted-puck={crc:.2f}  (n_contact={nr})", flush=True)
    print(f"  GATE (contacted-puck > 0.42, the image-WM ceiling): mono={'PASS' if cmc>0.42 else 'fail'} rel={'PASS' if crc>0.42 else 'fail'}", flush=True)
    print(f"  VERDICT: {'object-relational dynamics IS the lever -> 0.42 ceiling was PERCEPTION' if crc>0.42 and crc>cmc+0.1 else 'oracle-state dynamics does not separate -> bottleneck is contact/objective, not structure' if max(cmc,crc)<=0.5 else 'monolithic already faithful with oracle states -> ceiling was PERCEPTION not dynamics structure'}", flush=True)


if __name__ == "__main__":
    main()
