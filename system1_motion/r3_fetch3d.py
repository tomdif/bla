#!/usr/bin/env python3
"""3D moat validation on FetchReach (gymnasium-robotics) -- the 3D analogue of r1_imitation_fails.

Question (same as 2D, re-validated NOT assumed): does a GATED world model reach SHIFTED 3D goals zero-shot
via CEM planning, while goal-conditioned behavior cloning trained on one region's demos fails on the other?

Pipeline mirrors the hard-won 2D recipe exactly:
  - wandering exploration (goal-agnostic, covers the whole 3D volume incl. the test region)  -> WM training data
  - scripted Cartesian expert demos in the TRAIN region only                                  -> BC training data
  - JEPA WM: plain-MSE latent prediction (stop-grad target) + variance hinge + 15x gripper / 5x target decode
    grounding (normalized [0,1]); CONVERGENCE GATE on gripper decode (cm) with early-abort/retry; held-out +
    OOD multi-step rollout (cm). out_dim=3 decode heads; positions normalized to a fixed 3D box.
  - CEM-MPC plans in 3D (cost = decoded gripper -> goal). BC is goal-conditioned on the 3D goal.
  - eval: success = final gripper within thresh_cm of goal, on TEST (shifted) vs TRAIN region.

  MUJOCO_GL=egl python3 -m system1_motion.r3_fetch3d --validate
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gymnasium as gym
try:
    import gymnasium_robotics
    try: gym.register_envs(gymnasium_robotics)
    except Exception: pass
except Exception:
    pass

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.objective import variance_hinge

IMG = 96
LO = np.array([1.15, 0.55, 0.35]); HI = np.array([1.55, 0.95, 0.75]); SPAN = HI - LO   # 3D box (m) for normalization
M2CM = float(np.mean(SPAN)) * 100.0          # 1 normalized-distance unit ~= this many cm (~40)
Y_SPLIT = 0.749                              # goal-box center y: test = goal.y < split (shifted), train = >=
ENVID = "FetchReach-v4"


# ----------------------------- env helpers -----------------------------
def make_env3d(seed=0):
    try:    env = gym.make(ENVID, render_mode="rgb_array", width=IMG, height=IMG)   # render small if supported
    except Exception: env = gym.make(ENVID, render_mode="rgb_array")
    env.reset(seed=seed); return env

def render3d(env):
    im = np.asarray(env.render())
    if im.shape[0] != IMG: im = np.asarray(Image.fromarray(im).resize((IMG, IMG), Image.BILINEAR))
    return im.transpose(2, 0, 1).copy()                       # [3,IMG,IMG] uint8

def norm3(p):  return ((np.asarray(p, np.float64) - LO) / SPAN).astype(np.float32)   # -> [0,1]^3
def region_of3d(goal): return "test" if goal[1] < Y_SPLIT else "train"
def expert_action3d(obs, gain=10.0):
    a = np.zeros(4, np.float32); a[:3] = np.clip(gain * (obs["desired_goal"] - obs["achieved_goal"]), -1, 1); return a


# ----------------------------- data collection -----------------------------
def collect_exploration(n_steps, seed=0, ep_len=50, log=print):
    """goal-AGNOSTIC wandering: drift toward random waypoints in the box (real action->consequence, broad 3D
    coverage incl. the test region). Records (frame, action, gripper_norm, target_norm, ep_id)."""
    env = make_env3d(seed); rng = np.random.RandomState(seed)
    F_, A, P, T, E = [], [], [], [], []; ep = 0; t0 = time.time()
    obs, _ = env.reset(seed=seed)
    wp = rng.uniform(LO, HI); steps_on_wp = 0
    for i in range(n_steps):
        F_.append(render3d(env)); P.append(norm3(obs["achieved_goal"])); T.append(norm3(obs["desired_goal"])); E.append(ep)
        if steps_on_wp >= rng.randint(6, 14):                 # pick a fresh waypoint -> keeps wandering everywhere
            wp = rng.uniform(LO, HI); steps_on_wp = 0
        a = np.zeros(4, np.float32); a[:3] = np.clip(8.0 * (wp - obs["achieved_goal"]), -1, 1)
        a[:3] += rng.normal(0, 0.25, 3)                       # exploration noise (causal: still drives the gripper)
        a = np.clip(a, -1, 1); A.append(a.astype(np.float32)); steps_on_wp += 1
        obs, _, term, trunc, _ = env.step(a)
        if (i + 1) % ep_len == 0 or term or trunc:
            ep += 1; obs, _ = env.reset(); wp = rng.uniform(LO, HI); steps_on_wp = 0
        if (i + 1) % 4000 == 0: log(f"  [explore] {i+1}/{n_steps} ({time.time()-t0:.0f}s)", flush=True)
    env.close()
    return (np.asarray(F_, np.uint8), np.asarray(A, np.float32), np.asarray(P, np.float32),
            np.asarray(T, np.float32), np.asarray(E, np.int64))

def collect_demos(n_demos, region, seed=0, ep_len=30, log=print):
    """scripted Cartesian expert episodes whose GOAL lies in `region`. Returns demo dicts (BC training data)."""
    env = make_env3d(seed); demos = []; tries = 0; s = seed
    while len(demos) < n_demos and tries < n_demos * 30:
        tries += 1; obs, _ = env.reset(seed=s); s += 1
        if region_of3d(obs["desired_goal"]) != region: continue
        g = norm3(obs["desired_goal"]); frames, actions, grip = [], [], []
        for _ in range(ep_len):
            frames.append(render3d(env)); grip.append(norm3(obs["achieved_goal"]))
            a = expert_action3d(obs); actions.append(a)
            obs, _, _, _, _ = env.step(a)
        demos.append({"frames": np.asarray(frames, np.uint8), "actions": np.asarray(actions, np.float32),
                      "gripper": np.asarray(grip, np.float32), "goal": g.astype(np.float32),
                      "final": norm3(obs["achieved_goal"]).astype(np.float32)})
    env.close()
    log(f"  [demos:{region}] {len(demos)} episodes ({tries} resets)", flush=True)
    return demos


# ----------------------------- world model (3D) -----------------------------
@torch.no_grad()
def rollout_error_cm(enc, dyn, dec_g, trans, device, horizon=8, n=512):
    F_, A, P, T, idx = trans; idxset = set(int(s) for s in idx)   # 5th element = precomputed consecutive-pair indices
    starts = np.array([s for s in idx if all((int(s) + k) in idxset for k in range(horizon))])
    if len(starts) == 0: return float("nan")
    rng = np.random.RandomState(123); starts = rng.choice(starts, min(n, len(starts)), replace=False)
    fr = torch.from_numpy(F_); ac = torch.from_numpy(A); po = torch.from_numpy(P); errs = []
    for i in range(0, len(starts), 256):
        bs = starts[i:i + 256]; z = enc(fr[bs].float().to(device) / 255.0)
        for k in range(horizon):
            z = dyn(z, ac[bs + k].to(device))
            d = (dec_g(z) - po[bs + k + 1].to(device)).pow(2).sum(-1).sqrt()     # norm-dist
            errs.append(d.mean().item() * M2CM)
    return float(np.mean(errs))

def train_wm3d(trans, steps, device, d_z=384, lr=3e-4, batch=128, log=print, tag="", seed=0,
               gate_cm=4.0, early_cm=7.0, max_attempts=6, rollout_eval=None):
    F_, A, P, T, E = trans; adim = A.shape[1]; idx = np.where(E[:-1] == E[1:])[0]
    batch = min(batch, max(8, len(idx)))
    fr = torch.from_numpy(F_); ac = torch.from_numpy(A); po = torch.from_numpy(P); tg = torch.from_numpy(T)
    split = int(0.9 * len(idx)); train_idx = idx[:split]
    held_idx = idx[split:] if (len(idx) - split) >= 64 else idx
    held = (F_, A, P, T, held_idx)                            # HELD-OUT pairs only (never measure rollout on trained data)
    if rollout_eval is not None:                              # DIVERSE off-dist reference (its own pair-indices)
        rF, rA, rP, rT, rE = rollout_eval; ood = (rF, rA, rP, rT, np.where(rE[:-1] == rE[1:])[0])
    else: ood = None
    early_step = int(0.45 * steps); last = None
    for attempt in range(max_attempts):
        torch.manual_seed(seed + 1000 * attempt); np.random.seed(seed + attempt)
        enc = ViTEncoder(IMG, 8, 3, d_z, 6).to(device); dyn = LatentDynamics(d_z, adim, 4).to(device)
        dec_g = DecodeHead(d_z, out_dim=3).to(device); dec_t = DecodeHead(d_z, out_dim=3).to(device)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(dyn.parameters()) + list(dec_g.parameters()) + list(dec_t.parameters()), lr=lr)
        brng = np.random.RandomState(0); t0 = time.time(); cur_cm = 99.0; stuck = False
        for step in range(steps):
            b = brng.choice(train_idx, batch)
            x0 = fr[b].float().to(device) / 255.0; x1 = fr[b + 1].float().to(device) / 255.0
            a = ac[b].to(device); p0 = po[b].to(device); g0 = tg[b].to(device)
            z_t = enc(x0)
            with torch.no_grad(): z_next = enc(x1)
            pred = F.mse_loss(dyn(z_t, a), z_next)
            hinge = variance_hinge(z_t)
            grip = F.mse_loss(dec_g(z_t), p0); tgl = F.mse_loss(dec_t(z_t), g0)
            loss = pred + 1.0 * hinge + 15.0 * grip + 5.0 * tgl
            opt.zero_grad(); loss.backward(); opt.step()
            if step == early_step or step % max(1, steps // 10) == 0 or step == steps - 1:
                cur_cm = grip.item() ** 0.5 * M2CM
                log(f"[wm{tag} a{attempt+1} {step}/{steps}] pred={pred.item():.4f} std={z_t.std(0).mean().item():.3f} "
                    f"grip_cm={cur_cm:.2f} tgt_cm={tgl.item()**0.5*M2CM:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if step == early_step and cur_cm > early_cm:
                log(f"[wm{tag} a{attempt+1}] EARLY-ABORT grip_cm={cur_cm:.2f}>{early_cm} -> reinit", flush=True); stuck = True; break
        if stuck: continue
        roll = rollout_error_cm(enc.eval(), dyn.eval(), dec_g.eval(), held, device)
        roll_ood = rollout_error_cm(enc.eval(), dyn.eval(), dec_g.eval(), ood, device) if ood is not None else float("nan")
        wm = {"enc": enc.eval(), "dyn": dyn.eval(), "dec_g": dec_g.eval(), "dec_t": dec_t.eval(),
              "adim": adim, "grip_cm": cur_cm, "rollout_cm": roll, "rollout_ood_cm": roll_ood, "attempts": attempt + 1}
        if cur_cm <= gate_cm:
            log(f"[wm{tag}] CONVERGED grip_cm={cur_cm:.2f}<={gate_cm} rollout_held={roll:.1f} rollout_OOD={roll_ood:.1f} (a{attempt+1})", flush=True)
            wm["converged"] = True; return wm
        log(f"[wm{tag} a{attempt+1}] GATE FAIL grip_cm={cur_cm:.2f}>{gate_cm}; retry", flush=True); last = wm
    raise RuntimeError(f"[wm{tag}] WM did not converge in {max_attempts} attempts (best grip_cm={last['grip_cm']:.2f}>{gate_cm}). Audit N2.")


@torch.no_grad()
def cem_plan3d(wm, z0, goal_norm, device, horizon=6, iters=4, pop=160, elite=20, terminal_w=4.0):
    adim = wm["adim"]
    mu = torch.zeros(horizon, adim, device=device); sigma = torch.ones(horizon, adim, device=device) * 0.5
    g = torch.tensor(goal_norm, device=device).float()
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, adim, device=device)).clamp(-1, 1)
        z = z0.expand(pop, -1).clone(); cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h]); d = (wm["dec_g"](z) - g[None]).pow(2).sum(-1).sqrt()
            cost = cost + d * (terminal_w if h == horizon - 1 else 1.0)
        e = seqs[cost.topk(elite, largest=False).indices]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    return mu[0].cpu().numpy()


# ----------------------------- behavior cloning (3D, goal-conditioned) -----------------------------
class BC3D(nn.Module):
    def __init__(self, adim):
        super().__init__(); self.enc = ViTEncoder(IMG, 8, 3, 384, 4)
        self.head = nn.Sequential(nn.Linear(384 + 3, 256), nn.ReLU(), nn.Linear(256, adim))
    def forward(self, x, g): return self.head(torch.cat([self.enc(x), g], -1))

def train_bc3d(demos, adim, device, steps, lr=3e-4, batch=128, log=print, seed=0):
    torch.manual_seed(seed)
    X = np.concatenate([d["frames"] for d in demos], 0).astype(np.float32) / 255.0
    A = np.concatenate([d["actions"] for d in demos], 0).astype(np.float32)
    G = np.concatenate([np.tile(d["goal"], (len(d["actions"]), 1)) for d in demos], 0).astype(np.float32)
    net = BC3D(adim).to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr)
    Xt, At, Gt = torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(G); rng = np.random.RandomState(0)
    for step in range(steps):
        b = rng.choice(len(A), batch)
        loss = F.mse_loss(net(Xt[b].to(device), Gt[b].to(device)), At[b].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1: log(f"[bc3d {step}/{steps}] loss={loss.item():.4f}", flush=True)
    return net.eval()


# ----------------------------- evaluation -----------------------------
@torch.no_grad()
def eval_method3d(method, models, region, n_eps, seed0, device, ep_len=40, thresh_cm=(5.0, 10.0)):
    env = make_env3d(seed0); succ = {t: 0 for t in thresh_cm}; finals = []; s = seed0 + 7000
    for e in range(n_eps):
        while True:
            obs, _ = env.reset(seed=s); s += 1
            if region_of3d(obs["desired_goal"]) == region: break
        g = norm3(obs["desired_goal"])
        for t in range(ep_len):
            x = torch.from_numpy(render3d(env).astype(np.float32) / 255.0)[None].to(device)
            if method == "wm_cem":
                wm = models["wm"]; z0 = wm["enc"](x); a = cem_plan3d(wm, z0, g, device)
            else:
                a = models["bc"](x, torch.tensor(g, device=device).float()[None]).cpu().numpy()[0]
            obs, _, _, _, _ = env.step(np.clip(a, -1, 1).astype(np.float32))
        d_cm = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])) * 100.0
        finals.append(d_cm)
        for t in thresh_cm:
            if d_cm <= t: succ[t] += 1
    env.close()
    return {f"succ@{int(t)}cm": succ[t] / n_eps for t in thresh_cm} | {"mean_cm": float(np.mean(finals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", type=int, default=18000); ap.add_argument("--demos", type=int, default=160)
    ap.add_argument("--wm-steps", type=int, default=6000); ap.add_argument("--bc-steps", type=int, default=5000)
    ap.add_argument("--eval-eps", type=int, default=30); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--gate-cm", type=float, default=4.0); ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:                                            # plumbing test only: tiny data, gate disabled
        args.explore, args.demos, args.wm_steps, args.bc_steps, args.eval_eps = 800, 8, 200, 200, 3
        args.gate_cm, args.max_attempts = 999.0, 1
    print(f"[r3] FetchReach 3D moat validation on {dev} | env={ENVID} IMG={IMG} split=y<{Y_SPLIT} smoke={args.smoke}", flush=True)

    print("[r3] collecting goal-agnostic exploration ...", flush=True)
    expl = collect_exploration(args.explore)
    print("[r3] collecting TRAIN-region expert demos ...", flush=True)
    demos = collect_demos(args.demos, "train")
    print("[r3] training GATED 3D world model on exploration ...", flush=True)
    wm = train_wm3d(expl, args.wm_steps, dev, tag="_3d", rollout_eval=expl,
                    gate_cm=args.gate_cm, early_cm=args.gate_cm * 1.6, max_attempts=args.max_attempts)
    print("[r3] training goal-conditioned BC on TRAIN demos ...", flush=True)
    bc = train_bc3d(demos, 4, dev, args.bc_steps)

    if not args.smoke:                                        # save checkpoints for a future 3D GUI
        import os; os.makedirs("runs/r3_ckpt", exist_ok=True)
        torch.save({k: wm[k].state_dict() for k in ("enc", "dyn", "dec_g", "dec_t")} |
                   {"adim": wm["adim"], "grip_cm": wm["grip_cm"], "rollout_ood_cm": wm["rollout_ood_cm"], "img": IMG},
                   "runs/r3_ckpt/wm3d.pt")
        torch.save({"state": bc.state_dict(), "adim": 4, "img": IMG}, "runs/r3_ckpt/bc3d.pt")
        print("[r3] saved checkpoints -> runs/r3_ckpt/", flush=True)

    print("\n[r3] ===== MOAT EVAL (zero-shot, frozen models) =====", flush=True)
    out = {}
    for region in ("train", "test"):
        out[("wm", region)] = eval_method3d("wm_cem", {"wm": wm}, region, args.eval_eps, 0, dev)
        out[("bc", region)] = eval_method3d("bc", {"bc": bc}, region, args.eval_eps, 0, dev)
        print(f"  [{region:5}] WM {out[('wm',region)]}   |   BC {out[('bc',region)]}", flush=True)
    wt = out[("wm", "test")]["succ@5cm"]; bt = out[("bc", "test")]["succ@5cm"]
    print(f"\n  GATE: grip_cm={wm['grip_cm']:.2f} held-rollout={wm['rollout_cm']:.1f}cm OOD-rollout={wm['rollout_ood_cm']:.1f}cm", flush=True)
    print(f"  MOAT @5cm on SHIFTED(test) goals: WM={wt:.2f} vs BC={bt:.2f}  ->  {'HOLDS' if wt-bt>=0.3 else 'WEAK/ABSENT'} (margin {wt-bt:+.2f})", flush=True)


if __name__ == "__main__":
    main()
