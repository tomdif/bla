#!/usr/bin/env python3
"""RICHER 3D moat test: a custom 3-DOF TORQUE-controlled arm (real gravity/momentum/coupling dynamics) --
the honest dynamics/causality stress test that FetchReach's Cartesian control could not provide.

Same question, same recipe as r1/r3_fetch3d, harder dynamics: does a GATED world model reach SHIFTED 3D
goals zero-shot via CEM planning (now planning through real torque dynamics) while goal-conditioned BC
trained on one region's demos fails on the other?

  MUJOCO_GL=egl python3 -m system1_motion.r3_torque --validate
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import mujoco
from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.objective import variance_hinge

IMG = 96
LO = np.array([-0.42, -0.42, 0.04]); HI = np.array([0.42, 0.42, 0.46]); SPAN = HI - LO
M2CM = float(np.mean(SPAN)) * 100.0                          # approx cm per normalized unit (gate metric only; eval uses true m)
RMIN, RMAX = 0.16, 0.36                                     # reliably-reachable shell radius (front zone)
ADIM = 3

ARM_XML = """
<mujoco model="arm3d">
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <default>
    <joint type="hinge" damping="0.7" armature="0.1" limited="true"/>
    <geom type="capsule" size="0.022" density="600" rgba="0.55 0.6 0.7 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="cam" pos="0.62 -0.62 0.62" xyaxes="0.7 0.7 0 -0.4 0.4 0.82"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" pos="0 0 0" rgba="0.18 0.2 0.24 1"/>
    <body name="base" pos="0 0 0.06">
      <joint name="j0" axis="0 0 1" range="-180 180"/>
      <geom fromto="0 0 -0.06 0 0 0.06" size="0.03"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="j1" axis="0 1 0" range="-100 100"/>
        <geom fromto="0 0 0 0.2 0 0"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" axis="0 1 0" range="-150 150"/>
          <geom fromto="0 0 0 0.2 0 0"/>
          <body name="ee" pos="0.2 0 0">
            <geom name="fingertip" type="sphere" size="0.028" rgba="0.25 0.95 0.4 1"/>
            <site name="ee" pos="0 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" mocap="true" pos="0.3 0.0 0.3">
      <geom name="target" type="sphere" size="0.032" rgba="1 0.32 0.32 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="3.0"/><motor joint="j1" gear="6.0"/><motor joint="j2" gear="4.0"/>
  </actuator>
</mujoco>
"""

def norm3(p): return ((np.asarray(p, np.float64) - LO) / SPAN).astype(np.float32)
def region_of3d(t): return "test" if t[1] < 0.0 else "train"     # y<0 (left) = shifted/test ; y>=0 = train


class Arm:
    def __init__(self, seed=0):
        self.m = mujoco.MjModel.from_xml_string(ARM_XML); self.d = mujoco.MjData(self.m)
        self.ren = mujoco.Renderer(self.m, IMG, IMG)
        self.eid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.rng = np.random.RandomState(seed)
    def ee(self):  return self.d.site_xpos[self.eid].copy()
    def tgt(self): return self.d.mocap_pos[0].copy()
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.rng.uniform(-0.6, 0.6, self.m.nq); self.d.qvel[:] = 0; mujoco.mj_forward(self.m, self.d)
    def set_target(self, xyz): self.d.mocap_pos[0] = xyz; mujoco.mj_forward(self.m, self.d)
    def sample_target(self, region=None):
        while True:                                            # FRONT zone (x>0) -> reliably reachable; split left/right by y
            c = self.rng.uniform([0.10, -0.35, 0.08], [0.38, 0.35, 0.42])
            if RMIN < np.linalg.norm(c) < RMAX and (region is None or region_of3d(c) == region): return c
    def step(self, ctrl, repeat=2):
        self.d.ctrl[:] = np.clip(ctrl, -1, 1)
        for _ in range(repeat): mujoco.mj_step(self.m, self.d)
    def render(self):
        self.ren.update_scene(self.d, camera="cam"); return self.ren.render().transpose(2, 0, 1).copy()
    def pd_reach(self, Kp=45.0, Kd=5.0):
        """privileged operational-space controller: Jacobian-transpose Cartesian PD + gravity/coriolis
        compensation -> torque ctrl. Reliably drives the ee to the target (greedy 1-step shooting could not:
        too myopic under damping to build sustained motion across the workspace)."""
        t, ee = self.tgt(), self.ee()
        jacp = np.zeros((3, self.m.nv)); mujoco.mj_jacSite(self.m, self.d, jacp, None, self.eid)
        vee = jacp @ self.d.qvel                                # ee Cartesian velocity
        Fcart = Kp * (t - ee) - Kd * vee                        # desired Cartesian force
        tau = jacp.T @ Fcart + self.d.qfrc_bias                 # + gravity/coriolis compensation
        return np.clip(tau / self.m.actuator_gear[:, 0], -1, 1).astype(np.float32)


# ----------------------------- data -----------------------------
def collect_exploration(n_steps, seed=0, ep_len=50, log=print):
    """goal-agnostic wandering: shoot toward random waypoints across the WHOLE workspace (+noise) -> broad
    (state x action) dynamics coverage incl. the test region. Records frames/actions/ee_norm/target_norm/ep."""
    arm = Arm(seed); F_, A, P, T, E = [], [], [], [], []; ep = 0; t0 = time.time()
    arm.reset(); arm.set_target(arm.sample_target()); on = 0   # visible target = current waypoint; arm chases it (+noise)
    for i in range(n_steps):
        F_.append(arm.render()); P.append(norm3(arm.ee())); T.append(norm3(arm.tgt())); E.append(ep)
        if on >= arm.rng.randint(8, 18): arm.set_target(arm.sample_target()); on = 0
        a = np.clip(arm.pd_reach() + arm.rng.normal(0, 0.35, ADIM), -1, 1).astype(np.float32)   # torque toward wp + explore noise
        A.append(a); arm.step(a); on += 1
        if (i + 1) % ep_len == 0:
            ep += 1; arm.reset(); arm.set_target(arm.sample_target()); on = 0
        if (i + 1) % 4000 == 0: log(f"  [explore] {i+1}/{n_steps} ({time.time()-t0:.0f}s)", flush=True)
    return (np.asarray(F_, np.uint8), np.asarray(A, np.float32), np.asarray(P, np.float32),
            np.asarray(T, np.float32), np.asarray(E, np.int64))

def collect_demos(n_demos, region, seed=0, ep_len=70, keep_cm=0.07, log=print):
    """shooting-expert episodes to TRAIN-region targets; KEEP ONLY episodes that reach (< keep_cm) -> clean demos."""
    arm = Arm(seed + 5); demos = []; tries = 0; reached = []
    while len(demos) < n_demos and tries < n_demos * 12:
        tries += 1; arm.reset(); tgt = arm.sample_target(region); arm.set_target(tgt); g = norm3(tgt)
        frames, actions, ee = [], [], []
        for _ in range(ep_len):
            frames.append(arm.render()); ee.append(norm3(arm.ee()))
            a = arm.pd_reach(); actions.append(a); arm.step(a)
        fd = float(np.linalg.norm(arm.ee() - tgt)); reached.append(fd)
        if fd > keep_cm: continue
        demos.append({"frames": np.asarray(frames, np.uint8), "actions": np.asarray(actions, np.float32),
                      "ee": np.asarray(ee, np.float32), "goal": g.astype(np.float32)})
    log(f"  [demos:{region}] {len(demos)} clean episodes / {tries} attempts (median reach {np.median(reached)*100:.1f}cm)", flush=True)
    return demos


# ----------------------------- world model (3D torque) -----------------------------
@torch.no_grad()
def rollout_error_cm(enc, dyn, dec_g, trans, device, horizon=8, n=512):
    F_, A, P, T, idx = trans; idxset = set(int(s) for s in idx)
    starts = np.array([s for s in idx if all((int(s) + k) in idxset for k in range(horizon))])
    if len(starts) == 0: return float("nan")
    rng = np.random.RandomState(123); starts = rng.choice(starts, min(n, len(starts)), replace=False)
    fr = torch.from_numpy(F_); ac = torch.from_numpy(A); po = torch.from_numpy(P)
    span = torch.tensor(SPAN, device=device).float(); errs = []
    for i in range(0, len(starts), 256):
        bs = starts[i:i + 256]; z = enc(fr[bs].float().to(device) / 255.0)
        for k in range(horizon):
            z = dyn(z, ac[bs + k].to(device))
            d = ((dec_g(z) - po[bs + k + 1].to(device)) * span).pow(2).sum(-1).sqrt()   # TRUE meters
            errs.append(d.mean().item() * 100.0)
    return float(np.mean(errs))

def train_wm3d(trans, steps, device, d_z=384, lr=3e-4, batch=128, log=print, tag="", seed=0,
               gate_cm=4.0, early_cm=7.0, max_attempts=6, rollout_eval=None):
    F_, A, P, T, E = trans; adim = A.shape[1]; idx = np.where(E[:-1] == E[1:])[0]
    batch = min(batch, max(8, len(idx)))
    fr = torch.from_numpy(F_); ac = torch.from_numpy(A); po = torch.from_numpy(P); tg = torch.from_numpy(T)
    span = torch.tensor(SPAN, device=device).float()
    split = int(0.9 * len(idx)); train_idx = idx[:split]
    held_idx = idx[split:] if (len(idx) - split) >= 64 else idx
    held = (F_, A, P, T, held_idx)
    ood = None
    if rollout_eval is not None:
        rF, rA, rP, rT, rE = rollout_eval; ood = (rF, rA, rP, rT, np.where(rE[:-1] == rE[1:])[0])
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
            pred = F.mse_loss(dyn(z_t, a), z_next); hinge = variance_hinge(z_t)
            grip = F.mse_loss(dec_g(z_t), p0); tgl = F.mse_loss(dec_t(z_t), g0)
            loss = pred + 1.0 * hinge + 15.0 * grip + 5.0 * tgl
            opt.zero_grad(); loss.backward(); opt.step()
            if step == early_step or step % max(1, steps // 10) == 0 or step == steps - 1:
                with torch.no_grad(): cur_cm = (((dec_g(z_t) - p0) * span).pow(2).sum(-1).sqrt().mean().item()) * 100.0  # TRUE cm
                log(f"[wm{tag} a{attempt+1} {step}/{steps}] pred={pred.item():.4f} std={z_t.std(0).mean().item():.3f} "
                    f"grip_cm={cur_cm:.2f} ({time.time()-t0:.0f}s)", flush=True)
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
def cem_plan3d(wm, z0, goal_norm, device, horizon=8, iters=5, pop=256, elite=32, terminal_w=6.0):
    adim = wm["adim"]
    mu = torch.zeros(horizon, adim, device=device); sigma = torch.ones(horizon, adim, device=device) * 0.6
    g = torch.tensor(goal_norm, device=device).float()
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, adim, device=device)).clamp(-1, 1)
        z = z0.expand(pop, -1).clone(); cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h]); d = (wm["dec_g"](z) - g[None]).pow(2).sum(-1).sqrt()
            cost = cost + d * (terminal_w if h == horizon - 1 else 1.0)
        e = seqs[cost.topk(elite, largest=False).indices]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    return mu[0].cpu().numpy()


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


# ----------------------------- eval (true meters) -----------------------------
@torch.no_grad()
def eval_method3d(method, models, region, n_eps, seed0, device, ep_len=50, thresh_cm=(5.0, 10.0)):
    arm = Arm(seed0 + 7000); succ = {t: 0 for t in thresh_cm}; finals = []
    for e in range(n_eps):
        arm.reset(); tgt = arm.sample_target(region); arm.set_target(tgt); g = norm3(tgt)
        for t in range(ep_len):
            x = torch.from_numpy(arm.render().astype(np.float32) / 255.0)[None].to(device)
            if method == "wm_cem":
                wm = models["wm"]; z0 = wm["enc"](x); a = cem_plan3d(wm, z0, g, device)
            else:
                a = models["bc"](x, torch.tensor(g, device=device).float()[None]).cpu().numpy()[0]
            arm.step(np.clip(a, -1, 1))
        d_cm = float(np.linalg.norm(arm.ee() - tgt)) * 100.0; finals.append(d_cm)
        for t in thresh_cm:
            if d_cm <= t: succ[t] += 1
    return {f"succ@{int(t)}cm": succ[t] / n_eps for t in thresh_cm} | {"mean_cm": float(np.mean(finals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", type=int, default=20000); ap.add_argument("--demos", type=int, default=160)
    ap.add_argument("--wm-steps", type=int, default=8000); ap.add_argument("--bc-steps", type=int, default=5000)
    ap.add_argument("--eval-eps", type=int, default=30); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--gate-cm", type=float, default=4.0); ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--testexpert", action="store_true")
    args = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.testexpert:                                        # instrument one trajectory to see WHY it doesn't reach
        for region in ("train", "test", None):
            arm = Arm(0); arm.reset(); tgt = arm.sample_target(region); arm.set_target(tgt)
            print(f"\n  region={region} target={np.round(tgt,3)} |t|={np.linalg.norm(tgt):.3f} start_ee={np.round(arm.ee(),3)}", flush=True)
            for t in range(70):
                a = arm.pd_reach(); arm.step(a)
                if t % 14 == 0 or t == 69:
                    print(f"    t={t:2d} ee={np.round(arm.ee(),3)} dist={np.linalg.norm(arm.ee()-tgt)*100:5.1f}cm "
                          f"qpos={np.round(arm.d.qpos,2)} |qvel|={np.linalg.norm(arm.d.qvel):.1f} a={np.round(a,2)}", flush=True)
        return
    if args.smoke:
        args.explore, args.demos, args.wm_steps, args.bc_steps, args.eval_eps = 800, 6, 200, 200, 3
        args.gate_cm, args.max_attempts = 999.0, 1
    print(f"[r3t] TORQUE 3D moat test on {dev} | 3-DOF arm IMG={IMG} split=y<0 smoke={args.smoke}", flush=True)
    print("[r3t] collecting goal-agnostic torque exploration ...", flush=True)
    expl = collect_exploration(args.explore)
    print("[r3t] collecting TRAIN-region shooting-expert demos ...", flush=True)
    demos = collect_demos(args.demos, "train")
    if len(demos) < 2: raise RuntimeError(f"[r3t] expert produced only {len(demos)} clean demos -- expert too weak; tune shoot/keep_cm.")
    print("[r3t] training GATED torque world model ...", flush=True)
    wm = train_wm3d(expl, args.wm_steps, dev, tag="_3dt", rollout_eval=expl, gate_cm=args.gate_cm,
                    early_cm=args.gate_cm * 1.6, max_attempts=args.max_attempts)
    print("[r3t] training goal-conditioned BC ...", flush=True)
    bc = train_bc3d(demos, ADIM, dev, args.bc_steps)
    if not args.smoke:
        import os; os.makedirs("runs/r3t_ckpt", exist_ok=True)
        torch.save({k: wm[k].state_dict() for k in ("enc", "dyn", "dec_g", "dec_t")} |
                   {"adim": wm["adim"], "grip_cm": wm["grip_cm"], "rollout_ood_cm": wm["rollout_ood_cm"], "img": IMG}, "runs/r3t_ckpt/wm3dt.pt")
        torch.save({"state": bc.state_dict(), "adim": ADIM, "img": IMG}, "runs/r3t_ckpt/bc3dt.pt")
        print("[r3t] saved checkpoints -> runs/r3t_ckpt/", flush=True)

    print("\n[r3t] ===== MOAT EVAL (zero-shot, frozen, real torque dynamics) =====", flush=True)
    out = {}
    for region in ("train", "test"):
        out[("wm", region)] = eval_method3d("wm_cem", {"wm": wm}, region, args.eval_eps, 0, dev)
        out[("bc", region)] = eval_method3d("bc", {"bc": bc}, region, args.eval_eps, 0, dev)
        print(f"  [{region:5}] WM {out[('wm',region)]}   |   BC {out[('bc',region)]}", flush=True)
    wt, bt = out[("wm", "test")]["succ@5cm"], out[("bc", "test")]["succ@5cm"]
    print(f"\n  GATE: grip_cm={wm['grip_cm']:.2f} held-rollout={wm['rollout_cm']:.1f}cm OOD-rollout={wm['rollout_ood_cm']:.1f}cm", flush=True)
    print(f"  MOAT @5cm SHIFTED(test): WM={wt:.2f} vs BC={bt:.2f} -> {'HOLDS' if wt-bt>=0.3 else 'WEAK/ABSENT'} (margin {wt-bt:+.2f})", flush=True)


if __name__ == "__main__":
    main()
