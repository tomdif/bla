#!/usr/bin/env python3
"""Imagination-trained policy proposer (Dreamer-style analytic gradient) -- the differentiator over CEM.

The world model (dyn, dec_g) is differentiable, so we train a goal+zone-conditioned policy by BACKPROPAGATING
the imagined H-step cost through the FROZEN world model. Unlike random-sampling CEM/MPPI (which can't find a
coherent multi-step detour around a no-go zone), gradient descent follows the zone-penalty gradient and learns
the detour. This is the concrete demonstration that CEM belongs as a refiner, not the core planner.

Pipeline: load the torque world model -> build a buffer of real latents -> train policy in imagination ->
A/B CEM vs POLICY on (a) plain reach, (b) FORBIDDEN-ZONE task (reach goal while keeping the ee out of a zone
on the direct path). Run: MUJOCO_GL=egl python3 -m system1_motion.imagination_policy --train --ab
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from system1_motion.r3_torque import Arm, load_wm3d, norm3, region_of3d, SPAN, LO, HI, IMG, ADIM, RMIN, RMAX

CKPT = "runs/r3t_ckpt/wm3dt.pt"
ZR_W = 0.10                                                    # zone radius in WORLD meters
W_PEN = 1.5                                                   # zone penalty weight (vs normalized distance ~<=1)
span_t = None                                                 # set on device


def zr_norm():  return ZR_W / float(np.mean(SPAN))            # zone radius in normalized units


# ----------------------------- policy -----------------------------
class ImaginationPolicy(nn.Module):
    def __init__(self, d_z=384, adim=ADIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z + 3 + 4, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, adim))
    def forward(self, z, g, zone):                            # g[B,3] zone[B,4]=(center3, radius)
        return torch.tanh(self.net(torch.cat([z, g, zone], -1)))


def imag_cost(ee, g, zone):                                   # ee,g normalized [B,3]; zone[B,4]
    d = (ee - g).norm(dim=-1)
    zc, zr = zone[:, :3], zone[:, 3]
    pen = W_PEN * torch.exp(-((ee - zc).pow(2).sum(-1)) / (2 * zr.clamp_min(0.02) ** 2)) * (zr > 0).float()
    return d + pen


class ValueHead(nn.Module):                                   # predicts return-to-go (=-cost-to-go) for lambda-returns
    def __init__(self, d_z=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z + 3 + 4, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
    def forward(self, z, g, zone): return self.net(torch.cat([z, g, zone], -1)).squeeze(-1)


@torch.no_grad()
def roll_policy(pol, wm, z0, g, zone, device, H=8):
    """roll the policy through the WM; return its action chunk + decoded ee path (normalized) for verification."""
    gt = torch.tensor(g, device=device).float()[None]; zt = torch.tensor(zone, device=device).float()[None]
    z = z0; acts = []; ees = [wm["dec_g"](z)[0].cpu().numpy()]
    for _ in range(H):
        a = pol(z, gt, zt); acts.append(a[0].cpu().numpy()); z = wm["dyn"](z, a); ees.append(wm["dec_g"](z)[0].cpu().numpy())
    return np.array(acts), np.array(ees)


def sample_goals(n, rng):
    out = []
    while len(out) < n:
        c = rng.uniform([0.10, -0.35, 0.08], [0.38, 0.35, 0.42])
        if RMIN < np.linalg.norm(c) < RMAX: out.append(norm3(c))
    return np.asarray(out, np.float32)


def train_policy(wm, device, steps=6000, H=12, gamma=0.97, lr=3e-4, batch=256, zone_frac=0.5, log=print):
    for k in ("enc", "dyn", "dec_g", "dec_t"):                # FREEZE the world model (grad still flows THROUGH it)
        for p in wm[k].parameters(): p.requires_grad_(False)
    rng = np.random.RandomState(0)
    log("[pol] building real-latent buffer ...", flush=True)
    arm = Arm(0); Z = []
    with torch.no_grad():
        for _ in range(2500):
            arm.reset()
            x = torch.from_numpy(arm.render().astype(np.float32) / 255.0)[None].to(device)
            Z.append(wm["enc"](x))
    arm.close(); Zbuf = torch.cat(Z); log(f"[pol] {Zbuf.shape[0]} latents; training policy in imagination ...", flush=True)
    G = torch.from_numpy(sample_goals(8000, rng)).to(device); zrn = zr_norm()
    pol = ImaginationPolicy().to(device); opt = torch.optim.AdamW(pol.parameters(), lr=lr); t0 = time.time()
    for step in range(steps):
        z = Zbuf[rng.randint(0, Zbuf.shape[0], batch)]
        g = G[rng.randint(0, G.shape[0], batch)]
        with torch.no_grad(): ee0 = wm["dec_g"](z)
        use = (torch.from_numpy(rng.rand(batch)).to(device) < zone_frac).float()
        zc = 0.5 * (ee0 + g) + torch.from_numpy(rng.normal(0, 0.04, (batch, 3)).astype(np.float32)).to(device)
        zone = torch.cat([zc, (torch.full((batch, 1), zrn, device=device) * use[:, None])], -1)
        zc_z, tot = z, 0.0
        for h in range(H):
            a = pol(zc_z, g, zone); zc_z = wm["dyn"](zc_z, a); ee = wm["dec_g"](zc_z)
            tot = tot + (gamma ** h) * imag_cost(ee, g, zone)
        loss = tot.mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % max(1, steps // 12) == 0 or step == steps - 1:
            with torch.no_grad():                            # report imagined FINAL-step dist (cm) reach + zone breach
                fin_cm = ((ee - g) * span_t).pow(2).sum(-1).sqrt().mean().item() * 100
            log(f"[pol {step}/{steps}] imag_cost={loss.item():.3f} imag_final_dist={fin_cm:.1f}cm ({time.time()-t0:.0f}s)", flush=True)
    return pol.eval()


def train_policy_v2(wm, device, steps=6000, H=10, gamma=0.97, lam=0.95, lr=3e-4, batch=256,
                    zone_frac=0.5, areg=0.03, log=print):
    """Dreamer-style actor-critic in imagination: lambda-returns bootstrapped by a value head (denser/longer
    signal than naive full-H backprop), + action regularization to discourage exploiting WM error with extreme actions."""
    for k in ("enc", "dyn", "dec_g", "dec_t"):
        for p in wm[k].parameters(): p.requires_grad_(False)
    rng = np.random.RandomState(0); arm = Arm(0); Z = []
    log("[polv2] building real-latent buffer ...", flush=True)
    with torch.no_grad():
        for _ in range(2500):
            arm.reset(); x = torch.from_numpy(arm.render().astype(np.float32) / 255.0)[None].to(device); Z.append(wm["enc"](x))
    arm.close(); Zbuf = torch.cat(Z); G = torch.from_numpy(sample_goals(8000, rng)).to(device); zrn = zr_norm()
    pol = ImaginationPolicy().to(device); val = ValueHead().to(device)
    oa = torch.optim.AdamW(pol.parameters(), lr=lr); ov = torch.optim.AdamW(val.parameters(), lr=lr)
    log("[polv2] training actor-critic in imagination ...", flush=True); t0 = time.time()
    for step in range(steps):
        z = Zbuf[rng.randint(0, Zbuf.shape[0], batch)]; g = G[rng.randint(0, G.shape[0], batch)]
        with torch.no_grad(): ee0 = wm["dec_g"](z)
        use = (torch.from_numpy(rng.rand(batch)).to(device) < zone_frac).float()
        zc = 0.5 * (ee0 + g) + torch.from_numpy(rng.normal(0, 0.04, (batch, 3)).astype(np.float32)).to(device)
        zone = torch.cat([zc, torch.full((batch, 1), zrn, device=device) * use[:, None]], -1)
        zcur = z; states = [z]; rews = []; acts = []
        for h in range(H):
            a = pol(zcur, g, zone); acts.append(a); zcur = wm["dyn"](zcur, a)
            rews.append(-imag_cost(wm["dec_g"](zcur), g, zone)); states.append(zcur)
        Vb = [val(s, g, zone).detach() for s in states]       # detached bootstrap -> actor grad flows only via rewards
        ret = [None] * H; last = Vb[H]
        for h in reversed(range(H)):                           # Dreamer lambda-return (rews keep grad to the policy)
            last = rews[h] + gamma * ((1 - lam) * Vb[h + 1] + lam * last); ret[h] = last
        actor_loss = -torch.stack(ret).mean() + areg * torch.stack(acts).pow(2).mean()
        oa.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); oa.step()
        Vc = [val(states[h].detach(), g, zone) for h in range(H)]   # critic graph independent of policy -> no in-place conflict
        tgt = [None] * H; last = Vb[H]
        for h in reversed(range(H)):
            last = rews[h].detach() + gamma * ((1 - lam) * Vb[h + 1] + lam * last); tgt[h] = last
        critic_loss = sum(F.mse_loss(Vc[h], tgt[h]) for h in range(H)) / H
        ov.zero_grad(); critic_loss.backward(); nn.utils.clip_grad_norm_(val.parameters(), 1.0); ov.step()
        if step % max(1, steps // 12) == 0 or step == steps - 1:
            with torch.no_grad(): fin_cm = ((wm["dec_g"](zcur) - g) * span_t).pow(2).sum(-1).sqrt().mean().item() * 100
            log(f"[polv2 {step}/{steps}] actor={actor_loss.item():.3f} critic={critic_loss.item():.3f} "
                f"imag_final_dist={fin_cm:.1f}cm ({time.time()-t0:.0f}s)", flush=True)
    return pol.eval()


# ----------------------------- planners for eval -----------------------------
@torch.no_grad()
def cem_zone(wm, z0, g, zone, device, horizon=8, iters=5, pop=256, elite=32, terminal_w=6.0):
    mu = torch.zeros(horizon, wm["adim"], device=device); sigma = torch.ones(horizon, wm["adim"], device=device) * 0.6
    gt = torch.tensor(g, device=device).float(); zt = torch.tensor(zone, device=device).float()[None]
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, wm["adim"], device=device)).clamp(-1, 1)
        z = z0.expand(pop, -1).clone(); cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h]); ee = wm["dec_g"](z)
            c = imag_cost(ee, gt[None].expand(pop, -1), zt.expand(pop, -1))
            cost = cost + c * (terminal_w if h == horizon - 1 else 1.0)
        e = seqs[cost.topk(elite, largest=False).indices]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    return mu[0].cpu().numpy()


@torch.no_grad()
def eval_planner(which, wm, pol, region, n_eps, device, zone_on, ep_len=26, seed0=7000):
    arm = Arm(seed0); reach5 = reach10 = avoided = 0; finals = []; zrn = zr_norm(); n_steps = n_fallback = 0
    for e in range(n_eps):
        arm.reset(); tgt = arm.sample_target(region); arm.set_target(tgt); g = norm3(tgt)
        zc = 0.5 * (norm3(arm.ee()) + g); zr = zrn if zone_on else 0.0
        zone = np.concatenate([zc, [zr]]).astype(np.float32)
        gt = torch.tensor(g, device=device).float()[None]; zt = torch.tensor(zone, device=device).float()[None]; min_zone = 9.9
        for t in range(ep_len):
            x = torch.from_numpy(arm.render().astype(np.float32) / 255.0)[None].to(device); z0 = wm["enc"](x); n_steps += 1
            if which == "policy":
                a = pol(z0, gt, zt)[0].cpu().numpy()
            elif which == "verify":                            # governor: reject exploited proposals, fall back to CEM
                acts, ees = roll_policy(pol, wm, z0, g, zone, device, H=8)
                motion = float(np.linalg.norm(np.diff(ees, axis=0) * SPAN, axis=1).max()) * 100   # max predicted cm/step
                sat = float(np.mean(np.abs(acts[0]) > 0.95))
                if motion > 18.0 or sat > 0.6:                 # implausible predicted motion / saturated -> untrustworthy
                    a = cem_zone(wm, z0, g, zone, device); n_fallback += 1
                else:
                    a = acts[0]
            else:
                a = cem_zone(wm, z0, g, zone, device)
            arm.step(np.clip(a, -1, 1))
            min_zone = min(min_zone, float(np.linalg.norm(norm3(arm.ee())[:3] - zc)))
        d_cm = float(np.linalg.norm(arm.ee() - tgt)) * 100; finals.append(d_cm)
        reach5 += d_cm <= 5; reach10 += d_cm <= 10; avoided += (min_zone > zr) if zone_on else 1
    arm.close()
    out = {"reach@5": reach5 / n_eps, "reach@10": reach10 / n_eps, "avoided_zone": avoided / n_eps, "mean_cm": float(np.mean(finals))}
    if which == "verify": out["cem_fallback"] = round(n_fallback / max(1, n_steps), 2)
    return out


def main():
    global span_t
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true"); ap.add_argument("--ab", action="store_true")
    ap.add_argument("--v2", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--policy-file", default="runs/r3t_ckpt/policy.pt")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-eps", type=int, default=25); ap.add_argument("--action-repeat", type=int, default=12)
    args = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    import system1_motion.r3_torque as R; R.AR = args.action_repeat          # match the WM's control step
    span_t = torch.tensor(SPAN, device=dev).float()
    wm = load_wm3d(CKPT, dev); print(f"[pol] loaded WM (grip_cm={wm['grip_cm']:.2f}) AR={R.AR}", flush=True)
    if args.smoke: args.steps = 300; args.eval_eps = 5
    pf = "runs/r3t_ckpt/policy_v2.pt" if args.v2 else args.policy_file

    pol = None
    if args.v2:
        pol = train_policy_v2(wm, dev, steps=args.steps); torch.save({"state": pol.state_dict()}, pf); print(f"[polv2] saved {pf}", flush=True)
    elif args.train or args.smoke:
        pol = train_policy(wm, dev, steps=args.steps); torch.save({"state": pol.state_dict()}, pf); print(f"[pol] saved {pf}", flush=True)
    if args.ab or args.smoke or args.verify:
        if pol is None:
            pol = ImaginationPolicy().to(dev); pol.load_state_dict(torch.load(pf, map_location=dev)["state"]); pol.eval()
        ne = args.eval_eps; whichset = ("cem", "policy", "verify") if args.verify else ("cem", "policy")
        print(f"\n[pol] ===== A/B {'+verify ' if args.verify else ''}({ne} eps, policy={pf.split('/')[-1]}) =====", flush=True)
        for zone_on in (False, True):
            tag = "ZONE-TASK" if zone_on else "plain-reach"
            for which in whichset:
                res = eval_planner(which, wm, pol, "test", ne, dev, zone_on)
                print(f"  [{tag:11} test ] {which:6} {res}", flush=True)


if __name__ == "__main__":
    main()
