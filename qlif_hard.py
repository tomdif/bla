#!/usr/bin/env python3
"""Q-LIF harder task — does v1 (aux floors) BEAT plain SIGReg-JEPA?

v0/v1 were validated but their BENEFIT was undemonstrated: on the easy task plain
SIGReg already cleared the floors, so the aux floors had no gap to close. This task
manufactures a gap, using the grounding lesson: a JEPA encoder only encodes what the
prediction loss pays for.

Task (periodic + partially observed):
  angle th on a CIRCLE, driven by a HIDDEN velocity v; action a (torque) changes v:
      v_{t+1} = 0.9 v_t + 0.3*torque + noise ;  th_{t+1} = th_t + v_{t+1} + noise
  encoder sees a 3-FRAME STACK of NOISY [cos th, sin th]. So velocity IS recoverable
  (position differences) but is second-order + noisy, so next-latent prediction is
  dominated by tracking POSITION and UNDER-ENCODES velocity. The aux_vel / aux_action
  floors must force it in -> the regime where v1 can beat plain SIGReg.

Floors / probes (held-out, proxy bits):
  position : z -> [cos th, sin th]   (always recoverable; topology-sensitive readout, noisy so it won't saturate)
  velocity : z -> v                  (the UNDER-ENCODED, action-relevant variable -- where v1 should win)
  action   : (z,a) -> v_{t+1}  beats best single channel  (a affects v_{t+1}; marginal => gaming-resistant)

Methods: supervised (positive control), sigreg_jepa (plain baseline), qlif_v1 (aux floors),
broken_ownership (decoupled action). PRE-REGISTERED v1-benefit test:
  velocity_bits(v1) - velocity_bits(sigreg) >= V1_MARGIN  AND controls stay honest.
Self-contained, CPU. Emits q_lif_hard_gate.json.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

STD_FLOOR = 0.10; RANK_FLOOR = 1.0; B_POS = 0.5; B_VEL = 0.3; B_ACT = 0.3
V1_MARGIN = 0.5; INTRINSIC = 3.0   # th(circle=2) + v(1)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6000); ap.add_argument("--T", type=int, default=14); ap.add_argument("--S", type=int, default=3)
ap.add_argument("--D", type=int, default=16); ap.add_argument("--obs_noise", type=float, default=0.15)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lam_sig", type=float, default=1.0)
ap.add_argument("--lam_aux", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="q_lif_hard_gate.json")
args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
IN = args.S * 2


def sigreg_lewm(z, num_proj=1024, knots=17):
    if z.dim() == 2: z = z.unsqueeze(0)
    _, b, d = z.shape
    if b < 4: return z.new_zeros(())
    zf = z.float(); t = torch.linspace(0, 3, knots); dt = 3.0 / (knots - 1)
    w = torch.full((knots,), 2 * dt); w[0] = dt; w[-1] = dt; w = w * torch.exp(-t.square() / 2.0)
    a = torch.randn(d, num_proj); a = a.div_(a.norm(p=2, dim=0)); x_t = (zf @ a).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - torch.exp(-t.square() / 2.0)).square() + x_t.sin().mean(-3).square()
    return ((err @ w) * float(b)).mean()


def variance_hinge(z, gamma=1.0):
    return torch.clamp(gamma - z.std(0), min=0.0).mean()


def gen(decouple=False):
    """returns flat tensors over valid (stack ending at t) for t in [S-1, T-2]:
       X[N,IN] stacked noisy cos/sin, pos[N,2]=cos/sin th_t, vel[N,1]=v_t,
       act[N,1]=recorded a_t, vnext[N,1]=v_{t+1}, posnext[N,2]=cos/sin th_{t+1}."""
    rng = np.random.RandomState(args.seed + (7 if decouple else 0))
    th = np.zeros((args.n, args.T), np.float32); v = np.zeros((args.n, args.T), np.float32); a = np.zeros((args.n, args.T), np.float32)
    th[:, 0] = rng.uniform(0, 2 * math.pi, args.n); v[:, 0] = rng.randn(args.n) * 0.1
    for t in range(args.T - 1):
        torque = rng.randn(args.n) * 0.5
        a[:, t] = (rng.randn(args.n) * 0.5) if decouple else torque       # recorded action
        v[:, t + 1] = 0.9 * v[:, t] + 0.3 * torque + rng.randn(args.n) * 0.02
        th[:, t + 1] = th[:, t] + v[:, t + 1] + rng.randn(args.n) * 0.02
    cs = np.stack([np.cos(th), np.sin(th)], -1).astype(np.float32)        # [n,T,2] clean position
    obs = cs + rng.randn(args.n, args.T, 2).astype(np.float32) * args.obs_noise
    X, pos, vel, act, vnext, posnext = [], [], [], [], [], []
    for t in range(args.S - 1, args.T - 1):
        X.append(obs[:, t - args.S + 1:t + 1].reshape(args.n, -1))        # [n, S*2]
        pos.append(cs[:, t]); vel.append(v[:, t:t + 1]); act.append(a[:, t:t + 1])
        vnext.append(v[:, t + 1:t + 2]); posnext.append(cs[:, t + 1])
    f = lambda L: torch.tensor(np.concatenate(L, 0))
    return f(X), f(pos), f(vel), f(act), f(vnext), f(posnext)


def enc_mlp():
    return nn.Sequential(nn.Linear(IN, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, args.D))


def eff_rank(Z):
    lam = np.clip(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T)), 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0


def probe_mse(X, Y, steps=2500):
    n = len(X); idx = np.random.RandomState(0).permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 2e-3); g = torch.Generator().manual_seed(0); trt = torch.tensor(tr)
    for _ in range(steps):
        bb = trt[torch.randint(0, len(tr), (256,), generator=g)]
        loss = ((net(Xt[bb]) - Yt[bb]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((net(Xt[torch.tensor(te)]) - Yt[te]) ** 2).mean())


def bits(base, probe):
    return round(0.5 * math.log2(max(base, 1e-9) / max(probe, 1e-12)), 3)


# data (shared)
X, pos, vel, act, vnext, posnext = gen()
Xb, posb, velb, actb, vnextb, posnextb = gen(decouple=True)
N = len(X); ntr = int(0.8 * N)


def embed(f, X_):
    f.eval()
    with torch.no_grad():
        return f(X_).cpu().numpy()


def train_supervised(X_, pos_, vel_):
    f = enc_mlp(); head = nn.Linear(args.D, 3); opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), 2e-3)
    Y = torch.cat([pos_, vel_], -1); g = torch.Generator().manual_seed(0)
    for _ in range(args.steps):
        i = torch.randint(0, len(X_), (256,), generator=g)
        loss = ((head(f(X_[i])) - Y[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return f


def train_jepa(X_, act_, vnext_, pos_, vel_, aux=False):
    """plain single-ownership SIGReg JEPA (aux=False) or + health-gated aux floors (aux=True).
    'next latent' target = encoder of the stack shifted by one (approx via a learned predictor
    to vnext/pos is NOT used for L_pred; we use latent self-prediction across a +1 index)."""
    f = enc_mlp(); g_ = nn.Sequential(nn.Linear(args.D + 1, 128), nn.GELU(), nn.Linear(128, args.D))
    auxp = nn.Sequential(nn.Linear(args.D, 64), nn.GELU(), nn.Linear(64, 2))
    auxv = nn.Sequential(nn.Linear(args.D, 64), nn.GELU(), nn.Linear(64, 1))
    auxa = nn.Sequential(nn.Linear(args.D + 1, 64), nn.GELU(), nn.Linear(64, 1))
    P = list(f.parameters()) + list(g_.parameters()) + ([*auxp.parameters(), *auxv.parameters(), *auxa.parameters()] if aux else [])
    opt = torch.optim.Adam(P, 2e-3); gen_ = torch.Generator().manual_seed(0)
    # build "next stack" target index: the sample whose stack starts one step later is at +n rows
    # (gen concatenates by time block; consecutive time blocks are n apart). Use that for z_next.
    nblk = args.n
    al = 0.2 * float(vel_.var())                                # hinge allowed (velocity / pos share scale ~1)
    for _ in range(args.steps):
        i = torch.randint(0, len(X_) - nblk, (256,), generator=gen_)
        z_t = f(X_[i]); z_n = f(X_[i + nblk]); a_i = act_[i]
        L = ((g_(torch.cat([z_t, a_i], -1)) - z_n.detach()) ** 2).mean() + args.lam_sig * sigreg_lewm(z_t)
        if aux:
            if float(z_t.std(0).mean()) > STD_FLOOR:
                Lp = ((auxp(z_t) - pos_[i]) ** 2).mean(); Lv = ((auxv(z_t) - vel_[i]) ** 2).mean()
                La = ((auxa(torch.cat([z_t, a_i], -1)) - vnext_[i]) ** 2).mean()
                L = L + args.lam_aux * (torch.clamp(Lp - al, min=0.) + torch.clamp(Lv - al, min=0.) + torch.clamp(La - al, min=0.))
            else:
                L = L + variance_hinge(z_t)
        opt.zero_grad(); L.backward(); opt.step()
    return f


def diagnostic(Z, P, V, A, VN):
    er = round(eff_rank(Z), 2)
    h = {"std": round(float(np.median(Z.std(0))), 3), "eff_rank": er, "dim_cost": round(er / INTRINSIC, 2)}
    h["collapse"] = bool(h["std"] < STD_FLOOR or er < RANK_FLOOR)
    te = slice(ntr, N)
    Zt, Pt, Vt, At, VNt = Z[te], P[te].numpy(), V[te].numpy(), A[te].numpy(), VN[te].numpy()
    pos_bits = bits(float(((Pt - Pt.mean(0)) ** 2).mean()), probe_mse(Zt, Pt)) if not h["collapse"] else 0.0
    vel_bits = bits(float(((Vt - Vt.mean(0)) ** 2).mean()), probe_mse(Zt, Vt)) if not h["collapse"] else 0.0
    base_vn = float(((VNt - VNt.mean(0)) ** 2).mean())
    best_single = min(probe_mse(Zt, VNt), probe_mse(At, VNt), base_vn)
    act_bits = bits(best_single, probe_mse(np.concatenate([Zt, At], 1), VNt)) if not h["collapse"] else 0.0
    fl = {"position_bits": pos_bits, "velocity_bits": vel_bits, "action_bits": act_bits,
          "position_pass": (not h["collapse"]) and pos_bits >= B_POS,
          "velocity_pass": (not h["collapse"]) and vel_bits >= B_VEL,
          "action_pass": (not h["collapse"]) and act_bits >= B_ACT}
    v = "pass" if (not h["collapse"] and fl["position_pass"] and fl["velocity_pass"] and fl["action_pass"]) else "fail"
    return {"latent_health": h, "floors": fl, "verdict": v}


print(f"Q-LIF HARD | obs_noise={args.obs_noise} S={args.S} | v1-benefit margin={V1_MARGIN} bits\n", flush=True)
runs = {}
runs["positive(supervised)"] = diagnostic(embed(train_supervised(X, pos, vel), X), pos, vel, act, vnext)
runs["sigreg_jepa(plain)"] = diagnostic(embed(train_jepa(X, act, vnext, pos, vel, aux=False), X), pos, vel, act, vnext)
runs["qlif_v1(aux_floors)"] = diagnostic(embed(train_jepa(X, act, vnext, pos, vel, aux=True), X), pos, vel, act, vnext)
runs["broken_ownership_v1"] = diagnostic(embed(train_jepa(Xb, actb, vnextb, posb, velb, aux=True), Xb), posb, velb, actb, vnextb)

print(f"  {'run':24} {'std':>5} {'eff_rank':>8} {'dim_cost':>8} {'pos':>6} {'vel':>6} {'act':>6} verdict", flush=True)
for name, r in runs.items():
    h, f = r["latent_health"], r["floors"]
    print(f"  {name:24} {h['std']:5.2f} {h['eff_rank']:8.2f} {h['dim_cost']:8.2f} "
          f"{f['position_bits']:6.2f} {f['velocity_bits']:6.2f} {f['action_bits']:6.2f} {r['verdict']}", flush=True)

vel_sig = runs["sigreg_jepa(plain)"]["floors"]["velocity_bits"]; vel_v1 = runs["qlif_v1(aux_floors)"]["floors"]["velocity_bits"]
act_sig = runs["sigreg_jepa(plain)"]["floors"]["action_bits"]; act_v1 = runs["qlif_v1(aux_floors)"]["floors"]["action_bits"]
controls = {"positive_pass": runs["positive(supervised)"]["verdict"] == "pass",
            "broken_ownership_fail": runs["broken_ownership_v1"]["floors"]["action_pass"] is False}
sigreg_under = vel_sig < B_VEL or act_sig < B_ACT
v1_benefit = (vel_v1 - vel_sig >= V1_MARGIN) or (act_v1 - act_sig >= V1_MARGIN)
print(f"\n=== v1-benefit test ===", flush=True)
print(f"  controls honest: positive_pass={controls['positive_pass']} broken_ownership_fail={controls['broken_ownership_fail']}", flush=True)
print(f"  plain SIGReg under-encodes (the gap exists)?: {sigreg_under}  (vel_bits={vel_sig}, act_bits={act_sig})", flush=True)
print(f"  v1 closes it?  d_vel={round(vel_v1-vel_sig,3)}  d_act={round(act_v1-act_sig,3)}  (margin {V1_MARGIN})", flush=True)
verdict = ("V1 BEATS SIGReg" if (v1_benefit and sigreg_under and all(controls.values()))
           else ("NO GAP (task still too easy)" if not sigreg_under else "v1 does NOT close the gap"))
print(f"  => {verdict}", flush=True)
json.dump({"runs": runs, "controls": controls, "sigreg_under_encodes": sigreg_under,
           "d_vel": round(vel_v1 - vel_sig, 3), "d_act": round(act_v1 - act_sig, 3), "verdict": verdict},
          open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}", flush=True)
