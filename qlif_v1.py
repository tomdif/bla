#!/usr/bin/env python3
"""Q-LIF v1 — differentiable auxiliary-head hinge floors (now a TRAINING loss).

v0 was a diagnostic. v1 makes the floors a training signal: auxiliary heads on the
latent, each a HINGE floor (carry at least enough predictive quality, then stop),
GATED by latent health (floors off + variance repair when collapsed).

THE CANARY (the only thing that matters): once the floor is a loss, the model may
GAME it. The broken-ownership control is the trip-wire:
    if Q-LIF v1 makes broken-ownership PASS the action floor, v1 is INVALID
    (it found a shortcut instead of enforcing causal action ownership).
The v0 diagnostic's action floor is MARGINAL by construction -- (z,a) must beat
z-alone -- so leaking the future into z helps both equally and yields no marginal
gain. v1 must not break that. We verify by re-running the identical v0 diagnostic +
all controls on the v1-trained models.

v1 milestone is NOT higher bits. It is: v1 preserves/improves the candidate while
keeping every v0 control honest (positive pass, constant fail, broken-ownership fail).
Emits q_lif_v1_gate.json. Self-contained, CPU.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

STD_FLOOR = 0.10; RANK_FLOOR = 1.0; B_XY = 0.5; B_ACTION = 0.3; INTRINSIC_DIM = 2.0
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6000); ap.add_argument("--T", type=int, default=10)
ap.add_argument("--D", type=int, default=32); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--noise", type=float, default=0.03); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--lam_sig", type=float, default=1.0); ap.add_argument("--lam_xy", type=float, default=1.0)
ap.add_argument("--lam_act", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="q_lif_v1_gate.json")
args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)


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


def gen(decouple_action=False):
    rng = np.random.RandomState(args.seed + (7 if decouple_action else 0))
    s = np.zeros((args.n, args.T, 2), np.float32); a = np.zeros((args.n, args.T, 2), np.float32)
    s[:, 0] = rng.randn(args.n, 2)
    for t in range(args.T - 1):
        delta = rng.randn(args.n, 2) * 0.7
        a[:, t] = (rng.randn(args.n, 2) * 0.7) if decouple_action else delta
        s[:, t + 1] = s[:, t] + delta + rng.randn(args.n, 2) * args.noise
    R = rng.randn(2, args.D).astype(np.float32) / math.sqrt(2)
    return torch.tensor((s @ R).astype(np.float32)), torch.tensor(s), torch.tensor(a)


def enc_mlp():
    return nn.Sequential(nn.Linear(args.D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, args.d))


def eff_rank(Z):
    lam = np.clip(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T)), 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0


@torch.no_grad()
def embed(f, o):
    f.eval(); return f(o.reshape(-1, args.D)).cpu().numpy()


def probe_mse(X, Y, steps=2500):
    n = len(X); idx = np.random.RandomState(0).permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 2e-3); g = torch.Generator().manual_seed(0); tr_t = torch.tensor(tr)
    for _ in range(steps):
        bb = tr_t[torch.randint(0, len(tr), (256,), generator=g)]
        loss = ((net(Xt[bb]) - Yt[bb]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((net(Xt[torch.tensor(te)]) - Yt[te]) ** 2).mean())


def bits(mse_base, mse_probe):
    return round(0.5 * math.log2(max(mse_base, 1e-9) / max(mse_probe, 1e-12)), 3)


def train_supervised(o, s):
    f = enc_mlp(); head = nn.Linear(args.d, 2); opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), 2e-3)
    O = o.reshape(-1, args.D); S = s.reshape(-1, 2); g = torch.Generator().manual_seed(0)
    for _ in range(args.steps):
        i = torch.randint(0, len(O), (256,), generator=g)
        loss = ((head(f(O[i])) - S[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return f


def train_qlif_v1(o, a, s):
    """single-ownership JEPA + health-gated aux-head HINGE floors."""
    f = enc_mlp(); g_ = nn.Sequential(nn.Linear(args.d + 2, 128), nn.GELU(), nn.Linear(128, args.d))
    aux_xy = nn.Sequential(nn.Linear(args.d, 64), nn.GELU(), nn.Linear(64, 2))
    aux_act = nn.Sequential(nn.Linear(args.d + 2, 64), nn.GELU(), nn.Linear(64, 2))
    params = list(f.parameters()) + list(g_.parameters()) + list(aux_xy.parameters()) + list(aux_act.parameters())
    opt = torch.optim.Adam(params, 2e-3); gen_ = torch.Generator().manual_seed(0)
    var_s = float(s.reshape(-1, 2).var(0).mean())
    allowed = 0.2 * var_s            # hinge floor: require explaining ~80% of variance, then stop
    for _ in range(args.steps):
        bi = torch.randint(0, args.n, (256,), generator=gen_); ti = torch.randint(0, args.T - 1, (256,), generator=gen_)
        z_t = f(o[bi, ti]); z_n = f(o[bi, ti + 1]); act = a[bi, ti]; s_t = s[bi, ti]; s_next = s[bi, ti + 1]
        L = ((g_(torch.cat([z_t, act], -1)) - z_n.detach()) ** 2).mean() + args.lam_sig * sigreg_lewm(z_t)
        if float(z_t.std(0).mean()) > STD_FLOOR:                 # latent-health GATE
            mse_xy = ((aux_xy(z_t) - s_t) ** 2).mean()
            mse_act = ((aux_act(torch.cat([z_t, act], -1)) - s_next) ** 2).mean()
            L = L + args.lam_xy * torch.clamp(mse_xy - allowed, min=0.0) + args.lam_act * torch.clamp(mse_act - allowed, min=0.0)
        else:
            L = L + variance_hinge(z_t)                          # collapsed -> repair only
        opt.zero_grad(); L.backward(); opt.step()
    return f


def diagnostic(Z_all, s, a):
    Z = Z_all.reshape(args.n, args.T, -1)
    er = round(eff_rank(Z_all), 2)
    health = {"std": round(float(np.median(Z_all.std(0))), 3), "eff_rank": er,
              "dim_cost": round(er / INTRINSIC_DIM, 2)}
    health["collapse"] = bool(health["std"] < STD_FLOOR or er < RANK_FLOOR)
    ntr = int(0.8 * args.n)
    zt = Z[ntr:, :-1].reshape(-1, Z.shape[-1])
    s_now = s[ntr:, :-1].reshape(-1, 2).numpy(); s_next = s[ntr:, 1:].reshape(-1, 2).numpy(); a_now = a[ntr:, :-1].reshape(-1, 2).numpy()
    base_xy = float(((s_now - s_now.mean(0)) ** 2).mean())
    xy_bits = bits(base_xy, probe_mse(zt, s_now)) if not health["collapse"] else 0.0
    base_next = float(((s_next - s_next.mean(0)) ** 2).mean())
    mse_za = probe_mse(np.concatenate([zt, a_now], 1), s_next)
    best_single = min(probe_mse(zt, s_next), probe_mse(a_now, s_next), base_next)
    act_bits = bits(best_single, mse_za) if not health["collapse"] else 0.0
    floors = {"predictive_xy_bits": xy_bits, "action_delta_bits": act_bits,
              "predictive_pass": (not health["collapse"]) and xy_bits >= B_XY,
              "action_pass": (not health["collapse"]) and act_bits >= B_ACTION}
    verdict = "pass" if (not health["collapse"] and floors["predictive_pass"] and floors["action_pass"]) else "fail"
    return {"latent_health": health, "floors": floors, "verdict": verdict}


print(f"Q-LIF v1 | aux-head hinge floors (health-gated) | thresholds std>={STD_FLOOR} xy_bits>={B_XY} action_bits>={B_ACTION}\n", flush=True)
o, s, a = gen(); o_b, s_b, a_b = gen(decouple_action=True)
runs = {}
runs["positive_control(supervised)"] = diagnostic(embed(train_supervised(o, s), o), s, a)
runs["constant_latent"] = diagnostic(np.zeros((args.n * args.T, args.d), np.float32), s, a)
runs["candidate_v1(sigreg+aux_floors)"] = diagnostic(embed(train_qlif_v1(o, a, s), o), s, a)
runs["broken_ownership_v1(decoupled)"] = diagnostic(embed(train_qlif_v1(o_b, a_b, s_b), o_b), s_b, a_b)

print(f"  {'run':38} {'std':>6} {'eff_rank':>9} {'dim_cost':>9} {'xy_bits':>8} {'act_bits':>9} verdict", flush=True)
for name, r in runs.items():
    h, f = r["latent_health"], r["floors"]
    print(f"  {name:38} {h['std']:6.3f} {h['eff_rank']:9.2f} {h['dim_cost']:9.2f} "
          f"{f['predictive_xy_bits']:8.3f} {f['action_delta_bits']:9.3f} {r['verdict']}", flush=True)

controls = {"positive_control_pass": runs["positive_control(supervised)"]["verdict"] == "pass",
            "constant_latent_fail": runs["constant_latent"]["verdict"] == "fail",
            "broken_ownership_fail": runs["broken_ownership_v1(decoupled)"]["floors"]["action_pass"] is False}
diag_valid = all(controls.values())
gamed = runs["broken_ownership_v1(decoupled)"]["floors"]["action_pass"] is True
print(f"\n=== v1 milestone gates ===", flush=True)
for k, v in controls.items():
    print(f"  {k}: {'OK' if v else 'VIOLATED'}", flush=True)
print(f"  v1 GAMED the action floor (broken-ownership passed)?: {'YES -> v1 INVALID' if gamed else 'no'}", flush=True)
print(f"\nDIAGNOSTIC VALID: {diag_valid}", flush=True)
print(f"v1 CANDIDATE verdict: {runs['candidate_v1(sigreg+aux_floors)']['verdict']}", flush=True)
v1_pass = diag_valid and runs["candidate_v1(sigreg+aux_floors)"]["verdict"] == "pass" and not gamed
print(f"v1 MILESTONE (controls honest AND candidate passes AND not gamed): {'EARNED' if v1_pass else 'NOT earned'}", flush=True)

json.dump({"runs": runs, "controls": controls, "v1_gamed_action_floor": gamed,
           "diagnostic_valid": diag_valid, "v1_milestone_earned": v1_pass,
           "slot_floor": "deferred — needs object-centric encoder"}, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}", flush=True)
