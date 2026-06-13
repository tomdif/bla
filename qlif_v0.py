#!/usr/bin/env python3
"""Q-LIF v0 — DIAGNOSTIC ONLY (not a training loss). The Q-LIF Resurrection Gate.

The session's lesson: a metric that can't catch a lie when handed one is worthless
(the standardized probe made a collapsed latent look best). So v0 is a diagnostic
whose VALIDITY is established by its own controls before it scores anything:
  positive control (supervised latent)  MUST pass
  constant latent                       MUST fail (collapse)
  broken ownership (decoupled actions)  MUST fail (action floor)
If any control misbehaves, the diagnostic is broken and no candidate verdict counts.

What it measures, in strict order (health BEFORE information, per the design):
  latent_health : std, effective_rank, collapse flag (raw latent, no standardization)
  predictive floor : does z encode state?  proxy bits from a held-out probe z -> s
  action floor : does (z,a) predict the next state beyond the best single channel?
                 (requires z to encode state AND the recorded action to be informative)
Bits are PROXY bits (held-out probe vs baseline), explicitly not certificates:
  bits = 0.5 * log2( MSE_baseline / MSE_probe )

Task: action-conditioned point mass.  s_{t+1} = s_t + a_t + noise;  o = R @ s (linear).
Self-contained, CPU, minutes. Produces q_lif_resurrection_gate.json.
Slot/object floor is DEFERRED to the object-centric rig (needs a slot encoder) and
is intentionally not faked here.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

# ---- pre-registered thresholds ----
STD_FLOOR = 0.10; RANK_FLOOR = 1.0; B_XY = 0.5; B_ACTION = 0.3   # bits

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6000); ap.add_argument("--T", type=int, default=10)
ap.add_argument("--D", type=int, default=32); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--noise", type=float, default=0.03); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="q_lif_resurrection_gate.json")
args = ap.parse_args()
dev = "cpu"; torch.manual_seed(args.seed); np.random.seed(args.seed)


def sigreg_lewm(z, num_proj=1024, knots=17):
    if z.dim() == 2: z = z.unsqueeze(0)
    _, b, d = z.shape
    if b < 4: return z.new_zeros(())
    zf = z.float(); t = torch.linspace(0, 3, knots); dt = 3.0 / (knots - 1)
    w = torch.full((knots,), 2 * dt); w[0] = dt; w[-1] = dt; window = torch.exp(-t.square() / 2.0); w = w * window
    a = torch.randn(d, num_proj); a = a.div_(a.norm(p=2, dim=0)); proj = zf @ a; x_t = proj.unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - window).square() + x_t.sin().mean(-3).square()
    return ((err @ w) * float(b)).mean()


def gen(decouple_action=False):
    """s_{t+1}=s_t+delta+noise. Normal: recorded a==delta. Broken: recorded a is random
    noise decoupled from the true delta -> action floor must fail."""
    rng = np.random.RandomState(args.seed + (7 if decouple_action else 0))
    s = np.zeros((args.n, args.T, 2), np.float32); a = np.zeros((args.n, args.T, 2), np.float32)
    s[:, 0] = rng.randn(args.n, 2)
    for t in range(args.T - 1):
        delta = rng.randn(args.n, 2) * 0.7
        a[:, t] = (rng.randn(args.n, 2) * 0.7) if decouple_action else delta   # recorded action
        s[:, t + 1] = s[:, t] + delta + rng.randn(args.n, 2) * args.noise
    R = rng.randn(2, args.D).astype(np.float32) / math.sqrt(2)
    o = (s @ R).astype(np.float32)
    return torch.tensor(o), torch.tensor(s), torch.tensor(a)


def enc_mlp():
    return nn.Sequential(nn.Linear(args.D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, args.d))


def eff_rank(Z):
    lam = np.clip(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T)), 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0


@torch.no_grad()
def embed(f, o):
    f.eval(); return f(o.reshape(-1, args.D)).cpu().numpy()


def probe_mse(X, Y, steps=2500):
    """fresh MLP X->Y, held-out MSE."""
    n = len(X); idx = np.random.RandomState(0).permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 2e-3); g = torch.Generator().manual_seed(0)
    tr_t = torch.tensor(tr)
    for _ in range(steps):
        b = tr_t[torch.randint(0, len(tr), (256,), generator=g)]
        loss = ((net(Xt[b]) - Yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
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


def train_sigreg_jepa(o, a):
    f = enc_mlp(); g_ = nn.Sequential(nn.Linear(args.d + 2, 128), nn.GELU(), nn.Linear(128, args.d))
    opt = torch.optim.Adam(list(f.parameters()) + list(g_.parameters()), 2e-3); gen_ = torch.Generator().manual_seed(0)
    for _ in range(args.steps):
        bi = torch.randint(0, args.n, (256,), generator=gen_); ti = torch.randint(0, args.T - 1, (256,), generator=gen_)
        z_t = f(o[bi, ti]); z_n = f(o[bi, ti + 1]); act = a[bi, ti]
        L = ((g_(torch.cat([z_t, act], -1)) - z_n.detach()) ** 2).mean() + args.lam * sigreg_lewm(z_t)  # single ownership
        opt.zero_grad(); L.backward(); opt.step()
    return f


def diagnostic(Z_all, s, a):
    """Z_all: [n*T, d]. Compute health + predictive + action floors on held-out pairs."""
    n = args.n; T = args.T
    Z = Z_all.reshape(n, T, -1)
    health = {"std": round(float(np.median(Z_all.std(0))), 3), "eff_rank": round(eff_rank(Z_all), 2)}
    health["collapse"] = bool(health["std"] < STD_FLOOR or health["eff_rank"] < RANK_FLOOR)
    # held-out split on trajectories
    ntr = int(0.8 * n)
    zt = Z[ntr:, :-1].reshape(-1, Z.shape[-1]); zn = None
    s_now = s[ntr:, :-1].reshape(-1, 2).numpy(); s_next = s[ntr:, 1:].reshape(-1, 2).numpy()
    a_now = a[ntr:, :-1].reshape(-1, 2).numpy()
    # predictive floor: does z encode current state?
    base_xy = float(((s_now - s_now.mean(0)) ** 2).mean())
    xy_bits = bits(base_xy, probe_mse(zt, s_now)) if not health["collapse"] else 0.0
    # action floor: does (z,a) predict next state beyond the BEST single channel?
    base_next = float(((s_next - s_next.mean(0)) ** 2).mean())
    mse_za = probe_mse(np.concatenate([zt, a_now], 1), s_next)
    mse_z = probe_mse(zt, s_next); mse_a = probe_mse(a_now, s_next)
    best_single = min(mse_z, mse_a, base_next)
    act_bits = bits(best_single, mse_za) if not health["collapse"] else 0.0
    floors = {"predictive_xy_bits": xy_bits, "action_delta_bits": act_bits,
              "predictive_pass": (not health["collapse"]) and xy_bits >= B_XY,
              "action_pass": (not health["collapse"]) and act_bits >= B_ACTION}
    verdict = "pass" if (not health["collapse"] and floors["predictive_pass"] and floors["action_pass"]) else "fail"
    return {"latent_health": health, "floors": floors, "verdict": verdict}


print(f"Q-LIF v0 | thresholds std>={STD_FLOOR} rank>={RANK_FLOOR} xy_bits>={B_XY} action_bits>={B_ACTION}\n", flush=True)
o, s, a = gen(); o_b, s_b, a_b = gen(decouple_action=True)

runs = {}
runs["positive_control(supervised)"] = diagnostic(embed(train_supervised(o, s), o), s, a)
runs["constant_latent"] = diagnostic(np.zeros((args.n * args.T, args.d), np.float32), s, a)
runs["candidate(sigreg_jepa)"] = diagnostic(embed(train_sigreg_jepa(o, a), o), s, a)
runs["broken_ownership(decoupled_action)"] = diagnostic(embed(train_sigreg_jepa(o_b, a_b), o_b), s_b, a_b)

for name, r in runs.items():
    h, f = r["latent_health"], r["floors"]
    print(f"  {name:36} std={h['std']:.3f} eff_rank={h['eff_rank']:.2f} collapse={h['collapse']} | "
          f"xy_bits={f['predictive_xy_bits']} action_bits={f['action_delta_bits']} -> {r['verdict']}", flush=True)

# control invariants (the diagnostic must catch lies)
controls = {
    "positive_control_pass": runs["positive_control(supervised)"]["verdict"] == "pass",
    "constant_latent_fail": runs["constant_latent"]["verdict"] == "fail",
    "broken_ownership_fail": runs["broken_ownership(decoupled_action)"]["floors"]["action_pass"] is False,
}
diag_valid = all(controls.values())
print(f"\n=== control invariants (validate the diagnostic itself) ===", flush=True)
for k, v in controls.items():
    print(f"  {k}: {'OK' if v else 'VIOLATED'}", flush=True)
print(f"DIAGNOSTIC VALID: {diag_valid}", flush=True)
print(f"CANDIDATE (sigreg_jepa) verdict: {runs['candidate(sigreg_jepa)']['verdict']}"
      f"{'' if diag_valid else '  (INVALID — controls misbehaved, candidate verdict does not count)'}", flush=True)

out = {"thresholds": {"std_floor": STD_FLOOR, "rank_floor": RANK_FLOOR, "B_xy": B_XY, "B_action": B_ACTION},
       "runs": runs, "controls": controls, "diagnostic_valid": diag_valid,
       "candidate_verdict": runs["candidate(sigreg_jepa)"]["verdict"],
       "slot_floor": "deferred — needs object-centric encoder, not faked in v0"}
json.dump(out, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}", flush=True)
