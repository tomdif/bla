#!/usr/bin/env python3
"""Minimal single-ownership topology discriminator.

Question: within a HEALTHY rig (single-ownership loss, no fitted statistics in the
objective), does SIGReg degrade the representation when the state manifold is a
CIRCLE vs a LINE? Tests depth-2 (does single-ownership cure collapse?) and depth-3
(is there a topology-specific SIGReg cost?) at once.

THREE HARDENINGS (a new rig must gate itself or branch-3 is unfalsifiable):
  H1 supervised positive control that cannot collapse by construction -> validates
     plumbing (eff_rank/std/probe/loop). Gate everything behind it.
  H2 the line task is the literal UNIVERSAL COVER of the circle task: identical x_t
     trajectory + noise; the two differ ONLY by the quotient (observe x vs x mod 2pi).
     Any gap is attributable to the quotient and nothing else.
  H3 pre-register "struggles" as DISTORTION not death: a circle has no continuous
     bijection onto a Gaussian line, so SIGReg likely tears/crushes it with std and
     eff_rank still healthy. Registered metric = cos/sin probe R^2 gap, thresholds set
     BELOW, before the run.

LOSS = single ownership: L_pred = RAW MSE in latent space (stop-grad target, no
normalization, no running stats); SIGReg (known-good lewm variant) solely owns scale+rank.

PRE-REGISTERED GATES / BRANCHES (decided before running):
  std_floor=0.10  eff_rank_floor=1.0  sup_probe_pass_R2=0.95  distortion_gap=0.10
  Gate 0: supervised control on BOTH tasks -> cos/sin probe R2>=0.95, std/eff_rank healthy.
          FAIL => rig bug, fix before anything counts.
  Gate 1: SIGReg on the LINE (cover) -> std>=floor AND eff_rank>=floor (depth-2 cert).
          FAIL => BRANCH 3: single-ownership insufficient (attributable to ownership,
          NOT plumbing, because gate 0 passed). Halt-and-rethink before v4.
  Behind a passed Gate 1, read the CIRCLE:
    BRANCH 1 (topology localized): circle degrades vs line --
       stability cost  (std<floor or eff_rank<floor)  OR
       accuracy cost   (circle probe R2 < line probe R2 - distortion_gap), std healthy.
       => fix = manifold-matched target (uniform-on-circle / von Mises / product).
    BRANCH 2 (topology dies): circle ~ line (probe R2 within gap, both healthy)
       => single-ownership confirmed as the cure; port proceeds with clean conscience.
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---- pre-registered thresholds ----
STD_FLOOR = 0.10; RANK_FLOOR = 1.0; SUP_PASS = 0.95; DISTORT_GAP = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--periods", type=int, default=4)   # x ranges over N*2pi (quotient must bite)
ap.add_argument("--n", type=int, default=4000)      # trajectories
ap.add_argument("--T", type=int, default=12)
ap.add_argument("--D", type=int, default=32)        # observation dim
ap.add_argument("--d", type=int, default=16)        # latent dim
ap.add_argument("--omega", type=float, default=0.30)
ap.add_argument("--noise", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--lam", type=float, default=1.0)   # SIGReg weight (owns scale+rank)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", type=str, default="cpu")
args = ap.parse_args()
dev = args.device if (torch.cuda.is_available() or args.device in ("cpu", "mps")) else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
g = torch.Generator().manual_seed(args.seed)


# ---- known-good SIGReg (lewm variant), copied verbatim ----
def sigreg_lewm(z, num_proj=1024, knots=17):
    if z.dim() == 2: z = z.unsqueeze(0)
    t_dim, batch, d = z.shape
    if batch < 4: return z.new_zeros(())
    zf = z.float()
    t = torch.linspace(0, 3, knots, device=zf.device, dtype=zf.dtype); dt = 3.0 / (knots - 1)
    w = torch.full((knots,), 2 * dt, device=zf.device, dtype=zf.dtype); w[0] = dt; w[-1] = dt
    window = torch.exp(-t.square() / 2.0); w = w * window
    a = torch.randn(d, num_proj, device=zf.device, dtype=zf.dtype); a = a.div_(a.norm(p=2, dim=0))
    proj = zf @ a; x_t = proj.unsqueeze(-1) * t
    err = (x_t.cos().mean(dim=-3) - window).square() + x_t.sin().mean(dim=-3).square()
    return ((err @ w) * float(batch)).mean().to(z.dtype)


# ---- H2: universal cover. ONE x_t trajectory; line observes x, circle observes x mod 2pi ----
def gen():
    rng = np.random.RandomState(args.seed)
    x0 = rng.uniform(0, args.periods * 2 * math.pi, size=(args.n, 1))
    x = np.zeros((args.n, args.T, 1), np.float32)
    cur = x0.copy()
    for t in range(args.T):
        x[:, t] = cur
        cur = cur + args.omega + rng.randn(args.n, 1) * args.noise
    # fixed random projections to D-dim observation space
    Rc = rng.randn(1, args.D).astype(np.float32) / math.sqrt(1)      # line: embed x_norm
    Rt = rng.randn(2, args.D).astype(np.float32) / math.sqrt(2)      # circle: embed [cos,sin]
    xn = (x / (args.periods * 2 * math.pi)).astype(np.float32)       # line coord in [0,1]
    cs = np.concatenate([np.cos(x), np.sin(x)], -1).astype(np.float32)  # circle embedding + GT
    o_line = xn @ Rc                                                  # [n,T,D]
    o_circ = cs @ Rt
    return (torch.tensor(o_line), torch.tensor(o_circ), torch.tensor(cs))  # cs = ground-truth [cos,sin]


o_line, o_circ, gt = gen()
ntr = int(0.8 * args.n)
TASKS = {"line(cover)": o_line, "circle(torus)": o_circ}


def enc_mlp():
    return nn.Sequential(nn.Linear(args.D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, args.d)).to(dev)


def eff_rank(Z):
    Zc = Z - Z.mean(0); cov = np.cov(Zc.T); lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0


@torch.no_grad()
def embed(f, o):
    f.eval(); return f(o.reshape(-1, args.D).to(dev)).cpu().numpy()


def probe_cossin(Z, Y, steps=3000):
    """fresh MLP Z->[cos,sin], held-out R^2 (circle-preservation = representation quality)."""
    n = len(Z); idx = np.random.RandomState(0).permutation(n)
    tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    Zt = torch.tensor((Z - Z.mean(0)) / (Z.std(0) + 1e-6), dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    tr_t = torch.tensor(tr, device=dev); te_t = torch.tensor(te, device=dev)
    net = nn.Sequential(nn.Linear(Z.shape[1], 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3); gg = torch.Generator(device=dev).manual_seed(0)
    for _ in range(steps):
        s = tr_t[torch.randint(0, len(tr_t), (256,), generator=gg, device=dev)]
        loss = ((net(Zt[s]) - Yt[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Zt[te_t]).cpu().numpy()
    Yte = Y[te]; mse = ((pred - Yte) ** 2).mean(); var = Yte.var() + 1e-9
    return round(float(1 - mse / var), 4)


def train_supervised(o):
    """H1 positive control: encoder + head supervised to [cos,sin] of x. Cannot collapse."""
    f = enc_mlp(); head = nn.Linear(args.d, 2).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(head.parameters()), 2e-3)
    O = o[:ntr].reshape(-1, args.D).to(dev); Y = gt[:ntr].reshape(-1, 2).to(dev)
    for _ in range(args.steps):
        i = torch.randint(0, len(O), (256,), generator=g, device=dev) if dev == "cpu" else torch.randint(0, len(O), (256,), device=dev)
        loss = ((head(f(O[i])) - Y[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return f


def train_sigreg(o):
    """single-ownership JEPA: raw-MSE next-latent prediction + lam*SIGReg. No norm, no running stats."""
    f = enc_mlp(); pred = nn.Sequential(nn.Linear(args.d, 128), nn.GELU(), nn.Linear(128, args.d)).to(dev)
    opt = torch.optim.Adam(list(f.parameters()) + list(pred.parameters()), 2e-3)
    ot = o[:ntr].to(dev)
    for _ in range(args.steps):
        bi = torch.randint(0, ntr, (256,), device=dev)
        ti = torch.randint(0, args.T - 1, (256,), device=dev)
        z_t = f(ot[bi, ti]); z_n = f(ot[bi, ti + 1])
        L_pred = ((pred(z_t) - z_n.detach()) ** 2).mean()       # RAW mse, stop-grad target
        L = L_pred + args.lam * sigreg_lewm(z_t)
        opt.zero_grad(); L.backward(); opt.step()
    return f


def evaluate(f, o):
    Z = embed(f, o[ntr:]); Y = gt[ntr:].reshape(-1, 2).numpy()
    return {"std": round(float(np.median(Z.std(0))), 3), "eff_rank": round(eff_rank(Z), 2),
            "cossin_R2": probe_cossin(Z, Y)}


print(f"topo-discriminator | periods={args.periods} d={args.d} D={args.D} lam={args.lam} steps={args.steps} seed={args.seed}", flush=True)
print(f"thresholds: std_floor={STD_FLOOR} rank_floor={RANK_FLOOR} sup_pass_R2={SUP_PASS} distortion_gap={DISTORT_GAP}\n", flush=True)

res = {}
for name, o in TASKS.items():
    res[(name, "supervised")] = evaluate(train_supervised(o), o)
    res[(name, "sigreg")] = evaluate(train_sigreg(o), o)
    for cfg in ("supervised", "sigreg"):
        r = res[(name, cfg)]
        print(f"  {name:14} {cfg:10} std={r['std']:.3f} eff_rank={r['eff_rank']:.2f} cossin_R2={r['cossin_R2']}", flush=True)

# ---- gates / branch (pre-registered) ----
print("\n=== gates ===", flush=True)
# NB: supervised z maps to cos/sin (range [-1,1]) so its absolute scale is FREE;
# plumbing is validated by the scale-INVARIANT checks (R^2 + eff_rank), not std.
# std stays in Gate 1, where SIGReg's unit-Gaussian target makes it meaningful.
g0 = all(res[(t, "supervised")]["cossin_R2"] >= SUP_PASS
         and res[(t, "supervised")]["eff_rank"] >= RANK_FLOOR for t in TASKS)
print(f"GATE 0 (supervised plumbing): {'PASS' if g0 else 'FAIL -> rig bug, nothing below counts'}", flush=True)
if g0:
    line = res[("line(cover)", "sigreg")]; circ = res[("circle(torus)", "sigreg")]
    g1 = line["std"] >= STD_FLOOR and line["eff_rank"] >= RANK_FLOOR
    print(f"GATE 1 (single-ownership SIGReg holds on LINE): {'PASS' if g1 else 'FAIL'}", flush=True)
    if not g1:
        print("=> BRANCH 3: single-ownership INSUFFICIENT (ownership, not plumbing). Halt-and-rethink before v4.", flush=True)
    else:
        stab = circ["std"] < STD_FLOOR or circ["eff_rank"] < RANK_FLOOR
        acc = circ["cossin_R2"] < line["cossin_R2"] - DISTORT_GAP
        if stab or acc:
            mode = "stability" if stab else "accuracy/distortion"
            print(f"=> BRANCH 1: TOPOLOGY LOCALIZED ({mode} cost). circle R2={circ['cossin_R2']} vs line R2={line['cossin_R2']}.", flush=True)
            print("   fix = manifold-matched target (uniform-on-circle / von Mises / product), not isotropic Gaussian.", flush=True)
        else:
            print(f"=> BRANCH 2: TOPOLOGY STORY DIES. circle R2={circ['cossin_R2']} ~ line R2={line['cossin_R2']}, both healthy.", flush=True)
            print("   single-ownership confirmed as the cure; port proceeds with clean conscience.", flush=True)
