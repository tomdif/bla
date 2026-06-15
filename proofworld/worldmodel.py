#!/usr/bin/env python3
"""proofworld.worldmodel -- Stage B: a proof-state WORLD MODEL on a z3-grounded domain, contrastive vs SIGReg.

This is the PREDICTIVE setting where SIGReg belongs (unlike retrieval). The world model predicts the next PROOF
STATE in latent space given an action; z3 grounds the dynamics (it computes which derivations are actually sound).

Domain (z3-grounded inequality propagation):
  * n reals with axioms x_i <= x_j (a random DAG of edges) and a few bases x_b >= 0.
  * z3 computes SOUND propagations: the action "carry >=0 from i to j" is valid iff axioms |= x_i <= x_j (this
    includes TRANSITIVELY-entailed edges, not just direct ones -- z3 does real work). Fake edges are not entailed.
  * STATE = the set of nodes proven >= 0; ACTION = a (src,dst) carry; valid action adds dst; GOAL = a target node.

Models (shared state-encoder E, action-encoder A, dynamics D):
  predicted next latent  z'_hat = D(E(s), A(a));   target latent  z' = E(s')   (JEPA: predict the representation)
  * CONTRASTIVE : InfoNCE(z'_hat, z') with in-batch negatives.
  * SIGReg      : ||predict z'_hat ~ z'|| (negative cosine) + lambda * SIGReg(Z)  -- sketched isotropic-Gaussian reg
                  (random 1D projections + Epps-Pulley CF normality test); NO negatives/stop-grad/EMA.

Evaluated as a WORLD MODEL (not retrieval): (1) multi-step ROLLOUT fidelity in latent, (2) latent ISOTROPY /
effective rank (collapse check), (3) downstream PROBE: linear read-out of steps-to-goal from frozen latents.

Run:  python3 -m proofworld.worldmodel
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, z3


# ----------------------------- z3-grounded world -----------------------------
def build_world(n=14, path_len=6, n_fake=18, extra_p=0.08, seed=0):
    """sparse DAG so depth (dist-to-goal) varies, with a guaranteed multi-hop base->goal path AND genuinely
    non-entailed fakes. z3 verifies which CANDIDATE direct carries are sound (entailed) vs invalid."""
    rng = np.random.RandomState(seed)
    X = [z3.Real(f"x{i}") for i in range(n)]
    mids = sorted(rng.choice(range(1, n - 1), size=path_len - 1, replace=False).tolist())
    path = [0] + mids + [n - 1]
    real_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]      # one guaranteed depth-(path_len) route
    bases = [0]
    axioms = [X[i] <= X[j] for (i, j) in real_edges] + [X[b] >= 0 for b in bases]
    def entails(i, j):
        s = z3.Solver(); s.set("timeout", 2000)
        for a in axioms: s.add(a)
        s.add(z3.Not(X[i] <= X[j])); return s.check() == z3.unsat
    allpairs = [(i, j) for i in range(n) for j in range(n) if i < j and (i, j) not in real_edges]
    rng.shuffle(allpairs)
    fake = [p for p in allpairs if not entails(*p)][:n_fake]    # GENUINELY invalid carries (z3-rejected), no shortcuts
    candidates = sorted(set(real_edges) | set(fake))
    sound = set(real_edges)                                     # only the sparse path carries are sound -> real depth
    return dict(n=n, sound=sound, actions=candidates, bases=set(bases), goal=n - 1)


def step(state, a, world):
    src, dst = a
    if a in world["sound"] and src in state and dst not in state:
        return state | {dst}
    return state                                    # invalid / no-op (the model must learn this too)


def dist_to_goal(state, world):
    """min carries to establish the goal (BFS on sound edges from the established frontier)."""
    if world["goal"] in state: return 0
    import collections
    adj = collections.defaultdict(list)
    for (i, j) in world["sound"]: adj[i].append(j)
    dq = collections.deque((u, 0) for u in state); seen = set(state)
    while dq:
        u, d = dq.popleft()
        if u == world["goal"]: return d
        for v in adj[u]:
            if v not in seen: seen.add(v); dq.append((v, d + 1))
    return world["n"]                               # unreachable sentinel


def gen_transitions(world, n_traj=1200, max_len=10, seed=1):
    rng = np.random.RandomState(seed); n = world["n"]; acts = world["actions"]
    trans = []           # (state_set, action, next_state_set, dist(state))
    for _ in range(n_traj):
        state = set(world["bases"])
        for _ in range(max_len):
            a = acts[rng.randint(len(acts))]
            s2 = step(state, a, world)
            trans.append((frozenset(state), a, frozenset(s2), dist_to_goal(state, world)))
            state = s2
            if world["goal"] in state: break
    return trans


def feat_state(state, world):
    n = world["n"]; v = np.zeros(2 * n, np.float32)
    for u in state: v[u] = 1.0
    v[n + world["goal"]] = 1.0
    return v

def feat_action(a, world):
    n = world["n"]; v = np.zeros(2 * n, np.float32); v[a[0]] = 1.0; v[n + a[1]] = 1.0
    return v


# ----------------------------- world model + objectives -----------------------------
class WM(nn.Module):
    def __init__(self, din, d=48):
        super().__init__()
        self.E = nn.Sequential(nn.Linear(din, 96), nn.ReLU(), nn.Linear(96, d))
        self.A = nn.Sequential(nn.Linear(din, 96), nn.ReLU(), nn.Linear(96, d))
        self.D = nn.Sequential(nn.Linear(2 * d, 96), nn.ReLU(), nn.Linear(96, d))
    def enc(self, s): return self.E(s)
    def predict(self, s, a): return self.D(torch.cat([self.E(s), self.A(a)], -1))


def sigreg(Z, n_proj=64):
    d = Z.shape[1]; U = torch.randn(d, n_proj); U = U / (U.norm(dim=0, keepdim=True) + 1e-8)
    p = Z @ U; p = (p - p.mean(0)) / (p.std(0) + 1e-5)
    ts = torch.linspace(-5, 5, 33)
    tx = p.unsqueeze(-1) * ts
    re = torch.cos(tx).mean(0); im = torch.sin(tx).mean(0)
    g = torch.exp(-ts ** 2 / 2); w = torch.exp(-ts ** 2)
    return (((re - g) ** 2 + im ** 2) * w).mean()


def train_wm(objective, S, A, S2, epochs=300, d=48, tau=0.1, lam=3.0, seed=0):
    torch.manual_seed(seed)
    m = WM(S.shape[1], d); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    S, A, S2 = map(lambda z: torch.tensor(z), (S, A, S2)); N = len(S)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i + 256]
            zp = m.predict(S[idx], A[idx]); zt = m.enc(S2[idx])
            if objective == "contrastive":
                zpn = nn.functional.normalize(zp, -1); ztn = nn.functional.normalize(zt, -1)
                loss = nn.functional.cross_entropy(zpn @ ztn.t() / tau, torch.arange(len(idx)))
            else:  # SIGReg
                zpn = nn.functional.normalize(zp, -1); ztn = nn.functional.normalize(zt, -1)
                pred = (1 - (zpn * ztn).sum(-1)).mean()
                loss = pred + lam * sigreg(torch.cat([zp, zt], 0))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


# ----------------------------- world-model evaluations -----------------------------
def effective_rank(Z):
    Z = Z - Z.mean(0); C = (Z.T @ Z) / max(1, len(Z) - 1)
    ev = np.linalg.eigvalsh(C); ev = np.clip(ev, 1e-12, None); p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))     # entropy-based effective rank (isotropy proxy)

def rollout_fidelity(m, world, ks=(1, 2, 3, 4), n_roll=200, seed=7):
    rng = np.random.RandomState(seed); acts = world["actions"]; out = {k: [] for k in ks}
    with torch.no_grad():
        for _ in range(n_roll):
            state = set(world["bases"]); seq = []
            for _ in range(max(ks)):
                a = acts[rng.randint(len(acts))]; s2 = step(state, a, world); seq.append((state, a, s2)); state = s2
            z = m.enc(torch.tensor(feat_state(seq[0][0], world)[None]))   # start latent
            for t, (s, a, s2) in enumerate(seq, 1):
                z = m.D(torch.cat([z, m.A(torch.tensor(feat_action(a, world)[None]))], -1))   # roll dynamics in latent
                ztrue = m.enc(torch.tensor(feat_state(s2, world)[None]))
                cos = torch.nn.functional.cosine_similarity(z, ztrue).item()
                if t in out: out[t].append(cos)
    return {k: float(np.mean(v)) for k, v in out.items()}

def probe_r2(m, S, dist, seed=0):
    with torch.no_grad():
        Z = m.enc(torch.tensor(S)).numpy()
    rng = np.random.RandomState(seed); idx = rng.permutation(len(Z)); cut = int(0.8 * len(Z))
    tr, te = idx[:cut], idx[cut:]
    Xtr = np.hstack([Z[tr], np.ones((len(tr), 1))]); Xte = np.hstack([Z[te], np.ones((len(te), 1))])
    w, *_ = np.linalg.lstsq(Xtr, dist[tr], rcond=None)
    pred = Xte @ w; ss_res = ((dist[te] - pred) ** 2).sum(); ss_tot = ((dist[te] - dist[te].mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-9))


def main():
    print("=== proofworld.worldmodel :: Stage B proof-state world model (z3-grounded) -- contrastive vs SIGReg ===\n")
    world = build_world(seed=0)
    print(f"  world: n={world['n']} nodes, {len(world['sound'])} z3-entailed sound carries, "
          f"{len(world['actions'])} actions, bases={sorted(world['bases'])}, goal={world['goal']}")
    trans = gen_transitions(world)
    S = np.stack([feat_state(s, world) for s, a, s2, dd in trans])
    A = np.stack([feat_action(a, world) for s, a, s2, dd in trans])
    S2 = np.stack([feat_state(s2, world) for s, a, s2, dd in trans])
    dist = np.array([dd for s, a, s2, dd in trans], np.float32)
    print(f"  transitions: {len(trans)} (state,action,next_state); dist-to-goal range [{dist.min():.0f},{dist.max():.0f}]\n")
    rows = {}
    for obj in ("contrastive", "sigreg"):
        accs = {"erank": [], "probe": [], "roll1": [], "roll2": [], "roll3": [], "roll4": []}
        for seed in (0, 1, 2):
            m = train_wm(obj, S, A, S2, seed=seed)
            with torch.no_grad():
                Z = m.enc(torch.tensor(S)).numpy()
            accs["erank"].append(effective_rank(Z))
            accs["probe"].append(probe_r2(m, S, dist, seed))
            rf = rollout_fidelity(m, world, seed=seed)
            for k in (1, 2, 3, 4): accs[f"roll{k}"].append(rf[k])
        rows[obj] = {k: float(np.mean(v)) for k, v in accs.items()}
    print(f"  {'metric':28}{'contrastive':>13}{'SIGReg':>10}   (3 seeds)")
    labels = [("erank", "latent effective rank (isotropy)"), ("probe", "downstream probe R^2 (steps->goal)"),
              ("roll1", "rollout cos @1 step"), ("roll2", "rollout cos @2"),
              ("roll3", "rollout cos @3"), ("roll4", "rollout cos @4 (stability)")]
    for key, lab in labels:
        c, s = rows["contrastive"][key], rows["sigreg"][key]
        print(f"  {lab:28}{c:>13.3f}{s:>10.3f}")
    print(f"\n  READ: this is the PREDICTIVE world-model task (predict next proof-state latent), SIGReg's actual home.")
    print(f"  Higher effective rank = more isotropic (less collapsed); higher rollout cos @k = more STABLE multi-step")
    print(f"  imagination; higher probe R^2 = better-conditioned latent for a downstream value head. z3 grounds the")
    print(f"  dynamics; the world model only IMAGINES -- in Stage C the kernel would verify every executed rollout.")


if __name__ == "__main__":
    main()
