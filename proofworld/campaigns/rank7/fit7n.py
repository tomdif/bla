"""fit7 with enough rows: exact mod a 15-bit prime, numpy elimination; verify on held-out coefficients."""
import sys, itertools
import numpy as np
from collections import defaultdict
P = 32749
p = 7; KQ = int(sys.argv[1]); K = p * KQ
def orbit(a): return min(a % p, (-a) % p)
G = defaultdict(lambda: np.zeros(KQ + 1, dtype=np.int64))
for a in range(0, 5000):
    T = a * (a + 1) // 2
    if T > K: break
    r = 0
    while T + 3 * r * r <= K + 50:
        for e, s in ((T + 3*r*r + 3*a*r + r, (-1) ** a),) + (((T + 3*r*r + 3*a*r - r - a, -(-1) ** a),) if r >= 1 else ()):
            if 0 <= e <= K: G[(e % p, orbit(a))][e // p] += s
        r += 1
def mul(a, b): return np.convolve(a, b)[:KQ + 1] % P
def prod(rs, m):
    f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1
    for n in range(1, KQ + 1):
        if n % m in rs:
            g = f.copy(); g[n:] = (g[n:] - f[:KQ + 1 - n]) % P; f = g
    return f
def inv(f):
    g = np.zeros(KQ + 1, dtype=np.int64); g[0] = pow(int(f[0]), P - 2, P)
    for n in range(1, KQ + 1):
        g[n] = (-int(np.dot(f[1:n + 1] % P, g[n - 1::-1] % P) % P) * int(g[0])) % P
    return g
def pw(f, e):
    r = np.zeros(KQ + 1, dtype=np.int64); r[0] = 1
    b = f if e >= 0 else inv(f)
    for _ in range(abs(e)): r = mul(r, b)
    return r
J7 = prod({0}, 7); J = {k: mul(prod({k, 7 - k}, 7), J7) for k in (1, 2, 3)}
def Phi(k):
    s = np.zeros(KQ + 1, dtype=np.int64); n = 0
    while 7 * n * n <= KQ:
        den = np.zeros(KQ + 1, dtype=np.int64); den[0] = 1
        for i in range(n + 1):
            if k + 7 * i <= KQ:
                f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1; f[k + 7 * i] = P - 1; den = mul(den, f)
        for i in range(n):
            if 7 - k + 7 * i <= KQ:
                f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1; f[7 - k + 7 * i] = P - 1; den = mul(den, f)
        t = np.zeros(KQ + 1, dtype=np.int64); t[7 * n * n] = 1
        s = (s + mul(t, inv(den))) % P
        n += 1
    return s
Ph = {k: Phi(k) for k in (1, 2, 3)}
def shift(f, sh):
    g = np.zeros(KQ + 1, dtype=np.int64)
    if sh >= 0: g[sh:] = f[:KQ + 1 - sh]
    else: g[:KQ + 1 + sh] = f[-sh:]
    return g
basis = {}
for j in (1, 2, 3):
    for k in (1, 2, 3):
        for sh in (-2, -1, 0, 1):
            basis[f"q^{sh}*J7{j}*Phi{k}"] = shift(mul(J[j], Ph[k]), sh)
cache = {}
def PW(name, f, e):
    if (name, e) not in cache: cache[(name, e)] = pw(f, e)
    return cache[(name, e)]
for es in itertools.product(range(-2, 4), repeat=4):
    if sum(es) != 2: continue
    f = mul(mul(PW('J7', J7, es[0]), PW('J1', J[1], es[1])), mul(PW('J2', J[2], es[2]), PW('J3', J[3], es[3])))
    for sh in (-1, 0, 1):
        basis[f"q^{sh}*J7^{es[0]}J71^{es[1]}J72^{es[2]}J73^{es[3]}"] = shift(f, sh)
names = list(basis); nb = len(names)
Mfit = KQ - 30
A = np.stack([basis[nm][:Mfit] for nm in names], axis=1) % P   # Mfit x nb
print("basis", nb, "rows", Mfit, flush=True)
def rref_solve(A, b):
    m = np.concatenate([A, b[:, None]], axis=1) % P
    rows, cols = m.shape; piv = []; r = 0
    for c in range(cols - 1):
        nz = np.nonzero(m[r:, c])[0]
        if len(nz) == 0: continue
        i = r + nz[0]
        m[[r, i]] = m[[i, r]]
        m[r] = (m[r] * pow(int(m[r, c]), P - 2, P)) % P
        f = m[:, c].copy(); f[r] = 0
        m = (m - np.outer(f, m[r])) % P
        piv.append(c); r += 1
        if r == rows: break
    if np.any(m[r:, -1]): return None, len(piv)
    x = np.zeros(cols - 1, dtype=np.int64)
    for i, c in enumerate(piv): x[c] = m[i, -1]
    return x, len(piv)
B = np.stack([basis[nm] for nm in names], axis=1) % P
def lift(v): v = int(v); return v if v < P // 2 else v - P
for key in sorted(G):
    tgt = G[key] % P
    if not tgt.any(): continue
    x, rk = rref_solve(A, tgt[:Mfit])
    if x is None: print(key, "NO FIT (rank %d)" % rk, flush=True); continue
    ok = np.all((B[Mfit:KQ - 3] @ x - tgt[Mfit:KQ - 3]) % P == 0)
    used = {names[j]: lift(x[j]) for j in range(nb) if x[j]}
    print(key, "FIT" if ok else "overfit", "rank", rk, used if len(used) <= 10 else f"{len(used)} terms", flush=True)
