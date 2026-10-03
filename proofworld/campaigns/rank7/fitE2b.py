"""Fit the 7-dissection components R_k of R(ζ7;q) (Durfee) to span{theta, wt 1/2} ⊕ span{q^s Φ_k}, per ζ-coordinate."""
import sys, itertools
import numpy as np
P = 32749
KQ = int(sys.argv[1]); K = 7 * KQ + 7
# R(ζ;q) coefficients N(m, n) via rank counting: R(z;q) = Σ_n q^{n²}/((zq)_n (q/z)_n); compute as polynomials in z mod z^7-1
# represent element of Z[ζ] as length-7 vector mod (z^7 - 1), reduce to basis 1..ζ^5 at the end (ζ^6 = -1-...-ζ^5)
# E^2 and F(ζ) = (ζq)(ζ⁻¹q)(q), over Z[z]/(z^7-1), mod P
def mulz(f, c, e):   # f * (1 - z^c q^e)
    g = f.copy()
    for t in range(K, e - 1, -1):
        g[t] = (g[t] - np.roll(f[t - e], c)) % P
    return g
E = np.zeros((K + 1, 7), dtype=np.int64); E[0, 0] = 1
Fz = np.zeros((K + 1, 7), dtype=np.int64); Fz[0, 0] = 1
for n in range(1, K + 1):
    E = mulz(E, 0, n)
    Fz = mulz(mulz(mulz(Fz, 0, n), 1, n), 6, n)
E2 = np.zeros((K + 1, 7), dtype=np.int64)
e0 = E[:, 0]
E2[:, 0] = np.convolve(e0, e0)[:K + 1] % P
R = None
# reduce mod Φ7: v = Σ v_i ζ^i, ζ^6 = -(1+...+ζ^5)
def red(v): return [int(v[i] - v[6]) % P for i in range(6)]
comps = {('E2', k): [[red(E2[7 * m + k])[i] for m in range(KQ + 1)] for i in range(6)] for k in range(7)}
comps.update({('F', k): [[red(Fz[7 * m + k])[i] for m in range(KQ + 1)] for i in range(6)] for k in range(7)})
# basis
def mul(a, b): return np.convolve(a, b)[:KQ + 1] % P
def prod(rs, m):
    f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1
    for nn in range(1, KQ + 1):
        if nn % m in rs:
            g = f.copy(); g[nn:] = (g[nn:] - f[:KQ + 1 - nn]) % P; f = g
    return f
def inv(f):
    g = np.zeros(KQ + 1, dtype=np.int64); g[0] = pow(int(f[0]), P - 2, P)
    for nn in range(1, KQ + 1):
        g[nn] = (-int(np.dot(f[1:nn + 1] % P, g[nn - 1::-1] % P) % P) * int(g[0])) % P
    return g
def pw(f, e):
    r = np.zeros(KQ + 1, dtype=np.int64); r[0] = 1
    b = f if e >= 0 else inv(f)
    for _ in range(abs(e)): r = mul(r, b)
    return r
J7 = prod({0}, 7); J = {k: mul(prod({k, 7 - k}, 7), J7) for k in (1, 2, 3)}
def Phi(k):
    s = np.zeros(KQ + 1, dtype=np.int64); m = 0
    while 7 * m * m <= KQ:
        den = np.zeros(KQ + 1, dtype=np.int64); den[0] = 1
        for i in range(m + 1):
            if k + 7 * i <= KQ:
                f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1; f[k + 7 * i] = P - 1; den = mul(den, f)
        for i in range(m):
            if 7 - k + 7 * i <= KQ:
                f = np.zeros(KQ + 1, dtype=np.int64); f[0] = 1; f[7 - k + 7 * i] = P - 1; den = mul(den, f)
        t = np.zeros(KQ + 1, dtype=np.int64); t[7 * m * m] = 1
        s = (s + mul(t, inv(den))) % P
        m += 1
    return s
def shift(f, sh):
    g = np.zeros(KQ + 1, dtype=np.int64)
    if sh >= 0: g[sh:] = f[:KQ + 1 - sh]
    else: g[:KQ + 1 + sh] = f[-sh:]
    return g
basis = {}
basis["1"] = shift(np.eye(1, KQ + 1, 0, dtype=np.int64)[0], 0)
cache = {}
def PW(nm, f, e):
    if (nm, e) not in cache: cache[(nm, e)] = pw(f, e)
    return cache[(nm, e)]
for es in itertools.product(range(-2, 4), repeat=4):
    if sum(es) != 2: continue
    f = mul(mul(PW('J7', J7, es[0]), PW('J1', J[1], es[1])), mul(PW('J2', J[2], es[2]), PW('J3', J[3], es[3])))
    for sh in (-1, 0, 1):
        basis[f"q^{sh}*J7^{es[0]}J71^{es[1]}J72^{es[2]}J73^{es[3]}"] = shift(f, sh)
names = list(basis); nb = len(names)
for k, (cf, Jn) in {0: ([1,0,0,0,0,0], 3), 1: ([0,0,1,1,1,1], 2), 3: ([0,0,0,-1,-1,0], 1)}.items():
    Jser = J[Jn]
    ok = all(all((comps[('F', k)][i][m] - cf[i] * int(Jser[m])) % P == 0 for m in range(KQ)) for i in range(6))
    print("F class", k, "= coeff·J7%d:" % Jn, ok, flush=True)
for k in (2, 4, 5, 6):
    print("F class", k, "zero:", all(all(comps[('F', k)][i][m] % P == 0 for m in range(KQ)) for i in range(6)), flush=True)
Mfit = KQ - 25
print("basis", nb, "rows", Mfit, flush=True)
A = np.stack([basis[nm][:Mfit] for nm in names], axis=1) % P
Bf = np.stack([basis[nm] for nm in names], axis=1) % P
def rref_solve(A, b):
    m = np.concatenate([A, b[:, None]], axis=1) % P
    rows, cols = m.shape; piv = []; r = 0
    for c in range(cols - 1):
        nz = np.nonzero(m[r:, c])[0]
        if len(nz) == 0: continue
        i = r + nz[0]; m[[r, i]] = m[[i, r]]
        m[r] = (m[r] * pow(int(m[r, c]), P - 2, P)) % P
        f = m[:, c].copy(); f[r] = 0
        m = (m - np.outer(f, m[r])) % P
        piv.append(c); r += 1
        if r == rows: break
    if np.any(m[r:, -1]): return None
    x = np.zeros(cols - 1, dtype=np.int64)
    for i, c in enumerate(piv): x[c] = m[i, -1]
    return x
def lift(v): v = int(v); return v if v < P // 2 else v - P
for k in [c for c in comps if c[0] == 'E2']:
    for i in range(1):
        tgt = np.array([c % P for c in comps[k][i]], dtype=np.int64)
        if not tgt.any(): print(f"{k} ζ^{i}: ZERO", flush=True); continue
        x = rref_solve(A, tgt[:Mfit])
        if x is None: print(f"{k} ζ^{i}: NO FIT", flush=True); continue
        ok = np.all((Bf[Mfit:KQ - 3] @ x - tgt[Mfit:KQ - 3]) % P == 0)
        used = {names[j]: lift(x[j]) for j in range(nb) if x[j]}
        print(f"{k} ζ^{i}:", "FIT" if ok else "overfit", used if len(used) <= 8 else f"{len(used)} terms", flush=True)
