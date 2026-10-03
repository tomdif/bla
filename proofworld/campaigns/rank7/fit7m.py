"""p=7 orbit pieces vs basis, solved over GF(P) (fast, exact-modular); fit on M coefficients, verify on the rest."""
import sys, itertools
from collections import defaultdict
P = 2147483647
p = 7; KQ = int(sys.argv[1]) if len(sys.argv) > 1 else 110; K = p * KQ
def orbit(a): return min(a % p, (-a) % p)
G = defaultdict(lambda: [0] * (KQ + 1))
for a in range(0, 1000):
    T = a * (a + 1) // 2
    if T > K: break
    r = 0
    while T + 3 * r * r <= K + 50:
        for e, s in ((T + 3*r*r + 3*a*r + r, (-1) ** a),) + (((T + 3*r*r + 3*a*r - r - a, -(-1) ** a),) if r >= 1 else ()):
            if 0 <= e <= K: G[(e % p, orbit(a))][e // p] += s
        r += 1
def mul(a, b):
    c = [0] * (KQ + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(KQ + 1 - i): c[i + j] = (c[i + j] + x * b[j]) % P
    return c
def prod(rs, m):
    f = [0] * (KQ + 1); f[0] = 1
    for n in range(1, KQ + 1):
        if n % m in rs:
            for k in range(KQ, n - 1, -1): f[k] = (f[k] - f[k - n]) % P
    return f
def inv(f):
    g = [0] * (KQ + 1); g[0] = pow(f[0], P - 2, P)
    for n in range(1, KQ + 1): g[n] = (-sum(f[i] * g[n - i] for i in range(1, n + 1)) * g[0]) % P
    return g
def pw(f, e):
    r = [0] * (KQ + 1); r[0] = 1
    b = f if e >= 0 else inv(f)
    for _ in range(abs(e)): r = mul(r, b)
    return r
J7 = prod({0}, 7); J = {k: mul(prod({k, 7 - k}, 7), J7) for k in (1, 2, 3)}
def Phi(k):
    s = [0] * (KQ + 1); n = 0
    while 7 * n * n <= KQ:
        t = [0] * (KQ + 1); t[7 * n * n] = 1
        den = [0] * (KQ + 1); den[0] = 1
        for i in range(n + 1):
            f = [0] * (KQ + 1); f[0] = 1
            if k + 7 * i <= KQ: f[k + 7 * i] = P - 1
            den = mul(den, f)
        for i in range(n):
            f = [0] * (KQ + 1); f[0] = 1
            if 7 - k + 7 * i <= KQ: f[7 - k + 7 * i] = P - 1
            den = mul(den, f)
        t = mul(t, inv(den))
        s = [(x + y) % P for x, y in zip(s, t)]
        n += 1
    return s
Ph = {k: Phi(k) for k in (1, 2, 3)}
def shift(f, sh):
    return [0] * sh + f[:KQ + 1 - sh] if sh >= 0 else f[-sh:] + [0] * (-sh)
basis = {}
for j in (1, 2, 3):
    for k in (1, 2, 3):
        for sh in (-2, -1, 0, 1):
            basis[f"q^{sh}*J7{j}*Phi{k}"] = shift(mul(J[j], Ph[k]), sh)
pw_cache = {}
def PW(f, name, e):
    if (name, e) not in pw_cache: pw_cache[(name, e)] = pw(f, e)
    return pw_cache[(name, e)]
for es in itertools.product(range(-2, 4), repeat=4):
    if sum(es) != 2: continue
    f = mul(mul(PW(J7, 'J7', es[0]), PW(J[1], 'J1', es[1])), mul(PW(J[2], 'J2', es[2]), PW(J[3], 'J3', es[3])))
    for sh in (-1, 0, 1):
        basis[f"q^{sh}*J7^{es[0]}J71^{es[1]}J72^{es[2]}J73^{es[3]}"] = shift(f, sh)
names = list(basis); nb = len(names)
print("basis size", nb, flush=True)
def solve(rows, rhs):
    """Gaussian elimination mod P; return a solution vector or None."""
    m = [r[:] + [b] for r, b in zip(rows, rhs)]
    n = len(m[0]) - 1; piv = []; row = 0
    for col in range(n):
        sel = next((i for i in range(row, len(m)) if m[i][col]), None)
        if sel is None: continue
        m[row], m[sel] = m[sel], m[row]
        iv = pow(m[row][col], P - 2, P)
        m[row] = [x * iv % P for x in m[row]]
        for i in range(len(m)):
            if i != row and m[i][col]:
                c = m[i][col]; m[i] = [(x - c * y) % P for x, y in zip(m[i], m[row])]
        piv.append(col); row += 1
        if row == len(m): break
    for i in range(row, len(m)):
        if m[i][n]: return None
    x = [0] * n
    for i, col in enumerate(piv): x[col] = m[i][n]
    return x
Mfit = KQ - 25
rows = [[basis[nm][i] for nm in names] for i in range(Mfit)]
def lift(v): return v if v < P // 2 else v - P
for key in sorted(G):
    tgt = [x % P for x in G[key]]
    if not any(tgt): continue
    x = solve(rows, tgt[:Mfit])
    if x is None: print(key, "NO FIT", flush=True); continue
    ok = all((sum(x[j] * basis[names[j]][i] for j in range(nb)) - tgt[i]) % P == 0 for i in range(Mfit, KQ - 3))
    used = {names[j]: lift(x[j]) for j in range(nb) if x[j]}
    print(key, "FIT" if ok else "overfit(killed)", used if ok and len(used) < 12 else len(used), flush=True)
