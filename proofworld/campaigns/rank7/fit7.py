"""p=7: orbit pieces G_{c,o} of Σ_a w_a Hblk_a; fit to span{theta products of weight 1} ∪ {q^s J_{7,j} Φ_k}."""
from collections import defaultdict
from fractions import Fraction as Fr
import itertools, sympy
p = 7; KQ = 70; K = p * KQ
def orbit(a): return min(a % p, (-a) % p)
G = defaultdict(lambda: [0] * (KQ + 1))
for a in range(0, 400):
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
            for j in range(KQ + 1 - i): c[i + j] += x * b[j]
    return c
def prod(rs, m):
    f = [0] * (KQ + 1); f[0] = 1
    for n in range(1, KQ + 1):
        if n % m in rs:
            g = f[:]
            for k in range(n, KQ + 1): g[k] -= f[k - n]
            f = g
    return f
def inv(f):
    g = [Fr(0)] * (KQ + 1); g[0] = Fr(1, f[0])
    for n in range(1, KQ + 1): g[n] = -sum(f[i] * g[n - i] for i in range(1, n + 1)) / f[0]
    return g
def pw(f, e):
    r = [0] * (KQ + 1); r[0] = 1
    b = f if e >= 0 else inv(f)
    for _ in range(abs(e)): r = mul(r, b)
    return r
J7 = prod({0}, 7); J = {k: mul(prod({k, 7 - k}, 7), J7) for k in (1, 2, 3)}
def Phi(k):
    s = [Fr(0)] * (KQ + 1)
    n = 0
    while 7 * n * n <= KQ:
        t = [0] * (KQ + 1); t[7 * n * n] = 1
        for i in range(n + 1):
            f = [0] * (KQ + 1); f[0] = 1
            if k + 7 * i <= KQ: f[k + 7 * i] = -1
            t = mul(t, inv(f))
        for i in range(n):
            f = [0] * (KQ + 1); f[0] = 1
            if 7 - k + 7 * i <= KQ: f[7 - k + 7 * i] = -1
            t = mul(t, inv(f))
        s = [x + y for x, y in zip(s, t)]
        n += 1
    return s   # this is 1 + mock
P = {k: Phi(k) for k in (1, 2, 3)}
def shift(f, sh):
    if sh >= 0: return [0] * sh + f[:KQ + 1 - sh]
    return f[-sh:] + [0] * (-sh)
basis = {}
for j in (1, 2, 3):
    for k in (1, 2, 3):
        for sh in (-2, -1, 0, 1):
            basis[f"q^{sh} J7{j} Phi{k}"] = shift(mul(J[j], P[k]), sh)
for es in itertools.product(range(-2, 4), repeat=4):
    if sum(es) != 2: continue
    nm = f"J7^{es[0]} J71^{es[1]} J72^{es[2]} J73^{es[3]}"
    f = mul(mul(pw(J7, es[0]), pw(J[1], es[1])), mul(pw(J[2], es[2]), pw(J[3], es[3])))
    for sh in (-1, 0, 1):
        basis[nm + f" q^{sh}"] = shift(f, sh)
names = list(basis); M = 60
A = sympy.Matrix([[basis[nm][i] for nm in names] for i in range(M)])
for key in sorted(G):
    b = sympy.Matrix(G[key][:M])
    try:
        sol, params = A.gauss_jordan_solve(b)
        sol = sol.subs({pp: 0 for pp in params})
        res = [sum(sol[j] * basis[names[j]][i] for j in range(len(names))) - G[key][i] for i in range(KQ - 3)]
        ok = all(r == 0 for r in res)
        print(key, "FIT" if ok else "overfit", {names[j]: sol[j] for j in range(len(names)) if sol[j] != 0} if ok else "")
    except ValueError:
        print(key, "no fit")
