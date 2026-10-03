exec(open('spec_probe.py').read().split('print("M1"')[0])
from fractions import Fraction as Fr
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
J5 = prod({0}, 5); J51 = mul(prod({1, 4}, 5), J5); J52 = mul(prod({2, 3}, 5), J5)
def durf(a1, a2):  # Σ Q^{n²}/((q^a1;Q)_{n+1}(q^a2;Q)_n) - 1
    s = [Fr(0)] * (KQ + 1); s[0] = Fr(-1)
    n = 0
    while 5 * n * n <= KQ:
        t = [0] * (KQ + 1); t[5 * n * n] = 1
        for i in range(n + 1):
            f = [0] * (KQ + 1); f[0] = 1
            if a1 + 5 * i <= KQ: f[a1 + 5 * i] = -1
            t = mul(t, inv(f))
        for i in range(n):
            f = [0] * (KQ + 1); f[0] = 1
            if a2 + 5 * i <= KQ: f[a2 + 5 * i] = -1
            t = mul(t, inv(f))
        s = [x + y for x, y in zip(s, t)]
        n += 1
    return s
phi = durf(1, 4); psi = durf(2, 3)
print("psi", psi[:8])
psiq = psi[1:] + [0]
basis = {"J52psi/q": mul(J52, psiq), "J52phi": mul(J52, phi), "J5^2J51/J52": mul(mul(J5, J5), mul(J51, inv(J52))),
         "J5^2J52/J51": mul(mul(J5, J5), mul(J52, inv(J51))), "J5^2": mul(J5, J5), "J51": J51, "J52": J52,
         "J52^2/J51": mul(J52, mul(J52, inv(J51))), "J51^2/J52": mul(J51, mul(J51, inv(J52)))}
names = list(basis)
import sympy
for key in [(3, 1), (3, 2), (0, 1), (0, 0), (2, 0), (1, 1), (1, 2), (4, 0), (4, 2)]:
    tgt = G[key]
    N = 40
    A = sympy.Matrix([[basis[nm][i] for nm in names] for i in range(N)])
    b = sympy.Matrix([tgt[i] for i in range(N)])
    try:
        sol, params = A.gauss_jordan_solve(b)
        sol = sol.subs({p: 0 for p in params})
        res = [sum(sol[j] * basis[names[j]][i] for j in range(len(names))) - tgt[i] for i in range(KQ - 5)]
        ok = all(r == 0 for r in res)
        print(key, "fit" if ok else "nofit", {names[j]: sol[j] for j in range(len(names)) if sol[j] != 0})
    except ValueError:
        print(key, "inconsistent")
