"""evaluate fitted pieces numerically (Q = x, ζ = e^{2πi/7}) vs direct values."""
import cmath, math, re
exec(open('consist7.py').read().split("phi7 =")[0])   # symbols, parse, Rg, G, E2
x = 0.05
Z = cmath.exp(2j * math.pi / 7)
N = 400
def prodf(rs, m, Q):
    v = 1.0
    for n in range(1, N):
        if n % m in rs: v *= (1 - Q ** n)
    return v
def Jv(Q): return {'J7': prodf({0}, 7, Q), 'J71': prodf({0, 1, 6}, 7, Q), 'J72': prodf({0, 2, 5}, 7, Q), 'J73': prodf({0, 3, 4}, 7, Q)}
def Phiv(k, Q):
    s = 0.0; n = 0
    while n < 30:
        t = Q ** (7 * n * n)
        for i in range(n + 1): t /= (1 - Q ** (k + 7 * i))
        for i in range(n): t /= (1 - Q ** (7 - k + 7 * i))
        s += t; n += 1
    return s
# q-variable for fits is Q=x
jv = Jv(x)
subs = {J7: jv['J7'], J1: jv['J71'], J2: jv['J72'], J3: jv['J73'], q: x, z: Z,
        P1: Phiv(1, x), P2: Phiv(2, x), P3: Phiv(3, x)}
def ev(e): return complex(sp.N(e.subs(subs), 20))
# direct: series in q with q = x^{1/7}; class c piece = Σ_m coeff(7m+c) x^m
def orbit(a): return min(a % 7, (-a) % 7)
Gt = {}
for a in range(0, 200):
    T = a * (a + 1) // 2
    if T > 7 * 120: break
    r = 0
    while T + 3 * r * r <= 7 * 120:
        for e, s in ((T + 3*r*r + 3*a*r + r, (-1) ** a),) + (((T + 3*r*r + 3*a*r - r - a, -(-1) ** a),) if r >= 1 else ()):
            Gt[(e % 7, orbit(a))] = Gt.get((e % 7, orbit(a)), 0) + s * x ** (e // 7)
        r += 1
for key, expr in sorted(G.items()):
    print("G", key, "fit", round(ev(expr).real, 10), "true", round(Gt.get(key, 0), 10))
import numpy as np
K = 7 * 90
ser = np.zeros(K + 1, dtype=complex)
cur = np.zeros(K + 1, dtype=complex); cur[0] = 1
n = 0
while n * n <= K:
    ser[n * n:] += cur[:K + 1 - n * n]
    n += 1
    for c in (Z, 1 / Z):
        for t in range(n, K + 1): cur[t] += c * cur[t - n]
E = np.zeros(K + 1); E[0] = 1
for nn in range(1, K + 1):
    E[nn:] = E[nn:] - E[:K + 1 - nn].copy() if False else E[nn:] - np.concatenate([E[:K + 1 - nn]])
E2s = np.convolve(E, E)[:K + 1]
def cls(s, c): return sum(s[7 * m + c] * x ** m for m in range((K - c) // 7 + 1))
for k in range(7):
    print("R", k, "fit", np.round(ev(Rg[k]), 8), "true", np.round(cls(ser, k), 8))
for c in range(7):
    print("E2", c, "fit", round(ev(E2[c]).real, 8), "true", round(cls(E2s, c).real, 8))
