"""Dyson rank route check. R(z;q) = Σ q^{n²}/((zq;q)_n (q/z;q)_n) = (1−z)/E · Σ_n (−1)^n q^{n(3n+1)/2}/(1−zq^n)."""
import cmath
from collections import defaultdict
K = 80
def rank(l): return max(l) - len(l)
def parts(n, mx=None):
    if mx is None: mx = n
    if n == 0: yield []; return
    for k in range(min(n, mx), 0, -1):
        for p in parts(n - k, k): yield [k] + p
# combinatorial check of equidistribution
for p, r in [(5, 4), (7, 5)]:
    for n in range(r, 40, p):
        cnt = defaultdict(int)
        for l in parts(n): cnt[rank(l) % p] += 1
        assert len(set(cnt[i] for i in range(p))) == 1, (p, n, dict(cnt))
print("rank equidistribution mod 5 (5n+4) and mod 7 (7n+5) holds to n<40")
# Appell-Lerch form at a generic z, numerically as power series with complex coeffs
z = cmath.exp(0.7j) * 1.3
def ser_mul(a, b):
    c = [0j] * (K + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(K + 1 - i): c[i + j] += x * b[j]
    return c
def ser_inv(a):
    b = [0j] * (K + 1); b[0] = 1 / a[0]
    for n in range(1, K + 1): b[n] = -sum(a[k] * b[n - k] for k in range(1, n + 1)) / a[0]
    return b
one = [1] + [0] * K
R = [0j] * (K + 1)
for n in range(0, 10):
    if n * n > K: break
    den = one[:]
    for i in range(1, n + 1):
        f1 = [0j] * (K + 1); f1[0] = 1; f1[i] = -z if i <= K else 0
        f2 = [0j] * (K + 1); f2[0] = 1; f2[i] = -1 / z if i <= K else 0
        den = ser_mul(ser_mul(den, f1), f2)
    t = ser_inv(den); t = [0j] * (n * n) + t[:K + 1 - n * n]
    R = [a + b for a, b in zip(R, t)]
E = one[:]
for i in range(1, K + 1):
    f = [0j] * (K + 1); f[0] = 1; f[i] = -1; E = ser_mul(E, f)
S = [0j] * (K + 1)
for n in range(-12, 13):
    e = n * (3 * n + 1) // 2
    if e > K or e < 0: continue
    # 1/(1 - z q^n): for n >= 0 expand in q (n=0: constant 1/(1-z)); for n<0 use -z^{-1}q^{-n}/(1 - z^{-1}q^{-n})
    g = [0j] * (K + 1)
    if n == 0: g[0] = 1 / (1 - z)
    elif n > 0:
        for m in range(0, K // n + 1): g[m * n] += z ** m
    else:
        for m in range(1, K // (-n) + 1): g[m * (-n)] -= z ** (-m)
    term = ser_mul([0j] * e + [(-1) ** (n % 2)] + [0j] * (K - e), g)
    S = [a + b for a, b in zip(S, term)]
AL = [(1 - z) * x for x in ser_mul(S, ser_inv(E))]
print("Appell-Lerch form matches rank GF:", max(abs(a - b) for a, b in zip(R, AL)) < 1e-6)
print([round(abs(a-b),6) for a,b in zip(R,AL)][:10])
print([complex(round(x.real,4),round(x.imag,4)) for x in R[:5]])
print([complex(round(x.real,4),round(x.imag,4)) for x in AL[:5]])
d=[abs(a-b) for a,b in zip(R,AL)]; i=next(i for i,x in enumerate(d) if x>1e-6); print("first mismatch", i, d[i], abs(R[i]))
