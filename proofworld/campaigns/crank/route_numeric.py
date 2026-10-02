"""Planted-truth check for the Andrews–Garvan crank route (exact arithmetic)."""
from collections import Counter, defaultdict
def partitions(n, maxp=None):
    if maxp is None: maxp = n
    if n == 0: yield (); return
    for k in range(min(n, maxp), 0, -1):
        for rest in partitions(n - k, k): yield (k,) + rest
def crank(lam):
    w = lam.count(1)
    return max(lam) if w == 0 else sum(1 for x in lam if x > w) - w
N = 40
M = {n: Counter(crank(l) for l in partitions(n)) for n in range(1, N + 1)}
# (1) generating function: Σ_n Σ_m M(m,n) z^m q^n = (q;q)/((zq;q)(z^{-1}q;q)) for n ≥ 2
# expand RHS as Laurent polynomials in z (dict exponent -> int) per q-degree
def series_mul(a, b, K):
    c = [defaultdict(int) for _ in range(K + 1)]
    for i in range(K + 1):
        for e1, v1 in a[i].items():
            if v1 == 0: continue
            for j in range(K + 1 - i):
                for e2, v2 in b[j].items():
                    c[i + j][e1 + e2] += v1 * v2
    return c
K = 22
one = lambda: [defaultdict(int) for _ in range(K + 1)]
num = one(); num[0][0] = 1
for k in range(1, K + 1):                      # (q;q)
    f = one(); f[0][0] = 1; f[k][0] = -1; num = series_mul(num, f, K)
for z in (1, -1):                              # 1/(z^{±1} q;q) = Π 1/(1 - z^{±1} q^k)
    for k in range(1, K + 1):
        g = one()
        for t in range(0, K // k + 1): g[t * k][z * t] = 1
        num = series_mul(num, g, K)
ok1 = all({e: v for e, v in num[n].items() if v} == dict(M[n]) for n in range(2, K + 1))
print("(1) crank GF = (q;q)/((zq;q)(q/z;q)) for 2 ≤ n ≤", K, ":", ok1, "| n=1 anomaly: GF", dict((e, v) for e, v in num[1].items() if v), "vs", dict(M[1]))
# (2) equidistribution
for l, d in ((5, 4), (7, 5), (11, 6)):
    good = all(len({sum(v for m, v in M[n].items() if m % l == r) for r in range(l)}) == 1
               for n in range(d, N + 1, l))
    print(f"(2) crank equidistributed mod {l} for n ≡ {d}:", good, f"(n ≤ {N})")
# (3) decomposition by ω (number of ones): for w ≥ 1, crank = #{parts > w} − w; for w = 0, crank = largest part
print("(3) definition consistency checked implicitly; sample: crank(4,1)=", crank((4, 1)), "crank(3,3)=", crank((3, 3)), "crank(1,1,1)=", crank((1, 1, 1)))
