"""Class-0 orbit pieces of (2.18) at p=5 vs Ramanujan's φ."""
from collections import defaultdict
p = 5; K = 5 * 80; KQ = K // 5
def orbit(j): return min(j % p, (-j) % p)
G = defaultdict(lambda: [0] * (KQ + 1))
for n in range(0, 120):
    for j in range(0, n // 2 + 1):
        if (n*n - 3*j*j + n - j) % 2 == 0:
            e = (n*n - 3*j*j + n - j) // 2; s = (-1) ** (n + j)
            for zz in (n - 3*j, 3*j - n):
                if e <= K and e % 5 == 0: G[orbit(zz)][e // 5] += s * (0.5 if zz == 0 else 1) * (1 if True else 0)
    for j in range(1, n // 2 + 1):
        e = (n*n - 3*j*j + n + j) // 2; s = (-1) ** (n + j)
        for zz in (n - 3*j + 1, 3*j - n - 1):
            if e <= K and e % 5 == 0: G[orbit(zz)][e // 5] += s * (0.5 if zz == 0 else 1)
# NB: (2.18) has an overall 1/2 and pairs z^k + z^{-k}; we track per-monomial weight: coefficient of z^k.
def mul(a, b):
    c = [0] * (KQ + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(KQ + 1 - i): c[i + j] += x * b[j]
    return c
def prod(rs, m):  # prod_{n≡r mod m, n≥1} (1-q^n)
    f = [0] * (KQ + 1); f[0] = 1
    for n in range(1, KQ + 1):
        if n % m in rs:
            g = f[:]
            for k in range(n, KQ + 1): g[k] -= f[k - n]
            f = g
    return f
def inv(f):
    g = [0] * (KQ + 1); g[0] = 1 / f[0]
    for n in range(1, KQ + 1): g[n] = -sum(f[i] * g[n - i] for i in range(1, n + 1)) / f[0]
    return g
J52 = prod({2, 3, 0}, 5); J51 = prod({1, 4, 0}, 5)
# φ(q) = -1 + Σ q^{5n²}/((q;q⁵)_{n+1}(q⁴;q⁵)_n)
phi = [0] * (KQ + 1); phi[0] = -1
n = 0
while 5 * n * n <= KQ:
    t = [0] * (KQ + 1); t[5 * n * n] = 1
    for i in range(n + 1):
        f = [0] * (KQ + 1); f[0] = 1
        if 1 + 5 * i <= KQ: f[1 + 5 * i] = -1
        t = mul(t, inv(f))
    for i in range(n):
        f = [0] * (KQ + 1); f[0] = 1
        if 4 + 5 * i <= KQ: f[4 + 5 * i] = -1
        t = mul(t, inv(f))
    phi = [a + b for a, b in zip(phi, t)]
    n += 1
Jphi = mul(J52, phi)
print("G0", [round(x, 2) for x in G[0][:12]])
print("G1", G[1][:12]); print("G2", G[2][:12])
d = [a - b for a, b in zip(G[1], G[2])]
print("G1-G2", d[:15]); print("J52φ ", [round(x) for x in Jphi[:15]])
print("J51φ ", [round(x) for x in mul(J51, phi)[:15]])
