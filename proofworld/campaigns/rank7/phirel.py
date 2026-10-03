"""relations Φ_{7-k} vs Φ_k (exact integer series)."""
KQ = 400
def mul(a, b):
    c = [0] * (KQ + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(KQ + 1 - i): c[i + j] += x * b[j]
    return c
def divgeo(f, e):  # f/(1-q^e)
    g = f[:]
    for t in range(e, KQ + 1): g[t] += g[t - e]
    return g
def Phi(k):
    s = [0] * (KQ + 1); n = 0
    while 7 * n * n <= KQ:
        t = [0] * (KQ + 1); t[7 * n * n] = 1
        for i in range(n + 1): t = divgeo(t, k + 7 * i)
        for i in range(n): t = divgeo(t, 7 - k + 7 * i)
        s = [a + b for a, b in zip(s, t)]; n += 1
    return s
P = {k: Phi(k) for k in range(1, 7)}
for k in (1, 2, 3):
    a, b = P[k], P[7 - k]
    # test b = q^{s} a + poly(q) for small poly
    for s in range(-3, 4):
        sa = [0] * s + a[:KQ + 1 - s] if s >= 0 else a[-s:] + [0] * (-s)
        d = [b[i] - sa[i] for i in range(KQ - 5)]
        nz = [(i, v) for i, v in enumerate(d) if v]
        if len(nz) <= 4: print(f"Phi{7-k} = q^{s}·Phi{k} + {nz}")
