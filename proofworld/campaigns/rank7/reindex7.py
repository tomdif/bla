"""Test  q^s·G(c,o) + σ·M_k − J_{7,k} = 0  (or with J_{7,k} replaced by 0) for p=7."""
from collections import defaultdict
p = 7; KQ = 300; K = p * KQ
def orbit(a): return min(a % p, (-a) % p)
def hb_terms(a, lim):
    T = a * (a + 1) // 2; r = 0; out = []
    while T + 3 * r * r <= lim + 50:
        out.append((T + 3*r*r + 3*a*r + r, (-1) ** a))
        if r >= 1: out.append((T + 3*r*r + 3*a*r - r - a, -(-1) ** a))
        r += 1
    return out
G = defaultdict(lambda: [0] * (KQ + 1))
for a in range(0, 3000):
    if a * (a + 1) // 2 > K: break
    for e, s in hb_terms(a, K):
        if 0 <= e <= K: G[(e % p, orbit(a))][e // p] += s
def M(k):
    m = [0] * (KQ + 1)
    for b in range(0, 300):
        if 7 * b * (b + 1) // 2 - k * b > KQ: break
        for e, s in hb_terms(b, KQ):
            for sg in ([0] if b == 0 else [1, -1]):
                E = 7 * e + sg * k * b
                if 0 <= E <= KQ: m[E] += s
    return m
def Jk(k):
    j = [0] * (KQ + 1)
    for m in range(-60, 61):
        e = 7 * m * (m - 1) // 2 + k * m
        if 0 <= e <= KQ: j[e] += (-1) ** m
    return j
def sh(f, s): return [0] * s + f[:KQ + 1 - s] if s >= 0 else f[-s:] + [0] * (-s)
N = KQ - 10
for k in range(1, 7):
    Mk, J = M(k), Jk(k)
    for (c, o), g in sorted(G.items()):
        for s in range(-3, 4):
            gs = sh(g, s)
            for sig in (1, -1):
                for jj in (1, 0, -1):
                    if all(gs[i] + sig * Mk[i] - jj * J[i] == 0 for i in range(N)):
                        print(f"k={k}: q^{s}·G{(c, o)} + {sig}·M_{k} = {jj}·J_7,{k}")
