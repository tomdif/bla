"""Orbit pieces of Σ_a w_a(z) Hblk_a(q) (exact (2.18)·E form) vs specializations
M_k = Σ_b w_b(q^k) Hblk_b(q^5)  (= J_{5,k}(1+φ/ψ) by (2.18) at z=q^k, base q^5)."""
from collections import defaultdict
K = 5 * 70; KQ = K // 5
def hblk_terms(a, Kmax):
    # Hblk_a = (-1)^a q^{T(a)} Σ_r q^{r²+ar}(q^{2r²+2ar+r} - [r≥1] q^{2r²+2ar-r-a})
    T = a * (a + 1) // 2; out = []
    r = 0
    while T + 3*r*r + 3*a*r + r <= Kmax or r < 2:
        out.append((T + 3*r*r + 3*a*r + r, (-1) ** a))
        if r >= 1: out.append((T + 3*r*r + 3*a*r - r - a, -(-1) ** a))
        r += 1
        if T + 3*r*r > Kmax + 10: break
    return out
def orbit(a): return min(a % 5, (-a) % 5)
G = defaultdict(lambda: [0] * (KQ + 1))
for a in range(0, 200):
    if a * (a + 1) // 2 > K: break
    for e, s in hblk_terms(a, K):
        if 0 <= e <= K:
            G[(e % 5, orbit(a))][e // 5] += s
def M(k):
    m = [0] * (KQ + 1)
    for b in range(0, 100):
        if 5 * b * (b + 1) // 2 - k * b > KQ: break
        for e, s in hblk_terms(b, KQ):
            for sh in ([0] if b == 0 else [k * b, -k * b]):
                E = 5 * e + sh
                if 0 <= E <= KQ: m[E] += s
    return m
M1, M2 = M(1), M(2)
print("M1", M1[:16]); print("M2", M2[:16])
for key in sorted(G):
    print(key, G[key][:16])
