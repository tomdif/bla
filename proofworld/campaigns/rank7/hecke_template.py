"""Search  H_m = Σ_{n≥0} Σ_{|j|≤n/2} (−1)^{n+j} (z^{e} + z^{D−e}) q^{V(n,j)},  e = αn+βj+γ  (+ optional 2nd family V'(n,j))."""
import sys, itertools
from collections import defaultdict
exec(open('hecke_m.py').read().split("byc = defaultdict(dict)")[0])
target = {(n, c): v for n, d in H.items() for c, v in d.items() if v}
D = m - 1
def build(al, be, ga, fam2):
    S = defaultdict(int)
    for n in range(0, 2 * N + 2):
        for j in range(-(n // 2), n // 2 + 1):
            if (n * n - 3 * j * j + n - j) % 2: continue
            V = (n * n - 3 * j * j + n - j) // 2
            if 0 <= V <= N:
                e = al * n + be * j + ga
                S[(V, e)] += (-1) ** (n + j); S[(V, D - e)] += (-1) ** (n + j)
            if fam2 and j >= 1:
                V2 = (n * n - 3 * j * j + n + j) // 2
                if 0 <= V2 <= N:
                    e = fam2[0] * n + fam2[1] * j + fam2[2]
                    S[(V2, e)] += (-1) ** (n + j); S[(V2, D - e)] += (-1) ** (n + j)
    return {k: v for k, v in S.items() if v}
def check(S):
    keys = set(S) | set(target)
    return all(S.get(k, 0) == target.get(k, 0) for k in keys if k[0] <= N - 2)
for al, be, ga in itertools.product(range(-3, 4), range(-4, 5), range(-3, 4)):
    S = build(al, be, ga, None)
    if check(S): print("single family:", al, be, ga)
print("done single")
for al, be, ga, a2, b2, g2 in itertools.product(range(-2, 3), range(-3, 4), range(-2, 3), range(-2, 3), range(-3, 4), range(-2, 3)):
    S = build(al, be, ga, (a2, b2, g2))
    if check(S): print("two families:", (al, be, ga), (a2, b2, g2))
