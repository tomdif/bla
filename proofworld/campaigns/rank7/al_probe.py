"""Check E²·h1Z_c·... form:  E·(−1)^c q^{T(c)} PT_c  ==  Σ_s (−1)^s q^{C(s,2)} S_{c−s}   (per z^c, m = 1)
S_t = [z^t] Σ_n (−1)^n q^{n(3n+1)/2}/(1 − z q^n)."""
N = 80
def T(a): return a * (a + 1) // 2
def PT(c):  # (−1)^c q^{T(c)} Σ_r q^{r²+cr} α*_r(c)   (RHSz_c with sign and shift)
    s = [0] * (N + 1)
    for r in range(0, N):
        e = T(c) + 3 * r * r + 3 * c * r + r
        if e <= N: s[e] += (-1) ** c
        if r >= 1:
            e = T(c) + 3 * r * r + 3 * c * r - r - c
            if e <= N: s[e] -= (-1) ** c
    return s
E = [1] + [0] * N
for n in range(1, N + 1): E = [E[i] - (E[i - n] if i >= n else 0) for i in range(N + 1)]
def mul(a, b): return [sum(a[j] * b[i - j] for j in range(i + 1)) for i in range(N + 1)]
def S(t):
    s = [0] * (N + 1)
    if t >= 0:
        s[0] += 1
        for n in range(1, N):
            e = n * (3 * n + 1) // 2 + n * t
            if e <= N: s[e] += (-1) ** n
    else:
        u = -t
        for n in range(-N, 0):
            e = n * (3 * n + 1) // 2 - n * u
            if 0 <= e <= N: s[e] -= (-1) ** n
    return s
for c in range(0, 5):
    L = mul(E, PT(c))
    R = [0] * (N + 1)
    for sgen in range(-30, 31):
        e0 = sgen * (sgen - 1) // 2
        if e0 > N: continue
        St = S(c - sgen)
        for i in range(N + 1 - e0): R[i + e0] += (-1) ** sgen * St[i]
    print(c, L[:N - 5] == R[:N - 5])
from collections import Counter
def lhs_terms(c):
    out = []
    for s1 in range(-30, 31):
        p = s1 * (3 * s1 - 1) // 2
        for r in range(0, 30):
            for ty in ('P', 'M'):
                if ty == 'M' and r == 0: continue
                e = T(c) + 3 * r * r + 3 * c * r + (r if ty == 'P' else -r - c) + p
                if e <= N: out.append((e, (-1) ** (s1 + c) * (1 if ty == 'P' else -1), ('L', s1, r, ty)))
    return out
def rhs_terms(c):
    out = []
    for s in range(-30, 31):
        e0 = s * (s - 1) // 2; t = c - s
        if t >= 0:
            out.append((e0, (-1) ** s, ('R', s, 0)))
            for n in range(1, 30):
                e = e0 + n * (3 * n + 1) // 2 + n * t
                if e <= N: out.append((e, (-1) ** (s + n), ('R', s, n)))
        else:
            for n in range(-30, 0):
                e = e0 + n * (3 * n + 1) // 2 + n * t
                if 0 <= e <= N: out.append((e, -(-1) ** (s + n), ('R', s, n)))
    return [x for x in out if x[0] <= N]
for c in range(0, 3):
    L, R = lhs_terms(c), rhs_terms(c)
    cl = Counter(); cr = Counter()
    for e, sg, _ in L: cl[(e, sg)] += 1
    for e, sg, _ in R: cr[(e, sg)] += 1
    bij = all(cl[k] == cr[k] for k in set(cl) | set(cr) if k[0] < N - 10)
    print("c", c, "#L", len([x for x in L if x[0] < N - 10]), "#R", len([x for x in R if x[0] < N - 10]), "signed-multiset bijection:", bij)
