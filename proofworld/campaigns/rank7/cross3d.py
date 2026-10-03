"""Is J71·G(1,1) = −J72·M1 a signed bijection of 3D lattice points?  (variable x; base x^7)"""
from collections import Counter
N = 160
def T(a): return a * (a + 1) // 2
def orbit(a): return min(a % 7, (-a) % 7)
def J(k):   # Σ_m (−1)^m x^{7C(m,2)+km}
    out = []
    for m in range(-40, 41):
        e = 7 * m * (m - 1) // 2 + k * m
        if 0 <= e <= N: out.append((e, (-1) ** m, ('J', k, m)))
    return out
def Gpiece(c, o):
    out = []
    for a in range(0, 400):
        if orbit(a) != o or T(a) > 7 * N + 7: continue
        for r in range(0, 200):
            for ty in ('P', 'M'):
                if ty == 'M' and r == 0: continue
                e = T(a) + 3 * r * r + 3 * a * r + (r if ty == 'P' else -r - a)
                if e > 7 * N + 7: break
                if e % 7 == c: out.append((e // 7, (-1) ** a * (1 if ty == 'P' else -1), ('G', a, r, ty)))
    return out
def Mk(k):
    out = []
    for b in range(0, 200):
        if 7 * T(b) - k * b > N + 50: break
        for r in range(0, 100):
            for ty in ('P', 'M'):
                if ty == 'M' and r == 0: continue
                e = T(b) + 3 * r * r + 3 * b * r + (r if ty == 'P' else -r - b)
                for sg in ([0] if b == 0 else [1, -1]):
                    E = 7 * e + sg * k * b
                    if 0 <= E <= N: out.append((E, (-1) ** b * (1 if ty == 'P' else -1), ('M', b, r, ty, sg)))
    return out
def prod(A, B):
    return [(e1 + e2, s1 * s2, (l1, l2)) for e1, s1, l1 in A for e2, s2, l2 in B if e1 + e2 <= N]
L = prod(J(1), Gpiece(1, 1)); R = prod(J(2), Mk(1))
cl, cr = Counter(), Counter()
for e, s, _ in L: cl[(e, s)] += 1
for e, s, _ in R: cr[(e, -s)] += 1          # identity is L = −R
sl = Counter(); sr = Counter()
for e, s, _ in L: sl[e] += s
for e, s, _ in R: sr[e] -= s
print("series equal:", all(sl[e] == sr[e] for e in range(N - 20)))
print("pure signed bijection:", all(cl[k] == cr[k] for k in set(cl) | set(cr) if k[0] < N - 20))
print("#terms L", sum(1 for x in L if x[0] < N - 20), "R", sum(1 for x in R if x[0] < N - 20))
