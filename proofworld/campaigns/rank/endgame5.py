"""Numerical check of the ζ₅ endgame (exact arithmetic in Z[ζ] as vectors mod Φ5)."""
K = 300
# elements of Z[ζ]: tuples of 4 ints (basis 1,ζ,ζ²,ζ³), ζ⁴ = -1-ζ-ζ²-ζ³
def zadd(a, b): return tuple(x + y for x, y in zip(a, b))
def zneg(a): return tuple(-x for x in a)
def zmul(a, b):
    c = [0] * 8
    for i in range(4):
        for j in range(4): c[i + j] += a[i] * b[j]
    # reduce ζ^k for k>=4 via ζ^5=1 and ζ^4 = -1-ζ-ζ²-ζ³
    full = [0] * 5
    for k, v in enumerate(c): full[k % 5] += v
    return (full[0] - full[4], full[1] - full[4], full[2] - full[4], full[3] - full[4])
Z0 = (0, 0, 0, 0); Z1 = (1, 0, 0, 0)
def zint(n): return (n, 0, 0, 0)
def zpow(k):
    k %= 5
    return [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(-1,-1,-1,-1)][k]
# power series over Z[ζ] truncated at K
def smul(f, g):
    h = [Z0] * (K + 1)
    for i, a in enumerate(f):
        if a == Z0: continue
        for j in range(K + 1 - i):
            if g[j] != Z0: h[i + j] = zadd(h[i + j], zmul(a, g[j]))
    return h
def sint(lst): return [zint(x) for x in lst]
def poch(c, s):   # prod_{i>=0} (1 - c q^{s+i})
    f = [Z0] * (K + 1); f[0] = Z1
    for n in range(s, K + 1):
        g = f[:]
        for m in range(n, K + 1): g[m] = zadd(g[m], zneg(zmul(c, f[m - n])))
        f = g
    return f
def inv(f):
    g = [Z0] * (K + 1); g[0] = Z1   # f[0] = 1 assumed
    for n in range(1, K + 1):
        acc = Z0
        for i in range(1, n + 1): acc = zadd(acc, zmul(f[i], g[n - i]))
        g[n] = zneg(acc)
    return g
E = poch(Z1, 1)
z, zi = zpow(1), zpow(4)
F = smul(smul(poch(z, 1), poch(zi, 1)), E)
# R(ζ;q) Durfee
def durfee(x, y, base=1):
    tot = [Z0] * (K + 1)
    k = 0
    while base * k * k <= K:
        t = [Z0] * (K + 1); t[base * k * k] = Z1
        # 1/((xq^b;q^b)_k (yq^b;q^b)_k)
        for i in range(1, k + 1):
            for c in (x, y):
                fac = [Z0] * (K + 1); fac[0] = Z1
                if base * i <= K: fac[base * i] = zneg(c)
                t = smul(t, inv(fac))
        tot = [zadd(a, b) for a, b in zip(tot, t)]
        k += 1
    return tot
R1 = durfee(z, zi)
R2 = durfee(z, zi, 2)
def cls(f, r): return [f[n] if n % 5 == r else Z0 for n in range(K + 1)]
s = zadd(zpow(2), zpow(3)); t = zadd(zpow(1), zpow(4))
# F dissection: F = F0(q^5) + q s F1(q^5)
F0 = [Z0] * (K + 1); F1 = [Z0] * (K + 1)
for m in range(-20, 21):
    e0 = (25 * m * m - 5 * m) // 2; e1 = (25 * m * m + 15 * m) // 2 + 1
    sg = zint((-1) ** (m % 2))
    if 0 <= e0 <= K: F0[e0] = zadd(F0[e0], sg)
    if 0 <= e1 <= K: F1[e1] = zadd(F1[e1], zmul(sg, s))
print("F dissection:", all(F[n] == zadd(F0[n], F1[n]) for n in range(K + 1)))
# class 2 of F R(ζ;q) vs E^2
FR = smul(F, R1); E2s = smul(E, E)
print("class2(F R) = class2(E^2):", cls(FR, 2) == cls(E2s, 2))
# class 3,4 of F R(ζ;q^2) vs (s/2), (t/2) of E^3/E(q^2)
Eq2 = [Z0] * (K + 1)
for n in range(0, K // 2 + 1): Eq2[2 * n] = E[n]
th = smul(smul(E2s, E), inv(Eq2))
FR2 = smul(F, R2)
ok3 = all(zmul(zint(2), FR2[n]) == zmul(s, th[n]) for n in range(K + 1) if n % 5 == 3)
ok4 = all(zmul(zint(2), FR2[n]) == zmul(t, th[n]) for n in range(K + 1) if n % 5 == 4)
print("class3/4 of F R(ζ;q²):", ok3, ok4)
print("U53 R(ζ;q²) = 0:", all(R2[n] == Z0 for n in range(K + 1) if n % 5 == 3))
# HR selection: a(a+1)/2 + 3r²+3ar+r and a(a+1)/2+3r²+3ar-r-a ≡ 2 mod 5 ⇒ a ≡ 0
bad = [(a, r) for a in range(5) for r in range(5) for e in ((a*(a+1)//2 + 3*r*r+3*a*r+r), (a*(a+1)//2 + 3*r*r + 3*a*r - r - a)) if e % 5 == 2 and a % 5]
print("HR1 class-2 selection violations (residues):", bad)
# (2.19) G_a exponents: a(a+1)/2+3at+t(3t+1)/2 [+2a+2t+1]; class 3 ⇒ a ≡ ±2, class 4 ⇒ a ≡ ±1
b3 = set(); b4 = set()
for a in range(10):
    for tt in range(10):
        e = a*(a+1)//2 + 3*a*tt + tt*(3*tt+1)//2
        for ee in (e, e + 2*a + 2*tt + 1):
            if ee % 5 == 3: b3.add(a % 5)
            if ee % 5 == 4: b4.add(a % 5)
print("class-3 a residues:", sorted(b3), " class-4 a residues:", sorted(b4))
