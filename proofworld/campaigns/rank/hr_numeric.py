"""Check Garvan's Hecke-Rogers (2.18) in the form  j(z)·C(z) = (1−z)·E·HR(z)  as Laurent-in-z, power-in-q series.
C(z) = Σ_k z^k Σ_{n≥1} (−1)^{n−1} q^{n(3n−1)/2+|k|n}(1−q^n)   (= E·R(z;q))."""
from collections import defaultdict
K = 70
def add(d, key, v):
    d[key] += v
# rank GF check: E*R vs C
def parts(n, mx=None):
    if mx is None: mx = n
    if n == 0: yield []; return
    for k in range(min(n, mx), 0, -1):
        for p in parts(n - k, k): yield [k] + p
R = defaultdict(int)
for n in range(0, 26):
    for l in parts(n): R[(max(l) - len(l) if l else 0, n)] += 1
E = defaultdict(int)
for s in range(-10, 11):
    e = s * (3 * s - 1) // 2
    if e <= K: E[e] += (-1) ** (s % 2)
def mulq(a, b, Kmax):   # a: dict (k,n)->c ; b: dict n->c
    c = defaultdict(int)
    for (k, n), x in a.items():
        for m, y in b.items():
            if n + m <= Kmax: c[(k, n + m)] += x * y
    return c
C = defaultdict(int)
for k in range(-K, K + 1):
    for n in range(1, K):
        e = n * (3 * n - 1) // 2 + abs(k) * n
        if e > K: break
        C[(k, e)] += (-1) ** (n - 1)
        if e + n <= K: C[(k, e + n)] -= (-1) ** (n - 1)
for n, v in E.items(): C[(0, n)] += v
ER = mulq(R, E, 25)
print("E·R = C (q<=25):", all(ER[(k, n)] == C[(k, n)] for k in range(-26, 27) for n in range(26)))
# HR(z) from (2.18)
HR = defaultdict(int)
for n in range(0, 40):
    for j in range(0, n // 2 + 1):
        e = (n * n - 3 * j * j) // 2 + (n - j) // 2 if False else None
        e2 = (n * n - 3 * j * j) + (n - j)          # 2*exponent
        if e2 % 2 == 0 and e2 // 2 <= K:
            s = (-1) ** ((n + j) % 2)
            HR[(n - 3 * j, e2 // 2)] += s; HR[(3 * j - n, e2 // 2)] += s
    for j in range(1, n // 2 + 1):
        e2 = (n * n - 3 * j * j) + (n + j)
        if e2 // 2 <= K:
            s = (-1) ** ((n + j) % 2)
            HR[(n - 3 * j + 1, e2 // 2)] += s; HR[(3 * j - n - 1, e2 // 2)] += s
HR = {k: v / 2 for k, v in HR.items()}
# j(z) = Σ (−1)^m z^m q^{m(m−1)/2}
J = {}
for m in range(-15, 16):
    e = m * (m - 1) // 2
    if e <= K: J[(m, e)] = (-1) ** (m % 2)
def mul2(a, b, Kmax):
    c = defaultdict(float)
    for (k1, n1), x in a.items():
        for (k2, n2), y in b.items():
            if n1 + n2 <= Kmax: c[(k1 + k2, n1 + n2)] += x * y
    return c
L = mul2(J, C, 40)
EHR = mul2({(0, n): v for n, v in E.items()}, HR, 40)
Rt = defaultdict(float)
for (k, n), v in EHR.items(): Rt[(k, n)] += v; Rt[(k + 1, n)] -= v
bad = [(k, n) for k in range(-12, 13) for n in range(0, 30) if abs(L[(k, n)] - Rt[(k, n)]) > 1e-9]
print("j(z)·C(z) = (1−z)·E·HR(z) (q<=29, |z-deg|<=12):", not bad, bad[:5])
# reduced per-a identity for (2.18):  E·H_a = c_a·E + Σ_{m,n≥1} (−1)^{m+n} q^{T(m)+n(3n−1)/2} W_a(m,n)
def ser(): return defaultdict(int)
Kq = 60
Es = defaultdict(int)
for s in range(-10, 11):
    e = s*(3*s-1)//2
    if e <= Kq: Es[e] += (-1)**(s % 2)
def mul1(a, b):
    c = defaultdict(int)
    for i, x in a.items():
        for j, y in b.items():
            if i + j <= Kq: c[i+j] += x*y
    return c
ok = True
for a in range(0, 8):
    Ha = defaultdict(float)
    for (k, n), v in HR.items():
        if k == a and n <= Kq: Ha[n] += v
    lhs = mul1(Es, Ha)
    ca = defaultdict(int)
    for m in range(a+1, 20):
        e = m*(m-1)//2
        if e <= Kq: ca[e] += (-1)**((m+1) % 2)
    rhs = mul1(ca, Es)
    for m in range(1, 20):
        for n in range(1, 10):
            base = m*(m-1)//2 + n*(3*n-1)//2
            s = (-1)**((m+n) % 2)
            if m <= a: terms = [(n*(a-m+1), 1), (n*(a+m), -1)]
            else: terms = [(0, 1), (n, 1), (n*(a+m), -1), (n*(m-a), -1)]
            for e, c in terms:
                if base + e <= Kq: rhs[base+e] += s*c
    if any(abs(lhs[i] - rhs[i]) > 1e-9 for i in range(Kq+1)): ok = False; print("fail a=", a, [(i, lhs[i], rhs[i]) for i in range(Kq+1) if abs(lhs[i]-rhs[i])>1e-9][:4])
print("reduced 2-D identities for (2.18), a=0..7, to q^60:", ok)
