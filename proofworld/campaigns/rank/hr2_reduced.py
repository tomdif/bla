"""Reduced per-a form of Garvan's (2.19), using the proved per-b form of (2.18) in base q^2:
   Σ_{b∈Z} q^{(a−b)² + |b|(|b|+1)} PT_{|b|}(q²) = ψ(q)·q^{−a²}·Σ_{n≥|a|} q^{n(3n+1)/2}(1 − q^{2n+1}),
   PT_b(Q) = Σ_{r≥0} Q^{3r²+3br+r} − Σ_{r≥1} Q^{3r²+3br−r−b},  ψ(q) = Σ_{n∈Z} q^{2n²+n}."""
from collections import defaultdict
K = 150
def lhs_terms(a):
    t = []   # (exponent, sign, b, r, kind)
    for b in range(-40, 41):
        B = abs(b); base = (a - b) ** 2 + B * (B + 1)
        for r in range(0, 30):
            e = base + 2 * (3 * r * r + 3 * B * r + r)
            if e <= K: t.append((e, 1, b, r, 'P'))
            if r >= 1:
                e = base + 2 * (3 * r * r + 3 * B * r - r - B)
                if e <= K: t.append((e, -1, b, r, 'M'))
    return t
def rhs_terms(a):
    t = []
    A = abs(a)
    for n in range(-30, 31):
        for m in range(A, 40):
            e = 2 * n * n + n + m * (3 * m + 1) // 2 - a * a
            if 0 <= e <= K: t.append((e, 1, n, m, 'P'))
            e2 = e + 2 * m + 1
            if 0 <= e2 <= K: t.append((e2, -1, n, m, 'M'))
    return t
def ser(ts):
    d = defaultdict(int)
    for x in ts: d[x[0]] += x[1]
    return d
for a in range(0, 6):
    L, R = ser(lhs_terms(a)), ser(rhs_terms(a))
    ok = all(L[i] == R[i] for i in range(K + 1))
    nL = sum(1 for x in lhs_terms(a)); nR = sum(1 for x in rhs_terms(a))
    cancL = sum(abs(v) for v in L.values()); 
    print(f"a={a}: identity {ok};  #terms L={nL} R={nR};  sum|coeff| L={cancL} R={sum(abs(v) for v in R.values())}")
# normalized coordinates: 24e+4+24a^2 = 3U^2+V^2
def norm_L(a, b, r, kind):
    B = abs(b)
    if kind == 'P': e = (a-b)**2 + B*(B+1) + 2*(3*r*r+3*B*r+r)
    else: e = (a-b)**2 + B*(B+1) + 2*(3*r*r+3*B*r-r-B)
    return 24*e + 4 + 24*a*a
def rhs_point(n, m, kind): return (4*n+1, 6*m+1 if kind == 'P' else 6*m+5)
# unit rotations of ℤ[ω] acting on (U,V) with N = 3U²+V²: z = V + U√−3
def rotations(U, V):
    out = []
    for (x, y) in [(V, U), (-V, -U)]:            # z, -z
        cands = [(x, y), ((x - 3*y)//2 if (x-3*y) % 2 == 0 else None, (x + y)//2 if (x+y) % 2 == 0 else None),
                 ((x + 3*y)//2 if (x+3*y) % 2 == 0 else None, (y - x)//2 if (y-x) % 2 == 0 else None)]
        for c in cands:
            if None in c: continue
            out.append(c); out.append((c[0], -c[1]))   # with conjugation
    return out   # list of (V', U')
def find_rhs(a, U, V):
    hits = []
    for (V2, U2) in rotations(U, V):
        for U3 in (U2, -U2):
            if (U3 - 1) % 4 == 0 and V2 > 0:
                n = (U3 - 1)//4
                if V2 % 6 == 1 and (V2-1)//6 >= abs(a): hits.append((n, (V2-1)//6, 'P'))
                if V2 % 6 == 5 and (V2-5)//6 >= abs(a): hits.append((n, (V2-5)//6, 'M'))
    return set(hits)
for a in range(0, 3):
    Lt = lhs_terms(a); Rt = rhs_terms(a)
    Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
    stats = defaultdict(int)
    for (e, s, b, r, kind) in Lt:
        B = abs(b)
        if kind == 'P': U, V = 2*b - 4*a, 12*r + 6*B + 2
        else: U, V = 2*b - 4*a, 12*r + 6*B - 2   # guess; verify norm
        N = 3*U*U + V*V
        okN = (N == norm_L(a, b, r, kind))
        hits = [h for h in find_rhs(a, U, V) if Rsign.get(h) == s]
        stats[(kind, 'b>=0' if b >= 0 else 'b<0', okN, len(hits))] += 1
    print(a, dict(stats))
print("---- samples a=1 ----")
a = 1
Rt = rhs_terms(a); Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
for (e, s, b, r, kind) in sorted(lhs_terms(a), key=lambda t: (t[4], t[2] < 0, t[3], abs(t[2])))[:60]:
    B = abs(b)
    U, V = 2*b - 4*a, 12*r + 6*B + (2 if kind == 'P' else -2)
    hits = sorted(h for h in find_rhs(a, U, V) if Rsign.get(h) == s)
    print(kind, "b=%d r=%d" % (b, r), "->", hits)
print("---- injection + complement ----")
def Lmap(a, b, r, kind):
    B = abs(b); U = 2*b - 4*a; V = 12*r + 6*B + (2 if kind == 'P' else -2)
    V2, U2 = (V - 3*U)//2, (V + U)//2
    n = (U2 - 1)//4 if U2 % 4 == 1 else (-U2 - 1)//4
    if kind == 'P': assert V2 % 6 == 1; m = (V2 - 1)//6
    else: assert V2 % 6 == 5; m = (V2 - 5)//6
    return (n, m, kind)
for a in range(0, 5):
    Lt = lhs_terms(a); Rt = rhs_terms(a)
    Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
    img = [Lmap(a, b, r, k) for (e, s, b, r, k) in Lt]
    inj = len(set(img)) == len(img)
    signok = all(Rsign.get(Lmap(a, b, r, k)) == s for (e, s, b, r, k) in Lt)
    rest = {k: v for k, v in Rsign.items() if k not in set(img)}
    # does the rest cancel exponent-wise?
    d = defaultdict(int)
    for (n, m, k), v in rest.items():
        e = 2*n*n + n + m*(3*m+1)//2 - a*a + (2*m+1 if k == 'M' else 0)
        d[e] += v
    print(f"a={a}: injective {inj}, signs {signok}, rest cancels {all(v == 0 for e, v in d.items() if e <= K - 40)}  (#rest {len(rest)})")
    # try involution on rest: (n,m,P) <-> (n',m',M) via rotation by the other unit
    if a == 1:
        sample = sorted(rest.items(), key=lambda t: 2*t[0][0]**2+t[0][0]+t[0][1]*(3*t[0][1]+1)//2)[:16]
        print("   rest sample:", sample)
print("---- involution search on the complement ----")
def UV(n, m, k): return (4*n+1, 6*m+1 if k == 'P' else 6*m+5)
def from_UV(a, U, V):
    if U % 4 == 1: n = (U-1)//4
    elif U % 4 == 3: n = (-U-1)//4
    else: return None
    if V <= 0: return None
    if V % 6 == 1: return (n, (V-1)//6, 'P')
    if V % 6 == 5: return (n, (V-5)//6, 'M')
    return None
maps = {
 'rot+': lambda U, V: ((V + U)//2 if (V+U) % 2 == 0 else None, (V - 3*U)//2),
 'rot-': lambda U, V: ((U - V)//2 if (U-V) % 2 == 0 else None, (V + 3*U)//2),
 'rot+c': lambda U, V: (-(V + U)//2 if (V+U) % 2 == 0 else None, (V - 3*U)//2),
 'rot-c': lambda U, V: (-(U - V)//2 if (U-V) % 2 == 0 else None, (V + 3*U)//2),
 'rot2+': lambda U, V: ((V - U)//2 if (V-U) % 2 == 0 else None, (-V - 3*U)//2),
 'rot2-': lambda U, V: ((-V - U)//2 if (V+U) % 2 == 0 else None, (-V + 3*U)//2),
}
for a in range(0, 4):
    Lt = lhs_terms(a); Rt = rhs_terms(a)
    Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
    img = set(Lmap(a, b, r, k) for (e, s, b, r, k) in Lt)
    rest = {k: v for k, v in Rsign.items() if k not in img}
    for name, f in maps.items():
        good = 0; tot = 0
        for key, v in rest.items():
            U, V = UV(*key)
            U2, V2 = f(U, V)
            if U2 is None: continue
            tot += 1
            t = from_UV(a, U2, V2)
            if t is not None and t in rest and rest[t] == -v and t[1] >= abs(a): good += 1
        print(a, name, good, "/", len(rest))
print("---- criterion ----")
for a in range(0, 4):
    Lt = lhs_terms(a); Rt = rhs_terms(a)
    Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
    img = set(Lmap(a, b, r, k) for (e, s, b, r, k) in Lt)
    rest = {k: v for k, v in Rsign.items() if k not in img}
    cls = defaultdict(int)
    for key, v in rest.items():
        U, V = UV(*key)
        res = []
        for name in ('rot+', 'rot-'):
            U2, V2 = maps[name](U, V)
            t = None if U2 is None else from_UV(a, U2, V2)
            ok = t is not None and t in rest and rest[t] == -v
            res.append(ok)
        cls[(key[2], U % 4 == 1 and 'U=+' or 'U', (key[0] >= 0), tuple(res))] += 1
    print(a, sorted(cls.items()))
print("---- why the other rotation fails ----")
for a in range(0, 4):
    Lt = lhs_terms(a); Rt = rhs_terms(a)
    Rsign = {(x[2], x[3], x[4]): x[1] for x in Rt}
    img = set(Lmap(a, b, r, k) for (e, s, b, r, k) in Lt)
    rest = {k: v for k, v in Rsign.items() if k not in img}
    why = defaultdict(int)
    for key, v in rest.items():
        U, V = UV(*key)
        for name in ('rot+', 'rot-'):
            U2, V2 = maps[name](U, V)
            t = from_UV(a, U2, V2)
            if t is None: tag = 'V<=0 or bad residue'
            elif t[1] < abs(a): tag = 'm<|a|'
            elif t in img: tag = 'in image'
            elif t in rest: tag = 'rest' + ('(sign ok)' if rest[t] == -v else '(SAME sign)')
            else: tag = 'outside window'
            why[(name, tag)] += 1
    print(a, sorted(why.items()))
