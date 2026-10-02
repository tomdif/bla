from collections import defaultdict
Kq = 80
def HRcoef():
    HR = defaultdict(float)
    for n in range(0, 60):
        for j in range(0, n // 2 + 1):
            e2 = (n*n - 3*j*j) + (n - j)
            if e2 // 2 <= Kq:
                s = (-1) ** ((n + j) % 2)
                HR[(n - 3*j, e2 // 2)] += s/2; HR[(3*j - n, e2 // 2)] += s/2
        for j in range(1, n // 2 + 1):
            e2 = (n*n - 3*j*j) + (n + j)
            if e2 // 2 <= Kq:
                s = (-1) ** ((n + j) % 2)
                HR[(n - 3*j + 1, e2 // 2)] += s/2; HR[(3*j - n - 1, e2 // 2)] += s/2
    return HR
HR = HRcoef()
for a in range(0, 5):
    print(a, [(n, int(v)) for (k, n), v in sorted(HR.items(), key=lambda t: t[0][1]) if k == a and v != 0][:12])
Kq = 120
def T(m): return m*(m-1)//2
def P(n): return n*(3*n-1)//2
def terms_D(a):
    out = []   # (exponent, sign, m, n, type, u)
    for m in range(1, 40):
        for n in range(1, 25):
            base = T(m) + P(n); s = (-1)**((m+n) % 2)
            if m <= a:
                out.append((base + n*(a-m+1), s, m, n, 'A', m - n))
                out.append((base + n*(a+m), -s, m, n, 'B', m + n))
            else:
                out.append((base + n*(a+m), -s, m, n, 'C', m + n))
                out.append((base + n*(m-a), -s, m, n, 'D', m + n))
    return [t for t in out if t[0] <= Kq]
def ser_from(ts):
    d = defaultdict(int)
    for t in ts: d[t[0]] += t[1]
    return d
Es = defaultdict(int)
for s in range(-12, 13):
    if P(s) <= Kq: Es[P(s)] += (-1)**(s % 2)
def Hser(a):
    d = defaultdict(int); sg = (-1)**(a % 2); b = a*(a+1)//2
    for r in range(0, 20):
        e = b + 3*r*r + 3*r*a + r
        if e <= Kq: d[e] += sg
    for r in range(1, 20):
        e = b + 3*r*r + 3*r*a - r - a
        if e <= Kq: d[e] -= sg
    return d
def mul(x, y):
    c = defaultdict(int)
    for i, u in x.items():
        for j, v in y.items():
            if i + j <= Kq: c[i+j] += u*v
    return c
for a in range(0, 5):
    ts = terms_D(a)
    target = mul(Es, Hser(a))
    for m in range(a+1, 30):
        if T(m) <= Kq: target[T(m)] -= (-1)**((m+1) % 2)
    D = ser_from(ts)
    print(a, "identity:", all(D[i] == target[i] for i in range(Kq+1)), end="  ")
    for res in range(3):
        part = ser_from([t for t in ts if (t[5] - t[3]) % 3 == res])
        nz = [(i, part[i]) for i in range(Kq+1) if part[i]]
        print(f"res{res}: {'ZERO' if not nz else nz[:4]}", end="  ")
    print()
print("---- normalized coordinates ----")
Kq = 200
def Dpoints(a):
    pts = defaultdict(int)   # (X,Y) -> signed count, X odd, Y; includes c_a (identity: E·H_a = c_a + D')
    for m in range(1, 60):
        for n in range(1, 40):
            s = (-1)**((m+n) % 2)
            if m <= a:
                pts[(2*(m-n)-1, 2*n+a)] += s          # A
                pts[(2*(m+n)-1, 2*n+a)] += -s         # B
            else:
                pts[(2*(m+n)-1, 2*n+a)] += -s         # C
                pts[(2*(m+n)-1, 2*n-a)] += -s         # D
    for m in range(a+1, 60):
        pts[(2*m-1, a)] += (-1)**((m+1) % 2)          # c_a
    return {k: v for k, v in pts.items() if v and (k[0]**2 + 2*k[1]**2 - 1 - 2*a*a)//8 <= Kq}
def EHpoints(a):
    pts = defaultdict(int)
    for s in range(-30, 31):
        for r in range(0, 30):
            pts[(6*s-1, 6*r+3*a+1)] += (-1)**((s+a) % 2)
            if r >= 1: pts[(6*s-1, 6*r+3*a-1)] -= (-1)**((s+a) % 2)
    return {k: v for k, v in pts.items() if v and ((k[0]**2 + 2*k[1]**2)//3 - 1 - 2*a*a)//8 <= Kq}
def canon(X, Y): return (abs(X), abs(Y))
for a in range(0, 3):
    D = Dpoints(a); EH = EHpoints(a)
    # map EH points to (X,Y) via division by (1+sqrt-2) or (1-sqrt-2)
    img = defaultdict(int)
    for (x, y), v in EH.items():
        cands = []
        if (x - 2*y) % 3 == 0: cands.append(((x - 2*y)//3, (x + y)//3))
        if (x + 2*y) % 3 == 0: cands.append(((x + 2*y)//3, (y - x)//3))
        img[(tuple(sorted(canon(*c) for c in cands)), v)] += 1
    # aggregate D by canonical (|X|,|Y|)
    Dc = defaultdict(int)
    for (X, Y), v in D.items(): Dc[canon(X, Y)] += v
    nzD = {k: v for k, v in Dc.items() if v}
    print(a, "D canonical nonzero:", len(nzD), sorted(nzD.items())[:12])
    print(a, "EH sample:", sorted(EH.items(), key=lambda t: t[0][0]**2+2*t[0][1]**2)[:8])
