exec(open('spec_probe.py').read().split('print("M1"')[0])
def prod(rs, m):
    f = [0] * (KQ + 1); f[0] = 1
    for n in range(1, KQ + 1):
        if n % m in rs:
            g = f[:]
            for k in range(n, KQ + 1): g[k] -= f[k - n]
            f = g
    return f
def mul(a, b):
    c = [0] * (KQ + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(KQ + 1 - i): c[i + j] += x * b[j]
    return c
J5 = prod({0}, 5); J51 = mul(prod({1, 4}, 5), J5); J52 = mul(prod({2, 3}, 5), J5)
r1 = [G[(1, 2)][i] + M1[i] - J51[i] for i in range(KQ - 5)]
r3 = [([0] + G[(3, 1)])[i] + M2[i] - J52[i] for i in range(KQ - 5)]
print("G12 + M1 - J51 == 0:", all(x == 0 for x in r1))
print("qG31 + M2 - J52 == 0:", all(x == 0 for x in r3))
# term-level: list (exponent, sign, label) for each side
def terms_G(c, orb):
    out = []
    for a in range(0, 200):
        if orbit(a) != orb: continue
        T = a*(a+1)//2
        if T > K: break
        r = 0
        while T + 3*r*r > -1 and T + 3*r*r <= K + 50:
            for (e, s, ty) in ((T + 3*r*r + 3*a*r + r, (-1)**a, 'P'),) + (((T + 3*r*r + 3*a*r - r - a, -(-1)**a, 'M'),) if r >= 1 else ()):
                if e <= K and e % 5 == c: out.append((e // 5, s, ('G', a, r, ty)))
            r += 1
    return out
def terms_M(k):
    out = []
    for b in range(0, 100):
        T = b*(b+1)//2
        if 5*T - k*b > KQ: break
        r = 0
        while 5*(T + 3*r*r) - k*b <= KQ + 50:
            for (e, s, ty) in ((T + 3*r*r + 3*b*r + r, (-1)**b, 'P'),) + (((T + 3*r*r + 3*b*r - r - b, -(-1)**b, 'M'),) if r >= 1 else ()):
                for sg in ([0] if b == 0 else [1, -1]):
                    E = 5*e + sg*k*b
                    if 0 <= E <= KQ: out.append((E, s, ('M', b, r, ty, sg)))
            r += 1
    return out
tg = sorted(terms_G(1, 2))[:12]; tm = sorted(terms_M(1))[:12]
print("G12 terms:", tg); print("M1 terms:", tm)
from collections import defaultdict
def match(Gt, Mt, label):
    byE = defaultdict(list)
    for t in Gt: byE[t[0]].append(t)
    pairs = []
    for (E, s, lab) in Mt:
        cands = [g for g in byE[E] if g[1] == -s]
        if len(cands) == 1: pairs.append((lab, cands[0][2]))
    print(label, len(pairs), "unique pairs")
    pat = defaultdict(list)
    for m, g in pairs:
        _, b, r, ty, sg = m; _, a, r2, ty2 = g
        pat[(ty, sg, ty2)].append((b, r, a, r2))
    for key, v in pat.items():
        v = sorted(v)[:8]
        print(" ", key, v)
match(terms_G(1, 2), terms_M(1), "class1 orbit2 vs M1")
G31 = [(t[0] + 1, t[1], t[2]) for t in terms_G(3, 1)]
match(G31, terms_M(2), "class3 orbit1 (shifted) vs M2")
