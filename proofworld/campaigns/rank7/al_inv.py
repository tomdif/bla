exec(open('al_probe.py').read().split("for c in range(0, 5):")[0])
from collections import Counter, defaultdict
def Lpts(c, B):
    pts = {}
    for s1 in range(-B, B):
        X = 6 * s1 - 1
        for r in range(0, B):
            for ty, Y in (('P', 6 * r + 3 * c + 1), ('M', 6 * r + 3 * c - 1)):
                if ty == 'M' and r == 0: continue
                w = (-1) ** (s1 + c) * (1 if ty == 'P' else -1)
                pts[(X, Y)] = pts.get((X, Y), 0) + w
    return pts
def Rpts(c, B):
    pts = {}
    for s in range(-B, B):
        t = c - s
        ns = range(0, B) if t >= 0 else range(-B, 0)
        for n in ns:
            w = (-1) ** (s + n) if t >= 0 else -(-1) ** (s + n)
            U, V = 2 * (s - n) - 1, 2 * n + c
            pts[(U, V)] = pts.get((U, V), 0) + w
    return pts
c = 1; B = 25; NB = 24 * 60
L = {k: v for k, v in Lpts(c, B).items() if k[0] ** 2 + 2 * k[1] ** 2 <= NB}
R = {k: v for k, v in Rpts(c, B).items() if 3 * (k[0] ** 2 + 2 * k[1] ** 2) <= NB}
stats = Counter()
for (U, V), w in R.items():
    for name, (X, Y) in (('+', (U - 2 * V, U + V)), ('-', (U + 2 * V, U - V)), ('+c', (U - 2 * V, -(U + V))), ('-c', (U + 2 * V, -(U - V))),
                         ('n+', (-(U - 2 * V), U + V)), ('n-', (-(U + 2 * V), U - V))):
        if (X, Y) in L:
            stats[(name, 'same' if L[(X, Y)] == w else 'opp')] += 1
print(len(L), len(R)); print(sorted(stats.items()))
