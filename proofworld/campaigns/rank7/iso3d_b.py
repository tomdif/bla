"""Find isometries of Q = 2Y² − 6A² + 3W² mapping LHS points (J71·G(1,1)) to RHS points (J72·M1) with opposite sign."""
import random, itertools
from fractions import Fraction as Fr
from collections import defaultdict
NE = 70
def T(a): return a * (a + 1) // 2
L = []   # (A, Y, W, sign)
for a in range(0, 200):
    if min(a % 7, (-a) % 7) != 1: continue
    for r in range(0, 100):
        for ty, Y in (('P', 6 * r + 3 * a + 1), ('M', 6 * r + 3 * a - 1)):
            if ty == 'M' and r == 0: continue
            e = T(a) + 3 * r * r + 3 * a * r + (r if ty == 'P' else -r - a)
            if e % 7 != 1 or (e - 1) // 7 > NE: continue
            for m in range(-30, 31):
                eJ = 7 * m * (m - 1) // 2 + m
                if eJ < 0: continue
                E = (e - 1) // 7 + eJ
                if E <= NE:
                    L.append((a, Y, 14 * m - 5, (-1) ** a * (1 if ty == 'P' else -1) * (-1) ** m, E))
R = []
for b in range(0, 60):
    for r in range(0, 60):
        for ty, y in (('P', 6 * r + 3 * b + 1), ('M', 6 * r + 3 * b - 1)):
            if ty == 'M' and r == 0: continue
            eB = T(b) + 3 * r * r + 3 * b * r + (r if ty == 'P' else -r - b)
            for sg in ([0] if b == 0 else [1, -1]):
                eM = 7 * eB + sg * b
                if eM > NE: continue
                for m in range(-30, 31):
                    eJ = 7 * m * (m - 1) // 2 + 2 * m
                    if eJ < 0 or eM + eJ > NE: continue
                    A = 7 * b - 2 * sg if b > 0 else -2
                    R.append((A, 7 * y, 14 * m - 3, -((-1) ** b * (1 if ty == 'P' else -1) * (-1) ** m), eM + eJ))
def Qf(v): return 2 * v[1] ** 2 - 6 * v[0] ** 2 + 3 * v[2] ** 2
print("#L", len(L), "#R", len(R))
# sanity: same norm offset
offs = set(Qf(p) - 168 * p[4] for p in L) | set(Qf(p) - 168 * p[4] for p in R)
print("norm offsets", offs)
Rby = defaultdict(list)
for p in R: Rby[p[4]].append(p)
Rset = {(p[0], p[1], p[2]): p[3] for p in R}
Lset = {(p[0], p[1], p[2]): p[3] for p in L}
def solve3(Ls, Rs):
    import numpy as np
    Lm = np.array([[p[0], p[1], p[2]] for p in Ls], dtype=float).T
    Rm = np.array([[p[0], p[1], p[2]] for p in Rs], dtype=float).T
    if abs(np.linalg.det(Lm)) < 1e-9: return None
    return Rm @ np.linalg.inv(Lm)
import numpy as np
Qm = np.diag([-6.0, 2.0, 3.0])
Lby = defaultdict(list)
for p in L: Lby[p[4]].append(p)
results = {}
random.seed(2)
small = [p for p in L if p[4] <= 30]
for it in range(400000):
    Ls = random.sample(small, 3)
    mode = random.choice(('LR', 'LL'))
    pool = Rby if mode == 'LR' else Lby
    Rs = [random.choice(pool[p[4]]) if pool[p[4]] else None for p in Ls]
    if None in Rs: continue
    M = solve3(Ls, Rs)
    if M is None or not np.allclose(M.T @ Qm @ M, Qm, atol=1e-6): continue
    key = (mode, tuple(np.round(M, 4).flatten()))
    if key in results: continue
    target = Rset if mode == 'LR' else Lset
    good = 0
    for p in L:
        v = M @ np.array(p[:3], dtype=float)
        vr = tuple(int(round(x)) for x in v)
        if np.allclose(v, vr, atol=1e-6) and vr in target:
            want = p[3] if mode == 'LR' else -p[3]
            if target[vr] == want: good += 1
    results[key] = good
top = sorted(results.items(), key=lambda kv: -kv[1])[:12]
for (mode, key), g in top:
    print(mode, g, [Fr(x).limit_denominator(60) for x in key])
