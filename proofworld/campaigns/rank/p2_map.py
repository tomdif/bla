"""Check the explicit bijection for  dis5_c(Σ_{a≡±2k} Hblk_a)·q^{sh} + M_k − J_{5,k} = 0."""
from collections import Counter
def T(a): return a*(a+1)//2
def eG(a, r, ty): return T(a) + 3*r*r + 3*a*r + r if ty == 'P' else T(a) + 3*r*r + 3*a*r - r - a
def sG(a, ty): return (-1)**a * (1 if ty == 'P' else -1)
def check(k, c, sh, rho, B=40, R=40):
    # G terms (a,r,ty), class c, orbit ±2k
    G = {}
    for a in range(0, 5*B):
        if a % 5 not in ((2*k) % 5, (-2*k) % 5): continue
        for r in range(0, 5*R):
            for ty in ('P', 'M'):
                if ty == 'M' and r == 0: continue
                e = eG(a, r, ty)
                if e % 5 == c: G[(a, r, ty)] = ((e - c)//5 + sh, sG(a, ty))
    hit = Counter(); bad = 0; jl = []
    for b in range(0, B):
        for r in range(0, R):
            for ty in ('P', 'M'):
                if ty == 'M' and r == 0: continue
                for sg in ([0] if b == 0 else [1, -1]):
                    E = 5*eG(b, r, ty) + sg*k*b
                    s = sG(b, ty)
                    if sg == 1: a = 5*b - 2*k; rr = 5*r + (rho[0] if ty == 'P' else rho[2]); ty2 = 'M' if ty == 'P' else 'P'
                    else: a = 5*b + 2*k; rr = 5*r + (rho[1] if ty == 'P' else rho[3]); ty2 = 'M' if ty == 'P' else 'P'
                    if ty2 == 'M' and rr <= 0 or rr < 0:
                        jl.append((E, s, (b, r, ty, sg))); continue
                    key = (a, rr, ty2)
                    if key not in G or G[key][0] != E or G[key][1] != -s: bad += 1; continue
                    hit[key] += 1
    left = [(v[0], v[1], kk) for kk, v in G.items() if hit[kk] == 0 and kk[0] < 5*B - 30 and kk[1] < 5*R - 30]
    J = Counter()
    for m in range(-30, 31):
        J[(5*m*(m-1)//2 + k*m)] += (-1)**m
    tot = Counter()
    for e, s, _ in left + jl: tot[e] += s
    ok = all(tot[e] == J[e] for e in range(0, 200)) and all(v <= 1 for v in hit.values())
    print("k", k, "bad", bad, "dup", max(hit.values()), "leftover+J-type ok:", ok)
    print("  G leftovers:", sorted(left)[:8]); print("  M leftovers:", sorted(jl)[:8])
check(1, 1, 0, (2, 0, 0, -2))
check(2, 3, 1, (3, -1, 1, -3))
