"""Decompose each z^c coefficient of H_m into arithmetic-progression partial thetas Σ_{t≥0} s·q^{αt²+βt+γ}."""
import sys
exec(open('hecke_m.py').read().split("for c in sorted(byc):")[0].replace("m = int(sys.argv[1]); N = int(sys.argv[2])", "m = int(sys.argv[1]); N = int(sys.argv[2])"))
def decompose(d):
    terms = sorted(d.items()); used = set(); out = []
    while True:
        rest = [(n, v) for n, v in terms if n not in used]
        if not rest: break
        n0, v0 = rest[0]
        best = None
        for a in (1, 2, 3, 4, 5, 6):        # α (half-integers allowed via 2α)
            for b2 in range(-20, 30):
                # progression n_t = n0 + a t² + b2 t
                seq = [n0 + a * t * t + b2 * t for t in range(0, 12)]
                seq = [x for x in seq if x <= N]
                if len(seq) < 2 or seq[1] <= n0: continue
                if all(d.get(x, 0) == v0 for x in seq) and all(x not in used for x in seq):
                    if best is None or len(seq) > len(best[2]): best = (a, b2, seq)
        if best is None: out.append(("single", n0, v0)); used.add(n0); continue
        a, b2, seq = best
        out.append((f"{v0:+d}·Σ q^({a}t²+{b2}t+{n0})", len(seq))); used.update(seq)
    return out
for c in range(-5, 8):
    print(c, decompose(byc[c]))
