"""Numeric planted-truth / held-out check of the B2 Macdonald (Dyson) form of (q;q)_inf^10, and of the
mod-11 consequences, before any formal work. Exact integer arithmetic throughout."""
import sys, os, itertools
from fractions import Fraction
sys.path.insert(0, os.path.expanduser("~/bla"))
from proofworld import numeric

N = 400
def qq_pow(e, N):
    c = [0] * (N + 1); c[0] = 1
    for n in range(1, N + 1):
        for _ in range(e):
            for i in range(N, n - 1, -1): c[i] -= c[i - n]
    return c
Q10 = qq_pow(10, N)

def dyson(rx, ry, mod, K, N):
    """sum over x≡rx, y≡ry (mod `mod`) of x*y*(x^2-y^2) q^((x^2+y^2-5)/12), divided by K."""
    c = [Fraction(0)] * (N + 1)
    B = int((12 * N + 5) ** 0.5) + 2
    for x in range(-B, B + 1):
        if x % mod != rx % mod: continue
        for y in range(-B, B + 1):
            if y % mod != ry % mod: continue
            e2 = x * x + y * y - 5
            if e2 < 0 or e2 % 12: continue
            e = e2 // 12
            if e <= N: c[e] += Fraction(x * y * (x * x - y * y), K)
    return c

# --- search the residue classes / normalisation on the SEEN coefficients only (first 40)
seen = {n: Q10[n] for n in range(0, 40)}
held = {n: Q10[n] for n in range(40, N + 1)}
hits = []
for mod in (6, 12):
    for rx, ry in itertools.product(range(mod), repeat=2):
        raw = dyson(rx, ry, mod, 1, 39)
        if raw[0] == 0: continue
        K = raw[0] / Q10[0]
        if all(raw[n] / K == seen[n] for n in seen):
            hits.append((mod, rx, ry, K))
print("forms matching the 40 SEEN coefficients:", hits[:6], f"({len(hits)} total)")
mod, rx, ry, K = hits[0]
full = dyson(rx, ry, mod, K, N)
law = lambda n: full[n]
v = numeric.kill_test(law, held, min_support=50)
print(f"held-out check of (q;q)^10 = (1/{K}) Σ_(x≡{rx},y≡{ry} mod {mod}) xy(x²-y²) q^((x²+y²-5)/12):",
      v.status, f"({v.checked} unseen coefficients, n=40..{N})")

# --- the mod-11 consequences, numerically
bad10 = [a for a in range(N + 1) if a % 11 == 6 and Q10[a] % 11]
print("coeff (q;q)^10 at a≡6 (mod 11) nonzero mod 11:", bad10 or "none", f"(a≤{N})")
P = [0] * (N + 1); P[0] = 1
for k in range(1, N + 1):
    for i in range(k, N + 1): P[i] += P[i - k]
print("p(11n+6) mod 11 nonzero:", [11 * n + 6 for n in range((N - 6) // 11 + 1) if P[11 * n + 6] % 11] or "none",
      "| p(6), p(17), p(28) =", P[6], P[17], P[28])
# heart: a ≡ 6 (mod 11)  =>  x^2+y^2 = 12a+5 ≡ 0 (mod 11)  =>  x ≡ y ≡ 0 (mod 11)
heart = all(((x * x + y * y) % 11 != 0) or (x % 11 == 0 and y % 11 == 0) for x in range(11) for y in range(11))
print("heart over ZMod 11 (x²+y²=0 ⇒ x=y=0):", heart, "| 12*6+5 mod 11 =", (12 * 6 + 5) % 11)
