"""Formal consistency of F(ζ7)·R_guess = H18(ζ7), class by class. Prints the residual theta identities."""
import ast, re, itertools
import numpy as np
import sympy as sp
J7, J1, J2, J3, q, z = sp.symbols('J7 J71 J72 J73 q z')
P1, P2, P3 = sp.symbols('Phi1 Phi2 Phi3')
Jsym = {'J7': J7, 'J71': J1, 'J72': J2, 'J73': J3}
LOW = 6
def _trunc(rs):
    f = [1] + [0] * LOW
    for n in range(1, LOW + 1):
        if n % 7 in rs:
            f = [f[i] - (f[i - n] if i >= n else 0) for i in range(LOW + 1)]
    return sum(c * q ** i for i, c in enumerate(f))
def _phitrunc(k):
    # Σ_n q^{7n²}/((q^k;q^7)_{n+1}(q^{7-k};q^7)_n): only n = 0 below q^7 -> 1/(1-q^k)
    return sum(q ** (k * t) for t in range(LOW // k + 1))
LOWSUB = {J7: _trunc({0}), J1: _trunc({0, 1, 6}), J2: _trunc({0, 2, 5}), J3: _trunc({0, 3, 4}),
          P1: _phitrunc(1), P2: _phitrunc(2), P3: _phitrunc(3)}
def lowser(expr):
    e = sp.series(expr.subs(LOWSUB), q, 0, LOW).removeO()
    return [sp.expand(e).coeff(q, i) for i in range(LOW)]
def term(name):
    if name == "1": return sp.Integer(1)
    m = re.match(r"q\^(-?\d+)\*(.*)", name)
    s, rest = int(m.group(1)), m.group(2)
    core = term0(rest)
    if s >= 0: return q ** s * core
    low = lowser(core)
    return q ** s * (core - sum(low[i] * q ** i for i in range(-s)))
def term0(rest):
    e = sp.Integer(1)
    mm = re.match(r"(J7\d)\*Phi(\d)", rest)
    if mm:
        k = int(mm.group(2))
        ph = {1: P1, 2: P2, 3: P3, 4: q * P3 + 1 - q, 5: q**3 * P2 + 1 - q**3, 6: q**5 * P1 + 1 - q**5}[k]
        return e * Jsym[mm.group(1)] * ph
    mm = re.match(r"Phi(\d)", rest)
    if mm:
        k = int(mm.group(1))
        return {1: P1, 2: P2, 3: P3}[k]
    ex = re.match(r"J7\^(-?\d+)J71\^(-?\d+)J72\^(-?\d+)J73\^(-?\d+)", rest)
    a, b, c, d = map(int, ex.groups())
    return e * J7**a * J1**b * J2**c * J3**d
def parse(line):
    d = ast.literal_eval(line[line.index('{'):])
    return sum(v * term(k) for k, v in d.items())
# R_guess
Rg = {k: sp.Integer(0) for k in range(7)}
for line in open('R7_dissection_fit.txt'):
    m = re.match(r"R(\d) ζ\^(\d): (FIT|ZERO)", line)
    if not m: continue
    k, i = int(m.group(1)), int(m.group(2))
    if m.group(3) == 'FIT': Rg[k] += z**i * parse(line)
# orbit pieces
G = {}
for line in open('orbit_pieces_fit.txt'):
    m = re.match(r"\((\d), (\d)\) FIT", line)
    if m: G[(int(m.group(1)), int(m.group(2)))] = parse(line)
print("pieces:", sorted(G))
E2 = {}
for line in open('E2_dissection_fit.txt'):
    m = re.match(r"\('E2', (\d)\) ζ\^0: FIT", line)
    if m: E2[int(m.group(1))] = parse(line)
phi7 = sum(z**i for i in range(7))
def redz(e):
    e = sp.expand(e)
    poly = sp.Poly(e, z)
    r = sp.rem(poly, sp.Poly(phi7, z))
    return sp.expand(r.as_expr())
s1 = z**2 + z**3 + z**4 + z**5; s2 = z**3 + z**4
def w(o): return z**o + z**(7 - o) - 2
res = {}
for c in range(7):
    H = E2[c] + sum(w(o) * G.get((c, o), 0) for o in (1, 2, 3))
    Rm1 = Rg[c - 1] if c >= 1 else q * Rg[c + 6]
    Rm3 = Rg[c - 3] if c >= 3 else q * Rg[c + 4]
    FR = J3 * Rg[c] + s1 * J2 * Rm1 - s2 * J1 * Rm3
    D = redz(FR - H)
    D = sp.expand(D)
    parts = {}
    for ph in (P1, P2, P3):
        parts[str(ph)] = sp.simplify(D.coeff(ph))
    rest = sp.simplify(D.subs({P1: 0, P2: 0, P3: 0}))
    parts['theta'] = rest
    res[c] = parts
    print(f"class {c}:", {k: (0 if v == 0 else sp.factor(v)) for k, v in parts.items()}, flush=True)
