exec(open('consist7.py').read().split("E2 = {}")[0])
phi7 = sum(z**i for i in range(7))
def redz(e):
    e = sp.expand(e)
    return sp.expand(sp.rem(sp.Poly(e, z), sp.Poly(phi7, z)).as_expr())
qq = sp.Symbol('qq')   # the base variable q (Q = qq^7 is `q`)
def Fz(a):
    s1 = sum(z**((a * e) % 7) for e in (2, 3, 4, 5)); s2 = sum(z**((a * e) % 7) for e in (3, 4))
    return J3 + qq * s1 * J2 - qq**3 * s2 * J1
prodF = redz(sp.expand(Fz(1) * Fz(2) * Fz(3)))
print("prod F free of z:", sp.Poly(prodF, z).degree() <= 0)
P = sp.Poly(sp.expand(prodF), qq)
E2b = {c: 0 for c in range(7)}
for (dg,), co in P.terms():
    E2b[dg % 7] += co * q ** (dg // 7)
EQ7 = J1 * J2 * J3 / J7**2
E2b = {c: sp.simplify(v / EQ7) for c, v in E2b.items()}
print({c: sp.factor(v) for c, v in E2b.items()})
s1 = z**2 + z**3 + z**4 + z**5; s2 = z**3 + z**4
def w(o): return z**o + z**(7 - o) - 2
T = J1**3 * J2 * q + J1 * J3**3 - J2**3 * J3
for c in range(7):
    H = E2b[c] + sum(w(o) * G.get((c, o), 0) for o in (1, 2, 3))
    Rm1 = Rg[c - 1] if c >= 1 else q * Rg[c + 6]
    Rm3 = Rg[c - 3] if c >= 3 else q * Rg[c + 4]
    D = redz(J3 * Rg[c] + s1 * J2 * Rm1 - s2 * J1 * Rm3 - H)
    D = sp.expand(D)
    parts = [sp.simplify(D.coeff(ph)) for ph in (P1, P2, P3)] + [sp.factor(sp.simplify(D.subs({P1: 0, P2: 0, P3: 0})))]
    print(f"class {c}: Φ-parts {parts[:3]} theta {parts[3]}", flush=True)
