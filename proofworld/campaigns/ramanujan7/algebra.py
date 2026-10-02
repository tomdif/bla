import sympy as sp
x,y,z,t,Q,w = sp.symbols('x y z t Q w')
L = x - t*y - t**2 + t**5*z
def classes(expr):
    e = sp.Poly(sp.expand(expr), t)
    out = {}
    for (k,), c in e.terms():
        out.setdefault(k % 7, 0)
        out[k % 7] += c * t**k
    return {r: sp.expand(v) for r, v in out.items()}
C3 = classes(L**3)
for r in sorted(C3): print("cube class", r, ":", sp.factor(C3[r]))
s = sp.symbols('s')
Ls = x - s*y - s**2 + s**5*z
Nres = sp.expand(sp.resultant(s**7 - t**7, Ls, s))
P, rem = sp.div(Nres, sp.expand(L), t)
assert rem == 0
P = sp.expand(P)
rels = [x**2 - x*y**2 - Q*z, x - y**2 - Q*y*z**2, x**2*z - y - Q*z**2, x*y*z - 1]
G = sp.groebner(rels, z, y, x, Q, t, order='grevlex')
def red(e):
    e = sp.expand(e).subs(t**7, Q)
    e = sp.expand(e)
    # replace t^k with t^(k mod 7) Q^(k//7)
    pe = sp.Poly(sp.expand(e), t); out = 0
    for (k,), c in pe.terms(): out += c * t**(k % 7) * Q**(k//7)
    return sp.expand(G.reduce(sp.expand(out))[1])
Nr = red(Nres)
print("norm reduced:", sp.factor(Nr))
P5 = sum(c*t**k for (k,), c in sp.Poly(P, t).terms() if k % 7 == 5)
print("P class5 reduced:", sp.factor(red(P5)))
print("G:", G.exprs)
F = x**3*y + Q*y**3*z - Q**2*x*z**3 - 8*Q
print("F^2 - N reduces to:", red(F**2 - Nres))
# cofactor certificate for F^2 - N in ideal
q, r = sp.reduced(sp.expand(F**2 - Nr), G.exprs, z, y, x, Q, t, order='grevlex')
print("F^2-Nr remainder", r)
