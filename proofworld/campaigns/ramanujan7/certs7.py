import sympy as sp, json
x,y,z,t,T,w,s = sp.symbols('x y z t T w s')
L = x - t*y - t**2 + t**5*z
Nres = sp.expand(sp.resultant(s**7 - t**7, x - s*y - s**2 + s**5*z, s))
P, rem = sp.div(Nres, sp.expand(L), t); assert rem == 0
P = sp.expand(P)
# relations (T = t^7 in t-world)
R = {'R2': x**2 - x*y**2 - T*z, 'R4': x - y**2 - T*y*z**2, 'R5': x**2*z - y - T*z**2, 'R1': x*y*z - 1}
Rl = list(R.values())
G = sp.groebner(Rl, z, y, x, T, order='grevlex')
def toT(e):
    pe = sp.Poly(sp.expand(e), t); out = {}
    for (k,), c in pe.terms(): out[k % 7] = out.get(k % 7, 0) + c*T**(k//7)
    return {r: sp.expand(v) for r, v in out.items()}
Pc = toT(P)
Pn = {r: sp.expand(G.reduce(Pc.get(r, 0))[1]) for r in range(7)}
F = x**3*y + T*y**3*z - T**2*x*z**3 - 8*T
print("P5 normal form == 7(F+7T):", sp.expand(Pn[5] - 7*(F + 7*T)) == 0)
Pn[5] = sp.expand(7*(F + 7*T))
Nt = toT(Nres); assert set(Nt) == {0}
N = Nt[0]
print("N =", N)
# certificate (a): L * sum t^r Pn[r] - N(t^7)  in ideal, with T -> t^7
Q7 = sum(t**r * Pn[r] for r in range(7))
diff = sp.expand((L*Q7 - N).subs(T, t**7))
Rt = [sp.expand(r.subs(T, t**7)) for r in Rl]
q, r = sp.reduced(diff, Rt, z, y, x, t, order='grevlex')
print("cert (a) remainder:", r, " sizes:", [len(sp.Add.make_args(sp.expand(c))) for c in q])
# certificate (b): F^2 - N in ideal (T world)
qb, rb = sp.reduced(sp.expand(F**2 - N), Rl, z, y, x, T, order='grevlex')
print("cert (b) remainder:", rb, " sizes:", [len(sp.Add.make_args(sp.expand(c))) for c in qb])
json.dump({'Pn': {r: str(Pn[r]) for r in range(7)}, 'N': str(N), 'F': str(F),
           'certA': [str(c) for c in q], 'certB': [str(c) for c in qb]},
          open('/Users/thomasdifiore/bla/proofworld/campaigns/ramanujan7/certs7.json', 'w'))
for r_ in range(7): print(r_, len(sp.Add.make_args(Pn[r_])), Pn[r_] if len(str(Pn[r_])) < 200 else '')
# class-wise certificate in T-world
LQ = toT(sp.expand(L*Q7))   # but Q7 contains T; substitute T->t^7 first then reclass
LQ = toT(sp.expand((L*Q7).subs(T, t**7)))
cof = [0, 0, 0, 0]
ok = True
for r_ in range(7):
    D = sp.expand(LQ.get(r_, 0) - (N if r_ == 0 else 0))
    qq, rr = sp.reduced(D, Rl, z, y, x, T, order='grevlex')
    if rr != 0: ok = False; print("class", r_, "rem", rr)
    for i in range(4): cof[i] += t**r_ * qq[i].subs(T, t**7)
cof = [sp.expand(c) for c in cof]
chk = sp.expand(L*Q7.subs(T, t**7) - N.subs(T, t**7) - sum(c*r.subs(T, t**7) for c, r in zip(cof, Rl)))
print("classwise ok:", ok, "check:", chk == 0, "sizes:", [len(sp.Add.make_args(c)) for c in cof])
d = json.load(open('/Users/thomasdifiore/bla/proofworld/campaigns/ramanujan7/certs7.json'))
d['certA'] = [str(c) for c in cof]
json.dump(d, open('/Users/thomasdifiore/bla/proofworld/campaigns/ramanujan7/certs7.json', 'w'))
print("certB:", d['certB'])
