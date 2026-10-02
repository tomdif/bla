import sympy as sp, time
x,y,z,t,w = sp.symbols('x y z t w')
L = lambda s: x - s*y - s**2 + s**5*z
t0=time.time()
prod = 1
for j in range(7):
    prod = sp.expand(prod * L(w**j*t))
    # reduce w^k -> w^(k mod 7)
    prod = sp.expand(sum(c*w**(k%7)*rest for (k,), c, rest in []) ) if False else prod
pw = sp.Poly(prod, w)
red = 0
for (k,), c in pw.terms(): red += c*w**(k % 7)
red = sp.expand(red)
phi = sum(w**i for i in range(7))
q, r = sp.div(sp.Poly(red, w), sp.Poly(phi, w))
r = sp.expand(r.as_expr())
print("remainder (norm):", sp.factor(r) if len(str(r)) < 400 else str(r)[:400])
print("terms in w-free remainder:", len(sp.Add.make_args(r)), " cofactor terms:", len(sp.Add.make_args(sp.expand(q.as_expr()))), time.time()-t0)
q1, r1 = sp.div(sp.Poly(prod - red, w), sp.Poly(w**7 - 1, w))
print("w^7-1 cofactor terms:", len(sp.Add.make_args(sp.expand(q1.as_expr()))), "rem0:", r1.is_zero)
# alternative: single cofactor wrt Phi7 directly (w^7-1 = (w-1)Phi7)
q2, r2 = sp.div(sp.Poly(sp.expand(prod - r), w), sp.Poly(phi, w))
print("direct Phi7 cofactor terms:", len(sp.Add.make_args(sp.expand(q2.as_expr()))), "rem0:", r2.is_zero)
def lean(e):
    s = str(e).replace('**', '^')
    return s
cof = sp.expand(q2.as_expr())
lhs = " * ".join(f"(x - w^{j} * t * y - w^{(2*j)} * t^2 + w^{(5*j)} * t^5 * z)" for j in range(7))
open("/private/tmp/claude-501/-Users-thomasdifiore/b98906b4-449a-44a9-96bd-fd6647304d75/scratchpad/NormTest.lean","w").write(
f"""import Mathlib.Tactic
set_option maxHeartbeats 0 in
theorem norm7 {{R : Type*}} [CommRing R] (w x y z t : R)
    (hw : w ^ 6 + w ^ 5 + w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0) :
    {lhs} = {lean(r)} := by
  linear_combination ({lean(cof)}) * hw
""")
print("written; norm =", lean(r))
