#!/usr/bin/env python3
"""proofworld.scratchpad -- a computational scratchpad: COMPUTE the shape of a proof, then VERIFY it with the kernel.

Flagship routine: certify a polynomial inequality p(x) >= 0 by finding a Sum-Of-Squares certificate.
  1. PRE-SCREEN (z3): is p >= 0 even true? If z3 finds a counterexample, report FALSE -- never attempt a proof.
  2. COMPUTE (cvxpy SDP, UNTRUSTED): find a Gram matrix Q >= 0 with p = z^T Q z (z = monomial basis). Floating point.
  3. EXACTIFY: round Q to an EXACT rational certificate; if it doesn't reproduce p exactly, PROJECT onto the affine
     constraint space in rational arithmetic; LDL^T -> a weighted sum of squares p = sum_k w_k * (l_k . z)^2, w_k>=0.
  4. VERIFY (two independent kernels):
       * z3 checks the EXACT identity p - sum_k w_k (l_k.z)^2 == 0 over the reals, and every w_k >= 0;
       * Lean closes `p >= 0` by `nlinarith [sq_nonneg (l_k . z), ...]` -- the discovered squares become the hints.
The numerics are untrusted heuristics that find the SHAPE; soundness comes ONLY from the exact symbolic identity and
the two kernels. A non-SOS-but-true polynomial (Motzkin) is reported honestly; a false one is refuted, not faked.

Run:  python3 -m proofworld.scratchpad
"""
from __future__ import annotations
import os, re, subprocess, tempfile
import numpy as np, sympy as sp, z3

HERE = os.path.dirname(os.path.abspath(__file__)); LEAN_PROJECT = os.path.join(HERE, "lean")


# ---------------- sympy <-> z3 ----------------
def s2z(e, zv):
    e = sp.sympify(e)
    if e.is_Integer: return z3.RealVal(int(e))
    if e.is_Rational: return z3.Q(int(e.p), int(e.q))
    if e.is_Symbol: return zv[e.name]
    if e.is_Add: return z3.Sum(*[s2z(a, zv) for a in e.args])
    if e.is_Mul:
        r = z3.RealVal(1)
        for a in e.args: r = r * s2z(a, zv)
        return r
    if e.is_Pow:
        b = s2z(e.base, zv); n = int(e.exp); return b ** n
    raise ValueError(f"cannot convert {e}")


def z3_truth(p, vs):
    """pre-screen: is p >= 0 for all reals? return ('true', None) or ('false', counterexample) or ('unknown', None)."""
    zv = {v.name: z3.Real(v.name) for v in vs}; s = z3.Solver(); s.set("timeout", 5000)
    s.add(s2z(p, zv) < 0)
    r = s.check()
    return ("false", str(s.model())) if r == z3.sat else ("true", None) if r == z3.unsat else ("unknown", None)


# ---------------- SDP: find a Gram matrix (untrusted numerics) ----------------
def monomials(vs, d):
    from itertools import product
    out = []
    for exps in product(range(d + 1), repeat=len(vs)):
        if sum(exps) <= d:
            out.append(exps)
    return sorted(out, key=lambda e: (sum(e), e))

def find_gram(p, vs):
    import cvxpy as cp
    P = sp.Poly(p, *vs); deg = P.total_degree()
    if deg % 2: return None, None, "odd degree -> not SOS"
    d = deg // 2
    homog = len({sum(m) for m, _ in P.terms()}) == 1                 # tight basis for homogeneous polynomials
    z = [m for m in monomials(vs, d) if (sum(m) == d if homog else True)]
    n = len(z); pcoeff = {tuple(m): float(c) for m, c in P.terms()}
    groups = {}
    for i in range(n):
        for j in range(n):
            mu = tuple(a + b for a, b in zip(z[i], z[j]))
            groups.setdefault(mu, []).append((i, j))
    Q = cp.Variable((n, n), PSD=True)                                # Q >= 0 (allows rank-deficient certificates)
    cons = [cp.sum([Q[i, j] for (i, j) in pairs]) == pcoeff.get(mu, 0.0) for mu, pairs in groups.items()]
    ok = False
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            cp.Problem(cp.Minimize(0), cons).solve(solver=solver); ok = Q.value is not None
            if ok: break
        except Exception:
            continue
    if not ok: return None, None, "SDP infeasible (not a sum of squares at this degree)"
    return np.array(Q.value), z, None


# ---------------- exactify: exact rational certificate via affine projection + rational LDL ----------------
def _rational_ldl(Q):
    """LDL of an exact rational symmetric PSD matrix, handling SINGULAR (rank-deficient) cases. Returns weighted
    squares [(d_k, l_k)] with Q = sum_k d_k l_k l_k^T, d_k >= 0; or None if a pivot is negative (not PSD)."""
    n = Q.shape[0]; Q = Q.copy(); out = []
    for k in range(n):
        d = Q[k, k]
        if d == 0:
            if any(Q[i, k] != 0 for i in range(n)): return None     # zero pivot but nonzero col -> indefinite
            continue
        if d < 0: return None
        l = sp.Matrix([Q[i, k] / d for i in range(n)])
        out.append((d, l))
        for i in range(n):
            for j in range(n):
                Q[i, j] = sp.nsimplify(Q[i, j] - d * l[i] * l[j])
    return out

def exact_sos(p, vs, Qnum, z):
    n = len(z); zexpr = sp.Matrix([sp.prod([v ** e for v, e in zip(vs, m)]) for m in z])
    gram_ok = lambda Q: sp.expand(p - (zexpr.T * Q * zexpr)[0]) == 0
    # round, snapping near-zero entries to exactly 0 to kill SDP noise
    def rnd(x):
        r = sp.nsimplify(x, rational=True, tolerance=1e-6)
        return sp.Integer(0) if abs(float(x)) < 1e-6 else r
    Q = sp.Matrix(n, n, lambda i, j: rnd(Qnum[i, j])); Q = (Q + Q.T) / 2
    if not gram_ok(Q):
        Q = _project_exact(p, vs, Q, zexpr, n)
        if Q is None or not gram_ok(Q): return None
    sq = _rational_ldl(Q)
    if sq is None: return None
    return [(w, sp.expand((l.T * zexpr)[0])) for w, l in sq] or None

def _project_exact(p, vs, Q0, zexpr, n):
    """force exact constraints p = z^T Q z: pin Q's free entries to the rounded values, solve the rest exactly."""
    idx = [(i, j) for i in range(n) for j in range(i, n)]
    syms = sp.symbols(f"qv0:{len(idx)}"); Q = sp.zeros(n, n)
    for s, (i, j) in zip(syms, idx):
        Q[i, j] = s; Q[j, i] = s
    eqs = [sp.Eq(co, 0) for _, co in sp.Poly(sp.expand(p - (zexpr.T * Q * zexpr)[0]), *vs).terms()]
    sol = sp.solve(eqs, syms, dict=True)
    if not sol: return None
    sol = sol[0]
    subs = {s: Q0[idx[k][0], idx[k][1]] for k, s in enumerate(syms) if s not in sol}   # free -> rounded
    Qr = sp.zeros(n, n)
    for s, (i, j) in zip(syms, idx):
        val = sol[s].subs(subs) if s in sol else subs.get(s, 0)
        Qr[i, j] = Qr[j, i] = sp.nsimplify(val)
    return Qr


# ---------------- kernel verifications ----------------
def verify_z3(p, vs, squares):
    zv = {v.name: z3.Real(v.name) for v in vs}
    rhs = z3.Sum(*[s2z(w, zv) * (s2z(f, zv) ** 2) for w, f in squares]) if squares else z3.RealVal(0)
    s = z3.Solver(); s.set("timeout", 5000); s.add(s2z(p, zv) - rhs != 0)
    identity = s.check() == z3.unsat
    weights_ok = all(w >= 0 for w, _ in squares)
    return identity and weights_ok

def verify_lean_batch(items):
    """items: list of (name, p_lean, [forms_lean]). One Lean file: nlinarith with the discovered square hints."""
    body = ""
    for nm, plean, forms in items:
        vs = sorted(set(re.findall(r"[a-z]", plean)))
        binders = " ".join(vs)
        hints = ", ".join(f"sq_nonneg ({f})" for f in forms)
        body += f"theorem {nm} ({binders} : ℝ) : {plean} ≥ 0 := by nlinarith [{hints}]\n"
    src = "import Mathlib.Tactic\n" + body
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "S.lean"); open(f, "w").write(src)
        try:
            r = subprocess.run(["lake", "env", "lean", f], cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=200)
        except subprocess.TimeoutExpired:
            return {nm: False for nm, _, _ in items}
    out = r.stdout + r.stderr
    bad = {int(m.group(1)) for m in re.finditer(r"S\.lean:(\d+):\d+: error:", out)}
    return {items[k][0]: (2 + k) not in bad for k in range(len(items))}   # 1 import line + theorem k on line 2+k


def lean_str(e):
    return str(e).replace("**", "^")


# ---------------- driver ----------------
def certify(name, poly_str, log=print):
    p = sp.expand(sp.sympify(poly_str)); vs = sorted(p.free_symbols, key=lambda s: s.name)
    log(f"\n=== {name}:  {poly_str} >= 0  ? ===")
    truth, cx = z3_truth(p, vs)
    if truth == "false":
        log(f"  PRE-SCREEN (z3): FALSE -- counterexample {cx}. No proof attempted (computation saved the effort)."); return None
    log(f"  PRE-SCREEN (z3): no counterexample (looks true) -> search for an SOS certificate")
    Qnum, z, err = find_gram(p, vs)
    if Qnum is None:
        log(f"  SDP: {err}.  (true but NOT sum-of-squares -- would need Positivstellensatz multipliers; honest stop)"); return None
    squares = exact_sos(p, vs, Qnum, z)
    if not squares:
        log(f"  EXACTIFY: could not extract an exact rational certificate (numerics too far from a rational PSD point)."); return None
    log(f"  COMPUTED SOS shape (exact): {poly_str} = " + " + ".join(f"({sp.nsimplify(w)})*({f})^2" for w, f in squares))
    ok_z3 = verify_z3(p, vs, squares)
    log(f"  VERIFY z3 (exact identity + weights>=0): {'PASS' if ok_z3 else 'FAIL'}")
    return {"name": re.sub(r'[^a-zA-Z0-9]', '_', name).lower(), "p_lean": lean_str(p),
            "forms": [lean_str(f) for _, f in squares], "z3": ok_z3}


def main():
    print("=== proofworld.scratchpad :: compute the proof SHAPE (SOS via SDP), then VERIFY with the kernel ===")
    goals = [
        ("amgm2", "x**2 + y**2 - 2*x*y"),
        ("schur3", "x**2 + y**2 + z**2 - x*y - y*z - z*x"),
        ("quartic", "x**4 + y**4 - x**2*y**2"),
        ("biquad", "x**4 + 2*x**2*y**2 + y**4"),
        ("motzkin", "x**4*y**2 + x**2*y**4 + 1 - 3*x**2*y**2"),     # nonneg but NOT SOS (honest stop)
        ("false_one", "x**2 + y**2 - 3*x*y"),                       # FALSE (z3 refutes)
    ]
    certified = []
    for nm, ps in goals:
        c = certify(nm, ps)
        if c and c["z3"]: certified.append((c["name"], c["p_lean"], c["forms"]))
    print(f"\n=== FORMALIZE: Lean nlinarith with the SDP-discovered squares as hints ({len(certified)} certs) ===")
    if certified:
        res = verify_lean_batch(certified)
        for nm, ok in res.items():
            print(f"  Lean {nm:10}: {'VERIFIED (nlinarith closed it with the discovered hints)' if ok else 'FAILED'}")
    print("\n  The SDP numerics found the SHAPE (which squares); soundness came ONLY from the exact symbolic identity")
    print("  and two independent kernels (z3 + Lean). Non-SOS and false cases were reported honestly, never faked.")


if __name__ == "__main__":
    main()
