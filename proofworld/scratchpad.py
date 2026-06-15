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

def find_gram(p, vs, eps=0.0):
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
    Q = cp.Variable((n, n), symmetric=True)
    cons = [Q - eps * np.eye(n) >> 0]                                # Q >= eps*I (eps=0 -> PSD; eps>0 -> rounding room)
    cons += [cp.sum([Q[i, j] for (i, j) in pairs]) == pcoeff.get(mu, 0.0) for mu, pairs in groups.items()]
    ok = False
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            cp.Problem(cp.Minimize(0), cons).solve(solver=solver); ok = Q.value is not None
            if ok: break
        except Exception:
            continue
    if not ok: return None, None, "SDP infeasible"
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
    def rnd(x, t):
        return sp.Integer(0) if abs(float(x)) < t else sp.nsimplify(x, rational=True, tolerance=t)
    for tol in (3e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4):               # coarse->fine: coarse can snap onto the true rational
        Q0 = sp.Matrix(n, n, lambda i, j: rnd(Qnum[i, j], tol)); Q0 = (Q0 + Q0.T) / 2
        Q = Q0 if gram_ok(Q0) else _project_exact(p, vs, Q0, z)    # Peyrl-Parrilo: rational projection onto {p=z^TQz}
        if Q is None or not gram_ok(Q): continue
        sq = _rational_ldl(Q)                                      # exact PSD check (negative pivot -> reject)
        if sq:
            return [(w, sp.expand((l.T * zexpr)[0])) for w, l in sq]
    return None

def _project_exact(p, vs, Q0, z):
    """exact rational least-norm projection of Q0 onto the affine space {Q : p = z^T Q z}, via A,b matrices:
    vech(Qr) = vech(Q0) + A^T (A A^T)^{-1} (b - A vech(Q0)). Soundness is re-checked by gram_ok afterward."""
    n = len(z); idx = [(i, j) for i in range(n) for j in range(i, n)]
    P = sp.Poly(p, *vs); pc = {tuple(m): sp.Rational(c) for m, c in P.terms()}
    monos = set()
    for i in range(n):
        for j in range(n):
            monos.add(tuple(a + b for a, b in zip(z[i], z[j])))
    monos = sorted(monos)
    A = sp.zeros(len(monos), len(idx)); b = sp.zeros(len(monos), 1)
    for r, mu in enumerate(monos):
        b[r] = pc.get(mu, sp.Integer(0))
        for k, (i, j) in enumerate(idx):
            mult = (1 if tuple(a + c for a, c in zip(z[i], z[j])) == mu else 0)
            if i != j and tuple(a + c for a, c in zip(z[j], z[i])) == mu: mult += 1
            A[r, k] = mult
    v0 = sp.Matrix([Q0[i, j] for (i, j) in idx])
    try:
        corr = A.T * (A * A.T).inv() * (b - A * v0)
    except Exception:
        return None
    v = v0 + corr; Qr = sp.zeros(n, n)
    for k, (i, j) in enumerate(idx):
        Qr[i, j] = Qr[j, i] = sp.nsimplify(v[k])
    return Qr


# ---------------- certificate search: direct SOS, then Positivstellensatz multipliers ----------------
def sos_certificate(p, vs, log=print):
    """find (m, squares) with m*p = sum w_k (l_k.z)^2, m a positive multiplier. m=1 is direct SOS; otherwise
    m=(1+sum x^2)^k > 0 (Positivstellensatz) so m*p >= 0 and m > 0 imply p >= 0."""
    s = sp.expand(1 + sum(v ** 2 for v in vs))
    for m in [sp.Integer(1), s, sp.expand(s ** 2)]:
        q = sp.expand(m * p)
        for eps in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):                   # increasing margin -> rounding room (Peyrl-Parrilo)
            Qnum, z, err = find_gram(q, vs, eps)
            if Qnum is None:
                if eps == 0.0: break                               # not SOS at this multiplier -> next multiplier
                continue
            squares = exact_sos(q, vs, Qnum, z)
            if squares:
                return m, squares
    return None, None


# ---------------- kernel verifications ----------------
def verify_z3(p, vs, m, squares):
    zv = {v.name: z3.Real(v.name) for v in vs}
    rhs = z3.Sum(*[s2z(w, zv) * (s2z(f, zv) ** 2) for w, f in squares]) if squares else z3.RealVal(0)
    s = z3.Solver(); s.set("timeout", 5000); s.add(s2z(m, zv) * s2z(p, zv) - rhs != 0)   # exact identity m*p == SOS
    identity = s.check() == z3.unsat
    return identity and all(w >= 0 for w, _ in squares)

def verify_lean_batch(items):
    """items: (name, m_lean, p_lean, [(w_lean, form_lean)]). Structured proof: the multiplier identity m*p = SOS by
    ring, m > 0 by positivity, then nlinarith. Handles direct SOS (m=1) and Positivstellensatz (m=1+sum x^2) uniformly.
    Each theorem is a 4-line block so we map errors to the right goal."""
    blocks = []
    for nm, mlean, plean, sq in items:
        vs = sorted(set(re.findall(r"[a-z]", plean + " " + mlean)))
        binders = " ".join(vs)
        sos = " + ".join(f"({w})*({f})^2" for w, f in sq) if sq else "0"
        hints = ", ".join(["h", "hm"] + [f"sq_nonneg ({f})" for _, f in sq])
        blocks.append(f"theorem {nm} ({binders} : ℝ) : {plean} ≥ 0 := by\n"
                      f"  have h : ({mlean}) * ({plean}) = {sos} := by ring\n"
                      f"  have hm : (0:ℝ) < {mlean} := by positivity\n"
                      f"  nlinarith [{hints}]")
    src = "import Mathlib.Tactic\n" + "\n".join(blocks) + "\n"
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "S.lean"); open(f, "w").write(src)
        try:
            r = subprocess.run(["lake", "env", "lean", f], cwd=LEAN_PROJECT, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return {it[0]: False for it in items}
    out = r.stdout + r.stderr
    bad = {int(m.group(1)) for m in re.finditer(r"S\.lean:(\d+):\d+: error:", out)}
    res, line = {}, 2                                              # 1 import line; each block is 4 lines
    for nm, _, _, _ in items:
        res[nm] = not any(line + k in bad for k in range(4)); line += 4
    return res


def lean_str(e):
    return str(e).replace("**", "^")


# ---------------- driver ----------------
def certify(name, poly_str, log=print):
    p = sp.expand(sp.sympify(poly_str)); vs = sorted(p.free_symbols, key=lambda s: s.name)
    log(f"\n=== {name}:  {poly_str} >= 0  ? ===")
    truth, cx = z3_truth(p, vs)
    if truth == "false":
        log(f"  PRE-SCREEN (z3): FALSE -- counterexample {cx}. No proof attempted (computation saved the effort)."); return None
    log(f"  PRE-SCREEN (z3): no counterexample (looks true) -> search for an SOS / Positivstellensatz certificate")
    m, squares = sos_certificate(p, vs)
    if not squares:
        log(f"  no certificate found at the tried degrees (direct SOS and (1+sum x^2)^<=2 multipliers)."); return None
    kind = "direct SOS" if m == 1 else f"Positivstellensatz, multiplier ({sp.simplify(m)})"
    pref = poly_str if m == 1 else f"({sp.simplify(m)})*({poly_str})"
    log(f"  COMPUTED [{kind}] (exact): {pref} = " + " + ".join(f"({sp.nsimplify(w)})*({f})^2" for w, f in squares))
    ok_z3 = verify_z3(p, vs, m, squares)
    log(f"  VERIFY z3 (exact identity m*p == SOS + weights>=0): {'PASS' if ok_z3 else 'FAIL'}")
    return {"name": re.sub(r'[^a-zA-Z0-9]', '_', name).lower(), "m_lean": lean_str(m), "p_lean": lean_str(p),
            "sq": [(lean_str(sp.nsimplify(w)), lean_str(f)) for w, f in squares], "z3": ok_z3}


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
        if c and c["z3"]: certified.append((c["name"], c["m_lean"], c["p_lean"], c["sq"]))
    print(f"\n=== FORMALIZE: Lean (multiplier identity by ring, m>0 by positivity, nlinarith) ({len(certified)} certs) ===")
    if certified:
        res = verify_lean_batch(certified)
        for nm, ok in res.items():
            print(f"  Lean {nm:10}: {'VERIFIED (nlinarith closed it with the discovered hints)' if ok else 'FAILED'}")
    print("\n  The SDP numerics found the SHAPE (which squares); soundness came ONLY from the exact symbolic identity")
    print("  and two independent kernels (z3 + Lean). Non-SOS and false cases were reported honestly, never faked.")


if __name__ == "__main__":
    main()
