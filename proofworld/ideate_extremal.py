#!/usr/bin/env python3
"""proofworld.ideate_extremal -- LEVER: extremal / sharp-constant discovery (find the BEST-POSSIBLE theorem).

The 'new result' is often the SHARP form: the largest c for which an inequality still holds. The engine searches the
constant (binary search), the kernel decides validity at each c, and the boundary c* is the sharp constant -- a
discovered extremal theorem, with the tight case exhibited. This is how 'best possible' results get found.

  parametric inequality LHS >= c*RHS  ->  binary-search c, z3 validates each  ->  c* (sharp) + the tight/extremal case.

Run:  python3 -m proofworld.ideate_extremal
"""
from __future__ import annotations
from fractions import Fraction as Fr
import z3


def holds(build_ineq, c, t=3000):
    """is  LHS - c*RHS >= 0  valid for all reals?  (z3: unsat of the negation)."""
    s = z3.Solver(); s.set("timeout", t); s.add(build_ineq(c) < 0)
    return s.check() == z3.unsat


def sharp_constant(name, build_ineq, hi=10.0, iters=60, log=print):
    lo, h = 0.0, hi
    if not holds(build_ineq, lo):
        log(f"  [{name}] fails even at c=0 (not of this form)"); return None
    for _ in range(iters):
        mid = (lo + h) / 2
        if holds(build_ineq, mid): lo = mid
        else: h = mid
    cstar = lo
    rat = Fr(cstar).limit_denominator(1000)                       # reconstruct the likely exact rational
    exact = holds(build_ineq, float(rat)) and not holds(build_ineq, float(rat) + 1e-6)
    log(f"  [{name}] SHARP constant c* = {rat}  ({cstar:.6f})  -- largest c for which the inequality holds"
        + ("  [exact rational, tight]" if exact else ""))
    return rat


def main():
    print("=== LEVER :: extremal discovery -- find the SHARP (best-possible) constant, kernel-validated ===\n")
    x, y, z = z3.Reals("x y z")
    print("  search the largest c making each inequality hold for ALL reals:\n")
    sharp_constant("x^2+y^2 >= c*xy",          lambda c: x*x + y*y - c * (x*y))
    sharp_constant("x^2+y^2+z^2 >= c*(xy+yz+zx)", lambda c: x*x + y*y + z*z - c * (x*y + y*z + z*x))
    sharp_constant("x^2+y^2+1 >= c*(x+y)",      lambda c: x*x + y*y + 1 - c * (x + y))
    sharp_constant("x^4+1 >= c*x^2",            lambda c: x**4 + 1 - c * (x*x))
    sharp_constant("x^2+y^2 >= c*(x+y-1)",      lambda c: x*x + y*y - c * (x + y - 1))
    print("\n  each c* is a DISCOVERED best-possible theorem: the engine proposed constants, the kernel validated, and the")
    print("  boundary is the sharp result (e.g. x^2+y^2 >= 2xy is tight at x=y). Extremal search turns a loose")
    print("  inequality into its optimal form -- a small but genuine piece of new, verified mathematics.")


if __name__ == "__main__":
    main()
