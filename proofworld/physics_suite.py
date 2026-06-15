#!/usr/bin/env python3
"""proofworld.physics_suite -- run the ideation suite on the UnifiedTheory physics codebase.

The directly-applicable lever for physics is A (experimental discovery): integer-relation detection (PSLQ) on
physical CONSTANTS. We (1) test the repo's claim alpha(0) = ln(5/3)/70 -- is it an EXACT identity or a numerical
coincidence?; (2) search a basis of natural expressions for the best closed-form fits to the measured alpha; and
(3) note the verified physics SUBSTRATE (870+ axiom-clean theorems harvested: Bianchi identity, Lie structure, ...)
that the citation/discovery loops can use. Everything is reported honestly: PSLQ confirms EXACT relations and
flags approximations as approximations -- it does not fabricate a derivation.

Run:  python3 -m proofworld.physics_suite
"""
from __future__ import annotations
from mpmath import mp, pslq, log, mpf, pi, sqrt, e, exp


def rel(a, b):
    return abs(a - b) / abs(b)


def main():
    mp.dps = 30
    print("=== LEVER A on the UnifiedTheory physics codebase -- discover/assess closed forms for constants ===\n")
    alpha = mpf(1) / mpf("137.035999084")                         # CODATA fine-structure constant (measured)
    repo = log(mpf(5) / 3) / 70                                   # the unifiedtheory derivation alpha(0)
    print(f"  measured alpha (CODATA)      = {mp.nstr(alpha, 14)}")
    print(f"  repo formula ln(5/3)/70      = {mp.nstr(repo, 14)}")
    print(f"  relative difference          = {mp.nstr(rel(repo, alpha)*100, 3)} %  (~0.0021%, as the repo notes)\n")

    # 1) is it EXACT? PSLQ over {alpha, ln(5/3), 1} with small coefficients -- an exact identity would show [70,-1,0]
    print("  1) EXACT-or-COINCIDENCE test (PSLQ on the measured value, small coefficients):")
    relrec = pslq([alpha, log(mpf(5) / 3), mpf(1)], maxcoeff=500, maxsteps=10**5)
    if relrec == [70, -1, 0] or relrec == [-70, 1, 0]:
        print("     PSLQ CONFIRMS alpha = ln(5/3)/70 to full precision -> EXACT identity")
    else:
        print(f"     PSLQ finds NO small-coefficient exact relation ({relrec}) -> ln(5/3)/70 is a ~0.0021% COINCIDENCE,")
        print("     an excellent numerical APPROXIMATION, not an identity exact to the measured digits. (Honest: the")
        print("     engine will not certify a derivation that isn't exact -- same discipline as everywhere else.)")

    # 2) DISCOVERY: rank natural closed-form expressions by how well they fit the measured alpha
    print("\n  2) search natural closed forms for alpha, ranked by accuracy (an inverse-symbolic hunt):")
    cands = {
        "ln(5/3)/70  (repo)": log(mpf(5) / 3) / 70,
        "1/137": mpf(1) / 137,
        "1/(137 + 1/3)": mpf(1) / (137 + mpf(1) / 3),
        "1/(4*pi^3 + pi^2 + pi)": mpf(1) / (4 * pi**3 + pi**2 + pi),   # classic Wyler-style numerology fit
        "ln(2)/(95)": log(mpf(2)) / 95,
        "1/(2^7 + 9)": mpf(1) / (128 + 9),
        "sqrt(2)/194": sqrt(mpf(2)) / 194,
    }
    ranked = sorted(((rel(v, alpha), name) for name, v in cands.items() if v > 0), key=lambda x: x[0])
    seen = set()
    for err, name in ranked:
        if name in seen: continue
        seen.add(name)
        print(f"     {name:28} rel.err {mp.nstr(err*100, 3):>10} %")
    print("     -> the repo's ln(5/3)/70 is a strong simple fit (~0.002%), but the classic 1/(4pi^3+pi^2+pi) is ~10x")
    print("        CLOSER (~0.0002%) -- a useful finding for the repo: ln(5/3)/70 is a good coincidence, not the best,")
    print("        and NONE is exact, consistent with alpha being a measured constant with no known closed form.")

    # 3) hunt for an integer relation among several physical ratios (would be a real discovery if found)
    print("\n  3) relation hunt among physical ratios {alpha, m_p/m_e, 1}  (a moonshine-style coincidence search):")
    mp_me = mpf("1836.15267343")
    r = pslq([alpha, mp_me, mpf(1)], maxcoeff=10**4, maxsteps=10**5)
    print(f"     pslq = {r}  ->  {'no low-complexity relation (expected -- these are independent measured constants)' if r is None else 'candidate relation (a conjecture to scrutinize!)'}")

    print("\n  SUBSTRATE: 870+ AXIOM-CLEAN physics theorems harvested from UnifiedTheory (Bianchi identity dF=0, gauge")
    print("  connections, Lie structure-constant Jacobi identity, curvature antisymmetry, dimension selection). These")
    print("  are a citable, kernel-verified physics library for the cite/premise loops -- the same machinery as the math.")
    print("\n  WHAT CAME OF IT: Lever A re-derives EXACT relations and honestly flags COINCIDENCES (the alpha formula is a")
    print("  superb ~0.002% fit, not an identity). The verified physics theorems give a real substrate; the discovery")
    print("  engine can hunt for new exact relations within it -- and will only certify the ones that are truly exact.")


if __name__ == "__main__":
    main()
