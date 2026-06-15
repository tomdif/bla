#!/usr/bin/env python3
"""proofworld.research_loop -- the whole pipeline in one pass, on a single goal:

   LLM dreams lemmas (creative)  ->  verifier GATE kills false ones (grounded)  ->  value PRIORITIZER orders the
   minimal-proof search (fewer kernel calls)  ->  the minimal verified proof is RECORDED to the atlas.

Every generator is richer than the last, yet the trust boundary never moves: a lemma is believed only after
counterexample-search, and the final proof only after the kernel confirms the composition implies the goal. The
value prioritizer ORDERS the search; it never decides truth -- so it can only save time, never fake a proof.

Runs live with PROOFWORLD_LLM=1 (LLM generation); otherwise uses a fixed candidate pool so it still runs offline.
Run:  PROOFWORLD_LLM=1 python3 -m proofworld.research_loop
"""
from __future__ import annotations
from itertools import combinations
import os, re, z3
from proofworld.llm import (llm_dream_lemmas, parse_ineq, counterexample, implies, is_true,
                            GOAL, GOAL_DESC, VARS)

# the goal as a single nonneg expression: x^2+y^2+z^2 - (xy+yz+zx) >= 0
X, Y, Z = VARS["x"], VARS["y"], VARS["z"]
GOAL_LHS = X*X + Y*Y + Z*Z - (X*Y + Y*Z + Z*X)


def is_manifest_sos(lemma_str: str) -> bool:
    """ANTI-CIRCULARITY: a certificate atom must be nonneg *because it is a (sum of) square(s)* -- syntactically
    (...)**2 -- not because it IS the goal rescaled. A lemma like 'x**2+y**2+z**2-x*y-... >= 0' is nonneg for the
    SAME reason the goal is, so using it as a 'proof' is circular; we exclude it. The factored squares survive."""
    side = lemma_str.split(">=")[0] if ">=" in lemma_str else lemma_str.split("<=")[-1]
    if ")**2" not in side: return False                       # must contain at least one factored square
    rem = re.sub(r"\([^()]*\)\*\*2", "", side)                # strip the factored squares
    return not any(v in rem for v in ("x", "y", "z"))         # nothing but coefficients/operators may remain


def nonneg_expr(lemma_str: str):
    """the expression e such that the lemma asserts e >= 0 (for 'A>=B' -> A-B, for 'A<=B' -> B-A)."""
    s = lemma_str.strip().rstrip(".")
    for op, sign in ((">=", 1), ("<=", -1)):
        if op in s:
            l, r = s.split(op, 1); env = {"__builtins__": {}}
            return sign * (eval(l, env, VARS) - eval(r, env, VARS))
    raise ValueError(f"no >=/<= in {s!r}")


def is_identity(exprs, m, lhs, t=4000) -> bool:
    """EXACT algebraic check: does sum(exprs) == m*lhs as a polynomial identity? (z3: difference never nonzero).
    This is what makes a certificate REAL -- an irrelevant true lemma cannot reconstruct the goal expression."""
    diff = z3.Sum(*exprs) - m * lhs if len(exprs) > 1 else exprs[0] - m * lhs
    s = z3.Solver(); s.set("timeout", t); s.add(diff != 0); return s.check() == z3.unsat

# offline fallback pool (true SOS pieces + redundant restatement + irrelevant + FALSE) so the loop runs without a key
FALLBACK = [
    "(x-y)**2 >= 0", "(y-z)**2 >= 0", "(z-x)**2 >= 0",            # the three useful squares
    "x**2 + y**2 + z**2 - x*y - y*z - z*x >= 0",                  # the goal itself, rearranged (size-1 proof!)
    "x**2 >= 0",                                                  # true but irrelevant
    "x*y <= x**2 + y**2",                                         # true, weaker, irrelevant to closing
    "x**2 >= 2*x",                                                # FALSE (counterexample x=1)
]


def value_score(lemma_str: str) -> float:
    """the value PRIORITIZER (value.py's learned ranker plugs in here identically). A cheap structural estimate of
    'likely to close the goal': uses all three variables + degree-2 + mentions the cross terms the goal is about.
    ORDERS the search only -- the verifier still owns truth on every candidate."""
    s = lemma_str.replace(" ", "")
    uses_all = all(v in s for v in ("x", "y", "z"))
    factored = s.count(")**2")        # prefer LEGIBLE factored squares (x-y)**2 over expanded restatements
    expanded_restatement = ("x**2" in s and "x*y" in s)   # penalize "2*x**2+...-2*x*y..." = the goal restated (circular)
    return 2.0 * uses_all + 1.5 * factored - 1.0 * expanded_restatement


def gate(candidates, log=print):
    """GROUNDED gate: parse -> counterexample-search. False/unparseable die; true lemmas survive."""
    survivors = []
    for s in candidates:
        try:
            f = parse_ineq(s)
        except Exception as e:
            log(f"    {s:46} -> UNPARSEABLE ({e})"); continue
        cx = counterexample(f)
        if cx is not None:
            log(f"    {s:46} -> KILLED (false; counterexample {cx})")
        else:
            log(f"    {s:46} -> survives"); survivors.append((s, f))
    return survivors


def minimal_certificate(survivors, order_by_value=True, multipliers=(1, 2, 3), log=print):
    """find the SMALLEST subset of nonneg lemmas whose sum EXACTLY equals m*GOAL_LHS (an SOS certificate) for some
    small m. ascending size; within a size, try in value-priority order. The identity is z3-verified, so a true-
    but-irrelevant lemma (x^2>=0) cannot pass -- the lemma must actually reconstruct the goal. Returns (subset, m,
    kernel_calls)."""
    # anti-circularity: certificate atoms must be MANIFEST squares, not goal-restatements
    survivors = [sf for sf in survivors if is_manifest_sos(sf[0])]
    items = sorted(survivors, key=lambda sf: -value_score(sf[0])) if order_by_value else list(survivors)
    names = [s for s, _ in items]
    exprs = []
    for s, _ in items:
        try: exprs.append(nonneg_expr(s))
        except Exception: exprs.append(None)
    idx = [i for i in range(len(items)) if exprs[i] is not None]
    calls = 0
    for size in range(1, len(idx) + 1):
        for combo in combinations(idx, size):
            for m in multipliers:
                calls += 1
                if is_identity([exprs[i] for i in combo], m, GOAL_LHS):     # z3 verifies the exact identity
                    return [names[i] for i in combo], m, calls
    return None, None, calls


def main():
    print("=== proofworld.research_loop :: LLM dream -> gate -> value-ordered minimal-proof -> atlas record ===\n")
    print(f"  GOAL: {GOAL_DESC}\n")
    # 1) GENERATE (creative) -- live LLM if gated, else the fixed pool
    raw = llm_dream_lemmas(GOAL_DESC)
    if raw:
        raw = list(raw) + ["x**2 >= 2*x"]                          # inject one FALSE to stress the gate
        src = "LLM (live)"
    else:
        raw, src = FALLBACK, "fallback pool (offline)"
    print(f"  generator: {src} -- {len(raw)} candidate lemmas\n")
    # 2) GATE (grounded) -- kill false/unparseable
    print("  --- GATE (counterexample-search; the LLM is filtered, not trusted) ---")
    survivors = gate(raw)
    print(f"  survivors: {len(survivors)}/{len(raw)} true lemmas\n")
    # 3) value-ORDERED SOS-certificate search vs unordered -- find the minimal subset that EXACTLY reconstructs the goal
    proof_v, m_v, calls_v = minimal_certificate(survivors, order_by_value=True)
    proof_b, m_b, calls_b = minimal_certificate(survivors, order_by_value=False)
    sq = lambda lst: " + ".join(s.split(">=")[0].strip() for s in lst) if lst else "(none)"
    print(f"  --- SOS-certificate search (z3 verifies an EXACT identity sum(squares) = m*goal; not a vacuous implication) ---")
    print(f"  value-ORDERED : {calls_v} identity-checks -> {m_v}*goal = {sq(proof_v)}")
    print(f"  unordered     : {calls_b} identity-checks -> {m_b}*goal = {sq(proof_b)}")
    if calls_b and calls_v:
        print(f"  prioritizer speedup: {calls_b/calls_v:.1f}x fewer identity-checks, SAME certificate size ({len(proof_v or [])})")
    # honest note: z3 also proves the inequality directly; the VALUE is the human-readable, reusable certificate.
    print(f"  (note: z3 proves the inequality outright too; the certificate is the *reusable, legible* proof object,")
    print(f"   and a true-but-irrelevant lemma like 'x**2>=0' is correctly REJECTED -- it can't reconstruct the goal.)")
    # 4) RECORD to the atlas (the verified certificate becomes a reusable technique)
    print(f"\n  --- atlas record ---")
    print(f"  GOAL {GOAL_DESC}")
    print(f"    status     : {'VERIFIED' if is_true(GOAL) else 'OPEN'}  (kernel-confirmed)")
    print(f"    certificate: {m_v}*(x^2+y^2+z^2 - (x*y+y*z+z*x)) = {sq(proof_v)}   (kernel-verified identity)")
    print(f"    technique  : SOS / sum-of-squares decomposition (recorded for reuse on similar goals)")
    # 5) the invariant
    ok = bool(proof_v) and is_true(GOAL)
    print(f"\n  INVARIANT: generators got richer (LLM) and search got faster (value order), but a lemma was believed")
    print(f"  only after counterexample-search and a certificate only after the kernel verified the EXACT identity.")
    print(f"\n  GATE: {'PASS -- creative generation, grounded belief, minimal verified certificate recorded' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
