#!/usr/bin/env python3
"""proofworld.ideate_abduce -- LEVER C: obstruction-as-specification by ABDUCTION.

When the kernel can't close a goal from what's known, the breakthrough is usually a single missing idea. Abduction
finds it PRECISELY: the WEAKEST hypothesis H such that (known facts) + H entails the goal, and H is consistent. That
converts 'conjure an idea' into a sharp specification -- 'to prove G you need exactly H (and nothing stronger)'.
Then we search the library for something that already implies H (proof exists) or flag H as the genuine missing piece.

  goal G unprovable from facts F  ->  search candidate hypotheses  ->  keep H that are CONSISTENT and CLOSE G (z3)
  ->  among those, the WEAKEST (most general) = the minimal missing premise = the target to construct/find.

Run:  python3 -m proofworld.ideate_abduce
"""
from __future__ import annotations
import z3, itertools

A, B, C, D, E = z3.Reals("A B C D E")
VARS = [A, B, C, D, E]
FACTS = [A >= 0, A <= B]                                  # what is known
GOAL = C >= 0                                             # what we want; NOT provable from FACTS alone
FACT_DESC = "{A>=0, A<=B}"


def consistent(hyps, t=3000):
    s = z3.Solver(); s.set("timeout", t)
    for h in hyps: s.add(h)
    return s.check() == z3.sat

def entails(hyps, concl, t=3000):
    s = z3.Solver(); s.set("timeout", t)
    for h in hyps: s.add(h)
    s.add(z3.Not(concl)); return s.check() == z3.unsat


def candidate_hypotheses():
    cands = []
    names = {id(A): "A", id(B): "B", id(C): "C", id(D): "D", id(E): "E"}
    for X in VARS:
        cands.append((f"{names[id(X)]}>=0", X >= 0))
    for X, Y in itertools.permutations(VARS, 2):
        cands.append((f"{names[id(X)]}<={names[id(Y)]}", X <= Y))
    return cands


def main():
    print("=== LEVER C :: abduction -- find the WEAKEST missing premise that closes the goal ===\n")
    print(f"  known facts: {FACT_DESC}    goal: C >= 0")
    print(f"  goal provable from facts alone? {entails(FACTS, GOAL)}  (-> a missing idea is required)\n")
    cands = candidate_hypotheses()
    closing = [(d, h) for d, h in cands if not entails(FACTS, h)                       # not already known
               and consistent(FACTS + [h]) and entails(FACTS + [h], GOAL)]
    print(f"  candidate hypotheses that (consistently) CLOSE the goal: {[d for d, _ in closing]}")
    # exclude goal-RESTATEMENTS (circular: equivalent to the goal -- trivially the weakest, but no information)
    circular = [d for d, h in closing if entails([h], GOAL) and entails([GOAL], h)]
    noncirc = [(d, h) for d, h in closing if d not in circular]
    # weakest (most general) NON-circular premise: no other non-circular closing H' is strictly weaker GIVEN THE FACTS
    weakest = [(d, h) for d, h in noncirc
               if not any(entails(FACTS + [h], hp) and not entails(FACTS + [hp], h)
                          for dp, hp in noncirc if dp != d)]
    print(f"\n  SPECIFICATION (weakest NON-circular missing premise): {[d for d, _ in weakest]}")
    print(f"     -> to prove C>=0 from {FACT_DESC}, the MINIMAL idea needed is: {' or '.join(d for d,_ in weakest)}")
    print(f"     (circular options that close it but just restate the goal: {circular};")
    print(f"      stronger-than-needed options: {[d for d,_ in noncirc if (d,_) not in weakest and d not in [w for w,_ in weakest]]})")
    print(f"\n  is the spec already entailed by the facts? {any(entails(FACTS, h) for _, h in weakest)}")
    print(f"  -> NO: '{weakest[0][0]}' is the genuine missing fact -- the precise target to construct or find in the corpus.")
    print("\n  LEVER C: abduction pinpoints the minimal idea a proof needs -- not a vague 'something's missing', but a")
    print("  weakest sufficient premise, kernel-checked. That is the obstruction-as-specification that focuses search.")


if __name__ == "__main__":
    main()
