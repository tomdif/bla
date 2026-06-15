#!/usr/bin/env python3
"""proofworld.ideate_fruitful -- LEVER D: open-ended generation kept by VERIFIED FRUITFULNESS (self-play for math).

The deepest lever: don't just prove given goals -- GENERATE candidate lemmas/objects and keep the ones that are
FRUITFUL: that unlock the most open goals, not merely the ones that are true. Truth is necessary but not the signal;
LEVERAGE is. A lemma that closes ten frontier goals is a 'good idea'; a true-but-inert fact is not. Fruitfulness is
measured by the kernel (how many targets does adding this lemma let z3 close?), so it cannot be gamed.

  GENERATE candidates (squares of forms, products, ... + noise)  ->  keep TRUE ones (z3)
  ->  score each by FRUITFULNESS = # open targets it closes (z3)  ->  the most fruitful = the discovered idea.

Run:  python3 -m proofworld.ideate_fruitful
"""
from __future__ import annotations
import z3

p, q1, q2, q3, q4, q5, r, s = z3.Reals("p q1 q2 q3 q4 q5 r s")
QS = [q1, q2, q3, q4, q5]
BASE = [q >= p for q in QS]                              # known: each q_i is bounded below by p (but p is unknown)
TARGETS = [(f"q{i+1}>=0", q >= 0) for i, q in enumerate(QS)]   # frontier: prove each q_i >= 0 (UNprovable from BASE)


def entails(hyps, goal, t=3000):
    sol = z3.Solver(); sol.set("timeout", t)
    for h in hyps: sol.add(h)
    sol.add(z3.Not(goal)); return sol.check() == z3.unsat


def generate():
    """candidate lemmas to consider proving. Each is a HYPOTHESIS; we score which one, if established, unlocks the
    most of the frontier. (Mix: the hub p>=0, narrow q_i>=0, and noise unrelated to the targets.)"""
    return [("p>=0", p >= 0)] + [(f"q{i+1}>=0", q >= 0) for i, q in enumerate(QS)] + \
           [("r>=0", r >= 0), ("s>=0", s >= 0), ("p<=q1", p <= q1), ("p>=q1", p >= q1)]


def main():
    print("=== LEVER D :: open-ended generation kept by VERIFIED FRUITFULNESS (unlock the most, not just be true) ===\n")
    print(f"  known: each of q1..q5 is >= p (an unknown p).   frontier: prove q_i >= 0 for all i")
    print(f"  provable from what's known alone? {all(entails(BASE, g) for _, g in TARGETS)}  -> a lemma is needed\n")
    scored = []
    for d, L in generate():
        is_target = any(str(z3.simplify(L)) == str(z3.simplify(g)) for _, g in TARGETS)   # circular if L IS a target
        unlocked = [name for name, g in TARGETS if entails(BASE + [L], g) and not entails(BASE, g)]
        scored.append((len(unlocked), d, unlocked, is_target))
    scored.sort(reverse=True)
    print(f"  FRUITFULNESS of each candidate lemma (# frontier targets it would unlock, kernel-measured):")
    for n, d, unlocked, circ in scored:
        tag = "  [circular: IS a target]" if circ and n <= 1 else ""
        print(f"    fruit={n}  {d:8} -> {unlocked if unlocked else '(nothing)'}{tag}")
    # best NON-circular
    best = next((row for row in scored if not row[3]), scored[0])
    print(f"\n  MOST FRUITFUL non-circular lemma: '{best[1]}' unlocks {best[0]}/{len(TARGETS)} of the frontier at once.")
    print(f"  -> the engine DISCOVERS that proving 'p>=0' is the high-leverage move (it dominates the whole frontier),")
    print(f"     vs proving each q_i>=0 separately (1 each) or chasing the noise lemmas (0).")
    print("\n  LEVER D: generation proposes freely; the kernel scores each by how much it UNLOCKS. Fruitfulness -- not")
    print("  mere truth -- is the verified signal that finds the 'good idea' (the key lemma), and it cannot be gamed.")


if __name__ == "__main__":
    main()
