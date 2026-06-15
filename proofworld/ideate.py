#!/usr/bin/env python3
"""proofworld.ideate -- the IDEATION SUITE: every lever for conjuring new ideas, run together.

Each lever is a way to PROPOSE genuinely new mathematics, kept honest by the kernel (propose freely, verify, keep
what's verified/fruitful). None conjures a breakthrough on demand -- nothing can -- but together they maximize the
surface area for one to land, soundly, with every fragment certified.

  A  experiment  : data -> integer-relation (PSLQ) -> unexpected closed form          (ideate_experiment)
  B  reformulate : recast the problem; kernel certifies equivalent reframings         (ideate_reformulate)
  C  abduce      : the WEAKEST missing premise that closes a goal (obstruction spec)   (ideate_abduce)
  D  fruitful    : open-ended lemmas kept by how much they UNLOCK (verified leverage)  (ideate_fruitful)
  -  analogy     : carry one proof idea into new domains, kernel-checked               (ideate_analogy)
  -  extremal    : binary-search the SHARP best-possible constant                      (ideate_extremal)
  E  proposer    : the most capable engine (Opus 4.8) drives every generative step.

Run:  python3 -m proofworld.ideate            (full suite; add PROOFWORLD_LLM=1 for the live-proposer levers)
"""
from __future__ import annotations
import importlib

LEVERS = [
    ("A  experimental discovery (PSLQ)", "proofworld.ideate_experiment"),
    ("C  abduction (obstruction-as-spec)", "proofworld.ideate_abduce"),
    ("D  open-ended fruitfulness", "proofworld.ideate_fruitful"),
    ("B  reformulation search", "proofworld.ideate_reformulate"),
    ("analogy  cross-domain transfer", "proofworld.ideate_analogy"),
    ("extremal  sharp-constant discovery", "proofworld.ideate_extremal"),
]


def main():
    print("=" * 78)
    print("  proofworld IDEATION SUITE -- every lever for conjuring new, kernel-grounded ideas")
    print("=" * 78)
    for label, mod in LEVERS:
        print(f"\n\n>>>>>>>>>>  LEVER {label}  <<<<<<<<<<")
        try:
            importlib.import_module(mod).main()
        except Exception as e:
            print(f"  (lever {mod} errored: {e})")
    print("\n" + "=" * 78)
    print("  All levers PROPOSE freely; the kernel VERIFIES; only what's certified/fruitful is kept. The breakthrough")
    print("  spark is the proposer's (Lever E = Opus 4.8); the suite makes its creativity safe, grounded, cumulative,")
    print("  and targeted -- the most fertile honest ground for new mathematics to land.")
    print("=" * 78)


if __name__ == "__main__":
    main()
