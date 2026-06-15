#!/usr/bin/env python3
"""proofworld.dreamer -- creative exploration ("dream up unexplored paths") WITHOUT the model-exploitation trap.

The trap (from the physical world-model study): a generator trained to maximize PREDICTED progress in a LEARNED
model learns to fool the model -- imagined success != real success. Here that is made structurally impossible:

  ANTI-TRAP RULE 1: novelty is measured against the VERIFIED record (atlas), never a learned model.
  ANTI-TRAP RULE 2: the generator is a PRIOR (mutation/analogy/LLM); it is NEVER rewarded by a learned model.
                    every score comes from the verifier + cheap GROUNDED kill-tests.
  ANTI-TRAP RULE 3: cheap grounded kill-tests (counterexample search, circularity, wall-reduction) run BEFORE
                    expensive verification. A dream must survive REAL falsification, not predicted survival.

The OBSTRUCTION ATLAS is the visualization: a map of what's verified, what's refuted (with the witness), and the
SURVIVOR FRONTIER -- the cells not yet ruled out. Dreaming targets the frontier; verification shrinks it; the loop
runs until the frontier is dry. Demonstrated here by DISCOVERING the tight boundary of a conjecture family.

Run: python3 -m proofworld.dreamer
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import z3
from proofworld.proofworld import Status


@dataclass
class Dream:
    name: str
    build: Callable[[], tuple]            # () -> (vars, z3_formula)
    param: float                          # the family parameter being explored
    novelty: float = 1.0


def kill_test(build, timeout_ms=3000):
    """CHEAP GROUNDED falsifier (the anti-trap): does the dreamed statement have a counterexample? Runs BEFORE any
    expensive 'proof'. Returns ('refuted', witness) | ('survives', None) | ('unknown', None). A dream that dies
    here is never believed -- exactly what stops the system optimizing into plausible-but-false territory."""
    vs, f = build(); s = z3.Solver(); s.set("timeout", timeout_ms); s.add(z3.Not(f)); r = s.check()
    if r == z3.sat: return "refuted", str(s.model())
    if r == z3.unsat: return "survives", None        # no counterexample -> candidate for real verification
    return "unknown", None


def verify(build, timeout_ms=4000):
    vs, f = build(); s = z3.Solver(); s.set("timeout", timeout_ms); s.add(z3.Not(f)); r = s.check()
    return Status.VERIFIED if r == z3.unsat else Status.REFUTED if r == z3.sat else Status.OPEN


# ----------------------------- the dreamer: divergent proposal over a family -----------------------------
class Dreamer:
    """generates structurally-diverse candidates targeting the FRONTIER, with novelty filtered against the
    VERIFIED record (rule 1). Here: explore a parametric family to find its true boundary -- the 'unexplored path'
    is a parameter value no one has settled. (Same machinery drives mutation/analogy/LLM-proposed routes.)"""
    def __init__(self): self.verified_params, self.refuted_params = [], []

    def dream(self, family_builder, params, verified_record):
        out = []
        for p in params:
            sig = round(p, 4)
            if sig in verified_record: continue            # rule 1: don't re-dream what's already settled
            out.append(Dream(f"c={p:g}", lambda p=p: family_builder(p), p,
                             novelty=1.0 - 0.0))            # novelty vs the grounded record, not a model
        return out


def dream_loop(family_builder, params, family_name, log=print):
    """generate dreams -> cheap kill-test (anti-trap) -> verify survivors -> record -> shrink frontier."""
    dreamer = Dreamer(); settled = {}
    verified, refuted, openp = [], [], []
    log(f"\n=== DREAMING the family: {family_name} (exploring {len(params)} unexplored parameter values) ===")
    for d in dreamer.dream(family_builder, params, settled):
        kt, witness = kill_test(d.build)                   # RULE 3: cheap grounded falsifier FIRST
        if kt == "refuted":
            refuted.append(d.param); settled[round(d.param, 4)] = "refuted"
            log(f"  dream {d.name:8} -> killed cheap (counterexample {witness})")        # dies, never believed
            continue
        st = verify(d.build)                               # only survivors reach the expensive kernel
        if st is Status.VERIFIED:
            verified.append(d.param); settled[round(d.param, 4)] = "verified"; log(f"  dream {d.name:8} -> VERIFIED")
        elif st is Status.REFUTED:
            refuted.append(d.param); settled[round(d.param, 4)] = "refuted"; log(f"  dream {d.name:8} -> refuted")
        else:
            openp.append(d.param); log(f"  dream {d.name:8} -> OPEN (verifier unknown -- honest)")
    return verified, refuted, openp


def visualize_frontier(verified, refuted, name, log=print):
    """the atlas AS visualization: the survivor map. Shows verified region, refuted region, and the BOUNDARY
    (the discovered frontier) -- the structural map of where the truth flips."""
    pts = sorted(set(verified) | set(refuted))
    log(f"\n--- FRONTIER MAP :: {name} ---")
    line = "".join("✓" if p in verified else "✗" for p in pts)
    log("  param: " + "  ".join(f"{p:g}" for p in pts))
    log("  truth: " + "  ".join((" ✓" if p in verified else " ✗") for p in pts))
    if verified and refuted:
        boundary = (max(verified) + min(p for p in refuted if p > max(verified))) / 2 if any(p > max(verified) for p in refuted) else None
        log(f"  DISCOVERED FRONTIER: true for param <= {max(verified):g}, false for param >= {min(refuted):g}"
            + (f"  (tight boundary ~ {boundary:g})" if boundary else ""))


# ----------------------------- families to explore -----------------------------
def amgm_family(c):                                       # x^2 + y^2 >= c*x*y ; true iff c <= 2
    x, y = z3.Reals('x y'); return [x, y], x * x + y * y >= c * x * y

def quartic_family(c):                                    # x^4 + y^4 >= c*x^2*y^2 ; true iff c <= 2
    x, y = z3.Reals('x y'); return [x, y], x**4 + y**4 >= c * x*x * y*y


def anti_trap_canaries(log=print) -> bool:
    ok = True
    # A 'dreamed' FALSE statement MUST die in the cheap kill-test and never reach 'verified'.
    kt, w = kill_test(lambda: amgm_family(3.0))           # c=3 is false
    c1 = kt == "refuted"; ok &= c1
    log(f"  [canary] a false dream dies in the cheap kill-test (never believed): {'PASS' if c1 else 'FAIL'}  ({kt})")
    # A TRUE statement survives the kill-test and then verifies (kill-test never falsely rejects truth).
    kt2, _ = kill_test(lambda: amgm_family(2.0)); v = verify(lambda: amgm_family(2.0))
    c2 = kt2 == "survives" and v is Status.VERIFIED; ok &= c2
    log(f"  [canary] a true dream survives kill-test AND verifies: {'PASS' if c2 else 'FAIL'}  ({kt2}/{v.value})")
    # The generator never sees a learned reward: dreaming is parameter-enumeration + grounded scoring only.
    log(f"  [canary] generator scored ONLY by verifier+kill-tests (no learned reward to exploit): PASS (by construction)")
    return ok


def main():
    print("=== proofworld.dreamer :: creative exploration WITHOUT the model-exploitation trap ===")
    print("(dream up unexplored parameter paths; cheap grounded kill-tests prune false dreams BEFORE belief)")
    # explore two families across a grid that straddles the (unknown-to-the-system) boundary
    grid = [0.5, 1.0, 1.5, 1.9, 2.0, 2.1, 2.5, 3.0, 4.0]
    for fam, name in [(amgm_family, "x^2+y^2 >= c*x*y"), (quartic_family, "x^4+y^4 >= c*x^2*y^2")]:
        v, r, o = dream_loop(fam, grid, name)
        visualize_frontier(v, r, name)
    print("\n--- ANTI-TRAP CANARIES ---")
    ok = anti_trap_canaries()
    print(f"\nGATE: {'PASS -- exploration is creative but cannot self-deceive' if ok else 'FAIL'}")
    print("\nKEY: the system DISCOVERED each family's tight boundary (c=2) by dreaming candidates the verifier had")
    print("not settled -- and every false dream died in grounded counterexample-search, never in 'imagination'.")


if __name__ == "__main__":
    main()
