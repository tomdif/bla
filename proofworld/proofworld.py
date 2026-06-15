#!/usr/bin/env python3
"""proofworld -- a verifier-owns-truth proof-research instrument (the recommended MVP).

NOT a math chatbot, and NOT a learned proof-state dynamics model + imagination loop (that construct gets
*exploited* exactly like it did in the physical world-model study: a proposer trained to maximize predicted
progress in a learned model learns to fool the model, and proof-state "dynamics" is discrete/combinatorial and
the verifier already computes it exactly). Instead:

  proposers PROPOSE typed routes  ->  the VERIFIER (here: z3/SMT) owns truth  ->  the system ACCUMULATES verified
  techniques + an OBSTRUCTION ATLAS (why routes fail) + runs ANTI-HALLUCINATION CANARIES.

The verifier is the low-level dynamics (don't learn it). The learnable part is route VALUE / obstruction
classification -- the research-grade "why is this family doomed" reasoning that AlphaProof-style provers don't do.
The LLM is one (optional, env-gated) proposer among several; it never gets to assert truth.

Run: python3 -m proofworld.proofworld
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import os, z3


# ----------------------------- typed state -----------------------------
@dataclass
class Conjecture:
    name: str
    build: Callable[[], tuple]              # () -> (vars, z3_formula)  [formula = the statement to prove valid]
    features: tuple = ()                    # domain tags for the library proposer (e.g. "quadratic","nonneg")
    truth: Optional[bool] = None            # ground truth -- used ONLY to grade canaries, never to help solving

@dataclass
class ProofRoute:
    name: str
    source: str                            # which proposer produced it
    method: str                            # "smt" | "counterexample" | "lemma_smt"
    lemmas: tuple = ()                      # names of cited (must-be-verified) lemmas
    assume_goal: bool = False              # circularity bait: does this route assume the goal?
    predicted_value: float = 0.0

class Status(Enum):
    VERIFIED = "verified"; REFUTED = "refuted"; CIRCULAR = "circular"
    UNSOUND_LEMMA = "unsound_lemma"; OPEN = "open"

@dataclass
class VerifierResult:
    status: Status
    route: str
    counterexample: Optional[str] = None
    note: str = ""

@dataclass
class TechniqueCard:                        # accumulated reusable verified method
    name: str; method: str; applies_when: tuple; verified_on: list = field(default_factory=list); confidence: float = 0.0

@dataclass
class ObstructionCard:                      # accumulated "why a route fails" -- the atlas
    conjecture: str; route: str; status: str; why: str


# ----------------------------- the verifier (owns truth) -----------------------------
class Verifier:
    """z3 is the kernel. Proves validity (negation unsat), refutes with a concrete counterexample (model), and
    -- crucially -- REJECTS circular routes (a route may only cite lemmas already in the VERIFIED library)."""
    def __init__(self, library: dict, timeout_ms=4000):
        self.lib = library; self.timeout = timeout_ms

    def _equiv(self, f, g) -> bool:         # f <-> g valid?
        s = z3.Solver(); s.set("timeout", self.timeout); s.add(z3.Not(f == g)); return s.check() == z3.unsat

    def run(self, c: Conjecture, r: ProofRoute) -> VerifierResult:
        vars_, goal = c.build()
        # --- circularity / soundness audit BEFORE trusting anything ---
        if r.assume_goal:
            return VerifierResult(Status.CIRCULAR, r.name, note="route assumes the goal as a hypothesis")
        lemma_forms = []
        for lname in r.lemmas:
            if lname not in self.lib:       # citing an UN-verified lemma = unsound
                return VerifierResult(Status.UNSOUND_LEMMA, r.name, note=f"cites unverified lemma '{lname}'")
            lf = self.lib[lname]()[1]
            if self._equiv(lf, goal):       # citing (something equivalent to) the goal = circular
                return VerifierResult(Status.CIRCULAR, r.name, note=f"lemma '{lname}' is equivalent to the goal")
            lemma_forms.append(lf)
        # --- kernel check ---
        s = z3.Solver(); s.set("timeout", self.timeout)
        for lf in lemma_forms: s.add(lf)    # sound: only verified, non-circular lemmas reach here
        if r.method == "counterexample":
            s.add(z3.Not(goal)); res = s.check()
            if res == z3.sat: return VerifierResult(Status.REFUTED, r.name, counterexample=str(s.model()))
            if res == z3.unsat: return VerifierResult(Status.OPEN, r.name, note="no counterexample (statement may be true)")
            return VerifierResult(Status.OPEN, r.name, note="solver unknown")
        s.add(z3.Not(goal)); res = s.check()  # prove validity
        if res == z3.unsat: return VerifierResult(Status.VERIFIED, r.name)
        if res == z3.sat:   return VerifierResult(Status.REFUTED, r.name, counterexample=str(s.model()))
        return VerifierResult(Status.OPEN, r.name, note="solver returned unknown (honest 'don't know')")


# ----------------------------- proposers (all output the SAME typed route) -----------------------------
class LibraryProposer:
    name = "library"
    def __init__(self, techniques): self.tech = techniques
    def propose(self, c):
        out = []
        for t in self.tech.values():
            if set(t.applies_when) & set(c.features):
                out.append(ProofRoute(f"lib:{t.name}", self.name, t.method, predicted_value=0.6 + 0.3 * t.confidence))
        return out

class HeuristicProposer:
    name = "heuristic"
    def propose(self, c):
        return [ProofRoute("direct-smt", self.name, "smt", predicted_value=0.5),
                ProofRoute("counterexample-search", self.name, "counterexample", predicted_value=0.4)]

class LLMProposer:
    """OPTIONAL strong proposer. Env-gated (PROOFWORLD_LLM=1 + an API key). It NEVER asserts truth -- it only
    emits typed routes the verifier still checks. Off by default; the system runs without it."""
    name = "llm"
    def propose(self, c):
        if os.environ.get("PROOFWORLD_LLM") != "1": return []
        # (hook) call the LLM API here to propose a route/lemma; left disabled so the build runs offline + safe.
        return []


# ----------------------------- the research loop + atlas -----------------------------
class ProofWorld:
    def __init__(self, lemma_library: dict):
        self.lemmas = lemma_library                 # name -> Conjecture.build-like (verified facts the verifier may use)
        self.techniques: dict[str, TechniqueCard] = {}
        self.atlas: list[ObstructionCard] = []
        self.verifier = Verifier(lemma_library)
        self.proposers = [LibraryProposer(self.techniques), LLMProposer(), HeuristicProposer()]

    def attack(self, c: Conjecture, log=print) -> VerifierResult:
        routes, seen = [], set()
        for p in self.proposers:
            for r in p.propose(c):
                if r.name not in seen: seen.add(r.name); routes.append(r)
        routes.sort(key=lambda r: -r.predicted_value)             # try the most promising first (value, not random)
        best = None
        for r in routes:
            res = self.verifier.run(c, r); best = best or res
            if res.status is Status.VERIFIED:
                self._learn_technique(c, r); log(f"  [{c.name}] {r.source}/{r.name} -> VERIFIED"); return res
            if res.status is Status.REFUTED:
                self.atlas.append(ObstructionCard(c.name, r.name, "refuted", f"counterexample {res.counterexample}"))
                log(f"  [{c.name}] {r.name} -> REFUTED  (counterexample: {res.counterexample})"); return res
            self.atlas.append(ObstructionCard(c.name, r.name, res.status.value, res.note))   # why this route failed
        log(f"  [{c.name}] no route closed -> {best.status.value if best else 'no-route'}")
        return best or VerifierResult(Status.OPEN, "none")

    def _learn_technique(self, c, r):
        key = r.method + ":" + "|".join(sorted(c.features))
        t = self.techniques.get(key) or TechniqueCard(key, r.method, tuple(c.features))
        t.verified_on.append(c.name); t.confidence = min(1.0, 0.4 + 0.15 * len(t.verified_on))
        self.techniques[key] = t


# ----------------------------- a narrow domain: arithmetic inequalities/identities -----------------------------
def amgm2():       x, y = z3.Reals('x y');     return [x, y], x * x + y * y >= 2 * x * y
def amgm2_false(): x, y = z3.Reals('x y');     return [x, y], x * x + y * y >= 3 * x * y
def sq_nonneg():   x = z3.Real('x');           return [x], x * x >= 0
def cauchy2():     x, y, a, b = z3.Reals('x y a b'); return [x, y, a, b], (x*x + y*y) * (a*a + b*b) >= (x*a + y*b) ** 2
def int_false():   n = z3.Int('n');            return [n], n * n >= 2 * n
def shift_id():    x = z3.Real('x');           return [x], (x + 1) * (x + 1) == x * x + 2 * x + 1


CONJECTURES = [
    Conjecture("amgm2", amgm2, ("quadratic", "nonneg"), truth=True),
    Conjecture("sq_nonneg", sq_nonneg, ("quadratic", "nonneg"), truth=True),
    Conjecture("shift_id", shift_id, ("identity",), truth=True),
    Conjecture("cauchy2", cauchy2, ("quadratic", "nonneg"), truth=True),
    Conjecture("amgm2_FALSE", amgm2_false, ("quadratic",), truth=False),   # must be REFUTED, never verified
    Conjecture("int_FALSE", int_false, ("integer",), truth=False),         # must be REFUTED (n=1)
]
LIBRARY = {"sq_nonneg": sq_nonneg, "amgm2": amgm2}    # verified facts the verifier may cite


# ----------------------------- anti-hallucination canaries (the gate) -----------------------------
def canaries(log=print) -> bool:
    ok = True
    # 1) FAKE-THEOREM canary: a false statement must be REFUTED (counterexample), never VERIFIED.
    w = ProofWorld(LIBRARY); r = w.attack(Conjecture("fake", amgm2_false, ("quadratic",)), log=lambda *a: None)
    c1 = r.status is Status.REFUTED; ok &= c1
    log(f"  [canary] fake-theorem refuted (not 'proven'): {'PASS' if c1 else 'FAIL'}  ({r.status.value})")
    # 2) CIRCULARITY canary: a route that cites the goal as a lemma must be flagged CIRCULAR, not VERIFIED.
    g = Conjecture("amgm2", amgm2, ("quadratic", "nonneg"))
    res = ProofWorld({"amgm2": amgm2}).verifier.run(g, ProofRoute("self", "test", "lemma_smt", lemmas=("amgm2",)))
    c2 = res.status is Status.CIRCULAR; ok &= c2
    log(f"  [canary] circular route (assumes the goal) flagged: {'PASS' if c2 else 'FAIL'}  ({res.status.value})")
    # 3) UNSOUND-LEMMA canary: citing an unverified lemma is rejected.
    res = ProofWorld(LIBRARY).verifier.run(g, ProofRoute("x", "test", "lemma_smt", lemmas=("not_in_library",)))
    c3 = res.status is Status.UNSOUND_LEMMA; ok &= c3
    log(f"  [canary] unverified-lemma citation rejected: {'PASS' if c3 else 'FAIL'}  ({res.status.value})")
    # 4) SHUFFLE canary: corrupt the technique library; the system must STILL solve (verifier owns truth, not the lib).
    w = ProofWorld(LIBRARY); w.techniques["garbage"] = TechniqueCard("garbage", "smt", ("quadratic",), confidence=9.9)
    r = w.attack(Conjecture("amgm2", amgm2, ("quadratic", "nonneg")), log=lambda *a: None)
    c4 = r.status is Status.VERIFIED; ok &= c4
    log(f"  [canary] solves despite a corrupted technique library (truth=verifier, not library): {'PASS' if c4 else 'FAIL'}")
    return ok


def main():
    print("=== proofworld :: verifier-owns-truth proof-research loop (narrow arithmetic domain) ===\n")
    w = ProofWorld(LIBRARY); results = {}
    for c in CONJECTURES:
        res = w.attack(c); results[c.name] = res
    # honesty check: did the verifier ever disagree with ground truth?
    print("\n--- soundness audit (verifier vs ground truth) ---")
    bad = 0
    for c in CONJECTURES:
        st = results[c.name].status
        proved_true = st is Status.VERIFIED; refuted = st is Status.REFUTED
        wrong = (c.truth and refuted) or (c.truth is False and proved_true)
        bad += wrong
        print(f"  {c.name:12} truth={c.truth}  verdict={st.value:9} {'<-- WRONG' if wrong else ''}")
    print(f"  verifier soundness: {'OK (no false verdicts)' if bad == 0 else f'{bad} FALSE VERDICTS'}")
    print(f"\n--- accumulated technique library ({len(w.techniques)}) ---")
    for t in w.techniques.values(): print(f"  {t.name:28} conf={t.confidence:.2f} verified_on={t.verified_on}")
    print(f"\n--- obstruction atlas ({len(w.atlas)} entries) ---")
    for o in w.atlas[:8]: print(f"  {o.conjecture:12} {o.route:22} {o.status:12} {o.why[:60]}")
    print("\n--- ANTI-HALLUCINATION CANARIES (the gate) ---")
    ok = canaries()
    print(f"\nGATE: {'ALL CANARIES PASS -- trustworthy loop' if ok and bad == 0 else 'FAILED -- do not trust'}")


if __name__ == "__main__":
    main()
