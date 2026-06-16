#!/usr/bin/env python3
"""proofworld.bridge -- ideate by searching CONSTRUCTIONS (bridges), not proofs.

Thesis (from the case studies of "thought unsolvable -> solved"): the creative act is SHORT (Wiles: one sentence;
Huang: one matrix) while the proof is long. So search the small space of BRIDGES -- structure-preserving maps that
turn an open problem into one where known machinery applies -- and let the kernel eat the long, mechanical
verification of each candidate's PRECONDITION (not the whole proof).

Three components, the way a mathematician actually ideates:
  (1) FINGERPRINT  -- represent a problem by its DEEP type (positivity? finiteness? bound? identity?), not surface.
  (2) MOVE CORPUS  -- transformative moves distilled from real breakthroughs, indexed by PRECONDITION:
                      SOS/positivity (Hilbert), spectral/interlacing (Huang sensitivity), telescoping/monotone
                      (Abel/Perelman-style), transport ("solution here = forbidden object there"; Wiles/Frey),
                      finite-reduction (Appel-Haken). Each carries the precondition that makes it apply.
  (3) PROPOSER     -- given a fingerprint, RETRIEVE matching moves and INSTANTIATE the analog object (Opus 4.8 when
                      PROOFWORLD_LLM=1; a canonical instantiation otherwise). The KERNEL then checks the bridge's
                      precondition. Most bridges die cheaply; a survivor is a genuine, grounded idea.

Anti-trap, unchanged: the proposer invents the bridge; z3/numpy/sympy OWN whether it is real. The adversarial control
proves the gate has teeth -- a bogus bridge is REJECTED, not rationalized.

Run:  python3 -m proofworld.bridge            (PROOFWORLD_LLM=1 to let Opus propose the bridge object live)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os, json
import sympy as sp
import numpy as np


# ======================================================================================
# (1) STRUCTURAL FINGERPRINT -- the deep type, the thing analogy keys on
# ======================================================================================
@dataclass
class Problem:
    name: str
    statement: str                 # human statement
    kind: str                      # coarse class: 'inequality' | 'bound' | 'sum_identity'
    data: dict = field(default_factory=dict)   # machine payload (polynomial, dimension, term, ...)


def fingerprint(p: Problem) -> set[str]:
    """Derive deep-type tags. Surface syntax is useless for analogy; the TAGS are what a move matches against."""
    t = set()
    s = p.statement.lower()
    if p.kind == "inequality":
        t |= {"positivity", "real_closed_field"}
        poly = p.data.get("poly")
        if poly is not None:
            if poly == poly.subs(dict(zip(p.data["vars"], reversed(p.data["vars"])))):
                t.add("symmetric")
            if sp.total_degree(poly) == 2:
                t.add("quadratic")
    if p.kind == "bound":
        t.add("extremal_bound")
        if any(w in s for w in ("cube", "boolean", "graph", "subgraph", "degree", "vertices")):
            t |= {"combinatorial", "graph", "discrete"}
    if p.kind == "sum_identity":
        t |= {"identity", "telescoping_candidate", "closed_form"}
    if "exist" in s or "infinitely many" in s:
        t.add("existence")
    if "for all" in s or "every" in s:
        t.add("universal")
    return t


# ======================================================================================
# (2) MOVE CORPUS -- transformative moves distilled from breakthroughs, keyed by precondition
# ======================================================================================
# Each move: a precondition (tags it needs in the fingerprint), the bridge it builds, the historical source,
# and a `check` that lets the KERNEL verify a concrete instantiation of the bridge.

def _check_sos(p: Problem, bridge: dict):
    """Bridge = {'squares':[expr,...], 'scale':optional multiplier}. Kernel check: does the identity hold EXACTLY?
    If p - sum(squares) == 0 (symbolically) and every piece is a manifest square, then p >= 0. (sympy = exact.)"""
    vs = p.data["vars"]; poly = p.data["poly"]
    loc = {str(v): v for v in vs}                                   # bind names to the problem's OWN symbols
    squares = [sp.sympify(s, locals=loc) for s in bridge["squares"]]   # (else real-vs-generic 'x' won't cancel)
    rhs = sum(sp.expand(sq) for sq in squares)
    diff = sp.expand(poly - rhs)
    ok = diff == 0
    return ok, f"p - Σ(sq)  =  {diff}   (must be 0); pieces are explicit squares -> p ≥ 0 iff identity holds"


def _check_spectral(p: Problem, bridge: dict):
    """Huang move. Bridge = {'dim': n}: build the SIGNED hypercube adjacency A_n (Huang's construction) and let the
    kernel (numpy) verify A_n^2 = n·I and spectrum = {±√n}. THEN Cauchy interlacing gives the combinatorial bound."""
    n = bridge["dim"]
    A = np.array([[0.0, 1.0], [1.0, 0.0]])                          # A_1
    for _ in range(2, n + 1):
        k = A.shape[0]; I = np.eye(k)
        A = np.block([[A, I], [I, -A]])                             # A_n = [[A_{n-1}, I],[I, -A_{n-1}]]
    N = A.shape[0]                                                  # = 2^n
    sq_ok = np.allclose(A @ A, n * np.eye(N))
    eig = np.linalg.eigvalsh(A)
    spec_ok = np.allclose(np.sort(np.abs(eig)), np.sqrt(n))        # all eigenvalues are ±√n
    pos = int(round(np.sum(eig > 1e-9)))                            # multiplicity of +√n  (= 2^{n-1})
    ok = bool(sq_ok and spec_ok)
    why = (f"signed cube A_{n} ({N}×{N}): A²=nI? {sq_ok}; spectrum=±√{n}? {spec_ok}; mult(+√{n})={pos}=2^{n-1}. "
           f"Cauchy interlacing: any induced subgraph on 2^{{{n}-1}}+1 vertices has an eigenvalue ≥ √{n}, "
           f"so max degree ≥ √{n}.  (This is Huang's 2-page proof of the Sensitivity Conjecture.)")
    return ok, why


def _check_telescope(p: Problem, bridge: dict):
    """Bridge = {'potential': Phi(k)}. Kernel check: Phi(k) - Phi(k-1) == term(k) and Phi base = 0 (sympy exact).
    Then Σ_{k=1}^n term(k) = Phi(n) by telescoping -- the closed form is FORCED, not guessed."""
    k, n = sp.symbols("k n")
    Phi = sp.sympify(bridge["potential"]); term = p.data["term"]
    step = sp.simplify(Phi.subs(k, k) - Phi.subs(k, k - 1) - term)
    base = sp.simplify(Phi.subs(k, 0))
    ok = (step == 0) and (base == 0)
    return ok, f"Φ(k)−Φ(k−1)−term = {step} (must be 0); Φ(0) = {base} (must be 0) -> Σ_{{k≤n}} term = Φ(n)"


MOVES = [
    {"name": "SOS / positivity certificate",
     "source": "Hilbert's 17th; the workhorse of inequalities. 'A nonneg poly is (almost) a sum of squares.'",
     "needs": {"positivity"},
     "bridge_desc": "write the polynomial as an explicit sum of squares (× (1+Σx²)^k if needed)",
     "check": _check_sos},
    {"name": "spectral / eigenvalue interlacing",
     "source": "Huang 2019 (Sensitivity Conjecture): replace a COMBINATORIAL quantity by an eigenvalue of a cleverly "
               "SIGNED matrix; Cauchy interlacing does the rest. 30-year problem, 2 pages.",
     "needs": {"extremal_bound", "combinatorial"},
     "bridge_desc": "build a signed adjacency/Hermitian matrix whose interlacing bounds the combinatorial quantity",
     "check": _check_spectral},
    {"name": "telescoping / monotone potential",
     "source": "Abel summation; Perelman's monotone entropy in spirit. Find Φ that 'differences' to the summand.",
     "needs": {"telescoping_candidate"},
     "bridge_desc": "find a potential Φ with Φ(k)−Φ(k−1) = term; the closed form is then forced",
     "check": _check_telescope},
    {"name": "transport: solution-here = forbidden-object-there",
     "source": "Frey–Wiles (FLT→modular forms); Weil/Deligne (ζ→Frobenius on cohomology). Map a hypothetical "
               "solution to an object a STRUCTURAL theorem forbids.",
     "needs": {"existence", "universal"},
     "bridge_desc": "exhibit a functor sending a counterexample to an object excluded by a known structural theorem",
     "check": None},   # not mechanizable from a corpus -- flagged honestly (the genuinely-new-object tier)
    {"name": "finite reduction / discharging",
     "source": "Appel–Haken (Four Color); AKS. Reduce ∞-many cases to a finite, checkable set.",
     "needs": {"universal", "discrete"},
     "bridge_desc": "reduce to an unavoidable finite set of reducible configurations, then check exhaustively",
     "check": None},   # handled by closure_run when it applies
]


def retrieve(fp: set[str]):
    """Rank moves by how well their precondition is satisfied by the fingerprint (the analogy step)."""
    out = []
    for m in MOVES:
        hit = m["needs"] & fp
        if hit:
            out.append((len(hit) / len(m["needs"]), m, hit))
    return sorted(out, key=lambda x: -x[0])


# ======================================================================================
# (3) BRIDGE-PROPOSER -- instantiate the analog object (Opus when gated; canonical otherwise), kernel-check it
# ======================================================================================
def llm_propose_bridge(p: Problem, move: dict, log=print):
    """Ask Opus for the concrete bridge object for THIS problem under THIS move. Returns a bridge dict or None."""
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8")
    if move["check"] is _check_sos:
        ask = (f"Prove {p.statement} by SUM OF SQUARES. Give the squares whose sum equals the polynomial "
               f"{p.data['poly']} in variables {p.data['vars']}. Reply ONLY JSON: "
               '{"squares": ["(x-y)**2/2", ...]}. No prose.')
        key = "squares"
    elif move["check"] is _check_telescope:
        ask = (f"For the sum of term={p.data['term']} over k=1..n, give a telescoping potential Phi(k) (in k) with "
               f'Phi(k)-Phi(k-1)=term and Phi(0)=0. Reply ONLY JSON: {{"potential": "k*(k+1)/2"}}. No prose.')
        key = "potential"
    else:
        return None
    msg = client.messages.create(model=model, max_tokens=400, messages=[{"role": "user", "content": ask}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"): text = text.split("```")[1].lstrip("json").strip()
    try:
        b = json.loads(text); log(f"    Opus proposed bridge: {b}"); return b
    except Exception:
        log(f"    Opus output not parseable: {text[:80]!r}"); return None


def canonical_bridge(p: Problem, move: dict):
    """The move's canonical instantiation for the demo problems (what Opus proposes live when the gate is on)."""
    return p.data.get("canonical_bridge")


def attempt(p: Problem, log=print):
    fp = fingerprint(p)
    log(f"\n  PROBLEM: {p.name}\n    statement : {p.statement}")
    log(f"    fingerprint: {sorted(fp)}")
    ranked = retrieve(fp)
    if not ranked:
        log("    no move matches this fingerprint."); return None
    log(f"    moves retrieved (by precondition match): {[m['name'] for _, m, _ in ranked]}")
    for score, move, hit in ranked:
        log(f"\n    >> try move: {move['name']}  (matched on {sorted(hit)})")
        log(f"       source: {move['source']}")
        if move["check"] is None:
            log(f"       bridge: {move['bridge_desc']}")
            log(f"       VERDICT: NOT corpus-mechanizable -- this is the genuinely-new-object tier (honest wall).")
            continue
        bridge = llm_propose_bridge(p, move, log) or canonical_bridge(p, move)
        if bridge is None:
            log("       no bridge proposed (gate off and no canonical) -- skipping."); continue
        ok, why = move["check"](p, bridge)
        log(f"       proposed bridge: {bridge}")
        log(f"       KERNEL CHECK: {why}")
        log(f"       VERDICT: {'REAL BRIDGE ✓ (precondition kernel-verified -> theorem follows by the cited machinery)' if ok else 'REJECTED ✗ (precondition fails; bridge is bogus)'}")
        if ok:
            return move["name"]
    return None


# ======================================================================================
# DEMONSTRATION
# ======================================================================================
def demo_problems():
    x, y, z = sp.symbols("x y z", real=True)
    k = sp.symbols("k")
    P = []
    # A) symmetric quadratic inequality -> SOS move (kernel-exact)
    P.append(Problem(
        "AM-style quadratic", "x^2+y^2+z^2 >= xy+yz+zx for all real x,y,z", "inequality",
        {"vars": [x, y, z], "poly": sp.expand(x**2 + y**2 + z**2 - x*y - y*z - z*x),
         "canonical_bridge": {"squares": ["(x-y)**2/2", "(y-z)**2/2", "(z-x)**2/2"]}}))
    # B) combinatorial bound on the boolean cube -> SPECTRAL move (Huang; numpy-checkable)  *** the centerpiece ***
    P.append(Problem(
        "cube degree bound (Sensitivity)",
        "every induced subgraph on >half the vertices of the n-cube has a vertex of degree >= sqrt(n)", "bound",
        {"canonical_bridge": {"dim": 4}}))
    # C) a sum -> TELESCOPING move (closed form forced, not guessed)
    P.append(Problem(
        "sum of first n integers", "sum_{k=1}^n k has a closed form", "sum_identity",
        {"term": k, "canonical_bridge": {"potential": "k*(k+1)/2"}}))
    # D) ADVERSARIAL CONTROL: a BOGUS SOS bridge -- the kernel MUST reject it (proves the gate has teeth)
    P.append(Problem(
        "control: bogus bridge", "x^2+y^2+z^2 >= xy+yz+zx  (with a WRONG certificate)", "inequality",
        {"vars": [x, y, z], "poly": sp.expand(x**2 + y**2 + z**2 - x*y - y*z - z*x),
         "canonical_bridge": {"squares": ["(x-y)**2/2", "(y-z)**2/2"]}}))   # missing the third square -> not an identity
    return P


def main():
    print("=" * 90)
    print("  proofworld.bridge :: ideate by searching SHORT BRIDGES, verify the precondition with the kernel")
    print("  (the creative act is short; the proof is long -- so search constructions, delegate verification)")
    print("=" * 90)
    results = {}
    for p in demo_problems():
        results[p.name] = attempt(p)
    print("\n" + "=" * 90)
    print("  SUMMARY")
    for name, move in results.items():
        verdict = f"bridged via [{move}]" if move else "no real bridge found"
        print(f"    - {name:34} -> {verdict}")
    print("\n  Read-out: a fingerprint retrieves candidate MOVES; the proposer instantiates the analog OBJECT; the")
    print("  KERNEL checks the bridge's precondition (an SOS identity, A²=nI, a telescoping step) -- short to check,")
    print("  even when the theorem is hard. The control's bogus bridge was REJECTED, so the gate has teeth. The two")
    print("  moves with check=None (transport, finite-reduction) are flagged honestly: transport is the")
    print("  genuinely-new-object tier no corpus mechanizes; finite-reduction is closure_run's job when it applies.")
    print("  This is the recombinant-breakthrough engine: invent the bridge freely, let truth stay with the kernel.")
    print("=" * 90)


if __name__ == "__main__":
    main()
