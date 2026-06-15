#!/usr/bin/env python3
"""proofworld.research_hard -- research_loop on a goal z3 CANNOT close alone: the dreamed lemmas are LOGICALLY
LOAD-BEARING, not a legibility aid.

The goal is UNDERDETERMINED: about quantities whose value z3 cannot pin down without help, so z3 alone returns a
COUNTEREXAMPLE -- it genuinely cannot prove it. A proof exists only by citing established facts and chaining them.
This makes the search load-bearing, and it forces the discipline the SOS demo didn't need:

  SOUNDNESS GATE: a dreamed lemma is usable only if it is JUSTIFIED -- entailed by the established library. A lemma
  the LLM invents that is NOT entailed is rejected as UNSOUND, *even if assuming it would close the goal*. (You
  cannot assume your way to a proof: the anti-cheat that keeps a creative generator from "dreaming" the answer.)

  load-bearing check : z3 on the bare goal -> counterexample (cannot close)
  generate           : LLM dreams intermediate lemmas (relations among the quantities)
  soundness gate     : keep only lemmas ENTAILED by the established library (z3); reject unjustified ones
  composition        : minimal subset of JUSTIFIED lemmas that ENTAILS the goal (z3 owns truth) -> recorded

Run:  PROOFWORLD_LLM=1 python3 -m proofworld.research_hard
"""
from __future__ import annotations
from itertools import combinations
import os, json, z3

# quantities the goal is about (think: f(x), g(x), h(x) at a point, plus two unrelated quantities)
A, B, C, D, Ee = z3.Reals("A B C D E")
VARS = {"A": A, "B": B, "C": C, "D": D, "E": Ee}

# the ESTABLISHED LIBRARY: facts already proven/assumed in this world. The proof may cite ONLY these (or what they
# entail). Note the library does NOT directly state C>=0 -- that must be DERIVED by chaining.
LIBRARY = {
    "A >= 0":  A >= 0,
    "A <= B":  A <= B,
    "B <= C":  B <= C,
    "D <= 5":  D <= 5,        # decoy fact (irrelevant to the goal)
    "E >= -3": Ee >= -3,      # decoy fact
}
GOAL = C >= 0
GOAL_DESC = "C >= 0"
LIBRARY_DESC = "{A>=0, A<=B, B<=C, D<=5, E>=-3}"


def parse_rel(s: str):
    """parse an inequality over A,B,C,D,E into a z3 Bool."""
    s = s.strip().rstrip(".")
    for op in (">=", "<=", "=="):
        if op in s:
            l, r = s.split(op, 1); env = {"__builtins__": {}}
            lhs, rhs = eval(l, env, VARS), eval(r, env, VARS)
            return {">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs}[op]
    raise ValueError(f"no comparator in {s!r}")


def entails(hyps, concl, t=4000) -> bool:
    """z3: do the hypotheses entail the conclusion? (hyps & not concl unsat)."""
    s = z3.Solver(); s.set("timeout", t)
    for h in hyps: s.add(h)
    s.add(z3.Not(concl)); return s.check() == z3.unsat

def is_sound(lemma, t=4000) -> bool:
    """SOUNDNESS: is the lemma JUSTIFIED -- entailed by the established library? If not, citing it is unsound."""
    return entails(list(LIBRARY.values()), lemma, t)


def llm_dream_lemmas(log=print):
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        log("  (LLM gate off or no key -- using a fixed candidate pool so the demo still runs)")
        return ["A >= 0", "B >= 0", "A <= B", "B <= C", "C >= A", "D >= 0", "E >= 0"]   # mix: sound, derived, decoy, UNSOUND
    import anthropic
    client = anthropic.Anthropic()
    prompt = (
        f"Established facts (a 'library' of proven inequalities): {LIBRARY_DESC}.\n"
        f"GOAL to prove: {GOAL_DESC}.\n"
        "Propose up to 7 intermediate lemmas (inequalities among A,B,C,D,E) that, chained, would prove the goal "
        "from the established facts. Use ONLY variables A,B,C,D,E and one comparator (>=, <=) per lemma. "
        'Reply ONLY with a JSON array of strings, e.g. ["A >= 0", "B >= 0"]. No prose.'
    )
    msg = client.messages.create(model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-sonnet-4-6"),
                                 max_tokens=400, messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"): text = text.split("```")[1].lstrip("json").strip()
    try:
        return [str(s) for s in json.loads(text)]
    except Exception:
        log(f"  LLM output not JSON; head {text[:80]!r}"); return []


def is_goal_restatement(f, t=4000) -> bool:
    """ANTI-CIRCULARITY: a lemma logically EQUIVALENT to the goal (entails it AND is entailed by it) is just the
    goal restated -- not a proof. Excluded from the proof atoms so the search surfaces the explanatory chain."""
    return entails([f], GOAL, t) and entails([GOAL], f, t)


def minimal_proof(sound_lemmas, log=print):
    """smallest subset of JUSTIFIED, non-circular lemmas that entails the goal. ascending size; z3 owns truth."""
    usable = [(s, f) for s, f in sound_lemmas if not is_goal_restatement(f)]
    names = [s for s, _ in usable]; forms = [f for _, f in usable]
    calls = 0
    for size in range(1, len(names) + 1):
        for combo in combinations(range(len(names)), size):
            calls += 1
            if entails([forms[i] for i in combo], GOAL):
                return [names[i] for i in combo], calls
    return None, calls


def main():
    print("=== proofworld.research_hard :: a goal z3 CANNOT close alone -- dreamed lemmas are LOAD-BEARING ===\n")
    print(f"  established library: {LIBRARY_DESC}")
    print(f"  GOAL: {GOAL_DESC}\n")
    # 1) LOAD-BEARING CHECK: z3 on the bare goal (no library) cannot close it
    s = z3.Solver(); s.add(z3.Not(GOAL)); r = s.check()
    print(f"  z3 on the BARE goal (no lemmas): {'CANNOT CLOSE -- counterexample ' + str(s.model()) if r == z3.sat else r}")
    print(f"  -> the proof must CITE and CHAIN established facts; lemmas are logically necessary here.\n")
    # 2) GENERATE
    raw = llm_dream_lemmas()
    raw = list(raw) + ["C >= 5"]                       # inject an UNSOUND-but-HELPFUL canary (would close the goal!)
    print(f"  candidate lemmas dreamed: {raw}\n")
    # 3) SOUNDNESS GATE: keep only lemmas the library actually entails
    print("  --- SOUNDNESS GATE (a lemma is usable only if JUSTIFIED by the library) ---")
    sound, unsound_helpful_rejected = [], False
    for sstr in raw:
        try:
            f = parse_rel(sstr)
        except Exception as e:
            print(f"    {sstr:12} -> UNPARSEABLE ({e})"); continue
        if is_sound(f):
            print(f"    {sstr:12} -> JUSTIFIED (entailed by library)"); sound.append((sstr, f))
        else:
            would_close = entails([f], GOAL)
            note = "  <-- would CLOSE the goal, but UNJUSTIFIED: rejected (anti-cheat)" if would_close else ""
            print(f"    {sstr:12} -> UNSOUND (not entailed by library){note}")
            if would_close and sstr == "C >= 5": unsound_helpful_rejected = True
    # dedup sound lemmas by string
    seen = set(); sound = [(s, f) for s, f in sound if not (s in seen or seen.add(s))]
    restated = [s for s, f in sound if is_goal_restatement(f)]
    print(f"\n  justified lemmas: {len(sound)}" + (f"  (excluding {restated} as goal-restatements / circular)" if restated else "") + "\n")
    # 4) COMPOSITION: minimal justified, non-circular chain that entails the goal
    proof, calls = minimal_proof(sound)
    print(f"  --- minimal JUSTIFIED proof (non-circular; z3 owns truth on every trial; {calls} entailment-checks) ---")
    print(f"  proof: {' , '.join(proof) if proof else '(none found)'}  ==>  {GOAL_DESC}")
    # 5) RECORD + canaries
    print(f"\n  --- atlas record ---")
    print(f"  GOAL {GOAL_DESC}")
    print(f"    status: {'VERIFIED (by chaining justified facts)' if proof else 'OPEN'}")
    print(f"    proof : {' -> '.join(proof) if proof else '(none)'}")
    print(f"    technique: transitive bound chaining (cite-and-chain established facts)")
    print(f"\n  [canary] z3 could NOT close the bare goal alone (lemmas load-bearing): {'PASS' if r == z3.sat else 'FAIL'}")
    print(f"  [canary] an UNSOUND-but-helpful lemma (C>=5) was REJECTED, not used to cheat: {'PASS' if unsound_helpful_rejected else 'FAIL'}")
    ok = bool(proof) and r == z3.sat and unsound_helpful_rejected
    print(f"\n  GATE: {'PASS -- load-bearing lemmas, soundness enforced, minimal justified proof recorded' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
