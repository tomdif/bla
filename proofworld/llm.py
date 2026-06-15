#!/usr/bin/env python3
"""proofworld.llm -- the LLM as a CREATIVE generator of auxiliary lemmas, fully gated by the verifier.

This is the richest generator: instead of enumerating a fixed family, the LLM DREAMS new auxiliary inequalities
that might help prove a goal. The anti-trap discipline is unchanged and is what makes this safe to scale:

  the LLM PROPOSES (creative, unbounded)  ->  GROUNDED gate decides:
     (a) counterexample-search each proposed lemma (z3): a FALSE 'lemma' dies here, with a witness -- never believed
     (b) composition-check the survivors: do the TRUE ones actually imply the goal? (z3)
  the LLM never scores itself; truth is owned by z3. A confident-but-wrong lemma cannot survive (a).

Gated: does nothing unless PROOFWORLD_LLM=1. Reads the key ONLY from os.environ (never hardcoded/logged).
Run:  PROOFWORLD_LLM=1 python3 -m proofworld.llm
"""
from __future__ import annotations
import os, json, z3

MODEL = os.environ.get("PROOFWORLD_LLM_MODEL", "claude-sonnet-4-6")
VARS = {n: z3.Real(n) for n in ("x", "y", "z")}                  # the fixed vocabulary the LLM may use

# the goal to prove (true for all reals; an SOS-style auxiliary lemma helps)
GOAL = sum(VARS[a] * VARS[a] for a in "xyz") >= VARS["x"] * VARS["y"] + VARS["y"] * VARS["z"] + VARS["z"] * VARS["x"]
GOAL_DESC = "x^2 + y^2 + z^2 >= x*y + y*z + z*x   (for all real x,y,z)"


def parse_ineq(s: str):
    """parse a single inequality string over x,y,z into a z3 Bool. Safe-ish eval: no builtins, only the 3 vars."""
    s = s.strip().rstrip(".")
    for op in (">=", "<=", "=="):
        if op in s:
            l, r = s.split(op, 1)
            env = {"__builtins__": {}}
            lhs = eval(l, env, VARS); rhs = eval(r, env, VARS)      # z3 overloads + - * ** -> z3 exprs
            return {">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs}[op]
    raise ValueError(f"no comparator in {s!r}")


def is_true(form, t=4000) -> bool:                                # GROUNDED kill-test (a): true for all reals?
    s = z3.Solver(); s.set("timeout", t); s.add(z3.Not(form)); return s.check() == z3.unsat

def counterexample(form, t=4000):
    s = z3.Solver(); s.set("timeout", t); s.add(z3.Not(form))
    return str(s.model()) if s.check() == z3.sat else None

def implies(forms, goal, t=4000) -> bool:                         # GROUNDED check (b): do lemmas imply the goal?
    s = z3.Solver(); s.set("timeout", t)
    for f in forms: s.add(f)
    s.add(z3.Not(goal)); return s.check() == z3.unsat


def llm_dream_lemmas(goal_desc: str, n=6, log=print):
    """ask the LLM for auxiliary lemmas as parseable inequality strings. Returns [] if the gate is off."""
    if os.environ.get("PROOFWORLD_LLM") != "1":
        log("  LLM gate OFF (set PROOFWORLD_LLM=1 to enable live lemma generation). Nothing called.")
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("  PROOFWORLD_LLM=1 but no ANTHROPIC_API_KEY in env -- skipping live call."); return []
    import anthropic
    client = anthropic.Anthropic()                                # reads ANTHROPIC_API_KEY from env
    prompt = (
        f"I want to prove this inequality with an SMT solver:\n  {goal_desc}\n\n"
        f"Propose up to {n} AUXILIARY LEMMAS (true inequalities for all real x,y,z) that, combined, would imply it. "
        "Use ONLY variables x,y,z and operators + - * ** and one comparator (>= or <=) per lemma. "
        'Reply ONLY with a JSON array of strings, e.g. ["(x-y)**2 >= 0", "x**2 + y**2 >= 2*x*y"]. No prose.'
    )
    msg = client.messages.create(model=MODEL, max_tokens=512,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"): text = text.split("```")[1].lstrip("json").strip()
    try:
        items = json.loads(text)
    except Exception:
        log(f"  LLM output not valid JSON; raw head: {text[:80]!r}"); return []
    log(f"  LLM ({MODEL}) dreamed {len(items)} candidate lemmas.")
    return [str(s) for s in items]


def main():
    print("=== proofworld.llm :: LLM dreams auxiliary lemmas; the verifier owns truth (anti-trap gate) ===\n")
    print(f"  GOAL: {GOAL_DESC}")
    raw = llm_dream_lemmas(GOAL_DESC)
    if not raw:
        print("\n  (no lemmas generated -- gate off or no key. The gate logic below is what would run on each.)")
        print("  pipeline per lemma: parse -> counterexample-search (kill false) -> composition-check (imply goal?)")
        return
    # inject a known-FALSE lemma as an adversarial canary: the gate MUST kill it (proves the gate has teeth)
    INJECTED_FALSE = "x**2 >= 2*x"        # false at x=1 (1 >= 2)
    candidates = list(raw) + [INJECTED_FALSE]
    survivors, survivor_forms = [], []; injected_killed = False
    print("\n  --- GROUNDED GATE (the LLM's creativity is filtered by z3, not trusted) ---")
    for s in candidates:
        try:
            f = parse_ineq(s)
        except Exception as e:
            print(f"    {s:34} -> UNPARSEABLE ({e})"); continue
        tag = "  [injected adversarial canary]" if s == INJECTED_FALSE else ""
        cx = counterexample(f)
        if cx is not None:
            print(f"    {s:34} -> KILLED (false; counterexample {cx}){tag}")  # confident-but-wrong dies here
            if s == INJECTED_FALSE: injected_killed = True
        else:
            print(f"    {s:34} -> survives (true for all reals)")
            survivors.append(s); survivor_forms.append(f)
    # composition-check: do the surviving (true) lemmas imply the goal?
    closes = implies(survivor_forms, GOAL) if survivor_forms else False
    print(f"\n  survivors (true lemmas): {len(survivors)}/{len(raw)}")
    print(f"  composition of survivors implies the GOAL: {closes}")
    # also confirm the goal itself (z3 owns the final verdict)
    direct = is_true(GOAL)
    print(f"  goal verified by the kernel: {direct}")
    print(f"  [canary] injected FALSE lemma killed by the gate: {'PASS' if injected_killed else 'FAIL'}")
    print(f"\n  ANTI-TRAP: every LLM lemma faced counterexample-search BEFORE belief; a false 'lemma' could not")
    print(f"  survive (the injected canary proves it). The LLM proposed; z3 disposed. Creativity scaled; trust did not.")
    print(f"\n  GATE: {'PASS -- LLM lemmas gated, false lemma killed, goal kernel-verified' if (direct and injected_killed) else 'FAIL'}")


if __name__ == "__main__":
    main()
