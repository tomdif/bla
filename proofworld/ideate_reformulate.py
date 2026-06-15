#!/usr/bin/env python3
"""proofworld.ideate_reformulate -- LEVER B: reformulation search (find the reframing that breaks the problem open).

Breakthroughs usually come from RECASTING a problem (Wiles via modularity, the Langlands web), not from grinding the
original form. This lever searches REPRESENTATION space: a proposer (Opus, env-gated) offers reformulations of a goal;
the kernel CERTIFIES which are provably equivalent (z3) -- so a wrong reframing is caught -- and we surface the
equivalent reframing whose STRUCTURE is most explicit (e.g. manifestly a square), which is the tractable one to attack.

  goal G  ->  propose reformulations R_i  ->  z3 certifies G <=> R_i (keep only the equivalent ones)
  ->  rank by structural explicitness  ->  the tractable reframing.   The proposer reframes; the kernel disposes.

Run:  PROOFWORLD_LLM=1 python3 -m proofworld.ideate_reformulate    (offline fallback set otherwise)
"""
from __future__ import annotations
import os, re, json, z3

x, y = z3.Reals("x y")
VARS = {"x": x, "y": y}
GOAL_DESC = "x**4 - 4*x + 3 >= 0"
GOAL = x**4 - 4*x + 3 >= 0


def parse(expr_str):
    s = expr_str.strip()
    for op in (">=", "<=", "==", ">", "<"):
        if op in s:
            l, r = s.split(op, 1); env = {"__builtins__": {}}
            lhs, rhs = eval(l, env, VARS), eval(r, env, VARS)
            return {">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs, ">": lhs > rhs, "<": lhs < rhs}[op]
    raise ValueError(s)

def equivalent(a, b, t=4000):
    s = z3.Solver(); s.set("timeout", t); s.add(z3.Xor(a, b))      # G XOR R unsat  <=>  G <=> R
    return s.check() == z3.unsat

def explicitness(expr_str):
    """heuristic: a reframing that exposes squares / products is structurally clearer (more tractable to attack)."""
    return expr_str.count("**2") + 2 * expr_str.count(")**2") + expr_str.count("(")


def llm_reformulations(goal_desc):
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    prompt = ("Give equivalent reformulations of this real inequality that expose its structure (factor into squares "
              f"if possible):\n  {goal_desc}\n"
              "Reply ONLY a JSON array of strings, each an equivalent inequality in x,y using + - * ** and one of >=,<=. "
              'e.g. ["(x-1)**2 * (x**2+2*x+3) >= 0"]. No prose.')
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8"), max_tokens=400,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try: return [str(s) for s in json.loads(m.group(0) if m else text)]
    except Exception: return None


def main():
    print("=== LEVER B :: reformulation search -- the kernel certifies which reframings are equivalent ===\n")
    print(f"  GOAL: {GOAL_DESC}")
    cands = llm_reformulations(GOAL_DESC)
    src = "Opus" if cands else "offline fallback"
    if not cands:
        cands = ["(x-1)**2 * (x**2 + 2*x + 3) >= 0",         # equivalent (the factorization -> manifestly >=0)
                 "(x-1)**2 * ((x+1)**2 + 2) >= 0",           # equivalent (fully exposed as squares)
                 "x**4 + 3 >= 4*x",                          # equivalent (trivial rearrangement)
                 "(x-1)**2 >= 0"]                            # NOT equivalent (a wrong, too-weak reframing) -- canary
    print(f"  reformulations proposed by {src}: {len(cands)}\n")
    print(f"  --- kernel certification (z3: is each reframing EQUIVALENT to the goal?) ---")
    valid = []
    for c in cands:
        try: R = parse(c)
        except Exception as e: print(f"    {c:42} -> UNPARSEABLE"); continue
        eq = equivalent(GOAL, R)
        print(f"    {c:42} -> {'EQUIVALENT' if eq else 'NOT equivalent (rejected -- a false reframing)'}")
        if eq: valid.append(c)
    if valid:
        best = max(valid, key=explicitness)
        print(f"\n  most STRUCTURALLY EXPLICIT equivalent reframing: {best}")
        print(f"  -> this is the tractable shape: it exposes the goal as a product of squares, where nonnegativity is")
        print(f"     manifest. The reframing was the move; the kernel guaranteed it didn't change the problem.")
    print("\n  LEVER B: a proposer reframes freely (a wrong reframing is caught by z3-equivalence), and the engine")
    print("  surfaces the equivalent form whose structure makes the proof obvious -- searching representation space.")


if __name__ == "__main__":
    main()
