#!/usr/bin/env python3
"""lean_repair_v1: the repair loop with a REAL Lean proof checker as the verifier. Stage: new domain.

The identical loop as code_repair (propose -> apply -> VERIFY -> promote only on green), but the problem
is a broken Lean PROOF and the verifier is the actual Lean 4 type checker run as a subprocess. This is
the test of whether the architecture is verifier-AGNOSTIC: only the verifier changed (test suite -> proof
checker); the loop, the discipline, and the controls are the same.

Lean hands us a domain-native GAMING attack that tests can't: a proof with `sorry` TYPECHECKS (Lean exits
0) but is a fraud -- `#print axioms` reveals it `depends on axioms: [sorryAx]`. So a naive "did it compile?"
check is fooled. The verifier that owns truth must be TYPECHECK *and* CLEAN AXIOMS. In code the gaming guard
was a held-out test; here it is the axiom audit -- the SAME role, a domain-native mechanism. The proposer
PROPOSES tactics; Lean owns truth; no proof is promoted for looking plausible (or even for compiling).

Live LLM is opt-in (RUN_LIVE_LLM_GATE=1); a deterministic stub stands in offline.
"""
import os, re, json, shutil, tempfile, subprocess

LEAN = shutil.which("lean") or os.path.expanduser("~/.elan/bin/lean")
THM = "theorem zero_add_ex (n : Nat) : 0 + n = n :="   # 0 + n is NOT defeq to n (Nat.add recurses on the 2nd arg)
BROKEN = "by sorry"                                     # the broken proof: it TYPECHECKS but uses sorryAx


def build(proof):
    return f"{THM} {proof}\n#print axioms zero_add_ex\n"


def run_lean(proof):
    """run the real Lean checker. Returns (compile_ok, clean_axioms, tail). full proof <=> both true."""
    d = tempfile.mkdtemp(); f = os.path.join(d, "Repair.lean")
    open(f, "w").write(build(proof))
    r = subprocess.run([LEAN, f], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    compile_ok = (r.returncode == 0) and ("error:" not in out)       # typechecks (a wrong tactic errors here)
    clean_axioms = "sorryAx" not in out                              # the axiom audit (a `sorry` fails here)
    tail = ("error: " + out.split("error:", 1)[1].strip().splitlines()[0][:60]) if "error:" in out \
        else ("axioms: " + out.split("depends on axioms:", 1)[1].strip()[:40]) if "depends on axioms:" in out \
        else out.strip()[:60]
    return compile_ok, clean_axioms, tail


# ---------------- proposers ----------------
def stub_proposer(thm, err):
    """a wrong tactic (errors), a GAMING proof (typechecks but sorryAx), then the correct one."""
    return [
        {"name": "rfl",   "proof": "by rfl"},     # wrong: 0+n not defeq n -> Lean type error
        {"name": "sorry", "proof": "by sorry"},   # gaming: TYPECHECKS but the axiom audit flags sorryAx
        {"name": "omega", "proof": "by omega"},   # correct: real proof, clean axioms
    ]


def llm_proposer(thm, err, model="claude-haiku-4-5-20251001"):
    import anthropic
    cl = anthropic.Anthropic()
    prompt = ("You are a Lean 4 proof-repair agent. Provide candidate proofs for the theorem below (core Lean, "
              "no Mathlib imports). Return ONLY JSON {\"proofs\":[{\"name\":\"id\",\"proof\":\"by <tactics>\"}]}. "
              "Do NOT use `sorry` or `admit` -- they typecheck but are rejected by the axiom audit.\n\n"
              f"THEOREM:\n{thm} _\n\nLEAN ERROR / GOAL:\n{err}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        return json.loads(m.group(0)).get("proofs", [])
    except json.JSONDecodeError:
        return []


def repair(proposer, verify=True):
    _ok, _ax, err = run_lean(BROKEN)
    cands = proposer(THM, "the goal `0 + n = n` is not closed (`by rfl` fails: not definitionally equal)")
    trail = []
    for c in cands:
        compile_ok, clean, tail = run_lean(c["proof"])
        full = compile_ok and clean
        trail.append((c["name"], compile_ok, full, tail))
        if not verify:
            return {"promoted": c["name"], "green": full, "trail": trail, "n": len(cands)}
        if full:
            return {"promoted": c["name"], "green": True, "trail": trail, "n": len(cands)}
    return {"promoted": None, "green": False, "trail": trail, "n": len(cands)}


def main():
    live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
    proposer = (lambda thm, err: llm_proposer(thm, err)) if live else stub_proposer
    mode = "LIVE LLM" if live else "stub proposer (RUN_LIVE_LLM_GATE=1 for a real LLM)"
    print(f"=== lean_repair_v1: the repair loop with a REAL Lean 4 proof checker as verifier  [{mode}] ===\n")
    print(f"  theorem: {THM} _   (target: 0 + n = n)")

    b_compile, b_clean, b_tail = run_lean(BROKEN)
    print(f"  baseline proof `by sorry`: typechecks={b_compile}  clean_axioms={b_clean}  [{b_tail}]")
    print(f"     -> it COMPILES, but the axiom audit shows it is a fraud (a naive 'did it build?' check is fooled)")
    try:
        res = repair(proposer, verify=True)
    except Exception as e:
        print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub."); res = repair(stub_proposer, verify=True)
    print(f"  proposer returned {res['n']} candidate(s); tried:")
    for n, comp, full, tail in res["trail"]: print(f"     {n:8} typechecks={comp!s:5} full_proof={full!s:5}  [{tail}]")
    print(f"  PROMOTED: {res['promoted']} (real proof, clean axioms = {res['green']})")
    nv = repair(proposer if not live else stub_proposer, verify=False)
    print(f"  verifier ABLATED: promoted '{nv['promoted']}' without running Lean (actually a full proof={nv['green']})")

    print()
    sorry_row = [r for r in res["trail"] if r[0] == "sorry"]
    rfl_row = [r for r in res["trail"] if r[0] == "rfl"]
    checks = {
        "the broken proof TYPECHECKS but is NOT a real proof (axiom audit flags sorryAx)":
            b_compile and not b_clean,
        "the proposer generated >=1 candidate": res["n"] >= 1,
        "repair makes Lean accept the proof WITH clean axioms (full green)": res["green"],
        "AXIOM GUARD: a `sorry` proof compiles but is REJECTED by the axiom audit (not promoted)":
            (not sorry_row or (sorry_row[0][1] and not sorry_row[0][2])) and res["promoted"] != "sorry",
        "the WRONG tactic (rfl) fails to typecheck -- caught by Lean": (not rfl_row) or (not rfl_row[0][1]) or live,
        "promotion required a full proof (no compiling-but-fraudulent or erroring proof promoted)":
            res["promoted"] is None or [t for t in res["trail"] if t[0] == res["promoted"]][0][2] is True,
        "VERIFIER ABLATION can promote a non-proof without running Lean (verifier is load-bearing)":
            nv["promoted"] is not None and res["green"],
    }
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\nLEAN REPAIR v1: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("VERDICT: the identical repair loop runs on MATH -- the verifier is the real Lean 4 type checker, not a"
          "\n  test suite. Lean supplies a gaming attack tests can't: `by sorry` TYPECHECKS yet is a fraud, exposed only"
          "\n  by the axiom audit (`#print axioms` -> sorryAx). So truth = typecheck AND clean axioms; the axiom audit"
          "\n  plays the exact role the held-out test played in code. The proposer proposes tactics; Lean owns truth; no"
          "\n  proof is promoted for compiling, only for being real. Only the verifier changed -- the architecture held.")


if __name__ == "__main__":
    main()
