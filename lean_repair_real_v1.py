#!/usr/bin/env python3
"""lean_repair_real_v1: the repair loop on a REAL Mathlib-backed Lean repo (the SWE-bench of math).

The code_repair_v2 analog for Lean: take a REAL proven lemma in a real research repo (collatz-proven,
which is sorry-FREE -- the whole point of it), inject `sorry` into one lemma to create a genuine open
obligation in real Mathlib context, and run the repair loop to re-derive a real proof. The verifier is
the actual project build (`lake env lean` reusing the repo's cached Mathlib + project oleans) PLUS the
axiom audit (`#print axioms` must not show sorryAx). The original file is restored byte-for-byte at the
end; `lake env lean` only elaborates -- it never rewrites the olean cache -- so the repo is untouched.

Target lemma (Collatz2/SafePeakCore.lean): safePeakIterValue iteration composes over addition --
  safePeakIterValue (m + n) v = safePeakIterValue n (safePeakIterValue m v)
a real lemma with an honest inductive proof, re-derivable but not trivial.

Same discipline: the proposer proposes tactics; the real Lean build owns truth; a `sorry` typechecks but
is rejected by the axiom audit. Live LLM opt-in (RUN_LIVE_LLM_GATE=1); deterministic stub otherwise.
"""
import os, re, json, shutil, subprocess

REPO = os.path.expanduser("~/collatz-proven")
TARGET = os.path.join(REPO, "Collatz2/SafePeakCore.lean")
LEMMA = "safePeakIterValue_add"
ANCHOR = "safePeakIterValue (m + n) v = safePeakIterValue n (safePeakIterValue m v) := by"
TAIL = "/-! ## Descent Checking -/"
ORIG = open(TARGET).read()
assert ANCHOR in ORIG and TAIL in ORIG, "anchors not found -- file changed"
HEAD = ORIG[:ORIG.index(ANCHOR) + len(ANCHOR)]
REST = ORIG[ORIG.index(TAIL):]


def write_proof(proof_block):
    """splice a candidate proof in place of the original, with an axiom-audit probe after it."""
    body = f"{HEAD}\n{proof_block}\n\n#print axioms {LEMMA}\n\n{REST}"
    open(TARGET, "w").write(body)


def restore():
    open(TARGET, "w").write(ORIG)


def verify():
    """run the REAL project build on the file. Returns (compile_ok, clean_axioms, tail)."""
    r = subprocess.run(["lake", "env", "lean", TARGET], cwd=REPO, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    compile_ok = (r.returncode == 0) and ("error:" not in out)
    clean = "sorryAx" not in out
    if "error:" in out: tail = "error: " + out.split("error:", 1)[1].strip().splitlines()[0][:55]
    elif "depends on axioms:" in out: tail = "axioms: " + out.split("depends on axioms:", 1)[1].strip()[:45]
    elif "does not depend on any axioms" in out: tail = "axioms: (none)"
    else: tail = out.strip()[:55]
    return compile_ok, clean, tail


# ---------------- proposers ----------------
def stub_proposer(stmt, err):
    return [
        {"name": "simp_only",  "proof": "  simp [safePeakIterValue]"},                          # no induction -> unsolved
        {"name": "sorry",      "proof": "  sorry"},                                              # gaming: typechecks, sorryAx
        {"name": "induction",  "proof": ("  induction m generalizing v with\n"
                                         "  | zero => simp [safePeakIterValue]\n"
                                         "  | succ m ih => simp [safePeakIterValue, Nat.succ_add, ih]")},
    ]


def llm_proposer(stmt, err, model="claude-sonnet-4-6"):
    import anthropic
    cl = anthropic.Anthropic()
    ctx = ("def safePeakStepValue (v : Nat) : Nat := ...\n"
           "def safePeakIterValue : Nat -> Nat -> Nat\n"
           "  | 0, v => v\n  | k + 1, v => safePeakIterValue k (safePeakStepValue v)\n")
    prompt = ("You are a Lean 4 (+Mathlib) proof-repair agent. Prove the theorem below. Return ONLY JSON "
              "{\"proofs\":[{\"name\":\"id\",\"proof\":\"<SINGLE-LINE tactic proof, NO newlines, NO leading `by`>\"}]} "
              "with 3 to 4 DISTINCT candidates, including at least one using `induction m generalizing v`. "
              "Note the recursion is on the FIRST argument, so reducing safePeakIterValue (m+n) needs induction on m "
              "(generalizing v). Put each whole proof on ONE line, e.g. `induction m generalizing v with | zero => simp "
              "[safePeakIterValue] | succ k ih => simp [safePeakIterValue, Nat.succ_add, ih]`. Do NOT use `sorry`/`admit` "
              "(they typecheck but the axiom audit rejects them).\n\n"
              f"DEFINITION:\n{ctx}\nTHEOREM:\n{stmt} := by\n\nHINT/ERROR:\n{err}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=700,
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()      # strip markdown fences
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        proofs = json.loads(m.group(0)).get("proofs", [])
    except json.JSONDecodeError:
        return []
    return [{"name": p.get("name", "llm"), "proof": "  " + p["proof"].strip()} for p in proofs if p.get("proof")]


def repair(proposer, verify_on=True):
    cands = proposer(f"theorem {LEMMA} (m n v : Nat) : safePeakIterValue (m + n) v = "
                     "safePeakIterValue n (safePeakIterValue m v)",
                     "the goal does not reduce definitionally; induction on m (generalizing v) is needed")
    trail = []
    for c in cands:
        write_proof(c["proof"]); compile_ok, clean, tail = verify(); full = compile_ok and clean
        trail.append((c["name"], compile_ok, full, tail))
        if not verify_on:
            return {"promoted": c["name"], "green": full, "trail": trail, "n": len(cands)}
        if full:
            return {"promoted": c["name"], "green": True, "trail": trail, "n": len(cands)}
    return {"promoted": None, "green": False, "trail": trail, "n": len(cands)}


def main():
    live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
    proposer = (lambda s, e: llm_proposer(s, e)) if live else stub_proposer
    mode = "LIVE LLM" if live else "stub proposer (RUN_LIVE_LLM_GATE=1 for a real LLM)"
    print(f"=== lean_repair_real_v1: repair loop on a REAL Mathlib repo (collatz-proven)  [{mode}] ===")
    print(f"  target: {os.path.relpath(TARGET, REPO)} :: {LEMMA}")
    try:
        write_proof("  sorry"); b_compile, b_clean, b_tail = verify()
        print(f"  injected `sorry`: builds={b_compile}  clean_axioms={b_clean}  [{b_tail}]")
        print(f"     -> the project BUILDS with the sorry, but the axiom audit exposes the open obligation")
        try:
            res = repair(proposer, verify_on=True)
            if res["n"] == 0 and live:
                print("  LIVE LLM returned 0 parseable candidates; falling back to stub."); proposer = stub_proposer; res = repair(stub_proposer, verify_on=True)
        except Exception as e:
            print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub."); proposer = stub_proposer; res = repair(stub_proposer, verify_on=True)
        print(f"  proposer returned {res['n']} candidate(s); tried (real `lake env lean` each):")
        for n, comp, full, tail in res["trail"]: print(f"     {n:10} builds={comp!s:5} full_proof={full!s:5}  [{tail}]")
        print(f"  PROMOTED: {res['promoted']} (real proof, clean axioms = {res['green']})")
        nv = repair(proposer if not live else stub_proposer, verify_on=False)
        print(f"  verifier ABLATED: promoted '{nv['promoted']}' without building (actually a full proof={nv['green']})")
    finally:
        restore()
        assert open(TARGET).read() == ORIG, "RESTORE FAILED -- original not recovered!"
        print(f"  [restored {os.path.relpath(TARGET, REPO)} to original; repo untouched]")

    print()
    srow = [r for r in res["trail"] if r[0] == "sorry"]
    checks = {
        "injecting `sorry` BUILDS but is flagged by the axiom audit (a naive 'does it build?' is fooled)":
            b_compile and not b_clean,
        "the proposer generated >=1 candidate": res["n"] >= 1,
        "repair re-derives a REAL proof: the project build accepts it with clean axioms": res["green"],
        "AXIOM GUARD: a `sorry` builds but is REJECTED by the axiom audit (not promoted)":
            (not srow or (srow[0][1] and not srow[0][2])) and res["promoted"] != "sorry",
        "promotion required a full proof verified by the REAL repo build (no sorry/error promoted)":
            res["promoted"] is None or [t for t in res["trail"] if t[0] == res["promoted"]][0][2] is True,
        "VERIFIER ABLATION can promote a non-proof without building (verifier is load-bearing)":
            nv["promoted"] is not None and res["green"],
    }
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\nLEAN REPAIR (REAL REPO) v1: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("VERDICT: the repair loop runs on a REAL Mathlib-backed research repo -- the verifier is the actual project"
          "\n  build (`lake env lean` over cached Mathlib + project oleans) plus the axiom audit. A `sorry` injected into a"
          "\n  real proven lemma BUILDS (a naive check passes) but the axiom audit exposes it; the loop re-derives a real"
          "\n  inductive proof that the project build accepts with clean axioms. The original file is restored byte-for-byte."
          "\n  Same loop as the toy Lean and the code repos -- only the verifier scaled, to a real proof assistant on real math.")


if __name__ == "__main__":
    main()
