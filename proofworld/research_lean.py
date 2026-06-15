#!/usr/bin/env python3
"""proofworld.research_lean -- the FULLY load-bearing case: an INDUCTIVE goal, on the Lean kernel.

Here neither the facts nor the chaining are within z3's reach: the goal is about a recursively-defined function and
the proof needs INDUCTION, which is simply not in SMT. So z3/omega cannot close it even given every fact -- only a
verifier that accepts the `induction` tactic (Lean) can. We port research_hard's gate onto that kernel:

  LOAD-BEARING : the goal with non-inductive tactics (omega/simp/rfl) FAILS; only the inductive proof closes it.
  GENERATE     : candidate auxiliary lemmas (LLM-pluggable; a fixed pool here for determinism).
  SOUNDNESS    : a lemma is usable only if the Lean KERNEL can PROVE it (a real tactic, no `sorry`). A false-but-
                 helpful lemma -- one that WOULD close the goal if granted -- is REJECTED because it is unprovable.
                 (the anti-cheat: you cannot `sorry`/assume your way to a proof.)
  ANTI-CIRCULAR: a candidate equal to the goal is excluded as a restatement.
  COMPOSE      : the goal is proved BY INDUCTION citing the justified lemma; the Lean kernel confirms.

Run:  python3 -m proofworld.research_lean    (uses installed lean toolchain; ~1-3s per kernel call)
"""
from __future__ import annotations
import os, subprocess, tempfile
from proofworld.leankernel import TOOLCHAIN

DOUBLE = "def double : Nat → Nat\n  | 0 => 0\n  | (n+1) => double n + 2\n"   # double n = 2n, recursively

GOAL_STMT = "theorem G (n : Nat) : double n ≤ 2 * n + 10"
GOAL_PROP = "double n ≤ 2 * n + 10"
GOAL_DESC = "∀ n, double n ≤ 2*n + 10   (double defined recursively -> the proof needs INDUCTION)"

# the inductive proof skeleton, citing a named lemma for the recursion-unfolding step
IND = ("by\n  induction n with\n  | zero => simp [double]\n"
       "  | succ k ih => rw [show double (k+1) = double k + 2 from rfl]; omega")

# candidate auxiliary lemmas (a generator would dream these; fixed here for determinism)
CANDIDATES = [
    ("aux_unfold", "theorem aux_unfold (n : Nat) : double (n+1) = double n + 2", "needed for the inductive step"),
    ("aux_zero",   "theorem aux_zero : double 0 = 0",                            "true but irrelevant (decoy)"),
    ("aux_fh",     "theorem aux_fh (n : Nat) : double n ≤ n",               "FALSE but would close the goal (canary)"),
    ("aux_goal",   "theorem aux_goal (n : Nat) : double n ≤ 2 * n + 10",    "= the goal (circular)"),
]

TACTICS = ["by rfl", "by simp [double]", IND]      # the tactic-dreamer's repertoire (NO sorry -> cannot cheat)


def _lean(source: str, timeout=60):
    """run the Lean kernel on `source`; return (ok, has_sorry, first_error)."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "Check.lean")
        with open(f, "w") as fh: fh.write(source)
        try:
            p = subprocess.run(["elan", "run", TOOLCHAIN, "lean", f], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, False, "timeout"
    out = (p.stdout + p.stderr)
    err = next((ln.split("error:", 1)[1].strip() for ln in out.splitlines() if "error:" in ln), "")
    has_sorry = "sorry" in out
    return (p.returncode == 0 and "error:" not in out and not has_sorry), has_sorry, err


def prove_clean(statement: str):
    """SOUNDNESS: try to PROVE the statement with the kernel using real tactics (no sorry). Returns the winning
    tactic, or None if the kernel cannot prove it (false / out of reach) -- in which case it is NOT justified."""
    for tac in TACTICS:
        ok, _sorry, _err = _lean(DOUBLE + statement + " := " + tac + "\n")
        if ok:
            return tac
    return None


def prop_of(stmt: str) -> str:
    return stmt.split(") : ")[-1] if ") : " in stmt else stmt.split(" : ", 1)[-1]


def main():
    print("=== proofworld.research_lean :: INDUCTIVE goal on the Lean kernel -- facts AND chaining beyond z3 ===\n")
    print(f"  GOAL: {GOAL_DESC}\n")
    # 1) LOAD-BEARING: non-inductive tactics cannot close it
    print("  --- load-bearing check (omega/simp/rfl cannot do induction) ---")
    for tac in ["by omega", "by simp [double]", "by rfl"]:
        ok, _s, err = _lean(DOUBLE + GOAL_STMT + " := " + tac + "\n")
        print(f"    {GOAL_STMT.split(':')[0].strip()} := {tac:16} -> {'VERIFIED' if ok else 'FAILS'}" + (f"  ({err[:46]})" if not ok and err else ""))
    print("    => only an INDUCTIVE proof can close it; z3/omega are out of theory here.\n")
    # 2-4) SOUNDNESS gate + anti-circularity over the candidate lemmas
    print("  --- SOUNDNESS gate (a lemma is usable only if the KERNEL can prove it; no sorry) ---")
    justified = []
    for name, stmt, note in CANDIDATES:
        tac = prove_clean(stmt)
        if tac is not None:
            circular = prop_of(stmt).strip() == GOAL_PROP.strip()
            tag = "JUSTIFIED" + ("  (but CIRCULAR = the goal -> excluded as a restatement)" if circular else f"  [via {tac.split(chr(10))[0]}]")
            print(f"    {name:11} -> {tag}")
            if not circular: justified.append((name, stmt, tac))
        else:
            # is it nonetheless 'helpful' -- would granting it (sorry) close the goal? (the anti-cheat case)
            # judge on "no real error": the sorry is the deliberate granted assumption, not a failure.
            _hok, _hs, h_err = _lean(DOUBLE + "theorem G_if (n : Nat) : " + GOAL_PROP +
                                     " := by\n  have h : " + prop_of(stmt).strip() + " := by sorry\n  omega\n")
            helpful = (h_err == "")
            extra = "  <-- would CLOSE the goal if granted, but UNPROVABLE: REJECTED (anti-cheat)" if helpful else ""
            print(f"    {name:11} -> UNSOUND (kernel cannot prove it){extra}")
    print(f"\n  justified, non-circular lemmas: {[n for n, _, _ in justified]}\n")
    # 5) COMPOSE: prove the goal by induction, citing a justified lemma; the kernel confirms
    print("  --- compose: prove the goal BY INDUCTION citing the justified lemma (kernel owns truth) ---")
    cite = next((s for n, s, _ in justified if n == "aux_unfold"), None)
    source = DOUBLE + cite + " := by rfl\n" + \
        GOAL_STMT + " := by\n  induction n with\n  | zero => simp [double]\n  | succ k ih => rw [aux_unfold]; omega\n"
    ok, _s, err = _lean(source)
    print(f"    aux_unfold (rfl)  +  G by induction [rw aux_unfold; omega]  ->  {'VERIFIED by Lean' if ok else 'FAILED: ' + err}")
    # 6) record + canaries
    print(f"\n  --- atlas record ---")
    print(f"  GOAL {GOAL_DESC}")
    print(f"    status   : {'VERIFIED (Lean kernel, by induction)' if ok else 'OPEN'}")
    print(f"    proof    : induction on n; step rewrites via aux_unfold (double (n+1) = double n + 2)")
    print(f"    technique: structural induction + recursion-unfolding lemma")
    nonind_fail = not _lean(DOUBLE + GOAL_STMT + " := by omega\n")[0]
    canary_rejected = prove_clean("theorem aux_fh (n : Nat) : double n ≤ n") is None
    print(f"\n  [canary] goal NOT closable without induction (omega fails): {'PASS' if nonind_fail else 'FAIL'}")
    print(f"  [canary] false-but-helpful lemma (double n ≤ n) REJECTED as unprovable: {'PASS' if canary_rejected else 'FAIL'}")
    good = ok and nonind_fail and canary_rejected
    print(f"\n  GATE: {'PASS -- inductive proof, facts+chaining beyond z3, soundness enforced on the Lean kernel' if good else 'FAIL'}")


if __name__ == "__main__":
    main()
