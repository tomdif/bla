#!/usr/bin/env python3
"""proofworld.research_mathlib -- Mathlib behind the kernel: the gate on a goal needing a NONLINEAR tactic.

This deepens research_lean by putting Mathlib's tactic library behind the verifier. The goal is a nonlinear bound on
a recursively-defined function -- so it is out of reach for BOTH earlier kernels:
  * z3 / omega : cannot do induction, and cannot relate (k+1)^2 to k^2 (treats them as unrelated atoms);
  * CORE Lean  : has no `ring` / `nlinarith` to discharge the nonlinear step.
Only Mathlib's `nlinarith` (with induction) closes it. We reuse a BUILT Mathlib from one of the user's Lean
projects (no rebuild) via `lake env lean`, and run the same gate as research_hard / research_lean:

  LOAD-BEARING : omega fails; the inductive step closed by omega (no Mathlib) fails; only nlinarith closes it.
  SOUNDNESS    : a lemma is usable only if the Mathlib kernel PROVES it (real tactic, no sorry). A false-but-
                 helpful lemma (oddSum n <= n, would close the goal if granted) is REJECTED as unprovable.
  ANTI-CIRCULAR: a candidate equal to the goal is excluded.
  COMPOSE      : the goal is proved by induction + nlinarith, citing the justified unfolding lemma; kernel confirms.

Run:  python3 -m proofworld.research_mathlib   (reuses a built Mathlib; ~14s per kernel call)
"""
from __future__ import annotations
import os, glob, subprocess, tempfile

# ---- discover a project with a BUILT Mathlib whose toolchain is installed (no rebuild) ----
def _installed_toolchains():
    return {os.path.basename(p).split("---")[-1] for p in glob.glob(os.path.expanduser("~/.elan/toolchains/leanprover--lean4---*"))}

def find_mathlib_project():
    if os.environ.get("PROOFWORLD_MATHLIB_PROJECT"):
        return os.environ["PROOFWORLD_MATHLIB_PROJECT"]
    home = os.path.expanduser("~"); installed = _installed_toolchains()
    preferred = ["categorical_rh", "PlonkLean", "RamanujanTau", "unifiedtheory", "pfr", "carleson"]
    for name in preferred:
        proj = os.path.join(home, name)
        olean = os.path.join(proj, ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean")
        tc = ""
        try:
            tc = open(os.path.join(proj, "lean-toolchain")).read().strip().split(":")[-1]
        except OSError:
            continue
        if os.path.exists(olean) and tc in installed:
            return proj
    return None

PROJECT = find_mathlib_project()

PREAMBLE = ("import Mathlib.Tactic.Linarith\n"
            "def oddSum : ℕ → ℕ\n  | 0 => 0\n  | (n+1) => oddSum n + (2*n+1)\n")   # oddSum n = n^2

GOAL_STMT = "theorem G (n : ℕ) : oddSum n ≤ n^2 + n"
GOAL_PROP = "oddSum n ≤ n^2 + n"
GOAL_DESC = "∀ n, oddSum n ≤ n² + n   (oddSum recursive; nonlinear -> needs induction AND nlinarith)"

# inductive proof whose step is discharged by Mathlib's nlinarith (core Lean / omega cannot)
IND = ("by\n  induction n with\n  | zero => simp [oddSum]\n"
       "  | succ k ih => rw [show oddSum (k+1) = oddSum k + (2*k+1) from rfl]; nlinarith [ih]")

CANDIDATES = [
    ("aux_unfold", "theorem aux_unfold (n : ℕ) : oddSum (n+1) = oddSum n + (2*n+1)", "needed for the inductive step"),
    ("aux_zero",   "theorem aux_zero : oddSum 0 = 0",                                    "true but irrelevant (decoy)"),
    ("aux_fh",     "theorem aux_fh (n : ℕ) : oddSum n ≤ n",                "FALSE but would close the goal (canary)"),
    ("aux_goal",   "theorem aux_goal (n : ℕ) : oddSum n ≤ n^2 + n",        "= the goal (circular)"),
]
TACTICS = ["by rfl", IND, "by simp [oddSum]"]


def _lean(source: str, timeout=120):
    """run the Mathlib-backed kernel: `lake env lean <file>` from the project dir. (ok, has_sorry, first_error)."""
    if not PROJECT:
        return False, False, "no built Mathlib project found"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "Check.lean")
        with open(f, "w") as fh: fh.write(source)
        try:
            p = subprocess.run(["lake", "env", "lean", f], cwd=PROJECT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, False, "timeout"
    out = p.stdout + p.stderr
    err = next((ln.split("error:", 1)[1].strip() for ln in out.splitlines() if "error:" in ln), "")
    has_sorry = "sorry" in out
    return (p.returncode == 0 and "error:" not in out and not has_sorry), has_sorry, err


def prove_clean(statement: str):
    for tac in TACTICS:
        ok, _s, _e = _lean(PREAMBLE + statement + " := " + tac + "\n")
        if ok:
            return tac
    return None

def prop_of(stmt: str) -> str:
    return stmt.split(") : ")[-1] if ") : " in stmt else stmt.split(" : ", 1)[-1]


def main():
    print("=== proofworld.research_mathlib :: Mathlib behind the kernel -- nonlinear goal beyond z3 AND core Lean ===\n")
    if not PROJECT:
        print("  No built Mathlib project with an installed toolchain found. Set PROOFWORLD_MATHLIB_PROJECT."); return
    print(f"  reusing built Mathlib from: {PROJECT}")
    print(f"  GOAL: {GOAL_DESC}\n")
    # 1) LOAD-BEARING
    print("  --- load-bearing check ---")
    ok_omega, _s, e1 = _lean(PREAMBLE + GOAL_STMT + " := by omega\n")
    print(f"    G := by omega                         -> {'VERIFIED' if ok_omega else 'FAILS'}  (z3-style: no induction)")
    step_omega = (PREAMBLE + GOAL_STMT + " := by\n  induction n with\n  | zero => simp [oddSum]\n"
                  "  | succ k ih => rw [show oddSum (k+1) = oddSum k + (2*k+1) from rfl]; omega\n")
    ok_so, _s, e2 = _lean(step_omega)
    print(f"    G := induction; step by omega         -> {'VERIFIED' if ok_so else 'FAILS'}  (core Lean: can't relate (k+1)^2 to k^2)")
    print(f"    => only induction + Mathlib's nlinarith closes it.\n")
    # 2-4) SOUNDNESS gate + anti-circularity
    print("  --- SOUNDNESS gate (usable only if the Mathlib kernel PROVES it; no sorry) ---")
    justified = []
    for name, stmt, note in CANDIDATES:
        tac = prove_clean(stmt)
        if tac is not None:
            circular = prop_of(stmt).strip() == GOAL_PROP.strip()
            if circular:
                print(f"    {name:11} -> JUSTIFIED  (but CIRCULAR = the goal -> excluded as a restatement)")
            else:
                print(f"    {name:11} -> JUSTIFIED  [via {tac.splitlines()[0]}]"); justified.append((name, stmt, tac))
        else:
            _h, _hs, h_err = _lean(PREAMBLE + "theorem G_if (n : ℕ) : " + GOAL_PROP +
                                   " := by\n  have h : " + prop_of(stmt).strip() + " := by sorry\n  nlinarith [h]\n")
            extra = "  <-- would CLOSE the goal if granted, but UNPROVABLE: REJECTED (anti-cheat)" if h_err == "" else ""
            print(f"    {name:11} -> UNSOUND (kernel cannot prove it){extra}")
    print(f"\n  justified, non-circular lemmas: {[n for n, _, _ in justified]}\n")
    # 5) COMPOSE
    print("  --- compose: prove the goal by induction + nlinarith, citing the justified lemma (kernel owns truth) ---")
    source = (PREAMBLE + "theorem aux_unfold (n : ℕ) : oddSum (n+1) = oddSum n + (2*n+1) := by rfl\n" +
              GOAL_STMT + " := by\n  induction n with\n  | zero => simp [oddSum]\n"
              "  | succ k ih => rw [aux_unfold]; nlinarith [ih]\n")
    ok, _s, err = _lean(source)
    print(f"    aux_unfold (rfl) + G by induction [rw aux_unfold; nlinarith] -> {'VERIFIED by Mathlib kernel' if ok else 'FAILED: ' + err}")
    # 6) record + canaries
    print(f"\n  --- atlas record ---")
    print(f"  GOAL {GOAL_DESC}")
    print(f"    status   : {'VERIFIED (Mathlib kernel: induction + nlinarith)' if ok else 'OPEN'}")
    print(f"    proof    : induction on n; step unfolds via aux_unfold then nlinarith discharges the nonlinear bound")
    print(f"    technique: structural induction + nonlinear arithmetic (nlinarith)")
    canary_rejected = prove_clean("theorem aux_fh (n : ℕ) : oddSum n ≤ n") is None
    print(f"\n  [canary] goal needs Mathlib (omega + core-step both fail): {'PASS' if not ok_omega and not ok_so else 'FAIL'}")
    print(f"  [canary] false-but-helpful lemma (oddSum n ≤ n) REJECTED as unprovable: {'PASS' if canary_rejected else 'FAIL'}")
    good = ok and not ok_omega and not ok_so and canary_rejected
    print(f"\n  GATE: {'PASS -- nonlinear induction, beyond z3 AND core Lean, soundness enforced on the Mathlib kernel' if good else 'FAIL'}")


if __name__ == "__main__":
    main()
