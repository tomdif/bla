#!/usr/bin/env python3
"""proofworld.leankernel -- a REAL Lean kernel as the verifier, for theorems BEYOND decidable arithmetic.

z3 owns decidable arithmetic; Lean owns the rest (induction, lists, structures, eventually Mathlib + research
theorems). Here the verifier shells out to `lean` on a self-contained file: the Lean KERNEL decides truth. A
"tactic dreamer" proposes candidate proof scripts (`by omega`, `by simp`, `by induction ...`); Lean checks which
compile. Same discipline as the z3 side: the model PROPOSES tactics, the kernel OWNS truth -- a wrong tactic just
fails to compile (the grounded kill-test), and no false proof is ever accepted.

Run: python3 -m proofworld.leankernel    (uses the installed lean toolchain; ~3s/proof, no Mathlib needed)
"""
from __future__ import annotations
from dataclasses import dataclass
import os, subprocess, tempfile, glob

def _toolchain() -> str:
    """prefer an already-installed pinned toolchain so we never trigger a multi-minute elan download."""
    pref = os.environ.get("PROOFWORLD_LEAN_TOOLCHAIN")
    if pref: return pref
    installed = sorted(glob.glob(os.path.expanduser("~/.elan/toolchains/leanprover--lean4---v*")))
    for want in ("v4.30.0", ):                                     # known-good, present on this machine
        for p in installed:
            if p.endswith(want): return f"leanprover/lean4:{want}"
    if installed:                                                  # fall back to newest installed (no download)
        tag = installed[-1].split("---")[-1]; return f"leanprover/lean4:{tag}"
    return "leanprover/lean4:v4.30.0"

TOOLCHAIN = _toolchain()


@dataclass
class LeanResult:
    ok: bool                  # kernel accepted the proof
    status: str               # VERIFIED | PROOF_FAILED | ERROR
    detail: str               # first error line, if any


def lean_check(source: str, timeout=60) -> LeanResult:
    """write `source` to a temp .lean file and run the Lean kernel on it. ok iff lean exits 0 with no 'error:'.
    THIS is where truth lives -- the elaborator + kernel, not any learned model."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "Check.lean")
        with open(f, "w") as fh: fh.write(source)
        try:
            p = subprocess.run(["elan", "run", TOOLCHAIN, "lean", f],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return LeanResult(False, "ERROR", "timeout")
        out = (p.stdout + p.stderr).strip()
        err = next((ln for ln in out.splitlines() if "error:" in ln), "")
        msg = err.split("error:", 1)[1].strip() if err else ""        # strip the temp-path prefix
        if p.returncode == 0 and "error:" not in out:
            return LeanResult(True, "VERIFIED", "")
        return LeanResult(False, "PROOF_FAILED" if err else "ERROR", msg or (out.splitlines()[0] if out else "?"))


def verify_theorem(statement: str, proof: str, preamble: str = "") -> LeanResult:
    """statement e.g. 'theorem t (n : Nat) : n <= n + 1'; proof e.g. 'by omega'."""
    src = (preamble + "\n" if preamble else "") + f"{statement} := {proof}\n"
    return lean_check(src)


# ----------------------------- the tactic dreamer (propose proof scripts; Lean owns truth) -----------------------------
@dataclass
class Goal:
    name: str
    statement: str
    preamble: str = ""
    decidable_arith: bool = False     # could z3 also do it? (just for commentary)

def dream_proof(goal: Goal, candidate_tactics, log=print) -> LeanResult | None:
    """dream candidate proof scripts in order; the FIRST the Lean kernel accepts closes the goal. A wrong tactic
    fails to compile -- the grounded kill-test -- and is discarded. Nothing unverified is ever 'believed'."""
    log(f"\n  goal [{goal.name}]: {goal.statement}")
    for tac in candidate_tactics:
        r = verify_theorem(goal.statement, tac, goal.preamble)
        mark = "✓ VERIFIED" if r.ok else f"✗ {r.status}"
        log(f"    try {tac:38} -> {mark}" + (f"   ({r.detail[:54]})" if not r.ok and r.detail else ""))
        if r.ok:
            log(f"    => Lean kernel accepted '{tac}'. truth owned by the kernel, not a model.")
            return r
    log(f"    => no dreamed tactic closed it; honestly OPEN (no false proof accepted).")
    return None


GOALS = [
    # decidable arithmetic -- z3 could also do this; included to show the kernels agree
    (Goal("arith_le", "theorem g1 (n : Nat) : n <= n + 1", decidable_arith=True),
     ["by rfl", "by decide", "by simp", "by omega"]),
    # BEYOND decidable arithmetic: a fact about lists, proved by structural induction -- z3 has no theory for this
    (Goal("list_rev_rev", "theorem g2 (l : List Nat) : l.reverse.reverse = l"),
     ["by omega", "by rfl", "by decide", "by simp", "by induction l <;> simp [*]"]),
    # another non-arithmetic one: append is associative (induction)
    (Goal("list_append_assoc", "theorem g3 (a b c : List Nat) : (a ++ b) ++ c = a ++ (b ++ c)"),
     ["by omega", "by rfl", "by simp", "by induction a <;> simp [*]"]),
    # a FALSE statement: the kernel must reject EVERY dreamed proof (anti-trap canary)
    (Goal("false_succ", "theorem gbad (n : Nat) : n = n + 1"),
     ["by rfl", "by omega", "by simp", "by decide"]),
]


def main():
    print("=== proofworld.leankernel :: REAL Lean kernel verifier + tactic dreamer (beyond decidable arithmetic) ===")
    print(f"  toolchain: {TOOLCHAIN}  (truth is owned by the Lean elaborator+kernel)")
    results = {}
    for goal, tactics in GOALS:
        r = dream_proof(goal, tactics)
        results[goal.name] = (goal, r)
    print("\n--- summary ---")
    for name, (goal, r) in results.items():
        tag = "VERIFIED" if (r and r.ok) else "OPEN/REJECTED"
        kind = "decidable-arith (z3 too)" if goal.decidable_arith else "beyond z3 (induction/lists)"
        print(f"  {name:20} {tag:14} [{kind}]")
    # anti-trap canary: the false theorem must NOT be verified by any dreamed tactic
    false_ok = results["false_succ"][1] is None
    beyond = results["list_rev_rev"][1] and results["list_rev_rev"][1].ok
    print(f"\n  [canary] false theorem rejected by the kernel for every dreamed tactic: {'PASS' if false_ok else 'FAIL'}")
    print(f"  [canary] a theorem BEYOND decidable arithmetic was kernel-verified:      {'PASS' if beyond else 'FAIL'}")
    print(f"\n  GATE: {'PASS -- Lean kernel owns truth; tactic dreaming is grounded by real compilation' if (false_ok and beyond) else 'FAIL'}")


if __name__ == "__main__":
    main()
