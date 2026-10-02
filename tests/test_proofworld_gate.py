"""Tests for proofworld's trust boundary (proofworld.gate) and numeric kill-test (proofworld.numeric).

Each test encodes a specific way a verifier can certify something false -- several were live bugs in proofworld
before the gate existed (sorry accepted as VERIFIED; timeout/import failure counted as a proof; zero-evidence
"survives"; eval() on LLM output). Pure-Python tests run everywhere (CI has no Lean); the kernel-backed tests
skip unless a Lean toolchain is installed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proofworld import gate, numeric  # noqa: E402

LEAN = gate.lean_available()
MATHLIB_PROJECT = os.path.join(os.path.dirname(__file__), "..", "proofworld", "lean")
needs_lean = pytest.mark.skipif(not LEAN, reason="no Lean toolchain")
needs_mathlib = pytest.mark.skipif(
    not (LEAN and os.path.isdir(os.path.join(MATHLIB_PROJECT, ".lake", "packages", "mathlib"))),
    reason="no built Mathlib project")


# ---------------------------------------------------------------------------------------------- lint (pure)
@pytest.mark.parametrize("proof", [
    "by sorry", "by\n  admit", "by native_decide", "by exact Lean.ofReduceBool _ _ rfl",
    "by\n  set_option debug.skipKernelTC true in rfl", "cheat\naxiom cheat : False",
    "by trivial\ntheorem other : 1 = 2 := by sorry", "by run_tac pure ()", "by\n  #eval 1",
])
def test_lint_refuses_escape_hatches(proof):
    assert gate.lint(proof) is not None


@pytest.mark.parametrize("proof", [
    "by omega", "by induction n with\n  | zero => simp\n  | succ k ih => omega",
    "by simp [sorryFree_lemma]", "by exact axiomatic_thing", "by decide",
])
def test_lint_allows_ordinary_proofs(proof):
    assert gate.lint(proof) is None


def test_invalid_names_refused_without_running_lean():
    v = gate.check([gate.Claim("x : True := trivial\naxiom", ": True", "trivial")], controls=False, receipts=False)
    assert list(v.values())[0].status == "REFUSED"


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        gate.check([gate.Claim("a", ": True", "trivial"), gate.Claim("a", ": True", "trivial")],
                   controls=False, receipts=False)


# ---------------------------------------------------------------------------------- decision rules (pure)
def _verdicts(out, claims, timed_out=False, rc=0, allow_native=False):
    src, spans, qual = gate.assemble(claims)
    errs, stray, foot = gate.parse(out, spans, qual)
    return {c.name: gate._decide(c, errs, stray, foot, timed_out, rc, allow_native)[0] for c in claims}, spans


C = [gate.Claim("a", "(n : Nat) : n ≤ n + 1", "by omega"), gate.Claim("b", "(n : Nat) : n = n", "rfl")]
M = gate.MARK


def test_clean_footprints_are_proved():
    out = f"{M}\ta\tFOUND\tpropext\n{M}\tb\tFOUND\t\n"
    v, _ = _verdicts(out, C)
    assert v == {"a": "PROVED", "b": "PROVED"}


def test_missing_marker_is_not_a_pass():
    v, _ = _verdicts(f"{M}\ta\tFOUND\tpropext\n", C)          # b printed nothing
    assert v["b"] == "INCONCLUSIVE"


def test_timeout_is_not_a_pass():
    v, _ = _verdicts("", C, timed_out=True)
    assert set(v.values()) == {"INCONCLUSIVE"}


def test_error_outside_claims_voids_the_batch():
    # e.g. a failed import: the error sits on line 1, inside no claim's span (old decompose: "proved")
    out = f"{gate.FILE_BASENAME}:1:0: error: unknown module prefix 'Nope'\n{M}\ta\tFOUND\t\n{M}\tb\tFOUND\t\n"
    v, _ = _verdicts(out, C, rc=1)
    assert set(v.values()) == {"INCONCLUSIVE"}


def test_positionless_crash_voids_the_batch():
    out = f"uncaught exception: out of memory\n{M}\ta\tFOUND\t\n{M}\tb\tFOUND\t\n"
    v, _ = _verdicts(out, C)
    assert set(v.values()) == {"INCONCLUSIVE"}


def test_sorry_footprint_refused_and_error_failed():
    _, spans = _verdicts("", C)
    line_b = spans["b"][0]
    out = (f"{gate.FILE_BASENAME}:{line_b}:3: error: type mismatch\n"
           f"{M}\ta\tFOUND\tsorryAx\n{M}\tb\tFOUND\tsorryAx\n")
    v, _ = _verdicts(out, C, rc=1)
    assert v == {"a": "REFUSED", "b": "FAILED"}


def test_rc1_explained_by_another_claims_error_still_proves_the_good_one():
    _, spans = _verdicts("", C)
    out = f"{gate.FILE_BASENAME}:{spans['b'][0]}:3: error: nope\n{M}\ta\tFOUND\tpropext\n{M}\tb\tFOUND\tsorryAx\n"
    v, _ = _verdicts(out, C, rc=1)
    assert v["a"] == "PROVED" and v["b"] == "FAILED"


def test_native_axioms_refused_unless_explicitly_allowed():
    out = f"{M}\ta\tFOUND\tpropext,Foo._native.native_decide.ax_1\n{M}\tb\tFOUND\t\n"
    assert _verdicts(out, C)[0]["a"] == "REFUSED"
    assert _verdicts(out, C, allow_native=True)[0]["a"] == "PROVED"


def test_injected_axiom_refused():
    out = f"{M}\ta\tFOUND\tpropext,my_cheat\n{M}\tb\tFOUND\t\n"
    assert _verdicts(out, C, allow_native=True)[0]["a"] == "REFUSED"


def test_spans_point_at_claim_lines():
    src, spans, _ = gate.assemble(C, preamble="import Foo\ndef x := 1\n", namespace="N")
    lines = src.split("\n")
    for c in C:
        s, e = spans[c.name]
        assert lines[s - 1].startswith(f"theorem {c.name} ")
    assert src.startswith("import Lean\nimport Foo\n")              # imports stay first


# --------------------------------------------------------------------------------- benchmark parsers (pure)
def test_batch_parsers_fail_closed_on_stray_errors():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proofworld"))
    import build_proofset as bp
    spans = {"t1": (5, 7), "t2": (8, 10)}
    assert bp.parse_pass("pw_proofset_batch.lean:1:0: error: unknown package 'Mathlib'\n", spans) == \
        {"t1": False, "t2": False}
    assert bp.parse_pass("", spans) == {"t1": True, "t2": True}
    assert bp.parse_pass("@@TIMEOUT@@", spans) == {"t1": False, "t2": False}


# ------------------------------------------------------------------------------------------ numeric (pure)
def test_numeric_controls_pass():
    assert numeric.controls()["ok"]


def test_zero_evidence_is_insufficient_not_survives():
    law = numeric.compile_formula("n**3 + 1")
    v = numeric.kill_test(law, {4: 73}, domain=lambda n: n in (2, 3, 5, 7))
    assert v.status == "INSUFFICIENT" and v.checked == 0


def test_seen_counterexample_still_refutes():
    law = numeric.compile_formula("n**3 + 1")
    v = numeric.heldout_kill_test(law, seen={4: 73}, held={2: 9, 3: 28, 5: 126})
    assert v.status == "REFUTED"


def test_split_is_deterministic_and_disjoint():
    data = {n: n for n in range(100)}
    s1, h1 = numeric.split(data, salt="x")
    s2, h2 = numeric.split(data, salt="x")
    assert (s1, h1) == (s2, h2) and not set(s1) & set(h1) and len(s1) + len(h1) == 100


@pytest.mark.parametrize("expr", ["().__class__.__bases__", "__import__('os')", "9**9**9", "n.real",
                                  "open('x')", "lambda: 1", "2.5*n", "x + 1"])
def test_formula_sandbox(expr):
    with pytest.raises((numeric.UnsafeExpr, ZeroDivisionError)):
        numeric.compile_formula(expr)(3)


# ------------------------------------------------------------------------------------ kernel-backed tests
@needs_lean
def test_gate_controls_core_lean():
    assert gate.run_controls()["ok"]


@needs_lean
def test_leankernel_no_longer_accepts_sorry():
    from proofworld.leankernel import verify_theorem
    assert not verify_theorem("theorem t1 (n : Nat) : n = n + 1", "by sorry").ok
    assert not verify_theorem("theorem t2 : 2 + 2 = 4", "by native_decide").ok
    assert verify_theorem("theorem t3 (n : Nat) : n ≤ n + 1", "by omega").ok


@needs_mathlib
def test_decompose_import_failure_is_not_a_proof(monkeypatch):
    # old try_direct: a broken import left every attempt's span error-free -> goal "PROVED"
    from proofworld import decompose
    monkeypatch.setattr(decompose, "PREAMBLE", "import Mathlib.Tactic.DoesNotExist\n" + decompose.PREAMBLE)
    tac, _ = decompose.try_direct("pw_goal", "(n : ℕ) : n = n + 1")
    assert tac is None


@needs_mathlib
def test_decompose_scripted_split_verifies(monkeypatch):
    from proofworld import decompose
    dec = {"lemmas": [{"name": "f_succ", "sig": "(n : ℕ) : f (n+1) = f n + 2*n + 1"}],
           "main_proof": "by\n  induction n with\n  | zero => rfl\n  | succ k ih => rw [f_succ, ih]; ring"}
    monkeypatch.setattr(decompose, "opus_decompose", lambda *a, **k: dec)
    res = decompose.prove(*decompose.GOAL, log=lambda *a: None)
    assert res["ok"] and res["kind"] == "decomp"


@needs_mathlib
def test_decompose_cheating_split_refused(monkeypatch):
    from proofworld import decompose
    dec = {"lemmas": [], "main_proof": "by sorry"}
    monkeypatch.setattr(decompose, "opus_decompose", lambda *a, **k: dec)
    assert not decompose.prove(*decompose.GOAL, log=lambda *a: None)["ok"]


def test_imports_after_leading_block_comment_are_hoisted():
    pre = "/-\n# doc\nmore\n-/\nimport Foo\nimport Bar\n\ndef x := 1\n"
    imports, rest = gate._split_imports(pre)
    assert imports == ["import Foo", "import Bar"] and "def x := 1" in rest and "import" not in rest
