"""Regress published phase-gate verdicts against current artifact JSON.

Pinned values live in `phase_gates.json` alongside this file. For each
phase entry:

  1. Load the artifact JSON at `metrics_path` (relative to repo root).
     If the file does NOT exist, skip that phase (a fresh clone should
     still produce a green run; the pinned gates are reproduced from
     the published artifacts, not regenerated from scratch).
  2. For each assertion, walk the dotted `path` into the artifact JSON
     and apply the comparator `op` against `value`. Both positive and
     negative findings are pinned (e.g., Phase 18l's single-seed G1
     fail), so a flipped gate value triggers a test failure.

This is the lightweight (1-light) variant of the gates-as-tests fix.
The heavy variant — actually rerunning the canonical eval against a
pinned checkpoint — needs GPU and is deferred. This harness instead
guards against accidental edits to artifact JSONs and against tooling
drift that changes how summaries are computed.

Usage:

    pytest tests/regress/test_phase_gates.py -v

To add a new phase: append to phases in phase_gates.json. No code
changes required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATES_FILE = Path(__file__).parent / "phase_gates.json"


def _load_spec() -> dict:
    return json.loads(GATES_FILE.read_text())


def _walk(d: Any, dotted_path: str) -> Any:
    """Walk a dotted path into nested dicts/lists. Raises KeyError if missing."""
    cur = d
    for key in dotted_path.split("."):
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


def _cmp(actual: Any, op: str, expected: Any, tol: float = 0.05) -> bool:
    """Comparator dispatch. Returns True on pass."""
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "approx":
        if expected == 0:
            return abs(actual) <= tol
        return abs(actual - expected) / abs(expected) <= tol
    raise ValueError(f"Unknown op: {op!r}")


def _resolve_expected(assertion: dict, payload: dict) -> Any:
    """Resolve the expected value: either a literal or a derived field.

    - `value`: literal expected value.
    - `value_from` + `scale` (optional): expected = scale * walk(value_from).
      Lets one assertion express "N-times-smaller-than-baseline" without
      hardcoding the baseline magnitude.
    """
    if "value_from" in assertion:
        base = _walk(payload, assertion["value_from"])
        scale = assertion.get("scale", 1.0)
        return base * scale
    return assertion["value"]


def _collect_cases() -> list[tuple[str, str, dict, dict]]:
    """Build the parametrize input: one row per (phase, assertion)."""
    spec = _load_spec()
    cases: list[tuple[str, str, dict, dict]] = []
    for phase_name, phase in spec["phases"].items():
        metrics_path = REPO_ROOT / phase["metrics_path"]
        for a in phase.get("assertions", []):
            cases.append((phase_name, a["label"], phase, a))
    return cases


CASES = _collect_cases()


@pytest.mark.parametrize(
    ("phase_name", "label", "phase", "assertion"),
    CASES,
    ids=[f"{c[0]}::{c[1]}" for c in CASES],
)
def test_phase_gate(phase_name, label, phase, assertion):
    metrics_path = REPO_ROOT / phase["metrics_path"]
    if not metrics_path.exists():
        pytest.skip(
            f"{phase['metrics_path']} not present in this checkout "
            f"(fresh clone or pruned artifacts/). Pin re-validated when "
            f"file is restored.")

    payload = json.loads(metrics_path.read_text())

    try:
        actual = _walk(payload, assertion["path"])
    except (KeyError, IndexError, TypeError) as e:
        pytest.fail(
            f"[{phase_name}] could not resolve path {assertion['path']!r} "
            f"in {phase['metrics_path']}: {e}")

    expected = _resolve_expected(assertion, payload)
    tol = assertion.get("tol", 0.05)
    op = assertion["op"]

    ok = _cmp(actual, op, expected, tol=tol)
    if not ok:
        pytest.fail(
            f"[{phase_name}] gate '{label}' FAILED: "
            f"path={assertion['path']!r}  op={op}  actual={actual!r}  "
            f"expected={expected!r}"
            + (f"  tol={tol}" if op == 'approx' else ""))


def test_spec_schema_valid():
    """Schema sanity: each phase has metrics_path; each assertion has label/path/op."""
    spec = _load_spec()
    assert "phases" in spec
    valid_ops = {"==", "!=", "<", "<=", ">", ">=", "approx"}
    for phase_name, phase in spec["phases"].items():
        assert "metrics_path" in phase, f"{phase_name} missing metrics_path"
        for a in phase.get("assertions", []):
            assert "label" in a, f"{phase_name}: assertion missing label"
            assert "path" in a, f"{phase_name}: assertion {a.get('label')!r} missing path"
            assert "op" in a, f"{phase_name}: assertion {a['label']!r} missing op"
            assert a["op"] in valid_ops, (
                f"{phase_name}: bad op {a['op']!r} in {a['label']!r}")
            assert "value" in a or "value_from" in a, (
                f"{phase_name}: assertion {a['label']!r} needs `value` "
                f"or `value_from`")


def test_at_least_one_case_collected():
    """Guard against an empty harness: the JSON must yield at least one case."""
    assert len(CASES) >= 1, "No phase gates were collected from phase_gates.json"
