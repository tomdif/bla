#!/usr/bin/env python3
"""proofworld.numeric -- the cheap grounded kill-test, done so it cannot lie.

The numerical kill-test is proofworld's cheap falsifier before an expensive kernel call. Ported discipline
(from a review of the BootLoops harness, whose numeric gates certified false results in exactly these ways):

  * NO EVIDENCE IS NOT EVIDENCE. A law checked against zero (or too few) instances in its domain is
    INSUFFICIENT, never "survives". (Old conjecture.kill_test returned "survives" with checked == 0.)
  * INDEPENDENCE. Instances the proposer SAW cannot be what certify its proposal. `split` deals a
    deterministic held-out set; the proposer gets only `seen`, the kill-test's verdict comes from `held`.
  * PLANTED TRUTH FIRST. `controls()` runs a planted false law (must be REFUTED), a planted true law
    (must SURVIVE), an empty domain (must be INSUFFICIENT) and a hostile expression (must be rejected)
    before the kill-test is trusted on real data.
  * NO eval() ON MODEL OUTPUT. `{"__builtins__": {}}` is not a sandbox (`().__class__.__bases__...`).
    Formulas are parsed to an AST and only integer arithmetic on whitelisted variable names is allowed,
    with exponent / size caps so a hostile `9**9**9` cannot hang the process.

Statuses (closed vocabulary):  SURVIVES | REFUTED | INSUFFICIENT | UNEVALUABLE
A SURVIVES verdict is a reason to spend a kernel call, nothing more: the register is "numeric", never "proved".

Run:  python3 -m proofworld.numeric        (runs the planted controls; exit 0 pass / 1 fail)
"""
from __future__ import annotations

import ast
import hashlib
import operator
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

STATUSES = ("SURVIVES", "REFUTED", "INSUFFICIENT", "UNEVALUABLE")
MAX_EXP = 4096          # largest exponent a formula may use
MAX_BITS = 1 << 16      # largest intermediate integer (bits) before we refuse


class UnsafeExpr(ValueError):
    """the formula uses something other than whitelisted integer arithmetic."""


# ------------------------------------------------------------------------------------------------ safe formulas
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Div: lambda a, b: Fraction(a) / Fraction(b),
        ast.Pow: None}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _check_size(x):
    v = x.numerator if isinstance(x, Fraction) else x
    d = x.denominator if isinstance(x, Fraction) else 1
    if abs(v).bit_length() > MAX_BITS or d.bit_length() > MAX_BITS:
        raise UnsafeExpr("intermediate value too large")
    return x


def _validate(node: ast.AST, names: frozenset):
    if isinstance(node, ast.Expression):
        return _validate(node.body, names)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN:
            raise UnsafeExpr(f"operator {type(node.op).__name__} not allowed")
        _validate(node.left, names); _validate(node.right, names); return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UN:
            raise UnsafeExpr(f"unary {type(node.op).__name__} not allowed")
        _validate(node.operand, names); return
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:                      # bool is a subclass of int: excluded by `is`
            raise UnsafeExpr(f"constant {node.value!r} is not an integer")
        return
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise UnsafeExpr(f"unknown name {node.id!r} (allowed: {sorted(names)})")
        return
    raise UnsafeExpr(f"{type(node).__name__} not allowed")


def _eval(node: ast.AST, env: Dict[str, int]):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](_eval(node.operand, env))
    a, b = _eval(node.left, env), _eval(node.right, env)
    if isinstance(node.op, ast.Pow):
        if isinstance(b, Fraction):
            if b.denominator != 1:
                raise UnsafeExpr("non-integer exponent")
            b = b.numerator
        if abs(b) > MAX_EXP:
            raise UnsafeExpr(f"exponent {b} exceeds {MAX_EXP}")
        if (abs(a.numerator if isinstance(a, Fraction) else a).bit_length() or 1) * abs(b) > MAX_BITS:
            raise UnsafeExpr("power too large")
        return _check_size(Fraction(a) ** b if b < 0 else a ** b)
    if isinstance(node.op, (ast.FloorDiv, ast.Mod, ast.Div)) and b == 0:
        raise ZeroDivisionError("division by zero in formula")
    return _check_size(_BIN[type(node.op)](a, b))


def compile_formula(expr: str, names: Iterable[str] = ("n", "p")) -> Callable[..., object]:
    """parse + validate ONCE; returns f(**env) -> int|Fraction. Raises UnsafeExpr for anything else."""
    if not isinstance(expr, str) or len(expr) > 500:
        raise UnsafeExpr("formula must be a string of <= 500 chars")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpr(f"syntax: {e.msg}") from None
    allowed = frozenset(names)
    _validate(tree, allowed)

    def f(x=None, **env):
        """f(5) binds every allowed name to 5 (n and p are aliases); f(n=5, k=2) binds by name."""
        if x is not None:
            env = {name: x for name in allowed}
        unknown = set(env) - allowed
        if unknown:
            raise UnsafeExpr(f"unexpected variables {sorted(unknown)}")
        return _eval(tree, env)
    f.expr = expr
    return f


def safe_eval(expr: str, **env) -> object:
    """one-shot: safe_eval('n**3 + 1', n=5) == 126."""
    return compile_formula(expr, names=tuple(env) or ("n",))(**env)


# ------------------------------------------------------------------------------------------------ independence
def split(instances: Dict[Hashable, object], holdout: float = 0.4, salt: str = "proofworld") \
        -> Tuple[Dict[Hashable, object], Dict[Hashable, object]]:
    """deterministic seen/held-out split by keyed hash (stable across runs and Python's hash salting).
    The proposer must only ever be shown `seen`."""
    seen, held = {}, {}
    for k, v in instances.items():
        h = int(hashlib.sha256(f"{salt}|{k!r}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (held if h < holdout else seen)[k] = v
    return seen, held


# ------------------------------------------------------------------------------------------------ kill-test
@dataclass
class NumVerdict:
    status: str
    checked: int = 0
    counterexample: Optional[Tuple[object, object, object]] = None    # (input, expected, got)
    detail: str = ""
    register: str = "numeric"
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def survives(self) -> bool:
        return self.status == "SURVIVES"


def kill_test(law: Callable[[object], object], instances: Dict[object, object],
              domain: Callable[[object], bool] = lambda n: True, min_support: int = 3,
              provenance: Optional[dict] = None) -> NumVerdict:
    """does `law(n) == instances[n]` for EVERY instance in `domain`? Exact comparison (ints / Fractions).
    SURVIVES only with >= min_support in-domain instances; fewer is INSUFFICIENT (no evidence, not a pass)."""
    checked = 0
    for n, want in sorted(instances.items(), key=lambda kv: repr(kv[0])):
        if not domain(n):
            continue
        try:
            got = law(n)
        except (UnsafeExpr, ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
            return NumVerdict("UNEVALUABLE", checked, (n, want, None), f"{type(e).__name__}: {e}",
                              provenance=provenance or {})
        if got != want:
            return NumVerdict("REFUTED", checked, (n, want, got), "exact mismatch", provenance=provenance or {})
        checked += 1
    if checked < min_support:
        return NumVerdict("INSUFFICIENT", checked, None,
                          f"only {checked} in-domain instance(s) < min_support={min_support}; no evidence either way",
                          provenance=provenance or {})
    return NumVerdict("SURVIVES", checked, provenance=provenance or {})


def heldout_kill_test(law, seen: dict, held: dict, domain=lambda n: True, min_support: int = 3) -> NumVerdict:
    """refute on ANY data (seen counterexamples count), but SURVIVES only on held-out evidence."""
    on_seen = kill_test(law, seen, domain, min_support=0)
    prov = {"proposer_saw": len(seen), "heldout_total": len(held)}
    if on_seen.status in ("REFUTED", "UNEVALUABLE"):
        on_seen.provenance = {**prov, "refuted_on": "seen"}
        return on_seen
    v = kill_test(law, held, domain, min_support=min_support, provenance={**prov, "seen_checked": on_seen.checked})
    return v


# ------------------------------------------------------------------------------------------------ controls
def controls(log=None) -> dict:
    """planted truth: the kill-test is trusted only if it recovers known answers and rejects hostile input."""
    squares = {n: n * n for n in range(1, 30)}
    rows = []

    def rec(name, got, want):
        rows.append({"control": name, "want": want, "got": got, "ok": got == want})
        if log:
            log(f"  {'ok ' if got == want else 'BAD'} {name:28} want {want:13} got {got}")

    rec("planted_true_law", kill_test(compile_formula("n*n"), squares).status, "SURVIVES")
    rec("planted_false_law", kill_test(compile_formula("n*n + (n//20)"), squares).status, "REFUTED")
    rec("empty_domain", kill_test(compile_formula("n*n"), squares, domain=lambda n: n > 1000).status,
        "INSUFFICIENT")
    rec("thin_support", kill_test(compile_formula("n*n"), {2: 4}, min_support=3).status, "INSUFFICIENT")
    seen, held = split(squares)
    rec("heldout_nonempty", bool(held) and bool(seen), True)
    rec("heldout_true_law", heldout_kill_test(compile_formula("n*n"), seen, held).status, "SURVIVES")
    for hostile in ("().__class__.__bases__", "__import__('os').system('true')", "9**9**9", "n if n else 0",
                    "[n for n in range(3)]", "1.5*n", "True + n"):
        try:
            compile_formula(hostile)(n=3)
            got = "ACCEPTED"
        except (UnsafeExpr, ZeroDivisionError):
            got = "REJECTED"
        rec(f"hostile:{hostile[:18]}", got, "REJECTED")
    ok = all(r["ok"] for r in rows)
    return {"ok": ok, "rows": rows}


if __name__ == "__main__":
    r = controls(log=print)
    print(f"numeric controls: {'PASS' if r['ok'] else 'FAIL'}")
    sys.exit(0 if r["ok"] else 1)
