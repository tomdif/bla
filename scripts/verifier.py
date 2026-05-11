"""Verification-layer prototype for PAL candidate aggregation.

Given a problem and N candidate Python programs (each with code,
execution output, and a predicted answer), score each candidate by
structural and execution properties that correlate with correctness.
Then pick the highest-scoring candidate's answer.

Signal sources (offline-computable; no extra model calls):

  1. Computation evidence
       Penalize codes that don't perform any arithmetic operations
       (e.g., `answer = 42; print(answer)`). These are pure guesses.

  2. Input-use evidence
       Reward codes whose computation actually references the numbers
       from the problem (vs. binding them and discarding).

  3. Multi-step chain evidence
       Reward codes whose final `answer` is computed from earlier
       intermediates rather than assigned directly.

  4. Anti-echo filter
       Penalize predictions equal to a number already in the problem.

  5. Execution success — hard filter.

  6. Perturbation response (the killer signal)
       Modify one numeric literal in the code, re-execute, check
       whether the output changes. A code doing real computation
       responds to its inputs; a guess prints the same number
       regardless. Run perturbations on K numbers, count how many
       produce a different output.

Aggregation:
  combined_score = sum of weighted signals.
  Pick best score (or best per-bucket score for vote-style).

Usage:
  python3 scripts/verifier.py --input candidates.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import io
import json
import re
import signal
from typing import Any


ARITH_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow}


def _parse_safely(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _count_arith_ops(tree: ast.AST) -> int:
    """Count arithmetic ops in BinOp nodes anywhere in the AST."""
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in ARITH_OPS:
            n += 1
    return n


def _final_answer_uses_computation(tree: ast.AST) -> tuple[bool, int]:
    """Check if `answer` (or the last print arg) is derived from a chain
    of computation, not assigned directly to a literal.

    Returns: (is_computed, chain_depth)
      - is_computed: True if `answer` ultimately depends on a BinOp
      - chain_depth: how many intermediate variable hops before a BinOp
    """
    # Map variable name → its assigned expression
    assignments: dict[str, ast.AST] = {}
    for node in tree.body if hasattr(tree, "body") else []:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                assignments[tgt.id] = node.value

    # Find what `answer` is set to (or fall back to last print's arg)
    final_expr = assignments.get("answer")
    if final_expr is None:
        # Find last print call's first arg
        for node in reversed(list(ast.walk(tree))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                if node.args:
                    final_expr = node.args[0]
                break

    if final_expr is None:
        return (False, 0)

    # Walk the chain: if it's a Name pointing to another assignment, follow.
    # If it's a BinOp at any depth, it's computed.
    depth = 0
    seen = set()
    current = final_expr
    while True:
        if isinstance(current, ast.BinOp):
            return (True, depth)
        if isinstance(current, ast.Name):
            if current.id in seen:
                return (False, depth)
            seen.add(current.id)
            if current.id in assignments:
                current = assignments[current.id]
                depth += 1
                continue
        return (False, depth)


def _problem_numbers_used_in_computation(tree: ast.AST,
                                          problem_numbers: list[str]) -> set[str]:
    """Subset of problem_numbers that appear inside a BinOp in the AST.

    Numbers that are merely bound to a variable and never used in an
    arithmetic operation don't count.
    """
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for child in ast.walk(node):
                if isinstance(child, (ast.Num, ast.Constant)):
                    val = getattr(child, "n", None)
                    if val is None:
                        val = getattr(child, "value", None)
                    if isinstance(val, (int, float)):
                        s = str(int(val)) if val == int(val) else str(val)
                        if s in problem_numbers:
                            used.add(s)
    return used


_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                  for k in ("print", "range", "len", "sum", "max", "min", "sorted",
                           "abs", "round", "int", "str", "list", "tuple", "dict",
                           "set", "bool", "float", "map", "filter", "zip",
                           "enumerate", "reversed", "all", "any", "divmod", "pow")}


def _exec_safe(code: str, timeout: float = 0.5) -> str:
    buf = io.StringIO()
    g = {"__builtins__": _SAFE_BUILTINS}
    def _h(s, f): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
    except TimeoutError:
        return "ERROR: timeout"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return buf.getvalue().rstrip()


_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def perturbation_response(code: str, original_output: str,
                          problem_numbers: list[str], k: int = 3) -> dict:
    """Modify up to k distinct numeric literals that appear in the
    problem (inputs), one at a time, re-execute, count how many
    perturbations produce a different output.

    Critically: we only perturb numbers that are INPUTS (present in
    problem_numbers). Perturbing the final `answer = N` literal would
    trivially change the output without testing computation.

    A computation-bearing program responds to its inputs; a guess does
    not. Returns:
      n_perturbed   — how many perturbations were attempted
      n_responsive  — how many produced a different valid output
      responsiveness — n_responsive / n_perturbed
    """
    if not problem_numbers:
        return {"n_perturbed": 0, "n_responsive": 0, "responsiveness": 0.0}
    problem_set = set(problem_numbers)

    matches = list(_NUM_RE.finditer(code))
    if not matches:
        return {"n_perturbed": 0, "n_responsive": 0, "responsiveness": 0.0}

    # Pick unique literals that match a problem number
    seen_positions = []
    seen_values = set()
    for m in matches:
        if m.group(1) in problem_set and m.group(1) not in seen_values:
            seen_positions.append((m.start(), m.end(), m.group(1)))
            seen_values.add(m.group(1))
        if len(seen_positions) >= k:
            break

    n_resp = 0
    n_tried = 0
    for start, end, val in seen_positions:
        try:
            original_int = int(float(val))
        except ValueError:
            continue
        new_val = str(original_int + 17)
        new_code = code[:start] + new_val + code[end:]
        new_out = _exec_safe(new_code)
        if new_out.startswith("ERROR"):
            continue
        n_tried += 1
        if new_out != original_output:
            n_resp += 1
    return {
        "n_perturbed": n_tried,
        "n_responsive": n_resp,
        "responsiveness": n_resp / max(n_tried, 1),
    }


def score_candidate(code: str, output: str, pred: str | None,
                    problem_numbers: list[str]) -> dict:
    """Compute a structured score for a candidate. Returns dict with raw
    feature values and a combined score."""
    if output.startswith("ERROR") or pred is None:
        return {"runs": False, "score": -1e9,
                "ops": 0, "is_computed": False, "chain_depth": 0,
                "nums_in_op": 0, "echo": False, "responsiveness": 0.0}

    tree = _parse_safely(code)
    if tree is None:
        return {"runs": True, "score": -100, "ops": 0,
                "is_computed": False, "chain_depth": 0,
                "nums_in_op": 0, "echo": False, "responsiveness": 0.0}

    ops = _count_arith_ops(tree)
    is_computed, chain_depth = _final_answer_uses_computation(tree)
    nums_in_op = len(_problem_numbers_used_in_computation(tree, problem_numbers))
    echo = pred in set(problem_numbers)
    pert = perturbation_response(code, output, problem_numbers, k=3)

    # Combined score: heavily reward computation evidence + responsiveness
    score = (
        2.0 * ops                          # raw arithmetic ops
        + 3.0 * (1 if is_computed else 0)  # final answer is derived
        + 1.0 * chain_depth                # multi-step chain
        + 1.0 * nums_in_op                 # problem nums actually used
        - 1.5 * (1 if echo else 0)         # penalize echo
        + 4.0 * pert["responsiveness"]     # perturbation response is gold
    )
    return {
        "runs": True, "score": score,
        "ops": ops, "is_computed": is_computed, "chain_depth": chain_depth,
        "nums_in_op": nums_in_op, "echo": echo,
        "responsiveness": pert["responsiveness"],
    }


def aggregate(rec: dict, scoring_mode: str = "score") -> dict:
    """Apply the verifier to a record's candidates. Returns the chosen
    prediction + diagnostic info.

    scoring_mode:
      'score'    — pick the single candidate with the highest score
      'vote'     — bucket candidates by pred, sum scores per bucket, pick
                   the bucket with the highest total
      'mode'     — baseline: most common pred (ignores scores)
    """
    candidates = rec["candidates"]
    problem_numbers = rec.get("problem_numbers", [])

    scored = []
    for c in candidates:
        s = score_candidate(c.get("code", ""), c.get("output", ""),
                            c.get("pred"), problem_numbers)
        scored.append({"pred": c.get("pred"), **s})

    runnable = [s for s in scored if s["runs"]]
    if not runnable:
        return {"pred": None, "n_runnable": 0}

    if scoring_mode == "mode":
        votes = collections.Counter()
        for s in runnable:
            if s["pred"] is not None:
                votes[str(s["pred"]).rstrip(".")] += 1
        if not votes:
            return {"pred": None, "n_runnable": len(runnable)}
        pred, _ = votes.most_common(1)[0]
        return {"pred": pred, "n_runnable": len(runnable), "mode_vote": votes.most_common(1)[0][1]}

    if scoring_mode == "score":
        best = max(runnable, key=lambda s: s["score"])
        return {"pred": best["pred"], "n_runnable": len(runnable),
                "best_score": best["score"], "best_ops": best["ops"]}

    if scoring_mode == "vote":
        bucket_score = collections.defaultdict(float)
        bucket_count = collections.Counter()
        for s in runnable:
            if s["pred"] is None:
                continue
            key = str(s["pred"]).rstrip(".")
            bucket_score[key] += max(s["score"], 0) + 0.1
            bucket_count[key] += 1
        if not bucket_score:
            return {"pred": None, "n_runnable": len(runnable)}
        pred = max(bucket_score, key=bucket_score.get)
        return {"pred": pred, "n_runnable": len(runnable),
                "bucket_score": bucket_score[pred],
                "bucket_count": bucket_count[pred]}

    raise ValueError(f"unknown scoring_mode: {scoring_mode}")


def score_dataset(records: list[dict], scoring_mode: str) -> dict:
    correct = 0
    runnable_any = 0
    for r in records:
        gold = r["gold"].rstrip(".")
        out = aggregate(r, scoring_mode)
        if out["pred"] is not None:
            runnable_any += 1
            if str(out["pred"]).rstrip(".") == gold:
                correct += 1
    return {"correct": correct, "total": len(records),
            "runnable_any": runnable_any,
            "accuracy": correct / max(len(records), 1)}


def oracle_at_n(records: list[dict]) -> float:
    correct = 0
    for r in records:
        gold = r["gold"].rstrip(".")
        preds = [str(c.get("pred")).rstrip(".") for c in r["candidates"]
                 if c.get("pred") is not None]
        if gold in preds:
            correct += 1
    return correct / max(len(records), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    print(f"# Records: {len(records)}")
    print(f"# Oracle@N: {oracle_at_n(records)*100:.1f}%")
    print()
    print(f"{'strategy':25} {'correct/total':>15} {'acc%':>8}")
    print("-" * 50)
    for mode in ["mode", "score", "vote"]:
        r = score_dataset(records, mode)
        ratio = f"{r['correct']}/{r['total']}"
        print(f"{mode:25} {ratio:>15} {r['accuracy']*100:>7.1f}%")
    print()
    print("# Diagnostic: feature distributions over all candidates")
    ops_per_cand = []
    computed_per_cand = []
    resp_per_cand = []
    for r in records:
        for c in r["candidates"]:
            s = score_candidate(c.get("code", ""), c.get("output", ""),
                                c.get("pred"), r.get("problem_numbers", []))
            ops_per_cand.append(s["ops"])
            computed_per_cand.append(s["is_computed"])
            resp_per_cand.append(s["responsiveness"])
    n = len(ops_per_cand)
    print(f"  candidates total: {n}")
    print(f"  ops > 0: {sum(1 for x in ops_per_cand if x > 0)}/{n} = {sum(1 for x in ops_per_cand if x > 0)/n*100:.1f}%")
    print(f"  answer is computed: {sum(1 for x in computed_per_cand if x)}/{n} = {sum(1 for x in computed_per_cand if x)/n*100:.1f}%")
    print(f"  perturbation-responsive: {sum(1 for x in resp_per_cand if x > 0)}/{n} = {sum(1 for x in resp_per_cand if x > 0)/n*100:.1f}%")
    print(f"  responsiveness mean: {sum(resp_per_cand)/n:.3f}")


if __name__ == "__main__":
    main()
