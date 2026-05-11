"""In-distribution eval: tests whether the model has learned its own
training distribution. Uses the SAME generators as the curriculum but
with fresh seeds the model didn't see during training.

If a BLA model can't beat a base GPT-2 on its OWN training distribution,
the curriculum isn't teaching anything useful. If it does, then the
gap with GSM8K/HumanEval is "doesn't transfer" rather than "didn't
learn."

Usage:
  python3 scripts/phase6_eval_indist.py --ckpt runs/phase6/.../final.pt --output runs/phase6/eval_indist_X
  python3 scripts/phase6_eval_indist.py --baseline gpt2-xl --output runs/phase6/eval_indist_gpt2xl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system2_dca.curriculum_python import PythonExecutionSource
from system2_dca.curriculum_math import MathQASource, _fallback_synthetic_math
from system2_dca.curriculum_logic import _fallback_synthetic_logic
from scripts.phase6_eval import load_bla_checkpoint, load_baseline


# Use seed=999 so we don't collide with seed=0/1/2 used at training time.
EVAL_SEED = 999


def _normalize(s: str) -> str:
    return s.strip().lower()


def run_python_indist(generate, n: int = 50, max_new: int = 64) -> dict:
    """Sample 50 fresh Python programs from the same generators used at
    training time; score exact-match (after strip) on the predicted output."""
    src = PythonExecutionSource(n_examples=n, seed=EVAL_SEED)
    correct = 0
    tested = 0
    for ex in src:
        gen = generate(ex.prompt, max_new)
        # Take only the first line — model often runs on
        pred = gen.split("\n", 1)[0].strip()
        target = ex.target.strip()
        if _normalize(pred) == _normalize(target):
            correct += 1
        tested += 1
        if tested >= n:
            break
    return {"task": "python_indist", "accuracy": correct / max(tested, 1),
            "n_tested": tested}


def run_math_indist(generate, n: int = 50, max_new: int = 64) -> dict:
    """Use the math curriculum's synthetic fallback (seed=999) — covers the
    same problem templates the model saw at training time."""
    correct = 0
    tested = 0
    for ex in _fallback_synthetic_math(n, EVAL_SEED):
        gen = generate(ex.prompt, max_new)
        # Pull last number from generation; compare to last number in target.
        nums_gen = re.findall(r"-?\d+(?:\.\d+)?", gen)
        nums_tgt = re.findall(r"-?\d+(?:\.\d+)?", ex.target)
        if nums_gen and nums_tgt and nums_gen[-1] == nums_tgt[-1]:
            correct += 1
        tested += 1
        if tested >= n:
            break
    return {"task": "math_indist", "accuracy": correct / max(tested, 1),
            "n_tested": tested}


def run_logic_indist(generate, n: int = 50, max_new: int = 16) -> dict:
    """Synthetic propositional logic — same generators (modus ponens, etc.)."""
    correct = 0
    tested = 0
    for ex in _fallback_synthetic_logic(n, EVAL_SEED):
        gen = generate(ex.prompt, max_new)
        # Look for True/False/Uncertain in the first line
        first = gen.split("\n", 1)[0]
        pred = None
        for cand in ("True", "False", "Uncertain"):
            if cand.lower() in first.lower():
                pred = cand
                break
        if pred == ex.target:
            correct += 1
        tested += 1
        if tested >= n:
            break
    return {"task": "logic_indist", "accuracy": correct / max(tested, 1),
            "n_tested": tested}


RUNNERS = {
    "python": run_python_indist,
    "math": run_math_indist,
    "logic": run_logic_indist,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--suite", nargs="+", default=["python", "math", "logic"])
    p.add_argument("--n-per-task", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt and not args.baseline:
        raise SystemExit("Pass --ckpt or --baseline.")
    if args.ckpt and args.baseline:
        raise SystemExit("Pass only one of --ckpt or --baseline.")

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    if args.ckpt:
        print(json.dumps({"event": "loading_bla", "ckpt": args.ckpt}), flush=True)
        generate, _ = load_bla_checkpoint(args.ckpt, device)
        label = f"bla:{os.path.basename(args.ckpt)}"
    else:
        print(json.dumps({"event": "loading_baseline", "model": args.baseline}), flush=True)
        generate, _ = load_baseline(args.baseline, device)
        label = f"baseline:{args.baseline}"

    summary = {"label": label, "tasks": {}}
    for task in args.suite:
        runner = RUNNERS[task]
        t0 = time.time()
        result = runner(generate, args.n_per_task, args.max_new_tokens)
        result["elapsed_s"] = round(time.time() - t0, 1)
        summary["tasks"][task] = result
        print(json.dumps({"event": "task", **result}), flush=True)

    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
