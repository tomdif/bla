"""PAL eval that saves ALL N candidates per problem, with raw code,
execution output, and several rerank-relevant features. Used as input
for offline reranking experiments.

Saves a single JSONL where each line is a problem record:
  {question, gold, candidates: [{code, output, pred, features: {...}}]}
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import time
import contextlib
import builtins

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from scripts.phase6_eval_pal import exec_python, extract_python, parse_number
from scripts.phase6_eval_pal_sc import load_bla_sampling
from system2_dca.number_parser import extract_problem_numbers


def _problem_numbers(question: str) -> list[str]:
    return extract_problem_numbers(question)


def _code_features(code: str, problem_numbers: list[str], output: str) -> dict:
    """Compute features the reranker can use."""
    lines = [l for l in code.split("\n") if l.strip()]
    has_print = any("print" in l for l in lines)
    has_assignments = sum(1 for l in lines if "=" in l and "==" not in l)
    uses_nums = sum(1 for n in problem_numbers if n in code)
    op_count = sum(code.count(op) for op in ["+", "-", "*", "/", "//", "%"])
    code_len = len(code)
    pred = parse_number(output) if not output.startswith("ERROR") else None
    pred_in_problem = pred is not None and pred in problem_numbers
    return {
        "n_lines": len(lines),
        "code_len": code_len,
        "has_print": has_print,
        "n_assignments": has_assignments,
        "uses_n_problem_nums": uses_nums,
        "n_problem_nums": len(problem_numbers),
        "uses_fraction": uses_nums / max(len(problem_numbers), 1),
        "op_count": op_count,
        "ran_ok": not output.startswith("ERROR"),
        "pred_in_problem": pred_in_problem,
    }


def run_eval(generate, n: int = 200, max_new: int = 256, n_samples: int = 8) -> list:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        items.append(ex)

    records = []
    for i, ex in enumerate(items):
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        prob_nums = _problem_numbers(question)

        prompt = (
            "Write a Python program that prints the answer to this math problem.\n"
            "End with: print(answer)\n"
            f"Problem: {question}\n"
            "Python:\n"
        )

        candidates = []
        for s_idx in range(n_samples):
            gen = generate(prompt, max_new)
            code = extract_python(gen) or ""
            output = exec_python(code) if code else "ERROR: no code"
            pred = parse_number(output) if not output.startswith("ERROR") else None
            feats = _code_features(code, prob_nums, output)
            candidates.append({
                "code": code, "output": output[:200],
                "pred": pred, "features": feats,
            })

        records.append({
            "idx": i, "question": question, "gold": gold,
            "problem_numbers": prob_nums, "candidates": candidates,
        })
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "done": i + 1, "total": len(items)}), flush=True)

    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-per-task", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    print(json.dumps({"event": "loading", "ckpt": args.ckpt}), flush=True)
    generate, _ = load_bla_sampling(args.ckpt, device, args.temperature, args.top_p)

    t0 = time.time()
    records = run_eval(generate, args.n_per_task, args.max_new_tokens, args.n_samples)

    out_path = os.path.join(args.output, "candidates.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    elapsed = round(time.time() - t0, 1)
    print(json.dumps({"event": "done", "n": len(records), "elapsed_s": elapsed,
                      "output": out_path}), flush=True)


if __name__ == "__main__":
    main()
