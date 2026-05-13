"""RFT step 1: generate N candidates per GSM8K-train problem.

Uses a trained procedural-core checkpoint to produce many sampled PAL
candidates on the GSM8K-train split. Saves all (problem, code, output,
correct) records to JSONL for the next step (filter + reformat).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import contextlib
import builtins
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# B300 (sm_103) compatibility: cuDNN has no SDPA plan yet, force flash/math/mem-efficient backends
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

from scripts.phase6_eval_pal import exec_python, extract_python, parse_number
from scripts.phase6_eval_pal_sc import load_bla_sampling


def _extract_answer(answer_field: str) -> str:
    m = re.search(r"####\s*(-?\d[\d,\.]*)", answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-problems", type=int, default=7473,
                   help="number of GSM8K-train problems to generate on")
    p.add_argument("--n-samples", type=int, default=16,
                   help="candidates per problem")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)
    print(json.dumps({"event": "loading", "ckpt": args.ckpt}), flush=True)
    generate, _ = load_bla_sampling(args.ckpt, device, args.temperature, args.top_p)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= args.n_problems:
            break
        items.append(ex)

    out_path = os.path.join(args.output, "rft_candidates.jsonl")
    t0 = time.time()
    n_with_correct = 0
    total_correct = 0
    with open(out_path, "w") as fh:
        for i, ex in enumerate(items):
            question = ex["question"]
            gold = _extract_answer(ex["answer"])
            if not gold:
                continue

            prompt = (
                "Write a Python program that prints the answer to this math problem.\n"
                "End with: print(answer)\n"
                f"Problem: {question}\nPython:\n"
            )
            any_correct = False
            for _ in range(args.n_samples):
                gen = generate(prompt, args.max_new_tokens)
                code = extract_python(gen) or ""
                if not code:
                    continue
                output = exec_python(code)
                if output.startswith("ERROR"):
                    continue
                pred = parse_number(output)
                if pred is None:
                    continue
                correct = pred.rstrip(".") == gold.rstrip(".")
                if correct:
                    total_correct += 1
                    any_correct = True
                    # Save only correct candidates (RFT training data)
                    fh.write(json.dumps({
                        "problem": question,
                        "code": code,
                        "output": output,
                        "pred": pred,
                        "gold": gold,
                        "idx": i,
                    }) + "\n")
            if any_correct:
                n_with_correct += 1
            if (i + 1) % 100 == 0:
                print(json.dumps({
                    "event": "progress", "done": i + 1,
                    "total": len(items),
                    "n_with_correct": n_with_correct,
                    "total_correct": total_correct,
                    "elapsed_s": round(time.time() - t0, 1),
                }), flush=True)

    print(json.dumps({
        "event": "done",
        "n_problems": len(items),
        "n_with_correct": n_with_correct,
        "total_correct": total_correct,
        "elapsed_s": round(time.time() - t0, 1),
        "output": out_path,
    }), flush=True)


if __name__ == "__main__":
    main()
