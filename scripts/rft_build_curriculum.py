"""RFT step 2: build the bootstrapped curriculum file.

Reads rft_candidates.jsonl (already filtered to correct candidates) and
produces a JSONL in the standard curriculum format:
  {prompt: str, target: str, source: str, metadata: dict}

Format matches curriculum_word_to_python (Python-first PAL targets) so
the dataset class can mix it in directly.

Optionally also produces a CoT-PAL formatted version where targets
include the gold chain-of-thought (extracted from GSM8K-train answers
with calc markers stripped) BEFORE the Python.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _strip_calc(s: str) -> str:
    return re.sub(r"<<[^>]*>>", "", s)


def load_gsm8k_cot_map():
    """Return {problem_text: cot_prose} from GSM8K-train."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    out = {}
    for ex in ds:
        cot = _strip_calc(ex["answer"]).split("####")[0].strip()
        out[ex["question"]] = cot
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="rft_candidates.jsonl from rft_generate.py")
    p.add_argument("--output", required=True,
                   help="curriculum-format JSONL")
    p.add_argument("--mode", choices=["pal", "cot_pal"], default="cot_pal")
    args = p.parse_args()

    print(f"Loading candidates from {args.input}")
    records = []
    for line in open(args.input):
        records.append(json.loads(line))
    print(f"Loaded {len(records)} verified-correct candidates across "
          f"{len({r['idx'] for r in records})} unique problems")

    cot_map = load_gsm8k_cot_map() if args.mode == "cot_pal" else {}

    written = 0
    with open(args.output, "w") as fh:
        for r in records:
            q = r["problem"]
            code = r["code"]
            gold = r["gold"]

            if args.mode == "cot_pal":
                cot = cot_map.get(q, "")
                cot_steps = [l.strip() for l in cot.split("\n") if l.strip()]
                step_lines = []
                for i, line in enumerate(cot_steps, 1):
                    step_lines.append(f"Step {i}: {line}")
                target = (
                    "\n".join(step_lines)
                    + "\nPython:\n"
                    + code
                    + f"\nAnswer: {gold}"
                )
                prompt = (
                    "Solve this math problem. First show stepwise reasoning, then "
                    "write Python that prints the answer.\n"
                    "Format: Step 1: ... Step 2: ... Python:\n  step1 = ...\n  ...\n  print(answer)\n"
                    "  Answer: <number>\n"
                    f"Problem: {q}\nSolution:\n"
                )
                source = "rft_cot_pal"
            else:
                target = f"{code}\nAnswer: {gold}"
                prompt = (
                    "Write a Python program that prints the answer to this math problem.\n"
                    "End with: print(answer)\n"
                    f"Problem: {q}\nPython:\n"
                )
                source = "rft_pal"

            fh.write(json.dumps({
                "prompt": prompt, "target": target, "source": source,
                "metadata": {"gold": gold, "idx": r["idx"]},
            }) + "\n")
            written += 1

    print(f"Wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
