"""GSM8K training-set curriculum source v2 — extracts <<expr=result>>
calc markers from each answer and emits Python that performs those
exact computations.

Fixes the v1 weak-supervision flaw: v1 just bound problem numbers
to variables and printed the gold answer, so the model learned to
GUESS rather than COMPUTE. v2 emits Python that does the arithmetic
step by step, matching the model's expected output format.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


CALC_MARKER = re.compile(r"<<([^>=]+?)=([^>]+?)>>")
ANSWER_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")


def _extract_answer(answer_field: str) -> str:
    m = ANSWER_RE.search(answer_field)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def _strip_calc_markers(answer_field: str) -> str:
    return re.sub(r"<<[^>]*>>", "", answer_field)


def _normalize_num(s: str) -> str:
    """Strip dollar signs, commas; keep number."""
    return re.sub(r"[\$,]", "", s).strip()


def _calc_chain(answer_field: str) -> list[tuple[str, str]]:
    """Extract (expression, result) pairs from <<expr=result>> markers.
    Returns the chain in order of appearance."""
    out = []
    for m in CALC_MARKER.finditer(answer_field):
        expr = _normalize_num(m.group(1))
        res = _normalize_num(m.group(2))
        out.append((expr, res))
    return out


def _build_python(chain: list[tuple[str, str]], final_answer: str) -> str:
    """Build a Python program where each step is a variable bound to
    its expression's value. The final answer is the last step's result.
    """
    lines = []
    for i, (expr, res) in enumerate(chain, start=1):
        # Use the expression literally; the model learns to evaluate it
        lines.append(f"step{i} = {expr}")
    if chain:
        lines.append(f"answer = step{len(chain)}")
    else:
        # No chain — fall back to bare answer
        lines.append(f"answer = {final_answer}")
    lines.append("print(answer)")
    return "\n".join(lines)


class GSM8KTrainV2Source(CurriculumSource):
    """GSM8K-train problems with proper Python solutions extracted from
    <<expr=result>> calc markers. Two modes:
      'python' — problem -> step-by-step Python with real arithmetic
      'cot'    — problem -> chain-of-thought natural language
      'mixed'  — alternates (70% python, 30% cot)
    """

    name = "gsm8k_train_v2"

    def __init__(
        self,
        n_examples: int = 7000,
        seed: int = 0,
        mode: str = "python",  # default to python for fixing the bug
    ):
        self.n_examples = n_examples
        self.seed = seed
        self.mode = mode

    def __iter__(self) -> Iterator[CurriculumExample]:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        except Exception:
            return iter(())

        rng = random.Random(self.seed)
        produced = 0
        for i, ex in enumerate(ds):
            if produced >= self.n_examples:
                return
            q = ex["question"]
            raw_a = ex["answer"]
            ans = _extract_answer(raw_a)
            if not ans:
                continue
            chain = _calc_chain(raw_a)
            if not chain and self.mode == "python":
                # Skip problems without calc markers for python mode
                continue

            mode = self.mode
            if mode == "mixed":
                mode = "python" if rng.random() < 0.7 else "cot"

            if mode == "cot":
                cot_a = _strip_calc_markers(raw_a).split("####")[0].strip()
                prompt = (
                    "Solve this math problem step by step. End with: 'Answer: <number>'.\n"
                    f"Problem: {q}\nSolution:"
                )
                target = f"{cot_a}\nAnswer: {ans}"
                yield CurriculumExample(
                    prompt=prompt, target=target, source="gsm8k_train_v2_cot",
                    metadata={"gold": ans, "idx": i},
                )
            else:
                py = _build_python(chain, ans)
                prompt = (
                    "Write a Python program that prints the answer to this math problem.\n"
                    "End with: print(answer)\n"
                    f"Problem: {q}\nPython:\n"
                )
                target = f"{py}\nAnswer: {ans}"
                yield CurriculumExample(
                    prompt=prompt, target=target, source="gsm8k_train_v2_python",
                    metadata={"gold": ans, "idx": i, "n_steps": len(chain)},
                )
            produced += 1
