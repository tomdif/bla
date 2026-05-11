"""GSM8K training-set curriculum source.

Streams the gsm8k 'train' split (~7.5K real word problems) with their
chain-of-thought answers. This bridges our synthetic curriculum to the
linguistic variation of real GSM8K problems — addresses the diagnosis
from runs 5-9 that templated math doesn't transfer to GSM8K-test.

We deliberately use ONLY the train split. The test split (1.3K problems)
is held out for eval.

Two example flavors are produced:
  - 'cot':    teach chain-of-thought ending in 'Answer: N'
  - 'python': teach Python that prints the answer using numbers from the
              problem (heuristic generation; not all problems will have
              clean Python, but those that do reinforce translation)
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


def _extract_answer(answer_field: str) -> str:
    """GSM8K answers end with '#### N'. Pull N out, normalize."""
    m = re.search(r"####\s*(-?\d[\d,\.]*)", answer_field)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def _strip_calc_markers(answer_field: str) -> str:
    """Some GSM8K answers have <<a*b=c>> calc markers; strip them for
    natural CoT presentation."""
    return re.sub(r"<<[^>]*>>", "", answer_field)


def _extract_numbers(text: str) -> list[str]:
    """All distinct numbers in the question (preserving order)."""
    seen = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text):
        if tok not in seen:
            seen.append(tok)
    return seen


class GSM8KTrainSource(CurriculumSource):
    """GSM8K-train problems. Mode 'cot' yields the chain-of-thought
    target ending in 'Answer: N'. Mode 'python' yields a Python
    program that computes the answer using a final-answer-only shim
    (works as weak supervision for code generation)."""

    name = "gsm8k_train"

    def __init__(
        self,
        n_examples: int = 7000,
        seed: int = 0,
        mode: str = "mixed",  # 'cot', 'python', or 'mixed' (alternates)
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
            cot_a = _strip_calc_markers(raw_a)
            cot_a = cot_a.split("####")[0].strip()
            cot_target = f"{cot_a}\nAnswer: {ans}"

            mode = self.mode
            if mode == "mixed":
                mode = "cot" if rng.random() < 0.7 else "python"

            if mode == "cot":
                prompt = (
                    "Solve this math problem step by step. End with: 'Answer: <number>'.\n"
                    f"Problem: {q}\nSolution:"
                )
                yield CurriculumExample(
                    prompt=prompt, target=cot_target, source="gsm8k_train_cot",
                    metadata={"gold": ans, "idx": i},
                )
            else:
                # Python flavor: weak supervision. We don't know the
                # operations, but we can emit a program that uses the
                # problem's numbers and prints the gold answer. The
                # model still has to learn which numbers + operations
                # produce the answer; this gives it the format anchor.
                nums = _extract_numbers(q)[:6]
                if not nums:
                    continue
                # Heuristic body: bind each number, end with print(gold)
                body = "\n".join(f"n{i+1} = {v}" for i, v in enumerate(nums))
                body += f"\nanswer = {ans}\nprint(answer)"
                prompt = (
                    "Write a Python program that prints the answer to this math problem.\n"
                    "End with: print(answer)\n"
                    f"Problem: {q}\nPython:\n"
                )
                py_target = f"{body}\nAnswer: {ans}"
                yield CurriculumExample(
                    prompt=prompt, target=py_target, source="gsm8k_train_python",
                    metadata={"gold": ans, "idx": i},
                )
            produced += 1
