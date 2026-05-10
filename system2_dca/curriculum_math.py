"""MathQA curriculum source — pulls MetaMathQA from HuggingFace.

MetaMathQA is a derived dataset of math word problems with step-by-step
solutions. Pure procedural content (no factual claims about the world);
the model learns to derive numerical answers via reasoning chains.

Falls back to a synthetic small set if HuggingFace is unavailable.
"""

from __future__ import annotations

import random
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


class MathQASource(CurriculumSource):
    name = "math_qa"

    def __init__(
        self,
        n_examples: int = 10_000,
        seed: int = 0,
        dataset: str = "meta-math/MetaMathQA",
        split: str = "train",
        max_target_chars: int = 2000,
    ):
        self.n_examples = n_examples
        self.seed = seed
        self.dataset_name = dataset
        self.split = split
        self.max_target_chars = max_target_chars

    def __iter__(self) -> Iterator[CurriculumExample]:
        try:
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
            yielded = 0
            for ex in ds:
                question = ex.get("query") or ex.get("problem") or ex.get("question") or ""
                solution = ex.get("response") or ex.get("solution") or ex.get("answer") or ""
                if not question or not solution:
                    continue
                if len(solution) > self.max_target_chars:
                    solution = solution[: self.max_target_chars]
                prompt = f"# Math problem. Show your reasoning, then state the final numerical answer.\n{question.strip()}\n# Solution:\n"
                yield CurriculumExample(
                    prompt=prompt, target=solution.strip(), source=self.name,
                    metadata={"dataset": self.dataset_name},
                )
                yielded += 1
                if yielded >= self.n_examples:
                    return
        except Exception as exc:
            yield from _fallback_synthetic_math(self.n_examples, self.seed)


def _fallback_synthetic_math(n: int, seed: int) -> Iterator[CurriculumExample]:
    """Tiny synthetic math when HF is unavailable. Word-problem-flavored."""
    rng = random.Random(seed)
    yielded = 0
    while yielded < n:
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            text = f"If you have {a} and add {b}, what do you get?"
        elif op == "-":
            ans = a - b
            text = f"What is {a} minus {b}?"
        else:
            ans = a * b
            text = f"What is {a} times {b}?"
        sol = f"Step 1: compute {a} {op} {b}.\nStep 2: result is {ans}.\nFinal answer: {ans}"
        prompt = f"# Math problem. Show your reasoning, then state the final numerical answer.\n{text}\n# Solution:\n"
        yield CurriculumExample(
            prompt=prompt, target=sol, source="math_qa_synthetic",
            metadata={"a": a, "b": b, "op": op, "answer": ans},
        )
        yielded += 1
