"""Formal-logic curriculum source — pulls FOLIO (or ProofWriter) from
HuggingFace. Falls back to a synthetic propositional-logic generator if
unavailable.
"""

from __future__ import annotations

import random
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


class FormalLogicSource(CurriculumSource):
    name = "formal_logic"

    def __init__(
        self,
        n_examples: int = 5_000,
        seed: int = 0,
        dataset: str = "yale-nlp/FOLIO",
        split: str = "train",
    ):
        self.n_examples = n_examples
        self.seed = seed
        self.dataset_name = dataset
        self.split = split

    def __iter__(self) -> Iterator[CurriculumExample]:
        try:
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
            yielded = 0
            for ex in ds:
                premises = ex.get("premises") or ex.get("context") or ""
                conclusion = ex.get("conclusion") or ex.get("hypothesis") or ""
                label = ex.get("label") or ex.get("answer") or ""
                if not premises or not conclusion or not label:
                    continue
                prompt = (
                    "# Logical entailment. Decide whether the conclusion follows from the premises.\n"
                    f"Premises:\n{premises}\n"
                    f"Conclusion: {conclusion}\n"
                    "# Answer (True / False / Uncertain):\n"
                )
                yield CurriculumExample(
                    prompt=prompt, target=str(label), source=self.name,
                    metadata={"dataset": self.dataset_name},
                )
                yielded += 1
                if yielded >= self.n_examples:
                    return
        except Exception:
            yield from _fallback_synthetic_logic(self.n_examples, self.seed)


def _fallback_synthetic_logic(n: int, seed: int) -> Iterator[CurriculumExample]:
    """Tiny propositional-logic generator: modus ponens / modus tollens /
    syllogism. Synthetic predicates use abstract names (P, Q, R) to stay
    factual-free."""
    rng = random.Random(seed)
    schemas = [
        # modus ponens
        lambda p, q: (
            f"If {p}, then {q}. {p} is true.", f"{q}", "True"
        ),
        # modus tollens
        lambda p, q: (
            f"If {p}, then {q}. {q} is false.", f"{p}", "False"
        ),
        # affirming the consequent (invalid)
        lambda p, q: (
            f"If {p}, then {q}. {q} is true.", f"{p}", "Uncertain"
        ),
        # syllogism
        lambda p, q: (
            f"All {p} are {q}. Some thing X is {p}.", f"X is {q}", "True"
        ),
    ]
    placeholders = [
        ("the switch is on", "the light is lit"),
        ("it rains", "the ground gets wet"),
        ("a number is even", "it is divisible by 2"),
        ("the queue is empty", "the worker idles"),
        ("a sensor fires", "an alarm sounds"),
        ("knights", "honest"),
        ("triangles", "polygons"),
    ]
    yielded = 0
    while yielded < n:
        schema = rng.choice(schemas)
        p, q = rng.choice(placeholders)
        premise_text, conclusion_text, label = schema(p, q)
        prompt = (
            "# Logical entailment. Decide whether the conclusion follows from the premises.\n"
            f"Premises:\n{premise_text}\n"
            f"Conclusion: {conclusion_text}\n"
            "# Answer (True / False / Uncertain):\n"
        )
        yield CurriculumExample(
            prompt=prompt, target=label, source="formal_logic_synthetic",
            metadata={"label": label},
        )
        yielded += 1
