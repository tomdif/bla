"""GSM8K training-set curriculum source v3 — variable-reference chaining.

Improvement over v2: when a step's expression contains a number that
equals a previous step's result, substitute it with the variable name.
This forces the model to learn computational chains rather than
hardcoded literal sequences.

v2 emitted (bad):
    step1 = 48/2          # = 24
    step2 = 48 + 24       # 24 hardcoded; output independent of step1
    answer = step2
    print(answer)

v3 emits (good):
    step1 = 48/2          # = 24.0
    step2 = 48 + step1    # references step1 — perturbing input flows through
    answer = step2
    print(answer)

The model trained on v3 should produce code whose perturbation
responsiveness actually correlates with correctness, making
verification effective.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


CALC_MARKER = re.compile(r"<<([^>=]+?)=([^>]+?)>>")
ANSWER_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
NUM_TOKEN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _extract_answer(answer_field: str) -> str:
    m = ANSWER_RE.search(answer_field)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()


def _strip_calc_markers(answer_field: str) -> str:
    return re.sub(r"<<[^>]*>>", "", answer_field)


def _normalize_num(s: str) -> str:
    return re.sub(r"[\$,]", "", s).strip()


def _calc_chain(answer_field: str) -> list[tuple[str, str]]:
    out = []
    for m in CALC_MARKER.finditer(answer_field):
        expr = _normalize_num(m.group(1))
        res = _normalize_num(m.group(2))
        out.append((expr, res))
    return out


def _num_eq(a_str: str, b_str: str) -> bool:
    """Compare two numeric strings by value (handle 24 == 24.0)."""
    try:
        return float(a_str) == float(b_str)
    except ValueError:
        return False


def _chain_with_variables(chain: list[tuple[str, str]]) -> list[str]:
    """Rewrite each step's expression so that any number equaling a
    previous step's result is replaced with that step's variable name.

    For ties (multiple previous steps with the same result), prefer the
    most recent.

    Returns: list of code lines (one per step).
    """
    step_results: list[tuple[str, str]] = []  # (step_name, result_str)
    lines: list[str] = []
    for i, (expr, res) in enumerate(chain, start=1):
        # Find all numeric tokens in expr (with positions), replace
        # those that match a previous step's result.
        out_expr = ""
        last_end = 0
        for m in NUM_TOKEN.finditer(expr):
            out_expr += expr[last_end:m.start()]
            tok = m.group()
            # Find the most recent previous step whose result equals tok
            replacement = None
            for s_name, s_res in reversed(step_results):
                if _num_eq(s_res, tok):
                    replacement = s_name
                    break
            out_expr += replacement if replacement is not None else tok
            last_end = m.end()
        out_expr += expr[last_end:]
        var = f"step{i}"
        lines.append(f"{var} = {out_expr}")
        step_results.append((var, res))
    return lines


def _build_python(chain: list[tuple[str, str]], final_answer: str) -> str:
    if not chain:
        return f"answer = {final_answer}\nprint(answer)"
    lines = _chain_with_variables(chain)
    lines.append(f"answer = step{len(chain)}")
    lines.append("print(answer)")
    return "\n".join(lines)


class GSM8KTrainV3Source(CurriculumSource):
    """GSM8K-train with variable-reference Python solutions.

    Modes:
      'python' — problem -> step-by-step Python with variable references
      'cot'    — problem -> chain-of-thought natural language
      'mixed'  — alternates (70% python, 30% cot)
    """

    name = "gsm8k_train_v3"

    def __init__(
        self,
        n_examples: int = 7000,
        seed: int = 0,
        mode: str = "python",
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
                    prompt=prompt, target=target, source="gsm8k_train_v3_cot",
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
                    prompt=prompt, target=target, source="gsm8k_train_v3_python",
                    metadata={"gold": ans, "idx": i, "n_steps": len(chain)},
                )
            produced += 1
