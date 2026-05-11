"""Chain-of-thought-before-code curriculum source (CoT-PAL).

For GSM8K-train problems, emit targets of the form:

    Step 1: <natural-language reasoning>
    Step 2: <natural-language reasoning>
    ...
    Python:
    step1 = <expr>
    step2 = <expr using step1>
    ...
    answer = stepN
    print(answer)
    Answer: N

Trained on this, the model must commit to a reasoning chain BEFORE
emitting code. Three benefits over plain PAL:

  1. The reasoning narrative gives more tokens to "think with"
     before having to pick operations.
  2. The natural-language steps and the code steps must agree,
     which is itself a self-consistency signal at inference time.
  3. The reasoning lines act as a verifier surface: a critic could
     check whether step-N's English matches step-N's Python.

Implementation: parse GSM8K-train's answer field for `<<expr=result>>`
markers AND the surrounding prose. The prose between consecutive
markers (or before the first marker) is the reasoning for that step.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource
from .curriculum_gsm8k_v3 import (
    _extract_answer, _calc_chain, _chain_with_variables, _normalize_num,
)


CALC_MARKER = re.compile(r"<<([^>=]+?)=([^>]+?)>>")


def _segment_reasoning(answer_field: str) -> list[str]:
    """Split the answer text into prose segments, one per <<...>> marker.

    Returns the prose that motivates each calculation. The prose for
    marker N starts AFTER marker N-1's repeated result number (which
    GSM8K answers always include after a <<...>> marker), and ends
    just before the calc expression for marker N.

    Example raw text:
      'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\\n
       Natalia sold 48+24 = <<48+24=72>>72 clips altogether...'

    Step 1 prose: 'Natalia sold 48/2' (up to first marker)
    Step 2 prose: 'Natalia sold 48+24'
                  (skipping the '24 clips in May.\\nNatalia sold' prefix
                   that follows marker 1, find the last sentence-y bit
                   before marker 2)
    """
    matches = list(CALC_MARKER.finditer(answer_field))
    parts = []
    last_end = 0
    last_result = None
    for m in matches:
        chunk = answer_field[last_end:m.start()]
        # Strip leading whitespace
        chunk = chunk.lstrip()
        # If the chunk begins with the previous step's result, drop it
        if last_result is not None and chunk.startswith(last_result):
            chunk = chunk[len(last_result):]
        chunk = chunk.lstrip(" ,.\n")
        # Keep only the last sentence-ish piece (after the last '\n' or
        # '. ' — split on '. ' specifically to avoid breaking on decimals
        # like '0.2').
        for sep in ("\n", ". "):
            if sep in chunk:
                pieces = [p for p in chunk.split(sep) if p.strip()]
                if pieces:
                    chunk = pieces[-1]
        chunk = chunk.replace("\n", " ").strip()
        # Drop trailing connective words / equals signs that lead into the calc
        chunk = re.sub(r"\s*(=|is|equals|so|then)\s*$", "", chunk,
                       flags=re.IGNORECASE).rstrip(",;: ")
        if not chunk:
            chunk = f"Compute {m.group(1).strip()}"
        parts.append(chunk)
        last_end = m.end()
        last_result = _normalize_num(m.group(2))
    return parts


def _build_cot_python(prose_steps: list[str], chain_lines: list[str],
                     final_answer: str) -> str:
    """Combine prose Steps + Python step lines into a single target.

    Format:
      Step 1: <prose>
      Step 2: <prose>
      ...
      Python:
      step1 = <expr>
      step2 = <expr>
      ...
      answer = stepN
      print(answer)
      Answer: N
    """
    out = []
    for i, prose in enumerate(prose_steps, start=1):
        out.append(f"Step {i}: {prose}")
    out.append("Python:")
    for line in chain_lines:
        out.append(line)
    if chain_lines:
        out.append(f"answer = step{len(chain_lines)}")
    else:
        out.append(f"answer = {final_answer}")
    out.append("print(answer)")
    out.append(f"Answer: {final_answer}")
    return "\n".join(out)


class CoTPALSource(CurriculumSource):
    """GSM8K-train as Chain-of-Thought-then-Python targets.

    The prompt asks the model to produce stepwise reasoning followed
    by chained Python.
    """

    name = "cot_pal"

    def __init__(self, n_examples: int = 7000, seed: int = 0):
        self.n_examples = n_examples
        self.seed = seed

    def __iter__(self) -> Iterator[CurriculumExample]:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        except Exception:
            return iter(())

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
            if not chain:
                continue
            prose_steps = _segment_reasoning(raw_a)
            if len(prose_steps) != len(chain):
                # Mismatch (shouldn't happen often) — fall back to brief prose
                prose_steps = [f"Compute {expr}" for expr, _ in chain]
            chain_lines = _chain_with_variables(chain)
            target = _build_cot_python(prose_steps, chain_lines, ans)

            prompt = (
                "Solve this math problem. First show stepwise reasoning, then "
                "write Python that prints the answer.\n"
                "Format:\n"
                "  Step 1: ...\n"
                "  Step 2: ...\n"
                "  Python:\n"
                "  step1 = ...\n"
                "  step2 = ... step1 ...\n"
                "  answer = stepN\n"
                "  print(answer)\n"
                "  Answer: <number>\n\n"
                f"Problem: {q}\nSolution:\n"
            )
            yield CurriculumExample(
                prompt=prompt, target=target, source=self.name,
                metadata={"gold": ans, "idx": i, "n_steps": len(chain)},
            )
            produced += 1
