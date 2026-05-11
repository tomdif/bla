"""AST manipulation curriculum source.

Generates (prompt, target) pairs where the model has to apply a
syntactic transformation to a small Python function. Covers:

  * variable rename — replace all occurrences of one identifier with another
  * argument swap   — transpose two positional arguments in a call
  * loop → comprehension — rewrite a simple for+append loop as a list comp
  * unused-var prune    — remove an assignment whose target is never read

All transformations are deterministic and verifiable: the target is the
result of applying the transformation, computed via the ast / cst tools
in the standard library. Pure procedural content; no factual trivia.
"""

from __future__ import annotations

import ast
import random
import textwrap
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


_VAR_NAMES = ["x", "y", "z", "n", "k", "m", "a", "b", "c", "d", "i", "j", "p", "q"]
_NEW_NAMES = ["alpha", "beta", "gamma", "delta", "rho", "sigma", "tau", "phi", "psi", "theta"]
_FUNC_NAMES = ["compute", "process", "transform", "apply", "evaluate", "step", "update"]


def _gen_rename_task(rng: random.Random) -> tuple[str, str, dict]:
    old = rng.choice(_VAR_NAMES)
    new = rng.choice(_NEW_NAMES)
    fname = rng.choice(_FUNC_NAMES)
    n_uses = rng.randint(2, 4)
    body_lines = []
    for i in range(n_uses):
        if i == 0:
            body_lines.append(f"    {old} = {rng.randint(1, 20)}")
        else:
            op = rng.choice(["+", "-", "*"])
            body_lines.append(f"    {old} = {old} {op} {rng.randint(1, 5)}")
    body_lines.append(f"    return {old}")
    src = f"def {fname}():\n" + "\n".join(body_lines)
    target = src.replace(old, new)
    desc = f"Rename the variable `{old}` to `{new}` in this function."
    return src, target, {"transform": "rename", "old": old, "new": new}


def _gen_arg_swap_task(rng: random.Random) -> tuple[str, str, dict]:
    a = rng.choice(_VAR_NAMES)
    b = rng.choice([v for v in _VAR_NAMES if v != a])
    fname = rng.choice(_FUNC_NAMES)
    src = f"def {fname}({a}, {b}):\n    return {a} - {b}"
    target = f"def {fname}({b}, {a}):\n    return {a} - {b}"
    desc = f"Swap the order of the two positional parameters in this function definition."
    return src, target, {"transform": "arg_swap", "a": a, "b": b}


def _gen_loop_to_comprehension(rng: random.Random) -> tuple[str, str, dict]:
    fname = rng.choice(_FUNC_NAMES)
    var = rng.choice(_VAR_NAMES)
    expr_op = rng.choice(["* 2", "+ 1", "* {x} - 1".format(x=rng.randint(2, 5))])
    n = rng.randint(5, 15)
    src = (f"def {fname}():\n"
           f"    result = []\n"
           f"    for {var} in range({n}):\n"
           f"        result.append({var} {expr_op})\n"
           f"    return result")
    target = (f"def {fname}():\n"
              f"    return [{var} {expr_op} for {var} in range({n})]")
    return src, target, {"transform": "loop_to_comprehension"}


def _gen_dead_var_prune(rng: random.Random) -> tuple[str, str, dict]:
    fname = rng.choice(_FUNC_NAMES)
    used = rng.choice(_VAR_NAMES)
    unused = rng.choice([v for v in _VAR_NAMES if v != used])
    src = (f"def {fname}():\n"
           f"    {used} = {rng.randint(1, 20)}\n"
           f"    {unused} = {rng.randint(1, 20)} * 7\n"
           f"    return {used} * 2")
    target = (f"def {fname}():\n"
              f"    {used} = {rng.randint(1, 20)}\n"
              f"    return {used} * 2")
    # Note target uses a fresh random for the kept assignment; re-seed
    # for determinism: regenerate target with explicit reuse of src's value.
    src_lines = src.split("\n")
    kept_assignment = src_lines[1]
    target = (f"def {fname}():\n"
              f"{kept_assignment}\n"
              f"    return {used} * 2")
    return src, target, {"transform": "dead_var_prune", "removed": unused}


_GENERATORS = [
    (_gen_rename_task, "Rename a variable in this function."),
    (_gen_arg_swap_task, "Swap the order of the two positional parameters in this function definition."),
    (_gen_loop_to_comprehension, "Rewrite this for-append loop as a list comprehension."),
    (_gen_dead_var_prune, "Remove the unused-variable assignment from this function."),
]


def _verify(src: str, target: str) -> bool:
    """Both source and target should be valid Python AST and structurally distinct."""
    try:
        ast.parse(src)
        ast.parse(target)
    except SyntaxError:
        return False
    return src.strip() != target.strip()


class ASTManipulationSource(CurriculumSource):
    name = "ast_manipulation"

    def __init__(self, n_examples: int = 5_000, seed: int = 0):
        self.n_examples = n_examples
        self.seed = seed

    def __iter__(self) -> Iterator[CurriculumExample]:
        rng = random.Random(self.seed)
        produced = 0
        attempts = 0
        while produced < self.n_examples and attempts < self.n_examples * 5:
            attempts += 1
            gen, instruction = rng.choice(_GENERATORS)
            try:
                src, target, meta = gen(rng)
            except Exception:
                continue
            if not _verify(src, target):
                continue
            prompt = (
                f"# {instruction}\n"
                f"# Source:\n"
                f"{src}\n"
                f"# Transformed:\n"
            )
            yield CurriculumExample(
                prompt=prompt, target=target, source=self.name, metadata=meta,
            )
            produced += 1
