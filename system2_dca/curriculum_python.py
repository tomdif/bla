"""Synthetic Python execution traces for procedural-CPU pretraining.

Every example is (prompt, target) where:
  prompt = a small Python program (a function definition + a call)
  target = the program's stdout when executed

The model learns "given this code, predict its output" — a strict
procedural-execution task with no factual content. Programs cover:
  arithmetic    : f(x) = x*7 + 3 ; print(f(5))
  loops         : sum 1..n
  conditionals  : if/else branching
  list ops      : map / filter / sort
  recursion     : factorial / Fibonacci
  string ops    : reverse / count chars
"""

from __future__ import annotations

import builtins
import contextlib
import io
import random
import signal
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


_SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "print", "range", "len", "sum", "max", "min", "sorted", "abs", "round",
    "int", "str", "list", "tuple", "dict", "set", "bool", "float",
    "map", "filter", "zip", "enumerate", "reversed", "all", "any",
    "divmod", "pow",
)}


def _run(code: str, timeout: float = 1.0) -> str:
    """Execute Python source string in-process; ~1000x faster than subprocess.

    Returns stdout, or 'ERROR: ...' on failure. Safe-ish: restricted
    builtins, no imports, SIGALRM-enforced timeout. Intended only for
    the tiny synthetic snippets we generate.
    """
    buf = io.StringIO()
    g = {"__builtins__": _SAFE_BUILTINS}

    def _handler(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
    except TimeoutError:
        return "ERROR: timeout"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return buf.getvalue().rstrip()


# --- generators -----------------------------------------------------------

def _gen_arith(rng: random.Random) -> str:
    a, b, c = rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(1, 9)
    op = rng.choice(["+", "-", "*", "//", "%"])
    return f"x = {a}\ny = {b}\nz = (x {op} y) + {c}\nprint(z)"


def _gen_loop_sum(rng: random.Random) -> str:
    n = rng.randint(3, 30)
    return f"total = 0\nfor i in range(1, {n + 1}):\n    total += i\nprint(total)"


def _gen_conditional(rng: random.Random) -> str:
    n = rng.randint(-50, 50)
    return (f"x = {n}\n"
            f"if x > 0:\n    print('positive', x)\n"
            f"elif x < 0:\n    print('negative', x)\n"
            f"else:\n    print('zero')")


def _gen_list_op(rng: random.Random) -> str:
    nums = [rng.randint(-10, 30) for _ in range(rng.randint(4, 8))]
    op = rng.choice(["sum", "max", "min", "sorted", "reverse"])
    if op == "sum":
        return f"xs = {nums}\nprint(sum(xs))"
    if op == "max":
        return f"xs = {nums}\nprint(max(xs))"
    if op == "min":
        return f"xs = {nums}\nprint(min(xs))"
    if op == "sorted":
        return f"xs = {nums}\nprint(sorted(xs))"
    return f"xs = {nums}\nxs.reverse()\nprint(xs)"


def _gen_factorial(rng: random.Random) -> str:
    n = rng.randint(1, 8)
    return ("def fact(n):\n"
            "    if n <= 1: return 1\n"
            "    return n * fact(n-1)\n"
            f"print(fact({n}))")


def _gen_fib(rng: random.Random) -> str:
    n = rng.randint(2, 15)
    return ("def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            f"print(fib({n}))")


def _gen_str_op(rng: random.Random) -> str:
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "iota", "kappa"]
    s = rng.choice(words)
    op = rng.choice(["reverse", "upper", "len", "count_chars"])
    if op == "reverse":
        return f"s = '{s}'\nprint(s[::-1])"
    if op == "upper":
        return f"s = '{s}'\nprint(s.upper())"
    if op == "len":
        return f"s = '{s}'\nprint(len(s))"
    return f"s = '{s}'\nprint(sum(1 for c in s if c in 'aeiou'))"


def _gen_filter_map(rng: random.Random) -> str:
    n = rng.randint(5, 12)
    return (f"xs = list(range({n}))\n"
            "ys = [x*x for x in xs if x % 2 == 0]\n"
            "print(ys)")


_GENERATORS = [
    _gen_arith, _gen_loop_sum, _gen_conditional, _gen_list_op,
    _gen_factorial, _gen_fib, _gen_str_op, _gen_filter_map,
]


class PythonExecutionSource(CurriculumSource):
    """Yields synthetic Python (program, output) pairs by sampling generators
    and executing each program in a subprocess."""

    name = "python_execution"

    def __init__(self, n_examples: int = 10_000, seed: int = 0,
                 timeout: float = 3.0, max_retries: int = 5):
        self.n_examples = n_examples
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries

    def __iter__(self) -> Iterator[CurriculumExample]:
        rng = random.Random(self.seed)
        produced = 0
        while produced < self.n_examples:
            gen = rng.choice(_GENERATORS)
            code = gen(rng)
            attempts = 0
            output = _run(code, self.timeout)
            while output.startswith("ERROR") and attempts < self.max_retries:
                # discard and resample
                gen = rng.choice(_GENERATORS)
                code = gen(rng)
                output = _run(code, self.timeout)
                attempts += 1
            if output.startswith("ERROR"):
                continue
            prompt = f"# Predict the output of this Python program:\n{code}\n# Output:\n"
            yield CurriculumExample(
                prompt=prompt, target=output, source=self.name,
                metadata={"generator": gen.__name__, "code_len": len(code)},
            )
            produced += 1
