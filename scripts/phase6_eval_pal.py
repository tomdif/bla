"""Program-Aided Language eval (PAL).

Asks the model to write Python code that prints the answer, then EXECUTES
that code to extract the actual numeric result. Tests whether a small
procedural-trained model can solve math by delegating to a deterministic
backend — the 'simulate' action in the BLA router design.

This implements A1 from the first-principles analysis: no new training,
just a different inference loop on top of the existing checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import time
import contextlib
import builtins

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from scripts.phase6_eval import load_bla_checkpoint, load_baseline


SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "print", "range", "len", "sum", "max", "min", "sorted", "abs", "round",
    "int", "str", "list", "tuple", "dict", "set", "bool", "float",
    "map", "filter", "zip", "enumerate", "reversed", "all", "any",
    "divmod", "pow",
)}


def exec_python(code: str, timeout: float = 2.0) -> str:
    """Run code in-process; return stdout or 'ERROR: ...'."""
    buf = io.StringIO()
    g = {"__builtins__": SAFE_BUILTINS}
    def handler(signum, frame): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
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


def extract_python(generation: str) -> str | None:
    """Pull a code block from the model's generation. Tries common patterns:
      ```python ... ```
      ``` ... ```
      raw code starting after 'Python:' / 'Code:'
      everything before a non-code 'Answer:' line
    """
    # Try fenced code blocks first
    fence = re.search(r"```(?:python)?\n(.*?)```", generation, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # Try after 'Python:' or 'Code:' tag
    tag = re.search(r"(?:Python|Code)\s*:\s*\n(.*?)(?:\nAnswer:|\Z)",
                    generation, flags=re.DOTALL)
    if tag:
        return tag.group(1).strip()

    # Fallback: take everything that looks like code up to first non-indented
    # non-code line. Be permissive — model may emit raw code.
    lines = generation.split("\n")
    code_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        # heuristic: code line is indented, or starts with a Python keyword,
        # or contains '=' or 'print'
        if (line.startswith((" ", "\t"))
            or stripped.startswith(("def ", "for ", "if ", "while ", "import ",
                                     "from ", "return ", "print", "#"))
            or "=" in stripped):
            code_lines.append(line)
            started = True
        elif started:
            # First non-code line after code started → stop
            break
    code = "\n".join(code_lines).strip()
    return code if code else None


def parse_number(s: str) -> str | None:
    """Last number in the string, normalized (no trailing zeros etc.)."""
    m = list(re.finditer(r"-?\d+(?:\.\d+)?", s))
    if not m:
        return None
    n = m[-1].group(0)
    # Strip trailing .0 for int-equivalent
    if "." in n:
        try:
            f = float(n)
            if f == int(f):
                return str(int(f))
            return n
        except ValueError:
            return n
    return n


def run_math_pal(generate, n: int = 200, max_new: int = 256) -> dict:
    """GSM8K test split, solved via Python execution."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        items.append(ex)

    correct = 0
    code_extracted = 0
    code_ran = 0
    results = []
    for ex in items:
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)

        prompt = (
            "Write a Python program that prints the answer to this math problem.\n"
            "End with: print(answer)\n"
            f"Problem: {question}\n"
            "Python:\n"
        )
        gen = generate(prompt, max_new)
        code = extract_python(gen)
        if code:
            code_extracted += 1
            output = exec_python(code)
            if not output.startswith("ERROR"):
                code_ran += 1
            pred = parse_number(output)
        else:
            pred = None
            output = "<no code extracted>"

        ok = (pred is not None) and (pred.rstrip(".") == gold.rstrip("."))
        correct += int(ok)
        results.append({"question": question[:80], "gold": gold,
                        "pred": pred, "output": output[:80], "ok": ok})

    return {
        "task": "math_pal",
        "accuracy": correct / max(len(items), 1),
        "code_extracted": code_extracted,
        "code_ran": code_ran,
        "n_tested": len(items),
        "results": results,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--n-per-task", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.ckpt and not args.baseline:
        raise SystemExit("Pass --ckpt or --baseline.")

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    if args.ckpt:
        print(json.dumps({"event": "loading_bla", "ckpt": args.ckpt}), flush=True)
        generate, _ = load_bla_checkpoint(args.ckpt, device)
        label = f"bla-pal:{os.path.basename(args.ckpt)}"
    else:
        print(json.dumps({"event": "loading_baseline", "model": args.baseline}), flush=True)
        generate, _ = load_baseline(args.baseline, device)
        label = f"baseline-pal:{args.baseline}"

    t0 = time.time()
    result = run_math_pal(generate, args.n_per_task, args.max_new_tokens)
    result["elapsed_s"] = round(time.time() - t0, 1)
    summary = {"label": label, "tasks": {"math_pal": {k: v for k, v in result.items() if k != "results"}}}

    with open(os.path.join(args.output, "eval.json"), "w") as f:
        # Truncate per-example results for readability but keep summary
        json.dump({"summary": summary, "samples": result["results"][:20]}, f, indent=2)
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
