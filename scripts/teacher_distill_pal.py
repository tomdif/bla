"""Teacher-distill PAL solutions from Claude Haiku 4.5.

For each problem in the source pool (GSM8K-train + MATH-train + rephrasings),
ask the teacher to write a Python program that prints the answer. Verify
correctness by executing the code and comparing the printed number to gold.
Output JSONL matching the existing curriculum format:
  {"prompt": "Problem: ...\nPython:\n", "target": "<python>"}

Usage:
    python scripts/teacher_distill_pal.py \
        --source gsm8k_train \
        --start 0 --end 100 \
        --output runs/teacher_distill/v1.jsonl

Verification keeps only examples where the executed code printed a number
matching the gold answer (allowing common normalizations like int vs float,
stripping $ and ,).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import builtins, contextlib, io, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CLIENT = Anthropic()
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You write Python programs that solve math word problems. "
    "Output ONLY the Python program — no prose, no explanation, no code fences. "
    "The program must end with `print(answer)` where `answer` is the numeric result. "
    "Use only stdlib (math, fractions, itertools); no input(), no file I/O, no network. "
    "Keep the program self-contained and under 30 lines."
)


def render_problem(question: str) -> list:
    return [{"role": "user",
             "content": f"Problem: {question}\n\nWrite a Python program that prints the answer."}]


def parse_code(text: str) -> str:
    """Extract Python code from a teacher response. Handles fenced blocks
    and raw code."""
    fence = re.search(r"```(?:python)?\n?(.*?)```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def exec_python(code: str, timeout: float = 2.0) -> str:
    """Run code in a subprocess (thread-safe). Return stdout or 'ERROR: ...'."""
    try:
        r = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return f"ERROR: returncode={r.returncode}: {r.stderr.strip()[:200]}"
        return r.stdout.rstrip()
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"


def parse_number(s: str):
    m = list(re.finditer(r"-?\d+(?:\.\d+)?", s))
    if not m:
        return None
    n = m[-1].group(0)
    if "." in n:
        try:
            f = float(n)
            if f == int(f):
                return str(int(f))
            return n
        except ValueError:
            return n
    return n


def load_gsm8k_train(start: int, end: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    for i in range(start, min(end, len(ds))):
        ex = ds[i]
        gold_raw = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", gold_raw)
        yield {"question": ex["question"], "gold": gold, "idx": i, "source": "gsm8k_train"}


def load_math_train(start: int, end: int):
    """MATH dataset (Hendrycks et al). Pulls all 7 subjects, keeps only
    problems whose \\boxed{...} answer is a pure number (or fraction)."""
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra",
                "precalculus"]
    j = 0
    for sub in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", sub, split="train")
        except Exception:
            continue
        for i, ex in enumerate(ds):
            if j >= end:
                return
            m = re.search(r"\\boxed\{([^{}]+)\}", ex["solution"])
            if not m:
                continue
            gold = m.group(1).strip().strip("$").strip()
            if not re.fullmatch(r"-?\d+(?:\.\d+)?(/-?\d+)?", gold):
                continue
            if j >= start:
                yield {"question": ex["problem"], "gold": gold,
                       "idx": f"{sub}:{i}", "source": "math_train"}
            j += 1


def generate_one(problem: dict, max_retries: int = 2) -> dict | None:
    """Ask teacher, parse, verify. Return None on failure."""
    q = problem["question"]
    gold = problem["gold"]
    for attempt in range(max_retries):
        try:
            resp = CLIENT.messages.create(
                model=MODEL, max_tokens=600, temperature=0.0,
                system=SYSTEM_PROMPT, messages=render_problem(q),
            )
            text = resp.content[0].text
            code = parse_code(text)
            output = exec_python(code)
            if output.startswith("ERROR"):
                continue
            pred = parse_number(output)
            if pred is None:
                continue
            if pred.rstrip(".") != gold.rstrip("."):
                continue
            return {
                "prompt": f"Problem: {q}\nPython:\n",
                "target": code,
                "_meta": {"source": problem["source"], "idx": problem["idx"],
                          "gold": gold, "attempt": attempt + 1},
            }
        except Exception as e:
            time.sleep(1.0)
            continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["gsm8k_train", "math_train"],
                   default="gsm8k_train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.source == "gsm8k_train":
        problems = list(load_gsm8k_train(args.start, args.end))
    else:
        problems = list(load_math_train(args.start, args.end))
    print(f"loaded {len(problems)} problems from {args.source}", flush=True)

    t0 = time.time()
    kept = 0
    written = 0
    with open(args.output, "w") as out, \
         ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(generate_one, p): p for p in problems}
        for f in as_completed(futures):
            written += 1
            result = f.result()
            if result is not None:
                out.write(json.dumps(result) + "\n")
                out.flush()
                kept += 1
            if written % 25 == 0:
                rate = written / (time.time() - t0)
                eta = (len(problems) - written) / rate
                print(json.dumps({"done": written, "kept": kept,
                                  "keep_rate": round(kept / written, 3),
                                  "rate_per_s": round(rate, 2),
                                  "eta_s": round(eta, 0)}),
                      flush=True)

    print(json.dumps({"done": written, "kept": kept,
                      "keep_rate": round(kept / written, 3),
                      "elapsed_s": round(time.time() - t0, 1),
                      "output": args.output}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
