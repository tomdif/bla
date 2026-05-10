"""Phase 6 eval harness — measure a checkpoint against the four task
categories the gate requires:

  Math   — GSM8K (subset of test split)
  Code   — HumanEval (full)
  Proof  — MiniF2F-test (subset)
  Plan   — synthetic Sokoban-style task

Each runner returns (accuracy, n_tested) plus per-example records.
The runner is generation-agnostic: pass a callable
`generate(prompt: str, max_new_tokens: int) -> str` and it scores.

Run on a checkpoint:
  python3 scripts/phase6_eval.py --ckpt runs/phase6/run1/final.pt --suite math code

Run on GPT-2 1.5B baseline (download via HF):
  python3 scripts/phase6_eval.py --baseline gpt2-xl --suite math code
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system2_dca.procedural_core import ProceduralCore, ProceduralCoreConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="Phase 6 checkpoint (final.pt) to evaluate.")
    p.add_argument("--baseline", type=str, default=None,
                   help="HuggingFace model name to evaluate as baseline.")
    p.add_argument("--suite", nargs="+", default=["math", "code", "proof", "plan"],
                   choices=["math", "code", "proof", "plan"])
    p.add_argument("--n-per-task", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------- model loaders --------------------

def load_bla_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = state["config"]
    cfg = ProceduralCoreConfig(
        vocab_size=cfg_d["vocab"], d_model=cfg_d["d"],
        n_layers=cfg_d["n_layers"], n_heads=cfg_d["n_heads"],
        max_seq_len=cfg_d["seq_len"],
    )
    model = ProceduralCore(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(state["state_dict"])
    model.eval()

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int) -> str:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        if ids.shape[1] >= cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len + max_new_tokens:]
        out = ids
        for _ in range(max_new_tokens):
            logits = model.forward(out)[:, -1]
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id or out.shape[1] >= cfg.max_seq_len:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate, model


def load_baseline(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int) -> str:
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id,
        )
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate, model


# -------------------- task runners --------------------

def _final_number(text: str) -> Optional[str]:
    """Pull out the last numeric answer from a generation."""
    m = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    return m[-1].group(0) if m else None


def run_math(generate, n: int, max_new: int) -> dict:
    """GSM8K test split — small subset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        items = []
        for i, ex in enumerate(ds):
            if i >= n:
                break
            items.append(ex)
    except Exception as exc:
        return {"task": "math", "error": f"dataset load failed: {exc}", "accuracy": 0.0,
                "n_tested": 0, "results": []}

    correct = 0
    results = []
    for ex in items:
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        prompt = (f"Solve this math problem step by step. End with: 'Answer: <number>'.\n"
                  f"Problem: {question}\nSolution:")
        out = generate(prompt, max_new)
        pred = _final_number(out)
        ok = (pred is not None) and (pred.strip().rstrip(".") == gold.strip().rstrip("."))
        correct += int(ok)
        results.append({"question": question, "gold": gold, "pred": pred, "ok": ok})
    return {"task": "math", "accuracy": correct / max(len(items), 1),
            "n_tested": len(items), "results": results}


def run_code(generate, n: int, max_new: int) -> dict:
    """HumanEval — pass@1 with greedy decoding."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test", streaming=True)
        items = list(ds)
        if n > 0:
            items = items[:n]
    except Exception as exc:
        return {"task": "code", "error": f"dataset load failed: {exc}", "accuracy": 0.0,
                "n_tested": 0, "results": []}

    correct = 0
    results = []
    for ex in items:
        prompt = ex["prompt"]
        out = generate(prompt, max_new)
        # naive: take everything up to first 'def' or 'class' or empty line
        completion = out.split("\nclass ")[0].split("\ndef ")[0]
        full_solution = prompt + completion
        ok = _check_humaneval(ex, full_solution)
        correct += int(ok)
        results.append({"task_id": ex["task_id"], "ok": ok})
    return {"task": "code", "accuracy": correct / max(len(items), 1),
            "n_tested": len(items), "results": results}


def _check_humaneval(ex: dict, solution: str, timeout_s: float = 5.0) -> bool:
    import subprocess, sys, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(solution + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n")
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout_s,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_proof(generate, n: int, max_new: int) -> dict:
    """MiniF2F-test (Lean) — surface-form match because we don't run a
    proof checker yet. This is a proxy; Phase 8 wires the real Lean
    checker via the verification layer."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hoskinson-center/minif2f-lean4", split="test", streaming=True)
        items = []
        for i, ex in enumerate(ds):
            if i >= n:
                break
            items.append(ex)
    except Exception as exc:
        # Fall back to a small synthetic propositional set
        return _proof_synthetic(generate, n, max_new)

    correct = 0
    results = []
    for ex in items:
        statement = ex.get("formal_statement") or ex.get("informal_statement") or ""
        prompt = f"Provide a Lean 4 proof for the following statement:\n{statement}\n-- Proof:\n"
        out = generate(prompt, max_new)
        # Surrogate scoring: did the generation include 'sorry' or did it propose tactics?
        ok = ("sorry" not in out.lower()) and ("by" in out.lower() or "exact" in out.lower())
        correct += int(ok)
        results.append({"name": ex.get("name"), "produced_tactics": ok})
    return {"task": "proof", "accuracy": correct / max(len(items), 1),
            "n_tested": len(items), "results": results,
            "note": "surface-form score (no Lean checker yet); Phase 8 wires verification"}


def _proof_synthetic(generate, n: int, max_new: int) -> dict:
    cases = [
        ("If P implies Q, and P is true, prove Q.", "modus ponens"),
        ("If A is a subset of B, and B is a subset of C, prove A is a subset of C.", "transitivity"),
        ("If n is even, prove n^2 is even.", "even square"),
        ("Prove that the sum of two odd numbers is even.", "sum odds"),
    ]
    correct = 0
    results = []
    for i in range(min(n, len(cases))):
        statement, expected_keyword = cases[i]
        prompt = f"Prove the following statement step by step:\n{statement}\nProof:"
        out = generate(prompt, max_new)
        ok = expected_keyword in out.lower() or "therefore" in out.lower()
        correct += int(ok)
        results.append({"statement": statement, "ok": ok})
    return {"task": "proof_synthetic", "accuracy": correct / max(len(results), 1),
            "n_tested": len(results), "results": results}


def run_plan(generate, n: int, max_new: int) -> dict:
    """Synthetic planning: small grid, agent + goal, ask for action sequence."""
    import random
    rng = random.Random(0)
    correct = 0
    results = []
    for _ in range(n):
        ax, ay = rng.randint(0, 4), rng.randint(0, 4)
        gx, gy = rng.randint(0, 4), rng.randint(0, 4)
        if (ax, ay) == (gx, gy):
            gx = (gx + 1) % 5
        prompt = (f"You are an agent on a 5x5 grid. You are at ({ax},{ay}). The goal is at ({gx},{gy}).\n"
                  f"Output a sequence of moves (UP, DOWN, LEFT, RIGHT) one per line, then 'DONE'.\n"
                  f"Moves:\n")
        out = generate(prompt, max_new)
        ok = _validate_plan(out, (ax, ay), (gx, gy))
        correct += int(ok)
        results.append({"start": (ax, ay), "goal": (gx, gy), "ok": ok})
    return {"task": "plan", "accuracy": correct / max(len(results), 1),
            "n_tested": len(results), "results": results}


def _validate_plan(text: str, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    moves = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    x, y = start
    for line in text.split("\n"):
        line = line.strip().upper()
        if line == "DONE":
            return (x, y) == goal
        if line in moves:
            dx, dy = moves[line]
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                x, y = nx, ny
            else:
                return False
        else:
            continue
    return (x, y) == goal


# -------------------- main --------------------

RUNNERS: dict[str, Callable] = {
    "math": run_math, "code": run_code, "proof": run_proof, "plan": run_plan,
}


def main() -> None:
    args = parse_args()
    if not args.ckpt and not args.baseline:
        raise SystemExit("Pass --ckpt or --baseline.")
    if args.ckpt and args.baseline:
        raise SystemExit("Pass only one of --ckpt or --baseline.")

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    if args.ckpt:
        print(json.dumps({"event": "loading_bla", "ckpt": args.ckpt}), flush=True)
        generate, _ = load_bla_checkpoint(args.ckpt, device)
        label = f"bla:{os.path.basename(args.ckpt)}"
    else:
        print(json.dumps({"event": "loading_baseline", "model": args.baseline}), flush=True)
        generate, _ = load_baseline(args.baseline, device)
        label = f"baseline:{args.baseline}"

    summary = {"label": label, "tasks": {}}
    for task in args.suite:
        runner = RUNNERS[task]
        t0 = time.time()
        result = runner(generate, args.n_per_task, args.max_new_tokens)
        result["elapsed_s"] = round(time.time() - t0, 1)
        summary["tasks"][task] = {k: v for k, v in result.items() if k != "results"}
        summary["tasks"][task]["n_tested"] = result.get("n_tested", 0)
        print(json.dumps({"event": "task", "task": task, **summary["tasks"][task]}), flush=True)

    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
