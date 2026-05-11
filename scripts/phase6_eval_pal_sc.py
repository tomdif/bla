"""PAL eval with self-consistency.

For each GSM8K problem, sample N=8 Python programs at temperature 0.7,
execute each, majority-vote over the valid numeric answers. Standard
10-20pp boost on math benchmarks vs greedy decoding.
"""

from __future__ import annotations

import argparse
import collections
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

from system2_dca.procedural_core import ProceduralCore, ProceduralCoreConfig
from scripts.phase6_eval_pal import exec_python, extract_python, parse_number


def load_bla_sampling(ckpt_path: str, device: torch.device, temperature: float = 0.7,
                      top_p: float = 0.9, max_seq_len_override: int = None):
    """Load BLA checkpoint, return a generate(prompt, max_new) that uses
    temperature/top-p sampling instead of greedy."""
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
            logits = model.forward(out)[:, -1].float()
            logits = logits / max(temperature, 1e-5)
            # top-p
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            picked = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, picked)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id or out.shape[1] >= cfg.max_seq_len:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate, model


def run_math_pal_sc(generate, n: int = 200, max_new: int = 256, n_samples: int = 8) -> dict:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        items.append(ex)

    correct = 0
    code_ran_any = 0
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

        # Sample N programs, execute each, collect valid answers
        votes = collections.Counter()
        ran_any = False
        for _ in range(n_samples):
            gen = generate(prompt, max_new)
            code = extract_python(gen)
            if not code:
                continue
            output = exec_python(code)
            if output.startswith("ERROR"):
                continue
            ran_any = True
            pred = parse_number(output)
            if pred is not None:
                votes[pred] += 1

        if ran_any:
            code_ran_any += 1
        if votes:
            pred, _ = votes.most_common(1)[0]
        else:
            pred = None

        ok = (pred is not None) and (pred.rstrip(".") == gold.rstrip("."))
        correct += int(ok)
        results.append({
            "question": question[:80], "gold": gold, "pred": pred,
            "votes": dict(votes), "ok": ok,
        })

    return {
        "task": "math_pal_sc",
        "accuracy": correct / max(len(items), 1),
        "code_ran_any": code_ran_any,
        "n_samples": n_samples,
        "n_tested": len(items),
        "results": results,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-per-task", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    print(json.dumps({"event": "loading", "ckpt": args.ckpt,
                      "n_samples": args.n_samples, "temperature": args.temperature}),
          flush=True)
    generate, _ = load_bla_sampling(args.ckpt, device, args.temperature, args.top_p)

    t0 = time.time()
    result = run_math_pal_sc(generate, args.n_per_task, args.max_new_tokens, args.n_samples)
    result["elapsed_s"] = round(time.time() - t0, 1)
    summary = {"label": f"bla-pal-sc:{os.path.basename(args.ckpt)}",
               "tasks": {"math_pal_sc": {k: v for k, v in result.items() if k != "results"}}}

    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump({"summary": summary, "samples": result["results"][:30]}, f, indent=2)
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
