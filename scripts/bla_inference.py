"""End-to-end BLA inference pipeline.

Composes the existing brain pieces into a single inference loop:

    problem
      ↓
    procedural_core (System 2)
      ↓ generates Python (the SIMULATE action's body)
      ↓
    Python interpreter (the SIMULATE backend)
      ↓ produces execution output (the candidate claim)
      ↓
    PALCertifier (a certifier in the verification layer)
      ↓ scores the candidate and attaches a CertifierResult
      ↓
    CommitmentObject
      ↓ structured output with claim, evidence, certifier results,
      ↓ uncertainty, reasoning trace, reproducibility packet
      ↓
    return commitment

Before today's session we were calling procedural_core.forward() raw
and getting a flat answer. After today this script returns a structured
commitment whose certified property tells us if the verifier passed.

Usage:
    python3 scripts/bla_inference.py --ckpt path/to/final.pt --n 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from system2_dca.procedural_core import ProceduralCore, ProceduralCoreConfig
from verification.commitment import CommitmentObject
from verification.pal_certifier import PALCertifier
from verification.router_action import RouterAction, RouterActionType
from scripts.phase6_eval_pal import exec_python, extract_python, parse_number
from system2_dca.number_parser import extract_problem_numbers
import re


def load_procedural_core(ckpt_path: str, device: torch.device):
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
    def generate(prompt: str, max_new: int = 256) -> str:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        out = ids
        for _ in range(max_new):
            logits = model.forward(out)[:, -1].float()
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate


def simulate_action_handler(generate, problem: str,
                            problem_numbers: list[str]) -> dict:
    """Implements RouterActionType.SIMULATE for math problems: prompt the
    procedural core for Python, execute it, return a structured result."""
    prompt = (
        "Write a Python program that prints the answer to this math problem.\n"
        "End with: print(answer)\n"
        f"Problem: {problem}\nPython:\n"
    )
    raw_generation = generate(prompt, max_new=256)
    code = extract_python(raw_generation) or ""
    output = exec_python(code) if code else "ERROR: no code"
    pred = parse_number(output) if not output.startswith("ERROR") else None
    return {
        "code": code,
        "output": output,
        "pred": pred,
        "problem_numbers": problem_numbers,
        "raw_generation": raw_generation,
    }


def solve(problem: str, generate, certifier: PALCertifier) -> CommitmentObject:
    """End-to-end BLA solve: dispatch the SIMULATE action, build a
    CommitmentObject, attach the PAL certifier."""
    problem_numbers = extract_problem_numbers(problem)
    action = RouterAction(
        type=RouterActionType.SIMULATE,
        budget_flops=0,
        payload={"problem": problem},
        details={"problem_numbers": problem_numbers},
    )
    sim_result = simulate_action_handler(generate, problem, problem_numbers)

    commitment = CommitmentObject(
        claim=sim_result["pred"],
        evidence=[
            {"type": "python_code", "value": sim_result["code"]},
            {"type": "execution_output", "value": sim_result["output"]},
        ],
        reasoning_trace={
            "router_action": action.type.value,
            "raw_generation": sim_result["raw_generation"][:500],
        },
    )
    # Certify
    result = certifier.attach(commitment, sim_result)
    # Set commitment uncertainty from certifier confidence
    commitment.uncertainty = 1.0 - result.confidence
    return commitment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--score-threshold", type=float, default=3.0,
                   help="PAL certifier threshold for marking certified")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)
    print(json.dumps({"event": "loading", "ckpt": args.ckpt}), flush=True)
    generate = load_procedural_core(args.ckpt, device)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        items.append(ex)

    certifier = PALCertifier(score_threshold=args.score_threshold)
    correct = 0
    certified_and_correct = 0
    certified_total = 0
    commitments = []
    t0 = time.time()
    for ex in items:
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        commitment = solve(question, generate, certifier)
        pred = commitment.claim
        ok = pred is not None and str(pred).rstrip(".") == gold.rstrip(".")
        correct += int(ok)
        cert_passed = commitment.certified
        certified_total += int(cert_passed)
        if cert_passed and ok:
            certified_and_correct += 1
        commitments.append({
            "question": question[:80],
            "gold": gold,
            "pred": pred,
            "certified": cert_passed,
            "confidence": commitment.confidence,
            "uncertainty": commitment.uncertainty,
            "ok": ok,
        })

    summary = {
        "n_tested": len(items),
        "accuracy": correct / max(len(items), 1),
        "certified_total": certified_total,
        "certified_and_correct": certified_and_correct,
        "certifier_precision": certified_and_correct / max(certified_total, 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "bla_inference.json"), "w") as f:
        json.dump({"summary": summary, "commitments": commitments[:30]}, f, indent=2)
    print(json.dumps({"event": "done", **summary}, indent=2))


if __name__ == "__main__":
    main()
