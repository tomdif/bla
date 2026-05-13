"""BLA inference with verifier-in-loop refinement.

Instead of generating N=32 candidates blindly and voting, generate
candidates one at a time and stop as soon as one passes the verifier
threshold. If max_retries are exhausted without a passing candidate,
return the one with the highest verifier score.

Architecturally this is the BLA spec's adaptive-test-time-compute:

    while not commitment.certified and retries < budget:
        candidate = procedural_core.generate(prompt)
        certify(candidate)
        if pass: commit
        else: retry

Trade-off vs blind N-sample SC:
  + Stops early on easy problems → saves compute
  + Persists on hard ones → can use full budget when needed
  + Each retry is independent; no need to vote
  - First pass uses greedy (zero diversity); retries use sampling
  - Highly dependent on threshold calibration

Optional integration with retrieval: enable --rag to use TFIDFRetriever
demos in the prompt. This is the full assembly: RETRIEVE + iterated
SIMULATE + CERTIFY-or-RETRY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from system2_dca.procedural_core import ProceduralCore, ProceduralCoreConfig
from system2_dca.retrieval_memory import (
    TFIDFRetriever, build_gsm8k_train_index, format_few_shot_prompt,
)
from verification.commitment import CommitmentObject
from verification.pal_certifier import PALCertifier
from verification.router_action import RouterAction, RouterActionType
from scripts.phase6_eval_pal import exec_python, extract_python, parse_number
from system2_dca.number_parser import extract_problem_numbers


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
    def generate(prompt: str, max_new: int = 256,
                 temperature: float = 0.0, top_p: float = 0.9) -> str:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        if ids.shape[1] > cfg.max_seq_len - max_new:
            ids = ids[:, -(cfg.max_seq_len - max_new):]
        out = ids
        for _ in range(max_new):
            logits = model.forward(out)[:, -1].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                picked = torch.multinomial(sorted_probs, 1)
                next_id = sorted_idx.gather(-1, picked)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        consumed = tok.decode(ids[0], skip_special_tokens=True)
        return full[len(consumed):]

    return generate


def _attempt(prompt: str, generate, certifier: PALCertifier,
             problem_numbers: list[str], temperature: float):
    """Run one generation attempt, return (commitment-fragment, score)."""
    raw = generate(prompt, max_new=256, temperature=temperature)
    code = extract_python(raw) or ""
    output = exec_python(code) if code else "ERROR: no code"
    pred = parse_number(output) if not output.startswith("ERROR") else None
    candidate = {"code": code, "output": output, "pred": pred,
                 "problem_numbers": problem_numbers}
    # Score with certifier without attaching yet
    result = certifier.check(candidate)
    return {
        "code": code, "output": output, "pred": pred,
        "raw_generation": raw, "cert_passed": result.passed,
        "cert_confidence": result.confidence, "cert_details": result.details,
    }, result


def solve_loop(problem: str, generate, certifier: PALCertifier,
               retriever: TFIDFRetriever | None,
               max_retries: int = 8, k_demos: int = 3,
               retry_temperature: float = 0.7) -> CommitmentObject:
    """Iterated SIMULATE-and-CERTIFY. First pass greedy; retries sampled."""
    problem_numbers = extract_problem_numbers(problem)

    # Build prompt (optionally with retrieval)
    actions = []
    if retriever is not None:
        actions.append(RouterAction(type=RouterActionType.RETRIEVE,
                                    payload={"query": problem, "k": k_demos}))
        demos = retriever.lookup(problem, k=k_demos)
        prompt = format_few_shot_prompt(problem, demos, include_python=True)
    else:
        demos = []
        prompt = (
            "Write a Python program that prints the answer to this math problem.\n"
            "End with: print(answer)\n"
            f"Problem: {problem}\nPython:\n"
        )

    attempts = []
    best = None
    best_result = None
    # First pass: greedy (temperature 0)
    # Subsequent: sampled (temperature retry_temperature)
    for attempt_idx in range(max_retries + 1):
        temp = 0.0 if attempt_idx == 0 else retry_temperature
        actions.append(RouterAction(
            type=RouterActionType.SIMULATE,
            payload={"problem": problem, "attempt": attempt_idx, "temperature": temp},
        ))
        info, result = _attempt(prompt, generate, certifier,
                                problem_numbers, temp)
        attempts.append(info)
        if best is None or result.confidence > best_result.confidence:
            best = info
            best_result = result
        if result.passed:
            break

    # Build commitment around the BEST attempt
    commitment = CommitmentObject(
        claim=best["pred"],
        evidence=[
            {"type": "python_code", "value": best["code"]},
            {"type": "execution_output", "value": best["output"]},
            {"type": "retrieved_demos",
             "value": [{"q": d.question[:80], "sim": d.score} for d in demos]},
            {"type": "n_attempts", "value": len(attempts)},
            {"type": "all_attempts_preds",
             "value": [a["pred"] for a in attempts]},
        ],
        reasoning_trace={
            "router_actions": [a.type.value for a in actions],
            "n_attempts": len(attempts),
            "stopped_early": best_result.passed,
            "best_attempt_idx": attempts.index(best),
        },
    )
    # Attach the certifier with the best candidate so the test result lands
    certifier.attach(commitment, {"code": best["code"], "output": best["output"],
                                  "pred": best["pred"],
                                  "problem_numbers": problem_numbers})
    commitment.uncertainty = 1.0 - best_result.confidence
    return commitment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--retry-temp", type=float, default=0.7)
    p.add_argument("--score-threshold", type=float, default=6.0,
                   help="PAL certifier 'passed' threshold")
    p.add_argument("--rag", action="store_true",
                   help="Use TF-IDF retrieval to add demos to the prompt")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--index-cache", type=str,
                   default="/root/bla/runs/phase7/gsm8k_train_tfidf.pkl")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)

    retriever = None
    if args.rag:
        if os.path.exists(args.index_cache):
            print(json.dumps({"event": "loading_index", "path": args.index_cache}), flush=True)
            retriever = TFIDFRetriever.load(args.index_cache)
        else:
            print(json.dumps({"event": "building_index"}), flush=True)
            retriever = build_gsm8k_train_index()
            os.makedirs(os.path.dirname(args.index_cache), exist_ok=True)
            retriever.save(args.index_cache)

    print(json.dumps({"event": "loading_core", "ckpt": args.ckpt}), flush=True)
    generate = load_procedural_core(args.ckpt, device)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = [ex for i, ex in enumerate(ds) if i < args.n]

    certifier = PALCertifier(score_threshold=args.score_threshold)
    correct = 0
    cert_correct = 0
    cert_total = 0
    total_attempts = 0
    commitments = []
    t0 = time.time()
    for i, ex in enumerate(items):
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        commitment = solve_loop(
            question, generate, certifier, retriever,
            max_retries=args.max_retries, k_demos=args.k,
            retry_temperature=args.retry_temp,
        )
        pred = commitment.claim
        ok = pred is not None and str(pred).rstrip(".") == gold.rstrip(".")
        correct += int(ok)
        n_att = commitment.reasoning_trace.get("n_attempts", 1)
        total_attempts += n_att
        if commitment.certified:
            cert_total += 1
            if ok:
                cert_correct += 1
        commitments.append({
            "idx": i, "question": question[:80], "gold": gold, "pred": pred,
            "ok": ok, "certified": commitment.certified,
            "confidence": commitment.confidence,
            "n_attempts": n_att,
            "stopped_early": commitment.reasoning_trace.get("stopped_early", False),
        })
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "done": i + 1,
                              "running_acc": correct / (i + 1),
                              "avg_attempts": total_attempts / (i + 1)}),
                  flush=True)

    summary = {
        "n_tested": len(items),
        "max_retries": args.max_retries,
        "rag": bool(retriever is not None),
        "accuracy": correct / max(len(items), 1),
        "avg_attempts": total_attempts / max(len(items), 1),
        "cert_total": cert_total,
        "cert_correct": cert_correct,
        "cert_precision": cert_correct / max(cert_total, 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "bla_loop.json"), "w") as f:
        json.dump({"summary": summary, "commitments": commitments[:30]}, f, indent=2)
    print(json.dumps({"event": "done", **summary}, indent=2))


if __name__ == "__main__":
    main()
