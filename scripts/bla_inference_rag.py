"""BLA inference with retrieval-augmented prompting.

Adds the RETRIEVE action to the pipeline:

    problem
      ↓
    EntropyRouter → RETRIEVE
      ↓
    TFIDFRetriever.lookup(problem, k=K)
      ↓ returns K similar (problem, python_solution, answer) demos
      ↓
    format_few_shot_prompt(query, demos)
      ↓
    EntropyRouter → SIMULATE
      ↓
    procedural_core.generate(few_shot_prompt)
      ↓
    exec_python → output
      ↓
    PALCertifier
      ↓
    CommitmentObject(claim, evidence, certified, uncertainty)

Hypothesis: a small (500M) model with in-context demos of similar
problems will match/exceed plain greedy on GSM8K because each test
problem effectively gets few-shot supervision. Typical 2-5× gain
reported in PAL papers when retrieval is added.

Usage:
    python3 scripts/bla_inference_rag.py \\
        --ckpt runs/phase6/run13_500m_15k_v6_600k/final.pt \\
        --output runs/phase7/rag_run13 --n 200 --k 3
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
    def generate(prompt: str, max_new: int = 256) -> str:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        # Truncate if too long for the model (RAG prompts can be big)
        if ids.shape[1] > cfg.max_seq_len - max_new:
            ids = ids[:, -(cfg.max_seq_len - max_new):]
        out = ids
        for _ in range(max_new):
            logits = model.forward(out)[:, -1].float()
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(tok.decode(ids[0], skip_special_tokens=True)):]

    return generate


def solve_rag(problem: str, generate, retriever, certifier, k: int = 3,
              include_python: bool = True) -> CommitmentObject:
    """RAG-augmented BLA solve: retrieve demos, then simulate."""
    problem_numbers = extract_problem_numbers(problem)

    # Step 1: RETRIEVE
    retrieve_action = RouterAction(
        type=RouterActionType.RETRIEVE,
        payload={"query": problem, "k": k},
    )
    demos = retriever.lookup(problem, k=k)

    # Step 2: SIMULATE with demos in context
    simulate_action = RouterAction(
        type=RouterActionType.SIMULATE,
        payload={"problem": problem, "demos": [d.question[:80] for d in demos]},
    )
    prompt = format_few_shot_prompt(problem, demos, include_python=include_python)
    raw = generate(prompt, max_new=256)
    code = extract_python(raw) or ""
    output = exec_python(code) if code else "ERROR: no code"
    pred = parse_number(output) if not output.startswith("ERROR") else None

    sim_result = {
        "code": code, "output": output, "pred": pred,
        "problem_numbers": problem_numbers, "raw_generation": raw,
    }

    commitment = CommitmentObject(
        claim=pred,
        evidence=[
            {"type": "python_code", "value": code},
            {"type": "execution_output", "value": output},
            {"type": "retrieved_demos",
             "value": [{"q": d.question[:80], "sim": d.score} for d in demos]},
        ],
        reasoning_trace={
            "router_actions": [retrieve_action.type.value, simulate_action.type.value],
            "n_demos": len(demos),
            "top_demo_sim": demos[0].score if demos else 0.0,
            "raw_generation": raw[:500],
        },
    )
    result = certifier.attach(commitment, sim_result)
    commitment.uncertainty = 1.0 - result.confidence
    return commitment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=3, help="Number of retrieved demos")
    p.add_argument("--no-python", action="store_true",
                   help="Hide Python from demos, show only Q + Answer")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--index-cache", type=str,
                   default="/root/bla/runs/phase7/gsm8k_train_tfidf.pkl")
    p.add_argument("--score-threshold", type=float, default=3.0)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)

    # Build or load retriever
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
    items = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        items.append(ex)

    certifier = PALCertifier(score_threshold=args.score_threshold)
    correct = 0
    cert_correct = 0
    cert_total = 0
    commitments = []
    t0 = time.time()
    for i, ex in enumerate(items):
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        commitment = solve_rag(question, generate, retriever, certifier,
                               k=args.k, include_python=not args.no_python)
        pred = commitment.claim
        ok = pred is not None and str(pred).rstrip(".") == gold.rstrip(".")
        correct += int(ok)
        if commitment.certified:
            cert_total += 1
            if ok:
                cert_correct += 1
        commitments.append({
            "idx": i, "question": question[:80], "gold": gold, "pred": pred,
            "ok": ok, "certified": commitment.certified,
            "confidence": commitment.confidence,
            "top_demo_sim": commitment.reasoning_trace.get("top_demo_sim", 0),
        })
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "done": i + 1, "total": len(items),
                              "running_acc": correct / (i + 1)}), flush=True)

    summary = {
        "n_tested": len(items),
        "k_demos": args.k,
        "accuracy": correct / max(len(items), 1),
        "cert_total": cert_total,
        "cert_correct": cert_correct,
        "cert_precision": cert_correct / max(cert_total, 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "bla_rag.json"), "w") as f:
        json.dump({"summary": summary, "commitments": commitments[:30]}, f, indent=2)
    print(json.dumps({"event": "done", **summary}, indent=2))


if __name__ == "__main__":
    main()
