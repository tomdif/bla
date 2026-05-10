"""Phase 4a — retrieval-augmented vs parametric QA.

GPU-free fragment of Phase 4: tests whether external symbolic memory
reduces hallucination on factual questions, using GPT-2 small as both
the parametric baseline and the retrieval-augmented answerer.

The full Phase 4 (SEDD-style latent diffusion + B.L.A. diffusion +
retrieval) requires GPU and is deferred until pod is back up. Phase 4a
isolates the load-bearing question — *does external memory reduce
hallucination* — and tests it locally.

Each answer is wrapped in a CommitmentObject. Retrieval-augmented
answers carry the retrieved triples as `evidence`. Parametric answers
have empty evidence.

Gate (re-scoped from "≤ 50% of SEDD baseline"):
  RAG hallucination rate ≤ 50% of parametric hallucination rate.

Run:
  python3 scripts/phase4_qa.py \\
      --memory-db runs/phase3_facts/symbolic.db \\
      --output runs/phase4_qa
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa.factual_corpus import (
    COUNTRY_CAPITALS,
    PLANET_ORBITS,
    ELEMENT_ATOMIC_NUMBERS,
)
from system2_dca.symbolic_memory import SymbolicMemory
from verification import CommitmentObject


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-db", required=True)
    p.add_argument("--model", default="gpt2", help="HuggingFace model name (gpt2, gpt2-medium, distilgpt2)")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--top-k-retrieval", type=int, default=3)
    p.add_argument("--limit-per-relation", type=int, default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def build_test_set(limit: int | None = None) -> list[dict]:
    cases: list[dict] = []
    for country, capital, _ in COUNTRY_CAPITALS[: (limit if limit else None)]:
        cases.append({
            "predicate": "country_capital_is",
            "subject": country,
            "ground_truth": capital,
            "question": f"Q: What is the capital of {country}?\nA:",
        })
    for body, parent, _ in PLANET_ORBITS[: (limit if limit else None)]:
        cases.append({
            "predicate": "orbits",
            "subject": body,
            "ground_truth": parent,
            "question": f"Q: What does {body} orbit?\nA:",
        })
    for elem, z in ELEMENT_ATOMIC_NUMBERS[: (limit if limit else None)]:
        cases.append({
            "predicate": "atomic_number",
            "subject": elem,
            "ground_truth": str(z),
            "question": f"Q: What is the atomic number of {elem}?\nA:",
        })
    return cases


def load_llm(model_name: str):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tok, model


def generate(tok, model, prompt: str, max_new_tokens: int = 12) -> str:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    answer = full[len(prompt):].strip()
    # cut at newline / next Q
    for sep in ("\n", "Q:", "Question:"):
        idx = answer.find(sep)
        if idx >= 0:
            answer = answer[:idx]
    return answer.strip()


_NATURAL_LANGUAGE_TEMPLATE = {
    "country_capital_is": "The capital of {subject} is {obj}.",
    "orbits": "{subject} orbits {obj}.",
    "atomic_number": "{subject} has atomic number {obj}.",
}


def retrieve_context(sm: SymbolicMemory, case: dict, top_k: int) -> tuple[str, list]:
    """Use the symbolic memory to retrieve facts relevant to the question.
    Returns (context_string, evidence_triples).

    Format the retrieved triple as a declarative natural-language sentence
    so a small LLM can parse it without learning a custom DSL."""
    triples = sm.query_subject_predicate(case["subject"], predicate=case["predicate"])
    if not triples:
        resp = sm.query(case["question"], top_k=top_k)
        triples = resp["triples"]

    if not triples:
        return "", []

    selected = triples[: top_k]
    lines = []
    evidence = []
    template = _NATURAL_LANGUAGE_TEMPLATE.get(case["predicate"], "{subject} {predicate} {obj}.")
    for tr in selected:
        subj_name = case["subject"]
        obj_text = tr.get("object_value")
        if obj_text is None and tr.get("object_id"):
            obj_ent = sm._memoria.kg.get_entity(tr["object_id"])
            obj_text = obj_ent.name if obj_ent else "?"
        sentence = template.format(subject=subj_name, predicate=tr.get("predicate"), obj=obj_text)
        lines.append(sentence)
        evidence.append({
            "subject": subj_name,
            "predicate": tr.get("predicate"),
            "object": obj_text,
            "source_ref": tr.get("source_ref"),
        })
    return "\n".join(lines), evidence


def score(answer: str, ground_truth: str) -> bool:
    """Substring match, case-insensitive, with mild normalization."""
    a = answer.lower()
    g = ground_truth.lower()
    return g in a


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    sm = SymbolicMemory(db_path=args.memory_db)

    tok, model = load_llm(args.model)
    print(json.dumps({"event": "loaded", "model": args.model,
                      "n_params": sum(p.numel() for p in model.parameters())}), flush=True)

    cases = build_test_set(limit=args.limit_per_relation)
    print(json.dumps({"event": "test_set", "n": len(cases)}), flush=True)

    parametric_results = []
    rag_results = []
    commitments_para: list[CommitmentObject] = []
    commitments_rag: list[CommitmentObject] = []
    t0 = time.time()

    for i, case in enumerate(cases):
        # --- parametric ---
        prompt_para = case["question"]
        ans_para = generate(tok, model, prompt_para, args.max_new_tokens)
        ok_para = score(ans_para, case["ground_truth"])
        parametric_results.append({
            **case, "answer": ans_para, "correct": ok_para,
        })
        commitments_para.append(CommitmentObject(
            claim={"answer": ans_para, "question": case["question"]},
            evidence=[],
            reasoning_trace={"mode": "parametric", "model": args.model},
            uncertainty=1.0 - (1.0 if ok_para else 0.0),
        ))

        # --- RAG ---
        context, evidence = retrieve_context(sm, case, args.top_k_retrieval)
        prompt_rag = f"{context}\n{case['question']}" if context else case["question"]
        ans_rag = generate(tok, model, prompt_rag, args.max_new_tokens)
        ok_rag = score(ans_rag, case["ground_truth"])
        rag_results.append({
            **case, "answer": ans_rag, "correct": ok_rag,
            "n_retrieved": len(evidence),
        })
        commitments_rag.append(CommitmentObject(
            claim={"answer": ans_rag, "question": case["question"]},
            evidence=evidence,
            reasoning_trace={"mode": "retrieval_augmented", "model": args.model,
                             "retrieved_triples": len(evidence)},
            uncertainty=1.0 - (1.0 if ok_rag else 0.0),
        ))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            running_para = sum(1 for r in parametric_results if r["correct"]) / len(parametric_results)
            running_rag = sum(1 for r in rag_results if r["correct"]) / len(rag_results)
            print(json.dumps({
                "progress": i + 1,
                "parametric_acc": round(running_para, 3),
                "rag_acc": round(running_rag, 3),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)

    n = len(cases)
    para_acc = sum(1 for r in parametric_results if r["correct"]) / n
    rag_acc = sum(1 for r in rag_results if r["correct"]) / n
    para_hallucination = 1.0 - para_acc
    rag_hallucination = 1.0 - rag_acc

    # by predicate
    by_pred = {}
    for pred in {c["predicate"] for c in cases}:
        ps = [r for r in parametric_results if r["predicate"] == pred]
        rs = [r for r in rag_results if r["predicate"] == pred]
        by_pred[pred] = {
            "n": len(ps),
            "parametric_acc": sum(1 for r in ps if r["correct"]) / max(len(ps), 1),
            "rag_acc": sum(1 for r in rs if r["correct"]) / max(len(rs), 1),
        }

    summary = {
        "model": args.model,
        "n_questions": n,
        "parametric_accuracy": para_acc,
        "rag_accuracy": rag_acc,
        "parametric_hallucination": para_hallucination,
        "rag_hallucination": rag_hallucination,
        "rag_vs_parametric_hallucination_ratio": (rag_hallucination / max(para_hallucination, 1e-6)),
        "by_predicate": by_pred,
        "gate_ratio": 0.5,
        "gate_passed": rag_hallucination <= 0.5 * para_hallucination,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({"event": "summary", **summary}, indent=2))

    with open(os.path.join(args.output, "qa_audit.json"), "w") as f:
        json.dump({
            "summary": summary,
            "parametric": parametric_results,
            "rag": rag_results,
        }, f, indent=2)

    # save sample commitment objects
    with open(os.path.join(args.output, "sample_commitment_parametric.json"), "w") as f:
        f.write(commitments_para[0].to_json())
    with open(os.path.join(args.output, "sample_commitment_rag.json"), "w") as f:
        f.write(commitments_rag[0].to_json())


if __name__ == "__main__":
    main()
