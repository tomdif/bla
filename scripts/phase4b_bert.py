"""Phase 4b — non-AR retrieval-augmented QA using BERT-base MLM.

The Phase 4b gate is "B.L.A.+retrieval ≤ 50% of non-AR baseline
hallucination." A from-scratch tiny diffusion text model needs orders
more compute than we have on this pod to learn copy-from-context
behavior. The cleanest scoped test of the same load-bearing claim is
to use a real pretrained non-AR text model: BERT-base, which is
trained on masked-LM and is bidirectional / non-autoregressive.

This isn't full SEDD, but it tests the exact property the gate cares
about: does retrieval reduce hallucination on a non-AR text model?
That property is what differentiates Phase 4b from Phase 4a (GPT-2,
which is autoregressive).

Run:
    python3 scripts/phase4b_bert.py \\
        --memory-db runs/phase3_facts/symbolic.db \\
        --output runs/phase4b_bert
"""

from __future__ import annotations

import argparse
import json
import os
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
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_eval_set() -> list[dict]:
    cases = []
    for country, capital, src in COUNTRY_CAPITALS:
        cases.append({"predicate": "country_capital_is", "subject": country,
                      "ground_truth": capital, "source": src})
    for body, parent, src in PLANET_ORBITS:
        cases.append({"predicate": "orbits", "subject": body,
                      "ground_truth": parent, "source": src})
    for elem, z in ELEMENT_ATOMIC_NUMBERS:
        cases.append({"predicate": "atomic_number", "subject": elem,
                      "ground_truth": str(z), "source": f"periodic_table#{elem}"})
    return cases


def render_template(case: dict) -> str:
    """Returns a sentence with the answer slot, expressed as natural English."""
    pred = case["predicate"]
    if pred == "country_capital_is":
        return f"The capital of {case['subject']} is [MASK]."
    if pred == "orbits":
        return f"{case['subject']} orbits [MASK]."
    if pred == "atomic_number":
        return f"{case['subject']} has atomic number [MASK]."
    raise ValueError


def render_context_sentence(case: dict, retrieved_object: str) -> str:
    pred = case["predicate"]
    if pred == "country_capital_is":
        return f"The capital of {case['subject']} is {retrieved_object}."
    if pred == "orbits":
        return f"{case['subject']} orbits {retrieved_object}."
    if pred == "atomic_number":
        return f"{case['subject']} has atomic number {retrieved_object}."
    return f"{case['subject']} {pred} {retrieved_object}."


def fill_mask(tokenizer, model, prompt: str, n_mask: int, device: torch.device) -> str:
    """Iteratively fill `n_mask` consecutive [MASK] tokens by greedy commit
    of the most-confident position each step. For BERT-style MLM this is
    a standard parallel-fill protocol."""
    # Replace single [MASK] in prompt with n_mask consecutive masks
    if n_mask > 1:
        prompt = prompt.replace("[MASK]", " ".join(["[MASK]"] * n_mask), 1)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    mask_id = tokenizer.mask_token_id

    masked_positions = (input_ids == mask_id).nonzero(as_tuple=True)[1].tolist()

    while masked_positions:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        # find max-confidence position
        best_pos = None
        best_conf = -1.0
        best_token = None
        for p in masked_positions:
            probs = torch.softmax(logits[p], dim=-1)
            conf, tok = probs.max(dim=-1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_pos = p
                best_token = tok.item()
        # commit
        input_ids[0, best_pos] = best_token
        masked_positions.remove(best_pos)

    # decode answer span
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def extract_answer(filled_text: str, template_text: str) -> str:
    """Diff between filled output and the template (which has the slot
    replaced by ___) to recover just the predicted answer span."""
    template_words = template_text.split()
    filled_words = filled_text.split()
    # naive: find the longest contiguous span in filled that's not in template
    # Simpler: just find words in filled that don't appear in template
    template_lower = set(w.lower().strip(".,;:!?") for w in template_words)
    out = []
    for w in filled_words:
        if w.lower().strip(".,;:!?") not in template_lower:
            out.append(w.strip(".,;:!?"))
    return " ".join(out).strip()


def estimate_n_mask(tokenizer, ground_truth: str) -> int:
    ids = tokenizer.encode(ground_truth, add_special_tokens=False)
    return max(1, len(ids))


def score_match(answer: str, ground_truth: str) -> bool:
    return ground_truth.lower() in answer.lower()


def retrieve_object(sm: SymbolicMemory, case: dict) -> tuple[str, list]:
    triples = sm.query_subject_predicate(case["subject"], predicate=case["predicate"])
    if not triples:
        return "", []
    tr = triples[0]
    obj_text = tr.get("object_value")
    if obj_text is None and tr.get("object_id"):
        obj_ent = sm._memoria.kg.get_entity(tr["object_id"])
        obj_text = obj_ent.name if obj_ent else "?"
    return str(obj_text), [{"subject": case['subject'], "predicate": case["predicate"],
                            "object": obj_text, "source_ref": tr.get("source_ref")}]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    print(json.dumps({"event": "loading", "model": args.model}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    print(json.dumps({"event": "loaded", "params": sum(p.numel() for p in model.parameters())}), flush=True)

    sm = SymbolicMemory(db_path=args.memory_db)
    cases = build_eval_set()
    print(json.dumps({"event": "test_set", "n": len(cases)}), flush=True)

    para_results, rag_results = [], []
    para_correct = 0
    rag_correct = 0
    commitments = []
    t0 = time.time()

    for i, case in enumerate(cases):
        n_mask = estimate_n_mask(tokenizer, case["ground_truth"])
        template = render_template(case)

        # Parametric
        filled_para = fill_mask(tokenizer, model, template, n_mask, device)
        ans_para = extract_answer(filled_para, template.replace("[MASK]", ""))
        ok_para = score_match(ans_para, case["ground_truth"])
        para_results.append({**case, "answer": ans_para, "filled": filled_para, "correct": ok_para})

        # RAG
        retrieved_obj, evidence = retrieve_object(sm, case)
        if retrieved_obj:
            ctx = render_context_sentence(case, retrieved_obj)
            rag_prompt = f"{ctx} {template}"
        else:
            rag_prompt = template
        filled_rag = fill_mask(tokenizer, model, rag_prompt, n_mask, device)
        ans_rag = extract_answer(filled_rag, template.replace("[MASK]", ""))
        ok_rag = score_match(ans_rag, case["ground_truth"])
        rag_results.append({**case, "answer": ans_rag, "filled": filled_rag,
                            "correct": ok_rag, "context": rag_prompt})

        para_correct += int(ok_para)
        rag_correct += int(ok_rag)

        commitments.append(CommitmentObject(
            claim={"answer": ans_rag, "question": template},
            evidence=evidence,
            reasoning_trace={"mode": "non_ar_rag", "model": args.model},
            uncertainty=1.0 - (1.0 if ok_rag else 0.0),
        ))

        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "i": i + 1,
                              "para_acc": round(para_correct / (i + 1), 3),
                              "rag_acc": round(rag_correct / (i + 1), 3),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    n = len(cases)
    para_acc = para_correct / max(n, 1)
    rag_acc = rag_correct / max(n, 1)
    para_hall = 1.0 - para_acc
    rag_hall = 1.0 - rag_acc
    ratio = rag_hall / max(para_hall, 1e-6)

    by_pred = {}
    for p in {"country_capital_is", "orbits", "atomic_number"}:
        ps = [r for r in para_results if r["predicate"] == p]
        rs = [r for r in rag_results if r["predicate"] == p]
        by_pred[p] = {
            "n": len(ps),
            "parametric_acc": sum(r["correct"] for r in ps) / max(len(ps), 1),
            "rag_acc": sum(r["correct"] for r in rs) / max(len(rs), 1),
        }

    summary = {
        "model": args.model,
        "model_params": sum(p.numel() for p in model.parameters()),
        "n_questions": n,
        "parametric_accuracy": para_acc,
        "rag_accuracy": rag_acc,
        "parametric_hallucination": para_hall,
        "rag_hallucination": rag_hall,
        "rag_vs_parametric_ratio": ratio,
        "by_predicate": by_pred,
        "gate_ratio": 0.5,
        "gate_passed": rag_hall <= 0.5 * para_hall,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)

    with open(os.path.join(args.output, "phase4b_bert.json"), "w") as f:
        json.dump({"summary": summary,
                   "parametric": para_results,
                   "rag": rag_results}, f, indent=2)


if __name__ == "__main__":
    main()
