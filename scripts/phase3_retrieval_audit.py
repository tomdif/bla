"""Phase 3.4 — retrieval audit on a held-out factual question set.

Queries the SymbolicMemory built by `scripts/ingest_facts.py` with
paraphrased natural-language questions, then measures:

  * Retrieval precision @ k:
      ground-truth (subject, predicate, object) appears in top-k retrieved
      triples. Default k=5.
  * Provenance correctness:
      top-ranked retrieved triple's source_ref points to the same source
      we ingested for the ground-truth answer.
  * Mean reciprocal rank (MRR) of the correct triple.

Phase 3 gate (re-scoped from the roadmap, since end-to-end QA against
an LLM answerer is Phase 4):
  retrieval precision @ 5 ≥ 0.80
  provenance correctness ≥ 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from system1_jepa.factual_corpus import (
    COUNTRY_CAPITALS,
    PLANET_ORBITS,
    ELEMENT_ATOMIC_NUMBERS,
)
from system2_dca.symbolic_memory import SymbolicMemory


_QUERY_TEMPLATES = {
    "country_capital_is": [
        "What is the capital of {subject}?",
        "Capital of {subject}",
        "Which city is the capital of {subject}?",
    ],
    "orbits": [
        "What does {subject} orbit?",
        "What body does {subject} orbit?",
        "{subject} is in orbit around what?",
    ],
    "atomic_number": [
        "What is the atomic number of {subject}?",
        "Atomic number of {subject}",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="runs/phase3_facts/symbolic.db")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", default="balanced", choices=["speed", "balanced", "quality"])
    p.add_argument("--limit-per-relation", type=int, default=None)
    p.add_argument("--output", default="runs/phase3_facts/audit.json")
    return p.parse_args()


def build_test_queries(limit: int | None = None) -> list[dict]:
    """For each fact in the corpus, generate one paraphrased query +
    ground-truth (subject, predicate, object_or_value, source)."""
    cases: list[dict] = []
    for country, capital, src in COUNTRY_CAPITALS[: (limit if limit else None)]:
        cases.append({
            "predicate": "country_capital_is",
            "subject": country,
            "expected_object": capital,
            "expected_source": src,
            "query": f"What is the capital of {country}?",
        })
    for moon_or_planet, parent, src in PLANET_ORBITS[: (limit if limit else None)]:
        cases.append({
            "predicate": "orbits",
            "subject": moon_or_planet,
            "expected_object": parent,
            "expected_source": src,
            "query": f"What does {moon_or_planet} orbit?",
        })
    for elem, z in ELEMENT_ATOMIC_NUMBERS[: (limit if limit else None)]:
        cases.append({
            "predicate": "atomic_number",
            "subject": elem,
            "expected_value": str(z),
            "expected_source": f"periodic_table#{elem}",
            "query": f"What is the atomic number of {elem}?",
        })
    return cases


def evaluate_case(sm: SymbolicMemory, case: dict, kg, top_k: int, mode: str) -> dict:
    response = sm.query(case["query"], top_k=top_k, mode=mode)
    triples = response["triples"]

    # ground-truth match: same predicate AND
    # (object_id resolves to expected_object name) OR (object_value == expected_value)
    expected_obj = case.get("expected_object")
    expected_val = case.get("expected_value")
    expected_src = case.get("expected_source")

    correct_idx = None
    for idx, tr in enumerate(triples):
        if tr.get("predicate") != case["predicate"]:
            continue
        if expected_obj:
            obj_id = tr.get("object_id")
            if obj_id is None:
                continue
            obj_ent = kg.get_entity(obj_id)
            if obj_ent and obj_ent.name == expected_obj:
                correct_idx = idx
                break
        elif expected_val:
            if tr.get("object_value") == expected_val:
                correct_idx = idx
                break

    in_top_k = correct_idx is not None
    rank = correct_idx + 1 if correct_idx is not None else None
    reciprocal_rank = (1.0 / rank) if rank else 0.0

    provenance_match = False
    if in_top_k:
        provenance_match = triples[correct_idx].get("source_ref") == expected_src

    return {
        "query": case["query"],
        "predicate": case["predicate"],
        "subject": case["subject"],
        "in_top_k": in_top_k,
        "rank": rank,
        "reciprocal_rank": reciprocal_rank,
        "provenance_match": provenance_match,
        "n_retrieved": len(triples),
    }


def main() -> None:
    args = parse_args()
    sm = SymbolicMemory(db_path=args.db)
    kg = sm._memoria.kg

    cases = build_test_queries(limit=args.limit_per_relation)
    t0 = time.time()
    results = []
    for i, c in enumerate(cases):
        r = evaluate_case(sm, c, kg, top_k=args.top_k, mode=args.mode)
        results.append(r)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            running_p = sum(1 for x in results if x["in_top_k"]) / len(results)
            print(json.dumps({"progress": i + 1, "running_precision": round(running_p, 3),
                              "elapsed_s": round(elapsed, 1)}), flush=True)

    n = len(results)
    n_in_topk = sum(1 for r in results if r["in_top_k"])
    n_prov = sum(1 for r in results if r["provenance_match"])
    mrr = sum(r["reciprocal_rank"] for r in results) / max(n, 1)
    precision_at_k = n_in_topk / max(n, 1)
    provenance_correctness = n_prov / max(n_in_topk, 1) if n_in_topk else 0.0

    # per-predicate breakdown
    by_predicate: dict[str, list] = {}
    for r in results:
        by_predicate.setdefault(r["predicate"], []).append(r)
    breakdown = {}
    for pred, rs in by_predicate.items():
        ntp = sum(1 for x in rs if x["in_top_k"])
        prov = sum(1 for x in rs if x["provenance_match"])
        breakdown[pred] = {
            "n": len(rs),
            "precision_at_k": ntp / max(len(rs), 1),
            "provenance_correctness": prov / max(ntp, 1) if ntp else 0.0,
            "mrr": sum(x["reciprocal_rank"] for x in rs) / max(len(rs), 1),
        }

    summary = {
        "total_queries": n,
        "top_k": args.top_k,
        "mode": args.mode,
        "precision_at_k": precision_at_k,
        "provenance_correctness": provenance_correctness,
        "mrr": mrr,
        "by_predicate": breakdown,
        "gate_precision_at_k": 0.80,
        "gate_provenance": 0.95,
        "precision_passed": precision_at_k >= 0.80,
        "provenance_passed": provenance_correctness >= 0.95,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({"event": "summary", **summary}, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
