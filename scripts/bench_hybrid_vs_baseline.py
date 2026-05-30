"""Bench: hybrid JEPA-LLM harness vs. single-call LLM-with-context baseline.

Same model. Same information (both see the full StateStore as context).
Different routing:

  hybrid    — observe → predict → render (3 LLM calls, typed packets, guardrails)
  baseline  — 1 LLM call with the full state JSON in the user message

This isolates whether the architectural decomposition itself buys anything,
not whether having state buys anything (it does — that's RAG, not BLA).

Questions are factual single-turn lookups against the BLA-tracker seed.
Best case for the baseline; worst case for the hybrid (whose pitch is
multi-turn coherence + state updates over time, not isolated Q&A).

Grading is intentionally simple substring matching on lowercased text:
each question lists tokens that MUST appear in a correct answer and
tokens that MUST NOT appear (to catch hallucinations). Both responses
are printed in full so you can inspect them.

Usage:
    python3 scripts/bench_hybrid_vs_baseline.py [--model claude-haiku-4-5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bla.hybrid.bla_tracker import BLA_DOMAIN_SYSTEM_PROMPT, seed_bla_tracker
from bla.hybrid.llm_client import AnthropicLLMClient
from bla.hybrid.loop import HybridLoop
from bla.hybrid.predictor import LLMPredictor
from bla.hybrid.state import StateStore


# Each question has ground-truth criteria. "must" tokens are substrings
# that MUST appear in a correct lowercase answer (any 1 of any-of-N sublists
# counts as a hit); "must_not" catch hallucinations.
QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "Q1_18l_g1_result",
        "q": "Did Phase 18l G1 pass or fail single-seed? Give the observed number.",
        "must": [["fail"], ["0.47", "0.48", "0.477"]],
        "must_not": [["pass"]],
    },
    {
        "id": "Q2_18mu_verdict",
        "q": "What was the multi-seed verdict for Phase 18mu?",
        "must": [["acceptable", "swap"]],
        "must_not": [["fail"]],
    },
    {
        "id": "Q3_gate_18mu_g4",
        "q": "Did the gate_18mu_g4 (adapter mean Spearman >= 0.45) pass, and what was observed?",
        "must": [["pass", "true", "yes"], ["0.50", "0.501"]],
        "must_not": [["fail", "did not"]],
    },
    {
        "id": "Q4_locked_recipe",
        "q": "What is the locked recipe and on how many tasks was it validated?",
        "must": [["demo_no_cem"], ["4", "four"]],
        "must_not": [],
    },
    {
        "id": "Q5_hallucination_resistance",
        "q": "Tell me everything you know about gate_18l_g7_phantom.",
        "must": [["not", "don't", "no such", "doesn't", "do not", "isn't"]],
        "must_not": [["passed", "failed", "observed", "result was", "0.", "spearman"]],
    },
    {
        "id": "Q6_open_questions_count",
        "q": "How many open_questions does hyp_jepa_llm_state list?",
        "must": [["2", "two"]],
        "must_not": [["1", "3", "one", "three"]],
    },
    {
        "id": "Q7_relations",
        "q": "Which gates contradict phase_18l? Name them by id.",
        "must": [["gate_18l_g1_strong"]],
        "must_not": [],
    },
    {
        "id": "Q8_calibration",
        "q": "Is the hypothesis 'object-JEPA gives LLM a better latent state' considered high-confidence in the current state?",
        "must": [["0.6", "0.65", "moderate", "medium", "uncertain", "not high"]],
        "must_not": [["very high", "highly confident"]],
    },
]


_BASELINE_SYSTEM = """\
You are answering questions about an ongoing research project. Below is
the current state of the project as a JSON object store. Use ONLY this
state to answer — do not invent objects, gates, or numbers that aren't
present.

If something the user asks about isn't in the state, say so explicitly.
Be terse and reference object ids when relevant.
"""


def _grade(text: str, must: list[list[str]], must_not: list[list[str]]) -> dict[str, Any]:
    """Returns {hits, hit_groups, false_hits, score} where score is in [0,1]."""
    t = text.lower()
    hit_groups = 0
    for group in must:
        if any(tok.lower() in t for tok in group):
            hit_groups += 1
    false_hits = 0
    for group in must_not:
        if any(tok.lower() in t for tok in group):
            false_hits += 1
    # Score: fraction of must-groups satisfied minus a penalty per false hit
    must_total = max(len(must), 1)
    score = (hit_groups / must_total) - 0.5 * false_hits
    score = max(0.0, min(1.0, score))
    return {
        "hits": hit_groups,
        "must_total": len(must),
        "false_hits": false_hits,
        "score": score,
    }


def run_baseline(llm: AnthropicLLMClient, state: StateStore, question: str) -> tuple[str, float]:
    user_msg = (
        "STATE (JSON):\n"
        + json.dumps(state.to_dict(), indent=2, sort_keys=True)
        + "\n\nQUESTION:\n" + question
        + "\n\nAnswer in 1-3 sentences."
    )
    t0 = time.time()
    out = llm.complete(system=_BASELINE_SYSTEM, user=user_msg, max_tokens=400)
    return out, time.time() - t0


def run_hybrid(loop: HybridLoop, question: str) -> tuple[str, float]:
    t0 = time.time()
    rec = loop.step(question)
    return rec.render_text, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--verbose", action="store_true",
                    help="Print full responses per question")
    args = ap.parse_args()

    llm = AnthropicLLMClient(model=args.model)
    state = StateStore()
    seed_bla_tracker(state)

    # Fresh state per side so neither accumulates carry-over
    hybrid_state = StateStore()
    seed_bla_tracker(hybrid_state)
    predictor = LLMPredictor(llm=llm)
    loop = HybridLoop(
        llm=llm, predictor=predictor, state=hybrid_state,
        domain_preamble=BLA_DOMAIN_SYSTEM_PROMPT,
    )

    rows: list[dict[str, Any]] = []
    print(f"=== bench: model={args.model}  questions={len(QUESTIONS)} ===\n")

    for q in QUESTIONS:
        print(f"--- {q['id']} ---")
        print(f"Q: {q['q']}")
        try:
            base_out, base_t = run_baseline(llm, state, q["q"])
        except Exception as e:
            base_out, base_t = f"[error: {type(e).__name__}: {e}]", 0.0
        try:
            hyb_out, hyb_t = run_hybrid(loop, q["q"])
        except Exception as e:
            hyb_out, hyb_t = f"[error: {type(e).__name__}: {e}]", 0.0

        base_grade = _grade(base_out, q["must"], q["must_not"])
        hyb_grade = _grade(hyb_out, q["must"], q["must_not"])

        rows.append({
            "id": q["id"],
            "base_score": base_grade["score"],
            "hyb_score": hyb_grade["score"],
            "base_false_hits": base_grade["false_hits"],
            "hyb_false_hits": hyb_grade["false_hits"],
            "base_t": base_t,
            "hyb_t": hyb_t,
            "base_out": base_out,
            "hyb_out": hyb_out,
        })

        print(f"  baseline ({base_t:.1f}s, score={base_grade['score']:.2f}, "
              f"false_hits={base_grade['false_hits']}):")
        if args.verbose:
            for line in base_out.splitlines():
                print("    " + line)
        else:
            print("    " + (base_out[:200] + ("…" if len(base_out) > 200 else "")))
        print(f"  hybrid   ({hyb_t:.1f}s, score={hyb_grade['score']:.2f}, "
              f"false_hits={hyb_grade['false_hits']}):")
        if args.verbose:
            for line in hyb_out.splitlines():
                print("    " + line)
        else:
            print("    " + (hyb_out[:200] + ("…" if len(hyb_out) > 200 else "")))
        print()

    # Aggregate
    n = len(rows)
    base_avg = sum(r["base_score"] for r in rows) / n
    hyb_avg = sum(r["hyb_score"] for r in rows) / n
    base_t_avg = sum(r["base_t"] for r in rows) / n
    hyb_t_avg = sum(r["hyb_t"] for r in rows) / n
    base_fh = sum(r["base_false_hits"] for r in rows)
    hyb_fh = sum(r["hyb_false_hits"] for r in rows)
    base_wins = sum(1 for r in rows if r["base_score"] > r["hyb_score"])
    hyb_wins = sum(1 for r in rows if r["hyb_score"] > r["base_score"])
    ties = sum(1 for r in rows if r["hyb_score"] == r["base_score"])

    print("=" * 60)
    print(f"AGGREGATE  (n={n}, model={args.model})")
    print("-" * 60)
    print(f"  mean score      baseline={base_avg:.3f}   hybrid={hyb_avg:.3f}   "
          f"Δ={hyb_avg - base_avg:+.3f}")
    print(f"  false-hits (Σ)  baseline={base_fh}        hybrid={hyb_fh}")
    print(f"  mean latency    baseline={base_t_avg:.1f}s   hybrid={hyb_t_avg:.1f}s   "
          f"({hyb_t_avg / max(base_t_avg, 0.01):.1f}× slower)")
    print(f"  per-question:   baseline wins={base_wins}, hybrid wins={hyb_wins}, ties={ties}")
    print("=" * 60)

    # Save raw
    out_path = Path("artifacts/bench_hybrid_vs_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "n_questions": n,
        "summary": {
            "baseline_mean_score": base_avg,
            "hybrid_mean_score": hyb_avg,
            "delta": hyb_avg - base_avg,
            "baseline_false_hits_total": base_fh,
            "hybrid_false_hits_total": hyb_fh,
            "baseline_wins": base_wins,
            "hybrid_wins": hyb_wins,
            "ties": ties,
            "baseline_mean_latency_s": base_t_avg,
            "hybrid_mean_latency_s": hyb_t_avg,
        },
        "rows": rows,
    }, indent=2))
    print(f"raw saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
