"""Multi-turn bench: hybrid (incremental state) vs. baseline (conversation history).

THIS IS WHERE THE ARCHITECTURAL CLAIM LIVES. Single-turn Q&A doesn't
exercise the hybrid's pitch — multi-turn state evolution does.

Setup:
  - 10 turns: 5 UPDATE turns where user states info, 5 QUERY turns that
    require memory of prior UPDATE turns to answer correctly.
  - Hybrid: incrementally updates StateStore via OBSERVE → UPDATE per
    turn. State carries from turn to turn.
  - Baseline: single LLM call per turn with growing conversation history
    in messages[]; seed state JSON pinned in system prompt.
  - Both run the same model.

Pre-committed gates (set BEFORE running, per
[[feedback-proxy-vs-end-effect-gate]] + [[threshold-calibration-retrospective]]):

  HYBRID WINS IFF either is true at the end of the run:
    G1. Hybrid scores >= baseline on at least 6 of 10 turns AND
        aggregate hybrid score > aggregate baseline score
    G2. Hybrid total token cost is >= 30% lower than baseline,
        AND aggregate hybrid score >= aggregate baseline score - 0.05

  If neither G1 nor G2 passes:
    -> the architectural claim is UNSUPPORTED on this domain at this
       scale, and the right next action is to question the thesis, not
       add more layers to the harness.

Usage:
    python3 scripts/bench_hybrid_multiturn.py [--model claude-haiku-4-5]
"""
from __future__ import annotations

import argparse
import copy
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


# The 10 turns. UPDATE turns introduce new info; QUERY turns require
# correct memory of prior turns to answer. Memory-dependent turns are
# tagged in `depends_on` for the writeup.
TURNS: list[dict[str, Any]] = [
    {
        "i": 1,
        "kind": "UPDATE",
        "depends_on": [],
        "user": "I just ran a quick phase-19 sweep: the geometry adapter hit 0.52 mean Spearman across 3 seeds.",
        "must": [["0.52", "phase 19", "phase_19", "noted", "got it", "ok", "acknowledged", "tracked"]],
        "must_not": [["0.501 mean", "0.501 spearman"]],  # don't confuse with 18mu
    },
    {
        "i": 2,
        "kind": "QUERY",
        "depends_on": [1],
        "user": "Quick check: does that beat the phase 18mu adapter Spearman?",
        "must": [["yes", "beats", "higher", "exceeds", "above"], ["0.52"]],
        "must_not": [["no", "below", "worse"]],
    },
    {
        "i": 3,
        "kind": "UPDATE_CORRECTION",
        "depends_on": [1],
        "user": "Correction on phase 19: there was a bug in the eval. The real Spearman was 0.48, not 0.52.",
        "must": [["0.48"], ["correct", "update", "revis", "noted", "got it", "ok"]],
        "must_not": [],
    },
    {
        "i": 4,
        "kind": "QUERY",
        "depends_on": [1, 3],
        "user": "OK with the correction, does phase 19 still beat phase 18mu?",
        "must": [["no", "below", "worse", "doesn't", "does not"], ["0.48"]],
        "must_not": [["yes", "beats", "exceeds", "higher"]],
    },
    {
        "i": 5,
        "kind": "UPDATE",
        "depends_on": [],
        "user": "New hypothesis: phase 19's adapter overfit because we trained it on 800 samples instead of the standard 720.",
        "must": [["hypothesis", "added", "noted", "tracked", "overfit"]],
        "must_not": [],
    },
    {
        "i": 6,
        "kind": "QUERY",
        "depends_on": [5],
        "user": "Which of my active hypotheses involve overfitting?",
        "must": [["overfit", "phase 19", "800", "720"]],
        "must_not": [],
    },
    {
        "i": 7,
        "kind": "UPDATE_RETRACTION",
        "depends_on": [5],
        "user": "Actually I was wrong about phase 19's sample count — it also used 720 samples, same as everything else. My overfit hypothesis premise is bad.",
        "must": [["retract", "weaken", "premise", "revis", "no longer", "withdraw", "wrong"]],
        "must_not": [["confirmed", "still holds"]],
    },
    {
        "i": 8,
        "kind": "QUERY",
        "depends_on": [5, 7],
        "user": "Given the sample-count correction, what's the current status of my overfit hypothesis?",
        "must": [["weak", "retract", "no longer", "premise", "withdraw", "refuted", "broken", "invalid"]],
        "must_not": [["still active", "still holds", "supported"]],
    },
    {
        "i": 9,
        "kind": "UPDATE",
        "depends_on": [],
        "user": "Phase 18l G1 was retested with 5 seeds — it actually passes now at 0.51 mean Spearman, up from 0.478 single-seed.",
        "must": [["0.51"], ["pass", "now passes", "updated"]],
        "must_not": [],
    },
    {
        "i": 10,
        "kind": "QUERY_SYNTHESIS",
        "depends_on": [1, 3, 5, 7, 9],
        "user": "Give me a one-paragraph summary of all the updates I've shared in this conversation, in order. Be specific with numbers.",
        "must": [["0.48"], ["0.51"], ["retract", "weak", "withdrew", "premise", "wrong"]],
        "must_not": [["0.52"]],  # the corrected number is 0.48, not the original 0.52
    },
]


_BASELINE_SYSTEM = """\
You are answering questions about an ongoing BLA research project. Below
is the initial state of the project as a JSON object store, plus the
conversation history will accumulate as new turns are added.

Use only the state below AND what the user tells you in conversation.
Do not invent objects, gates, or numbers. When the user shares new
information or corrections, integrate them into your answers on
subsequent turns. Be terse — 1-3 sentences per response unless asked
for a summary.
"""


def _grade(text: str, must: list[list[str]], must_not: list[list[str]]) -> dict[str, Any]:
    t = text.lower()
    hit_groups = sum(1 for g in must if any(tok.lower() in t for tok in g))
    false_hits = sum(1 for g in must_not if any(tok.lower() in t for tok in g))
    must_total = max(len(must), 1)
    score = max(0.0, min(1.0, (hit_groups / must_total) - 0.5 * false_hits))
    return {
        "hits": hit_groups, "must_total": len(must),
        "false_hits": false_hits, "score": score,
    }


def run_hybrid(llm: AnthropicLLMClient, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = StateStore()
    seed_bla_tracker(state)
    predictor = LLMPredictor(llm=llm)
    loop = HybridLoop(
        llm=llm, predictor=predictor, state=state,
        domain_preamble=BLA_DOMAIN_SYSTEM_PROMPT,
        auto_apply_updates=True,
    )
    rows: list[dict[str, Any]] = []
    for turn in turns:
        t0 = time.time()
        usage_before = copy.deepcopy(llm.total_usage)
        try:
            rec = loop.step(turn["user"])
            answer = rec.render_text
            err = None
        except Exception as e:
            answer = f"[error: {type(e).__name__}: {e}]"
            err = str(e)
        dt = time.time() - t0
        usage_after = copy.deepcopy(llm.total_usage)
        usage_delta = {
            k: usage_after[k] - usage_before[k] for k in usage_after
        }
        rows.append({
            "i": turn["i"], "kind": turn["kind"],
            "answer": answer, "latency_s": dt, "usage": usage_delta,
            "n_objects": len(state), "error": err,
            "depends_on": turn.get("depends_on", []),
        })
    return rows


def run_baseline(llm: AnthropicLLMClient, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Single-call-per-turn with growing conversation history.

    System prompt holds the seed state JSON (pinned, cached). Messages
    accumulate user/assistant pairs as the conversation progresses.
    """
    # Build the system message: instructions + seed state JSON
    seed = StateStore()
    seed_bla_tracker(seed)
    system = (
        _BASELINE_SYSTEM
        + "\n\nINITIAL STATE (JSON):\n"
        + json.dumps(seed.to_dict(), indent=2, sort_keys=True)
    )
    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for turn in turns:
        history.append({"role": "user", "content": turn["user"]})
        t0 = time.time()
        usage_before = copy.deepcopy(llm.total_usage)
        try:
            # Reuse AnthropicLLMClient.complete() — but it takes a single
            # user string. We need multi-message history. Hit the SDK
            # directly via the underlying client.
            kwargs: dict[str, Any] = {
                "model": llm.model,
                "max_tokens": 500,
                "system": [{
                    "type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": history,
            }
            try:
                resp = llm._client.messages.create(**kwargs)
            except TypeError:
                resp = llm._client.messages.create(**kwargs)
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            # Track usage
            u = resp.usage
            for k in ("input_tokens", "output_tokens",
                      "cache_creation_input_tokens", "cache_read_input_tokens"):
                llm.total_usage[k] += getattr(u, k, 0) or 0
            llm.total_usage["n_calls"] += 1
            history.append({"role": "assistant", "content": answer})
            err = None
        except Exception as e:
            answer = f"[error: {type(e).__name__}: {e}]"
            err = str(e)
        dt = time.time() - t0
        usage_after = copy.deepcopy(llm.total_usage)
        usage_delta = {k: usage_after[k] - usage_before[k] for k in usage_after}
        rows.append({
            "i": turn["i"], "kind": turn["kind"],
            "answer": answer, "latency_s": dt, "usage": usage_delta,
            "history_len": len(history), "error": err,
            "depends_on": turn.get("depends_on", []),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== multi-turn bench: model={args.model}  turns={len(TURNS)} ===")
    print(f"PRE-COMMITTED GATES:")
    print(f"  G1: hybrid >= baseline on >= 6/10 turns AND total hybrid score > total baseline")
    print(f"  G2: hybrid total tokens 30% lower AND hybrid score >= baseline - 0.05")
    print(f"  if neither: architectural claim UNSUPPORTED on this domain.")
    print()

    # Separate client per side so total_usage doesn't cross-contaminate
    hyb_llm = AnthropicLLMClient(model=args.model)
    base_llm = AnthropicLLMClient(model=args.model)

    print("=== HYBRID RUN ===")
    hyb_rows = run_hybrid(hyb_llm, TURNS)
    print(f"  done. n_calls={hyb_llm.total_usage['n_calls']}, "
          f"in={hyb_llm.total_usage['input_tokens']}, "
          f"out={hyb_llm.total_usage['output_tokens']}")

    print("=== BASELINE RUN ===")
    base_rows = run_baseline(base_llm, TURNS)
    print(f"  done. n_calls={base_llm.total_usage['n_calls']}, "
          f"in={base_llm.total_usage['input_tokens']}, "
          f"out={base_llm.total_usage['output_tokens']}")
    print()

    # Grade
    per_turn: list[dict[str, Any]] = []
    for turn, hrow, brow in zip(TURNS, hyb_rows, base_rows):
        hg = _grade(hrow["answer"], turn["must"], turn["must_not"])
        bg = _grade(brow["answer"], turn["must"], turn["must_not"])
        per_turn.append({
            "i": turn["i"], "kind": turn["kind"],
            "depends_on": turn.get("depends_on", []),
            "hybrid": {"score": hg["score"], "false_hits": hg["false_hits"],
                       "answer": hrow["answer"], "latency": hrow["latency_s"],
                       "usage": hrow["usage"]},
            "baseline": {"score": bg["score"], "false_hits": bg["false_hits"],
                         "answer": brow["answer"], "latency": brow["latency_s"],
                         "usage": brow["usage"]},
        })

    # Print
    print("=" * 80)
    print(f"{'#':>2} {'kind':<22} {'deps':<10} {'base':>5} {'hyb':>5}  winner")
    print("-" * 80)
    for r in per_turn:
        deps = ",".join(str(d) for d in r["depends_on"]) or "—"
        bs = r["baseline"]["score"]
        hs = r["hybrid"]["score"]
        winner = "hybrid" if hs > bs else ("baseline" if bs > hs else "tie")
        print(f"{r['i']:>2} {r['kind']:<22} {deps:<10} {bs:>5.2f} {hs:>5.2f}  {winner}")
    print("-" * 80)

    base_avg = sum(r["baseline"]["score"] for r in per_turn) / len(per_turn)
    hyb_avg = sum(r["hybrid"]["score"] for r in per_turn) / len(per_turn)
    base_fh = sum(r["baseline"]["false_hits"] for r in per_turn)
    hyb_fh = sum(r["hybrid"]["false_hits"] for r in per_turn)
    base_wins = sum(1 for r in per_turn if r["baseline"]["score"] > r["hybrid"]["score"])
    hyb_wins = sum(1 for r in per_turn if r["hybrid"]["score"] > r["baseline"]["score"])
    ties = sum(1 for r in per_turn if r["hybrid"]["score"] == r["baseline"]["score"])
    hyb_ge_base = sum(1 for r in per_turn if r["hybrid"]["score"] >= r["baseline"]["score"])

    base_tok = base_llm.total_usage["input_tokens"] + base_llm.total_usage["output_tokens"]
    hyb_tok = hyb_llm.total_usage["input_tokens"] + hyb_llm.total_usage["output_tokens"]
    token_savings = (base_tok - hyb_tok) / max(base_tok, 1)

    print()
    print(f"AGGREGATE (n={len(per_turn)}, model={args.model})")
    print(f"  mean score      baseline={base_avg:.3f}   hybrid={hyb_avg:.3f}   "
          f"Δ={hyb_avg - base_avg:+.3f}")
    print(f"  false-hits (Σ)  baseline={base_fh}        hybrid={hyb_fh}")
    print(f"  per-turn wins   baseline={base_wins}, hybrid={hyb_wins}, ties={ties}")
    print(f"  hybrid >= base on {hyb_ge_base}/{len(per_turn)} turns")
    print()
    print(f"  total tokens    baseline={base_tok:>6}   hybrid={hyb_tok:>6}   "
          f"hybrid_savings={token_savings:+.1%}  ({'cheaper' if token_savings > 0 else 'more expensive'})")
    print(f"  total calls     baseline={base_llm.total_usage['n_calls']}   "
          f"hybrid={hyb_llm.total_usage['n_calls']}")

    # Gates
    g1 = hyb_ge_base >= 6 and hyb_avg > base_avg
    g2 = (token_savings >= 0.30) and (hyb_avg >= base_avg - 0.05)
    print()
    print("GATE EVALUATION:")
    print(f"  G1 (hybrid >= base on >=6 AND total > base): {g1}  "
          f"({hyb_ge_base}/10 ≥6={hyb_ge_base >= 6}, "
          f"avg {hyb_avg:.3f}>{base_avg:.3f}={hyb_avg > base_avg})")
    print(f"  G2 (>=30% token savings AND score within 0.05): {g2}  "
          f"(savings={token_savings:+.1%}, "
          f"score-gap={hyb_avg - base_avg:+.3f})")
    if g1 or g2:
        print("  VERDICT: HYBRID WINS — architectural claim supported on this bench")
    else:
        print("  VERDICT: HYBRID DOES NOT WIN — architectural claim UNSUPPORTED on this bench")

    if args.verbose:
        print()
        print("=" * 80)
        print("FULL RESPONSES")
        for r in per_turn:
            print("-" * 80)
            t = TURNS[r["i"] - 1]
            print(f"# T{r['i']}  {r['kind']}  (depends on {r['depends_on'] or 'none'})")
            print(f"USER: {t['user']}")
            print(f"--- BASELINE [score={r['baseline']['score']:.2f}] ---")
            print(r["baseline"]["answer"])
            print(f"--- HYBRID  [score={r['hybrid']['score']:.2f}] ---")
            print(r["hybrid"]["answer"])

    out_path = Path("artifacts/bench_hybrid_multiturn.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "summary": {
            "baseline_mean_score": base_avg, "hybrid_mean_score": hyb_avg,
            "baseline_false_hits": base_fh, "hybrid_false_hits": hyb_fh,
            "baseline_wins": base_wins, "hybrid_wins": hyb_wins, "ties": ties,
            "hybrid_ge_baseline_turns": hyb_ge_base,
            "baseline_total_tokens": base_tok, "hybrid_total_tokens": hyb_tok,
            "hybrid_token_savings_pct": token_savings * 100,
            "g1_pass": g1, "g2_pass": g2, "hybrid_wins_overall": g1 or g2,
        },
        "per_turn": per_turn,
    }, indent=2))
    print(f"raw saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
