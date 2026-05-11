"""Offline reranking experiments on PAL candidate data.

Loads candidates.jsonl (200 problems x N samples each) and tries
multiple aggregation strategies. Reports accuracy for each.

Strategies:
  oracle           — any candidate's pred matches gold (ceiling)
  mode             — mode vote over preds (baseline SC)
  mode_filter      — mode vote over preds that pass quality filter
  weighted_oplen   — vote weight = op_count + 1
  weighted_uses    — vote weight = uses_fraction
  filter_quality   — discard candidates with uses_fraction < 0.5
  filter_trivial   — discard candidates whose pred is just a problem number
  combo            — filter_quality + filter_trivial, then mode vote
"""

from __future__ import annotations

import argparse
import collections
import json


def load_records(path: str):
    out = []
    for line in open(path):
        out.append(json.loads(line))
    return out


def s_oracle(rec):
    gold = rec["gold"]
    return any(c["pred"] is not None and str(c["pred"]).rstrip(".") == gold.rstrip(".")
               for c in rec["candidates"])


def s_mode(rec):
    votes = collections.Counter()
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is not None:
            votes[str(p)] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def s_mode_filter_trivial(rec):
    """Mode but skip preds that are just a number from the problem."""
    votes = collections.Counter()
    nums = set(rec["problem_numbers"])
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        if p in nums:
            continue
        votes[str(p)] += 1
    if not votes:
        # Fall back to unfiltered mode
        return s_mode(rec)
    return votes.most_common(1)[0][0]


def s_filter_quality(rec, thresh: float = 0.5):
    """Mode over candidates where code uses >= thresh of problem numbers."""
    votes = collections.Counter()
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        if c["features"]["uses_fraction"] < thresh:
            continue
        votes[str(p)] += 1
    if not votes:
        return s_mode(rec)
    return votes.most_common(1)[0][0]


def s_weighted_oplen(rec):
    """Vote weighted by op_count + n_assignments."""
    votes = collections.Counter()
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        w = c["features"]["op_count"] + c["features"]["n_assignments"]
        votes[str(p)] += w
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def s_weighted_uses(rec):
    """Vote weighted by uses_fraction."""
    votes = collections.Counter()
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        w = c["features"]["uses_fraction"]
        votes[str(p)] += w
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def s_combo(rec, thresh: float = 0.5):
    """filter_trivial + filter_quality + mode."""
    votes = collections.Counter()
    nums = set(rec["problem_numbers"])
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        if p in nums:
            continue
        if c["features"]["uses_fraction"] < thresh:
            continue
        votes[str(p)] += 1
    if not votes:
        return s_mode_filter_trivial(rec)
    return votes.most_common(1)[0][0]


def s_combo_weighted(rec, thresh: float = 0.5):
    """combo but weighted by op_count + uses_fraction."""
    votes = collections.Counter()
    nums = set(rec["problem_numbers"])
    for c in rec["candidates"]:
        p = c.get("pred")
        if p is None:
            continue
        if p in nums:
            continue
        if c["features"]["uses_fraction"] < thresh:
            continue
        w = c["features"]["op_count"] + c["features"]["uses_fraction"] * 4
        votes[str(p)] += w
    if not votes:
        return s_mode_filter_trivial(rec)
    return votes.most_common(1)[0][0]


STRATEGIES = {
    "mode": s_mode,
    "mode_filter_trivial": s_mode_filter_trivial,
    "filter_quality": s_filter_quality,
    "weighted_oplen": s_weighted_oplen,
    "weighted_uses": s_weighted_uses,
    "combo": s_combo,
    "combo_weighted": s_combo_weighted,
}


def score(records, strategy_fn) -> tuple[int, int]:
    correct = 0
    total = 0
    for r in records:
        gold = r["gold"]
        pred = strategy_fn(r)
        if pred is not None and str(pred).rstrip(".") == gold.rstrip("."):
            correct += 1
        total += 1
    return correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    records = load_records(args.input)

    oracle_correct = sum(1 for r in records if s_oracle(r))
    print(f"# Records: {len(records)}")
    print(f"# Oracle: {oracle_correct}/{len(records)} = {oracle_correct/len(records)*100:.1f}%")
    print()
    print(f"{'strategy':25} {'correct/total':>15} {'acc%':>8}")
    print("-" * 50)
    for name, fn in STRATEGIES.items():
        c, t = score(records, fn)
        print(f"{name:25} {f'{c}/{t}':>15} {c/t*100:>7.1f}%")


if __name__ == "__main__":
    main()
