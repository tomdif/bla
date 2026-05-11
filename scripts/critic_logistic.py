"""Offline critic via logistic regression on verifier features.

Trains a small classifier that takes verifier features → P(correct),
then ranks candidates per problem. Uses leave-one-problem-out CV to
avoid contamination.

Goal: beat mode-vote's 4% / weighted-vote's 4.5% on run13 N=32.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.verifier import score_candidate


def extract_features(c, problem_numbers):
    s = score_candidate(c.get("code", ""), c.get("output", ""),
                        c.get("pred"), problem_numbers)
    if not s["runs"]:
        return None
    return [
        s["ops"],
        int(s["is_computed"]),
        s["chain_depth"],
        s["nums_in_op"],
        int(s["echo"]),
        s["responsiveness"],
        # interaction features
        s["ops"] * s["responsiveness"],
        s["chain_depth"] * s["responsiveness"],
    ]


def gold_match(pred, gold):
    if pred is None:
        return False
    try:
        return str(pred).rstrip(".") == gold.rstrip(".")
    except (AttributeError, TypeError):
        return False


def build_dataset(records):
    """Return list of per-problem (X, y, preds) tuples."""
    out = []
    for r in records:
        gold = r["gold"]
        problem_numbers = r.get("problem_numbers", [])
        X, y, preds = [], [], []
        for c in r["candidates"]:
            feats = extract_features(c, problem_numbers)
            if feats is None:
                continue
            X.append(feats)
            y.append(int(gold_match(c.get("pred"), gold)))
            preds.append(c.get("pred"))
        if X:
            out.append((np.array(X), np.array(y), preds, gold))
    return out


def loo_critic_eval(per_problem, verbose=False):
    """Leave-one-problem-out: train logistic regression on all-but-one
    problem's candidates, predict on held-out problem, pick top-1 ranked
    candidate's pred, check correctness."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(per_problem)
    correct = 0
    for i in range(n):
        # Gather train data from all but i
        train_X = np.vstack([per_problem[j][0] for j in range(n) if j != i])
        train_y = np.concatenate([per_problem[j][1] for j in range(n) if j != i])

        if train_y.sum() == 0:
            # No positives in train; default to mode
            pred = _mode_fallback(per_problem[i][2])
        else:
            scaler = StandardScaler()
            train_Xs = scaler.fit_transform(train_X)

            clf = LogisticRegression(class_weight="balanced", max_iter=500,
                                     solver="liblinear")
            clf.fit(train_Xs, train_y)

            test_X, test_y, test_preds, gold = per_problem[i]
            test_Xs = scaler.transform(test_X)
            probs = clf.predict_proba(test_Xs)[:, 1]
            # Tie-break: among top-k by prob, pick most common pred
            best_idx = int(probs.argmax())
            pred = test_preds[best_idx]

        gold = per_problem[i][3]
        if gold_match(pred, gold):
            correct += 1
        if verbose and i < 5:
            print(f"  problem {i}: gold={gold}, pred={pred}, ok={gold_match(pred, gold)}")

    return correct / n


def _mode_fallback(preds):
    import collections
    votes = collections.Counter()
    for p in preds:
        if p is not None:
            votes[str(p).rstrip(".")] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def feature_importance(per_problem):
    """Train on all data, report coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([p[0] for p in per_problem])
    y = np.concatenate([p[1] for p in per_problem])
    print(f"Total candidates: {len(X)}, positives: {int(y.sum())} ({y.mean()*100:.1f}%)")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear")
    clf.fit(Xs, y)

    names = ["ops", "is_computed", "chain_depth", "nums_in_op",
             "echo", "responsiveness", "ops*resp", "chain*resp"]
    print(f"\n{'feature':20} {'coef':>10}")
    for name, coef in zip(names, clf.coef_[0]):
        print(f"  {name:20} {coef:>10.3f}")
    print(f"  intercept            {clf.intercept_[0]:>10.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    print(f"# Records: {len(records)}")

    per_problem = build_dataset(records)
    print(f"# Problems with runnable candidates: {len(per_problem)}")

    feature_importance(per_problem)

    print()
    print("Leave-one-problem-out logistic-critic accuracy:")
    acc = loo_critic_eval(per_problem, verbose=args.verbose)
    print(f"  {acc*100:.1f}%")


if __name__ == "__main__":
    main()
