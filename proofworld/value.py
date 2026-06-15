#!/usr/bin/env python3
"""proofworld.value -- the learned ROUTE-VALUE head (the one endorsed learned component).

It ORDERS dreaming toward the frontier -- ranks candidate compositions by predicted closing-probability so the
dreamer tries promising chains first and calls the expensive kernel less. CRITICAL SAFETY LINE: it ranks ONLY;
it never asserts truth. The verifier still confirms every candidate, so a mis-ranking costs *time*, never
*correctness* -- there is nothing for the ranker to exploit. Trained on the accumulated verified/refuted traces.

Demo: a chain-closing family (lower bound a0>=0 + links a_i<=a_{i+1} => a_k>=0, among decoys). The ranker learns
"lower-bound + consecutive links" and finds the closing chain in far fewer kernel calls than brute enumeration --
with identical correctness. Run: python3 -m proofworld.value
"""
from __future__ import annotations
from itertools import combinations
import numpy as np, z3

K = 5                                                        # chain length: a0..aK
A = z3.Reals(' '.join(f"a{i}" for i in range(K + 1)))
LOWER = ("lower", A[0] >= 0)
LINKS = [(f"link{i}", A[i] <= A[i + 1]) for i in range(K)]
DECOYS = [(f"decoy{j}", A[(j * 2) % (K + 1)] <= A[(j * 2) % (K + 1)] + 3 + j) for j in range(4)]  # true but useless
POOL = [LOWER] + LINKS + DECOYS                              # the verified lemma pool
GOAL = A[K] >= 0                                             # closable only via the full chain


def implies(forms, goal, t=3000):                            # GROUNDED check (truth) -- also the training label
    s = z3.Solver(); s.set("timeout", t)
    for f in forms: s.add(f)
    s.add(z3.Not(goal)); return s.check() == z3.unsat

def feats(combo_names):                                      # cheap structural features of a composition
    has_lower = "lower" in combo_names
    links = sorted(int(n[4:]) for n in combo_names if n.startswith("link"))
    consec = 0                                               # longest run of consecutive links starting at 0
    if links and links[0] == 0:
        consec = 1
        for a, b in zip(links, links[1:]):
            if b == a + 1: consec += 1
            else: break
    return np.array([len(combo_names), float(has_lower), len(links), consec / (K + 1),
                     sum(n.startswith("decoy") for n in combo_names)], np.float32)


def train_ranker(seed=0):
    """label random sub-compositions by the GROUNDED implies-check; fit a tiny logistic ranker on the features."""
    rng = np.random.RandomState(seed); names = [n for n, _ in POOL]; X, y = [], []
    for _ in range(600):
        k = rng.randint(1, len(names) + 1); combo = tuple(rng.choice(names, k, replace=False))
        X.append(feats(combo)); y.append(float(implies([dict(POOL)[n] for n in combo], GOAL)))
    X, y = np.array(X), np.array(y); mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    w = np.zeros(Xn.shape[1]); b = 0.0
    for _ in range(4000):                                    # logistic regression by gradient descent
        p = 1 / (1 + np.exp(-(Xn @ w + b))); g = p - y
        w -= 0.05 * (Xn.T @ g / len(y) + 1e-3 * w); b -= 0.05 * g.mean()
    return (w, b, mu, sd), float((( (1/(1+np.exp(-(Xn@w+b))) > 0.5) == (y > 0.5)).mean()))

def score(model, combo):
    w, b, mu, sd = model; return float(1 / (1 + np.exp(-(((feats(combo) - mu) / sd) @ w + b))))


def main():
    print("=== proofworld.value :: learned route-value ranker (orders dreaming; NEVER decides truth) ===\n")
    model, acc = train_ranker(); print(f"  ranker trained on grounded traces (fit acc {acc:.2f})\n")
    names = [n for n, _ in POOL]; lemmas = dict(POOL)
    cands = []                                               # all compositions up to size K+1
    for sz in range(1, K + 2):
        cands += [tuple(c) for c in combinations(names, sz)]
    truth = {c: implies([lemmas[n] for n in c], GOAL) for c in cands}
    closing = [c for c in cands if truth[c]]
    # BRUTE order (enumeration) vs RANKED order -- count kernel calls until the first closing chain is found+verified
    brute_order = cands
    ranked_order = sorted(cands, key=lambda c: -score(model, c))
    def calls_until_closed(order):
        for i, c in enumerate(order, 1):
            if implies([lemmas[n] for n in c], GOAL): return i, c     # verifier confirms (truth owned here)
        return len(order), None
    bi, bc = calls_until_closed(brute_order); ri, rc = calls_until_closed(ranked_order)
    print(f"  total compositions: {len(cands)}   closing chains (ground truth): {len(closing)}")
    print(f"  BRUTE enumeration : {bi} kernel calls to first closing chain  {bc}")
    print(f"  RANKED by value   : {ri} kernel calls to first closing chain  {rc}")
    print(f"  speedup: {bi/ri:.1f}x fewer kernel calls -- SAME correctness (the verifier confirmed the winner)")
    print(f"\n  top-5 ranked compositions (value head's ordering):")
    for c in ranked_order[:5]:
        print(f"    score={score(model,c):.2f}  closes={truth[c]}  {c}")
    print(f"\n  SAFETY: the ranker only ORDERED the search. Truth was decided by the verifier on every candidate,")
    print(f"  so a bad ranking would have cost time, never a false 'proof'. Nothing for the ranker to exploit.")


if __name__ == "__main__":
    main()
