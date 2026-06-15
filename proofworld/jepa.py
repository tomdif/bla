#!/usr/bin/env python3
"""proofworld.jepa -- a JEPA-shaped premise retriever, A/B tested against the lexical baseline.

NOVEL use of JEPA here: premise selection as a Joint-Embedding Predictive Architecture. The GOAL is the context;
the CITED LEMMA is the target. A context-encoder + PREDICTOR predicts, in latent space, where useful premises live
(target = a separate lemma encoder); training is contrastive (InfoNCE with in-batch negatives -> no collapse).
This is exactly JEPA's shape -- predict the target's REPRESENTATION from the context, energy = -<pred, target> --
ported from pixels/proof-states to (goal -> cited-lemma). It NEVER decides truth (the kernel does); it only ranks
which lemmas to try, so it stays on the safe side of the trust boundary.

The test: hold out goals, rank all candidate premises, measure recall@k of the actually-cited lemmas, vs a lexical
(IDF token-overlap) baseline. Honest scope: only 359 goals / 1124 edges -- a SMALL-data probe. If JEPA does not beat
lexical here, the most likely reading is DATA-BOUND (too few pairs to learn non-lexical associations), not that the
architecture is useless; the same recipe at corpus scale (10^4-10^5 proofs) is where dual-encoder retrieval wins.

Run:  python3 -m proofworld.jepa
"""
from __future__ import annotations
import os, re, json, math
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PREMISES = os.path.join(HERE, "corpus", "premises.jsonl")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+")
IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*")

def toks(s):
    out = []
    for p in IDENT.findall(s):
        out += [w.lower() for w in (CAMEL.findall(p) or [p])]
    return out


def load():
    pairs = [json.loads(l) for l in open(PREMISES)]
    prem_pool = sorted({p for r in pairs for p in r["premises"]})
    return pairs, prem_pool


def build_vocab(pairs, prem_pool, min_df=2):
    df = {}
    for r in pairs:
        for t in set(toks(r["statement"])): df[t] = df.get(t, 0) + 1
    for p in prem_pool:
        for t in set(toks(p)): df[t] = df.get(t, 0) + 1
    kept = sorted(t for t, c in df.items() if c >= min_df)        # filter THEN enumerate (contiguous indices)
    vocab = {t: i for i, t in enumerate(kept)}
    return vocab, df

def featvec(s, vocab):
    v = np.zeros(len(vocab), np.float32)
    for t in toks(s):
        if t in vocab: v[vocab[t]] = 1.0
    return v


# ----------------------------- JEPA model -----------------------------
class Encoder(nn.Module):
    def __init__(self, din, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x): return self.net(x)

class JEPA(nn.Module):
    def __init__(self, din, d=64):
        super().__init__()
        self.ctx = Encoder(din, d)                                  # context (goal) encoder
        self.tgt = Encoder(din, d)                                  # target (premise) encoder
        self.pred = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # JEPA predictor
    def embed_goal(self, g):  return nn.functional.normalize(self.pred(self.ctx(g)), dim=-1)
    def embed_prem(self, p):  return nn.functional.normalize(self.tgt(p), dim=-1)


def train_jepa(edges_g, edges_p, din, epochs=400, d=64, tau=0.1, seed=0):
    torch.manual_seed(seed)
    m = JEPA(din, d); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    G = torch.tensor(np.stack(edges_g)); P = torch.tensor(np.stack(edges_p)); N = len(G)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i + 128]
            zg = m.embed_goal(G[idx]); zp = m.embed_prem(P[idx])
            logits = zg @ zp.t() / tau                              # in-batch InfoNCE (negatives = other premises)
            labels = torch.arange(len(idx))
            loss = nn.functional.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


# ----------------------------- baseline + eval -----------------------------
def lexical_scores(goal, prem_pool, idf):
    gt = set(toks(goal))
    return np.array([sum(idf.get(t, 0.0) for t in (gt & set(toks(p)))) for p in prem_pool])

def recall_at_k(ranked_idx, cited_set, ks):
    out = {}
    for k in ks:
        topk = set(ranked_idx[:k].tolist())
        hit = len(topk & cited_set)
        out[k] = hit / max(1, len(cited_set))
    return out


def run(seed):
    pairs, prem_pool = load()
    pidx = {p: i for i, p in enumerate(prem_pool)}
    vocab, df = build_vocab(pairs, prem_pool)
    n = len(pairs) or 1
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    Pfeat = np.stack([featvec(p, vocab) for p in prem_pool])
    # split goals
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pairs)); ntest = max(20, len(pairs) // 5)
    test_idx, train_idx = order[:ntest], order[ntest:]
    # train edges (goal feat, premise feat)
    eg, ep = [], []
    for gi in train_idx:
        gf = featvec(pairs[gi]["statement"], vocab)
        for pr in pairs[gi]["premises"]:
            eg.append(gf); ep.append(Pfeat[pidx[pr]])
    m = train_jepa(eg, ep, len(vocab), seed=seed)
    m.eval()
    with torch.no_grad():
        Z_prem = m.embed_prem(torch.tensor(Pfeat)).numpy()
    # CONTROL: premise popularity prior (train citation frequency, same ranking for every goal)
    freq = np.zeros(len(prem_pool))
    for gi in train_idx:
        for pr in pairs[gi]["premises"]: freq[pidx[pr]] += 1
    frank = np.argsort(-freq)
    ks = [5, 10, 20]
    jep = {k: [] for k in ks}; lex = {k: [] for k in ks}; pop = {k: [] for k in ks}
    for gi in test_idx:
        cited = {pidx[p] for p in pairs[gi]["premises"]}
        gf = featvec(pairs[gi]["statement"], vocab)
        with torch.no_grad():
            zg = m.embed_goal(torch.tensor(gf[None]))[0].numpy()
        jrank = np.argsort(-(Z_prem @ zg))
        lrank = np.argsort(-lexical_scores(pairs[gi]["statement"], prem_pool, idf))
        rj = recall_at_k(jrank, cited, ks); rl = recall_at_k(lrank, cited, ks); rp = recall_at_k(frank, cited, ks)
        for k in ks: jep[k].append(rj[k]); lex[k].append(rl[k]); pop[k].append(rp[k])
    agg = lambda d: {k: float(np.mean(v)) for k, v in d.items()}
    return agg(jep), agg(lex), agg(pop), len(test_idx), len(prem_pool)


def main():
    print("=== proofworld.jepa :: JEPA premise retriever vs lexical baseline (held-out recall@k) ===\n")
    seeds = [0, 1, 2]
    jacc = {k: [] for k in (5, 10, 20)}; lacc = {k: [] for k in (5, 10, 20)}; pacc = {k: [] for k in (5, 10, 20)}
    ntest = npool = 0
    for s in seeds:
        j, l, p, ntest, npool = run(s)
        for k in (5, 10, 20): jacc[k].append(j[k]); lacc[k].append(l[k]); pacc[k].append(p[k])
        print(f"  seed {s}:  JEPA r@10={j[10]:.3f}  |  lexical={l[10]:.3f}  |  popularity={p[10]:.3f}")
    print(f"\n  held-out goals/seed: {ntest}   candidate premises: {npool}   (random r@10 ~ {10/npool:.3f})\n")
    print(f"  {'metric':10} {'JEPA':>10} {'lexical':>10} {'popularity':>12} {'winner':>10}")
    for k in (5, 10, 20):
        jm, lm, pm = np.mean(jacc[k]), np.mean(lacc[k]), np.mean(pacc[k])
        best = max([("JEPA", jm), ("lexical", lm), ("popularity", pm)], key=lambda x: x[1])[0]
        print(f"  recall@{k:<4} {jm:>10.3f} {lm:>10.3f} {pm:>12.3f} {best:>10}")
    jm, lm, pm = np.mean(jacc[10]), np.mean(lacc[10]), np.mean(pacc[10])
    strong = jm > pm + 0.03 and jm > lm + 0.03
    verdict = ("JEPA HELPS, and it is GOAL-CONDITIONED -- it beats BOTH lexical AND the popularity prior, so it is "
               "learning which lemmas suit THIS goal, not just citing common ones" if strong else
               "JEPA ~ popularity prior -- the 'win' is mostly learning which lemmas are commonly cited (still useful, "
               "but not goal-specific)" if abs(jm - pm) <= 0.03 else
               "mixed -- see per-metric winners")
    print(f"\n  VERDICT @recall10: {verdict}")
    print("  NOTE: JEPA only RANKS premises; the Lean/z3 kernel still owns truth. Helpful = saves kernel calls, never")
    print("  changes correctness. Small-data caveat pre-registered: 359 goals is tiny for learning a joint embedding.")


if __name__ == "__main__":
    main()
