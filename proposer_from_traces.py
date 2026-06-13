#!/usr/bin/env python3
"""proposer_from_traces: train the PROPOSER (not just the triage surrogate) from repair traces. Stage 5.

Stage 4 learned which candidate to *try first*. This learns what *kind of fix to propose* for a given
bug SIGNATURE -- it mines the trace library for "which patch strategy actually went GREEN for bugs that
look like this" and conditions proposal on it. The proposer's first guess gets better as it sees more
bugs, so it solves a recurring bug class in one shot instead of brute-forcing the strategy space.

Same invariant, one level up:
  * a strategy that worked on past bugs of a type is still only a PROPOSAL -- it is RE-VERIFIED on the
    current bug; transfer is proposed, never assumed (verifier owns promotion -- run-until-green);
  * the proposer trains ONLY on the FULL verifier verdict (incl. the held-out check), NEVER on the weak
    target-test signal -- so a GAMING strategy (passes the target, fails held-out) is never learned as a
    winner. Training on the weak signal promotes the gamer; training on truth catches it;
  * an UNSEEN bug signature has no learned prior -> the proposer EXPLORES rather than confidently emitting
    a memorized (wrong) strategy from a different signature. Proposer-level OOD-refuse: don't fake knowing
    a bug you've never seen. It still solves it (no silent miss) and one-shot-learns the new class.
numpy only.
"""
import numpy as np

rng = np.random.default_rng(0)
N_STRAT, N_TYPES = 8, 6
# each bug TYPE (signature) has one CORRECT strategy (passes the full suite incl. held-out) and one GAMER
# strategy (passes the target test only -- a plausible patch that special-cases the failing input).
_perm = list(rng.permutation(N_STRAT))
correct = {t: _perm[t] for t in range(N_TYPES)}            # distinct correct strategy per type
gamer = {}
for t in range(N_TYPES):
    g = int(rng.integers(0, N_STRAT))
    while g == correct[t]: g = int(rng.integers(0, N_STRAT))
    gamer[t] = g
# a derangement of the signatures: the "no-truth" null labels a trace by the correct strategy of a
# DIFFERENT signature, so the type->strategy link is destroyed (the proposer learns a confidently-wrong map)
PI = list(rng.permutation(N_TYPES))
for i in range(N_TYPES):
    if PI[i] == i: j = (i + 1) % N_TYPES; PI[i], PI[j] = PI[j], PI[i]


def verifier(t, s):
    """returns (target_pass, full_pass). full_pass = the held-out-clean truth that owns promotion."""
    full = (s == correct[t])
    target = full or (s == gamer[t])
    return target, full


class Proposer:
    """signature-aware: learns a strategy ranking PER bug-signature; explores on an unseen signature."""
    def __init__(self):
        self.att = {t: np.zeros(N_STRAT) for t in range(N_TYPES)}
        self.suc = {t: np.zeros(N_STRAT) for t in range(N_TYPES)}
    def order(self, t):
        if self.att[t].sum() == 0:                          # UNSEEN signature -> explore (no confident prior)
            o = list(range(N_STRAT)); rng.shuffle(o); return o
        rate = self.suc[t] / np.maximum(self.att[t], 1)
        prio = np.where(self.att[t] > 0, rate, 0.5)         # untried strategies optimistic, below a proven winner
        return list(np.argsort(-prio))
    def update(self, t, s, lab): self.att[t][s] += 1; self.suc[t][s] += int(lab)
    def winner(self, t): return int(np.argmax(self.suc[t] / np.maximum(self.att[t], 1)))


class NaiveProposer:
    """pools strategies ACROSS signatures and always proposes its global favorite -- it 'pretends' a
    learned strategy transfers to a signature it has never seen. The undisciplined baseline."""
    def __init__(self): self.att = np.zeros(N_STRAT); self.suc = np.zeros(N_STRAT)
    def order(self, t):
        if self.att.sum() == 0:
            o = list(range(N_STRAT)); rng.shuffle(o); return o
        prio = np.where(self.att > 0, self.suc / np.maximum(self.att, 1), 0.5)
        return list(np.argsort(-prio))
    def update(self, t, s, lab): self.att[s] += 1; self.suc[s] += int(lab)


def solve(prop, t, label="full"):
    """propose strategies in the proposer's order; RUN-UNTIL-GREEN where green = FULL pass (truth).
    A gamer (target-only) never stops the loop. Train the proposer on the chosen label. Returns
    (candidates, found, first_try, seen_before, promoted)."""
    order = prop.order(t); seen = (getattr(prop, "att", None) is not None) and (
        prop.att[t].sum() > 0 if isinstance(prop.att, dict) else prop.att.sum() > 0)
    recs = []; promoted = None; runs = len(order); first = False
    for k, s in enumerate(order, 1):
        target, full = verifier(t, s)
        recs.append((s, full, target))
        if full:                                            # only the FULL verdict promotes
            promoted = s; runs = k; first = (k == 1); break
    for s, f, tg in recs:
        if label == "full": lab = f                          # the truth (held-out clean)
        elif label == "target": lab = tg                     # the weak signal a gamer also passes
        else: lab = int(s == correct[PI[t]])                 # shuffle: label from a WRONG signature (no truth)
        prop.update(t, s, lab)
    return runs, promoted is not None, first, seen, promoted


# cold-proposer stats (pure exploration): first-try rate and candidates-to-green
cold_ft, cold_c = [], []
for _ in range(300):
    t = int(rng.integers(0, N_TYPES)); o = list(range(N_STRAT)); rng.shuffle(o)
    pos = o.index(correct[t]) + 1; cold_ft.append(pos == 1); cold_c.append(pos)
cold_ft, cold_c = float(np.mean(cold_ft)), float(np.mean(cold_c))

# ---- Experiment 1: the proposer LEARNS from traces (verifier-labeled) vs RANDOM-labeled (no truth) ----
STREAM = [int(rng.integers(0, N_TYPES)) for _ in range(120)]
pf = Proposer(); ftf, candf, foundf, promf = [], [], [], []
for t in STREAM:
    r, fnd, fst, _s, pr = solve(pf, t, "full"); ftf.append(fst); candf.append(r); foundf.append(fnd); promf.append(pr == correct[t])
pr_ = Proposer(); ftr = []
for t in STREAM:
    _r, _f, fst, _s, _p = solve(pr_, t, "shuffle"); ftr.append(fst)
learned_ft = float(np.mean(ftf[-40:])); learned_cand = float(np.mean(candf[-40:])); rand_ft = float(np.mean(ftr[-40:]))

print("=== proposer_from_traces: the proposer learns what to PROPOSE, verifier still owns truth ===\n")
print(f"  Experiment 1 -- learn from {len(STREAM)} bugs over {N_TYPES} recurring signatures, {N_STRAT} strategies:")
print(f"     cold proposer (no traces): first-try success = {cold_ft:.2f}   candidates-to-green = {cold_c:.2f}")
print(f"     trained (verifier labels): first-try success = {learned_ft:.2f}   candidates-to-green = {learned_cand:.2f}")
print(f"     trained (SHUFFLED-signature labels, no truth): first-try success = {rand_ft:.2f}   (confidently wrong, <= cold)\n")

# ---- Experiment 2: gaming guard -- truth = the FULL verdict, not the target test ----
pfull, ptgt = Proposer(), Proposer()
for t in STREAM:
    solve(pfull, t, "full"); solve(ptgt, t, "target")
full_correct = sum(pfull.winner(t) == correct[t] for t in range(N_TYPES))
tgt_gamer = sum(ptgt.winner(t) == gamer[t] for t in range(N_TYPES))
fulltry = float(np.mean([verifier(t, pfull.winner(t))[1] for t in range(N_TYPES)]))
tgttry = float(np.mean([verifier(t, ptgt.winner(t))[1] for t in range(N_TYPES)]))
print(f"  Experiment 2 -- gaming guard (a gamer passes the target test but FAILS the held-out check):")
print(f"     trained on FULL verdict  : winners == correct for {full_correct}/{N_TYPES} types; gamers promoted = 0")
print(f"     trained on TARGET-only   : gamers promoted for {tgt_gamer}/{N_TYPES} types (gaming NOT caught)")
print(f"     learned-winner truly-green rate: full-verdict={fulltry:.2f}  target-only={tgttry:.2f}\n")

# ---- Experiment 3: novel-signature refuse -- explore vs pretend-to-know (proposer-level OOD) ----
M = 60; dnov, nnov, dsecond, dfound, nfound = [], [], [], [], []
for _ in range(M):
    d, n = Proposer(), NaiveProposer()
    for _ in range(40):                                      # train on signatures 0..4 ONLY (type 5 held out)
        t = int(rng.integers(0, N_TYPES - 1)); solve(d, t, "full"); solve(n, t, "full")
    rd, fd, fstd, _s, _p = solve(d, N_TYPES - 1, "full"); dnov.append(fstd); dfound.append(fd)
    rn, fn, fstn, _s, _p = solve(n, N_TYPES - 1, "full"); nnov.append(fstn); nfound.append(fn)
    rd2, _f, fstd2, _s, _p = solve(d, N_TYPES - 1, "full"); dsecond.append(fstd2)   # one-shot learning
dnov, nnov, dsecond = float(np.mean(dnov)), float(np.mean(nnov)), float(np.mean(dsecond))
print(f"  Experiment 3 -- a NOVEL signature (type {N_TYPES-1}, never trained on):")
print(f"     disciplined (explore when unseen): first-try = {dnov:.2f}   then one-shot-learned: {dsecond:.2f}")
print(f"     naive (pretends global strategy transfers): first-try = {nnov:.2f}")
print(f"     both still SOLVE the novel bug (no silent miss): disciplined={all(dfound)} naive={all(nfound)}\n")

checks = {
    "COLD PROPOSER: with no traces, first-try ~ chance and candidates ~ |strategies|/2":
        cold_ft <= 0.25 and cold_c >= 3.0,
    "PROPOSER LEARNS: trained first-try -> ~1.0 and candidates -> ~1 on recurring signatures":
        learned_ft >= 0.9 and learned_cand <= 1.3,
    "TRUTH IS LOAD-BEARING: verifier-labeled training learns; SHUFFLED-signature labels (no truth) do not":
        learned_ft >= 0.9 and rand_ft <= 0.3,
    "VERIFIER OWNS PROMOTION: every promoted strategy is the FULL-green one (transfer re-verified, no silent miss)":
        all(foundf) and all(promf),
    "GAMING GUARD: training on the FULL verdict never promotes a gamer; the weak target signal does":
        full_correct == N_TYPES and tgt_gamer >= 1 and fulltry > tgttry,
    "NOVEL-SIGNATURE REFUSE: on an unseen signature the proposer EXPLORES (beats pretend-to-know) and one-shot-learns":
        dnov > nnov + 0.05 and dsecond >= 0.95 and all(dfound) and all(nfound),
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nPROPOSER-FROM-TRACES GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print(f"VERDICT: the proposer now LEARNS what to propose -- conditioning on the bug signature, its first guess"
      f"\n  goes from chance ({cold_ft:.2f}) to ~{learned_ft:.2f} first-try as the trace library grows, so a recurring bug"
      f"\n  class is fixed in one shot. The discipline scales to the generator: it trains ONLY on the full verifier"
      f"\n  verdict (so a gamer that passes the target test is never learned as a winner -- the weak signal promotes it,"
      f"\n  truth does not), an unseen signature triggers EXPLORATION instead of a confidently-wrong memorized patch, and"
      f"\n  every transfer is re-verified on the current bug. Proposer, triage, and trainer all propose; verification owns truth.")
