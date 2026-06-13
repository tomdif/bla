#!/usr/bin/env python3
"""lean_proposer_from_traces: the trace-trained PROPOSER (Stages 3-5) on a REAL Lean 4 verifier.

Brings the learned-proposer discipline to math: the proposer learns which TACTIC closes which GOAL-SHAPE
from verifier-labeled traces, so its first-try success on a recurring goal-shape rises from chance to ~1.0.
The verifier is the actual Lean 4 type checker. Lean is deterministic, so every distinct (goal-shape,
tactic) pair is checked by REAL Lean exactly once and memoized -- real verification, no wasted runs -- and
the learning loop replays over that cache.

The discipline scales to math, with a domain-native gaming attack:
  * `by sorry` TYPECHECKS for every goal but `#print axioms` flags sorryAx -- so truth = typecheck AND
    clean axioms. Trained on the full verdict, `sorry` is never learned as a winner; trained on the weak
    typecheck-only signal, it IS promoted (it "closes" everything) -- the axiom audit is the gaming guard;
  * a tactic that worked on past goals of a shape is still RE-VERIFIED on the current goal (no silent miss);
  * an UNSEEN goal-shape triggers exploration, not a confidently-wrong memorized tactic (proposer-level OOD);
  * training on shuffled-shape labels (no true shape->tactic link) makes the proposer confidently wrong.
numpy only (plus the real `lean` binary).
"""
import os, shutil, tempfile, subprocess
import numpy as np

rng = np.random.default_rng(0)
LEAN = shutil.which("lean") or os.path.expanduser("~/.elan/bin/lean")

SHAPES = {
    "zero_add": "theorem t (n : Nat) : 0 + n = n :=",
    "list_app": "theorem t : [1,2] ++ [3] = [1,2,3] :=",
    "truth":    "theorem t : True :=",
    "lt":       "theorem t : (5 : Nat) < 9 :=",
    "add_comm": "theorem t (a b : Nat) : a + b = b + a :=",
    "succ":     "theorem t (n : Nat) : n + 1 = Nat.succ n :=",
}
# `by sorry` is index 0 so that under the WEAK typecheck-only signal it WINS ties against real proofs
# (it typechecks for every goal) -- exposing the gaming vulnerability the full axiom audit must close.
TACTICS = ["by sorry", "by rfl", "by omega", "by simp", "by decide", "by trivial",
           "by constructor", "by assumption"]
GAMER = 0
NAMES = list(SHAPES)


def run_lean(decl, proof):
    d = tempfile.mkdtemp(); f = os.path.join(d, "T.lean")
    open(f, "w").write(f"{decl} {proof}\n#print axioms t\n")
    r = subprocess.run([LEAN, f], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    compile_ok = (r.returncode == 0) and ("error:" not in out)     # typechecks (the weak, gameable signal)
    full = compile_ok and ("sorryAx" not in out)                   # + clean axioms = the truth
    return compile_ok, full


# ---- build the REAL Lean verifier cache: every (shape, tactic) checked once, memoized ----
print("=== lean_proposer_from_traces: trace-trained tactic proposer on a REAL Lean verifier ===\n")
print(f"  building verifier cache with real Lean ({len(SHAPES)}x{len(TACTICS)} = {len(SHAPES)*len(TACTICS)} checks)...")
FULL = {s: np.zeros(len(TACTICS), bool) for s in NAMES}
COMP = {s: np.zeros(len(TACTICS), bool) for s in NAMES}
for s in NAMES:
    for ti, tac in enumerate(TACTICS):
        c, fp = run_lean(SHAPES[s], tac); COMP[s][ti] = c; FULL[s][ti] = fp
closers = {s: [ti for ti in range(len(TACTICS)) if FULL[s][ti]] for s in NAMES}
print(f"  real full-proof closers per shape (the verifier's truth; `sorry` is index {GAMER}, never a closer):")
for s in NAMES: print(f"     {s:9} -> {[TACTICS[ti].replace('by ','') for ti in closers[s]]}")
assert all(closers[s] for s in NAMES), "every shape must have a real closer"


def verifier_full(s, ti):   return bool(FULL[s][ti])      # Lean accepts AND clean axioms (truth)
def verifier_target(s, ti): return bool(COMP[s][ti])      # Lean accepts (sorry passes this -> gaming)


class Proposer:
    """signature(shape)-aware tactic ranker; explores on an unseen shape."""
    def __init__(self): self.att = {s: np.zeros(len(TACTICS)) for s in NAMES}; self.suc = {s: np.zeros(len(TACTICS)) for s in NAMES}
    def order(self, s):
        if self.att[s].sum() == 0:
            o = list(range(len(TACTICS))); rng.shuffle(o); return o
        rate = self.suc[s] / np.maximum(self.att[s], 1)
        return list(np.argsort(-np.where(self.att[s] > 0, rate, 0.5)))
    def update(self, s, ti, lab): self.att[s][ti] += 1; self.suc[s][ti] += int(lab)
    def winner(self, s): return int(np.argmax(self.suc[s] / np.maximum(self.att[s], 1)))


def solve(prop, s, shuffle_map=None, label="full"):
    """propose tactics in order; RUN-UNTIL-GREEN where green = a real full proof (Lean accept + clean axioms).
    `sorry` (typechecks, sorryAx) never stops the loop. Train on the chosen label."""
    order = prop.order(s); recs = []; promoted = None; runs = len(order); first = False
    for k, ti in enumerate(order, 1):
        recs.append(ti)
        if verifier_full(s, ti): promoted = ti; runs = k; first = (k == 1); break
    for ti in recs:
        if label == "full":     lab = verifier_full(s, ti)
        elif label == "target": lab = verifier_target(s, ti)            # weak signal: a sorry passes this
        else:                   lab = verifier_full(shuffle_map[s], ti)  # shuffled shape: no true link
        prop.update(s, ti, lab)
    return runs, promoted, first


# ---- cold-proposer stats (pure exploration over the real closers) ----
cold_ft, cold_c = [], []
for _ in range(400):
    s = NAMES[int(rng.integers(0, len(NAMES)))]; o = list(range(len(TACTICS))); rng.shuffle(o)
    pos = min(o.index(ti) for ti in closers[s]) + 1; cold_ft.append(pos == 1); cold_c.append(pos)
cold_ft, cold_c = float(np.mean(cold_ft)), float(np.mean(cold_c))

# ---- Experiment 1: the proposer LEARNS (full labels) vs SHUFFLED-shape labels (no truth) ----
STREAM = [NAMES[int(rng.integers(0, len(NAMES)))] for _ in range(150)]
SHUF = {s: NAMES[(NAMES.index(s) + 2) % len(NAMES)] for s in NAMES}     # deranged shape map
pf = Proposer(); ftf, candf, okf = [], [], []
for s in STREAM:
    r, pr, fst = solve(pf, s, label="full"); ftf.append(fst); candf.append(r); okf.append(pr is not None and FULL[s][pr])
ps = Proposer(); fts = []
for s in STREAM:
    _r, _p, fst = solve(ps, s, shuffle_map=SHUF, label="shuffle"); fts.append(fst)
learned_ft, learned_c, shuf_ft = float(np.mean(ftf[-40:])), float(np.mean(candf[-40:])), float(np.mean(fts[-40:]))

# ---- Experiment 2: gaming guard -- full verdict vs typecheck-only (a sorry passes typecheck everywhere) ----
pfull, ptgt = Proposer(), Proposer()
for s in STREAM: solve(pfull, s, label="full"); solve(ptgt, s, label="target")
full_no_gamer = all(pfull.winner(s) != GAMER for s in NAMES)
tgt_gamer = sum(ptgt.winner(s) == GAMER for s in NAMES)
full_green = float(np.mean([FULL[s][pfull.winner(s)] for s in NAMES]))
tgt_green = float(np.mean([FULL[s][ptgt.winner(s)] for s in NAMES]))

# ---- Experiment 3: novel shape (add_comm, unique closer omega) held out -> explore vs pretend-to-know ----
# "pretend-to-know" = always apply the globally-best tactic (by real closer-count over training shapes),
# ignoring the signature. It is deterministic and confidently WRONG on a novel shape the favorite misses.
HELD = "add_comm"; TRAIN = [s for s in NAMES if s != HELD]
glob = np.array([sum(int(FULL[s][ti]) for s in TRAIN) for ti in range(len(TACTICS))])
naive_order = list(np.argsort(-glob)); naive_fav = naive_order[0]
naive_ft = 1.0 if FULL[HELD][naive_fav] else 0.0                # naive's confident first guess on the novel shape
naive_solves = any(FULL[HELD][ti] for ti in naive_order)        # run-until-green still finds a closer
M = 120; dnov, dsecond, dfound = [], [], []
for _ in range(M):
    d = Proposer()
    for _ in range(36):
        s = TRAIN[int(rng.integers(0, len(TRAIN)))]; solve(d, s, label="full")
    rd, pd, fd = solve(d, HELD, label="full"); dnov.append(fd); dfound.append(pd is not None)
    _r, _p, fd2 = solve(d, HELD, label="full"); dsecond.append(fd2)
dnov, nnov, dsecond = float(np.mean(dnov)), naive_ft, float(np.mean(dsecond))

print(f"\n  Experiment 1 -- learn over {len(STREAM)} goals, {len(NAMES)} recurring shapes, {len(TACTICS)} tactics:")
print(f"     cold (explore): first-try={cold_ft:.2f}  candidates-to-proof={cold_c:.2f}")
print(f"     trained (real Lean labels): first-try={learned_ft:.2f}  candidates-to-proof={learned_c:.2f}")
print(f"     trained (SHUFFLED-shape labels, no truth): first-try={shuf_ft:.2f}  (rides the general tactic, can't learn exceptions)")
print(f"  Experiment 2 -- gaming guard (`by sorry` typechecks for EVERY goal):")
print(f"     trained on FULL verdict: no shape's winner is `sorry` = {full_no_gamer}; winners truly-green={full_green:.2f}")
print(f"     trained on TYPECHECK-only: `sorry` promoted for {tgt_gamer}/{len(NAMES)} shapes; winners truly-green={tgt_green:.2f}")
print(f"  Experiment 3 -- novel shape '{HELD}' (unique closer: omega), held out of training:")
print(f"     disciplined (explore): first-try={dnov:.2f}  then one-shot-learned={dsecond:.2f}")
print(f"     naive (always proposes '{TACTICS[naive_fav].replace('by ','')}', the global favorite): first-try={nnov:.2f}"
      f"   both solved: d={all(dfound)} n={naive_solves}\n")

checks = {
    "PROPOSER LEARNS: trained first-try -> ~1.0 and candidates -> ~1, well above cold exploration":
        learned_ft >= 0.95 and learned_c <= 1.2 and learned_ft >= cold_ft + 0.25,
    "TRUTH IS LOAD-BEARING: real-Lean labels reach ~1.0; shuffled-shape labels plateau (can't learn the exceptions)":
        learned_ft >= 0.95 and shuf_ft <= 0.8 and (learned_ft - shuf_ft) >= 0.25,
    "VERIFIER OWNS PROMOTION: every promoted tactic is a real clean-axiom Lean proof; no goal left unsolved":
        all(okf) and all(p is not None for p in [solve(Proposer(), s, label="full")[1] for s in NAMES]),
    "AXIOM GUARD: the FULL verdict never promotes `sorry`; the typecheck-only signal does (gaming)":
        full_no_gamer and tgt_gamer >= 1 and full_green > tgt_green,
    "NOVEL-SHAPE REFUSE: unseen shape -> EXPLORE (beats confidently-wrong pretend-to-know), solved, one-shot-learn":
        (not FULL[HELD][naive_fav]) and dnov > nnov + 0.05 and dsecond >= 0.95 and all(dfound) and naive_solves,
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nLEAN PROPOSER-FROM-TRACES GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print(f"VERDICT: the trace-trained PROPOSER works on MATH against a real Lean checker -- it learns which tactic"
      f"\n  closes which goal-shape (first-try {cold_ft:.2f} -> {learned_ft:.2f}), so a recurring shape is proved in one shot."
      f"\n  The full discipline transferred unchanged: truth = the real Lean verdict (shuffled-shape labels plateau at"
      f"\n  general-tactic level and can't learn the exceptions), the axiom audit is the gaming guard (`sorry` typechecks"
      f"\n  everywhere but is never learned"
      f"\n  as a winner under the full verdict, only under the weak typecheck signal), an unseen goal-shape triggers"
      f"\n  exploration not a memorized tactic, and every promoted tactic is a re-verified real proof. Same loop, Lean truth.")
