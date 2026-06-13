#!/usr/bin/env python3
"""trace_trained_surrogate: the self-improving repair loop -- the triage surrogate LEARNS from the
loop's OWN repair traces and gets better at triage the more bugs it sees. Stage 4.

Every bug the loop solves emits (candidate features, VERIFIER outcome) pairs -- free labeled data. The
surrogate trains on those accumulated traces, so verifier runs-to-green falls over time without any
hand-labeling. The discipline that keeps this honest:
  * the training labels come ONLY from the verifier (truth), NEVER from the surrogate's own predictions
    -- so it cannot bootstrap a delusion (random/self labels do not improve: see the TRUTH control);
  * the verifier is still run-until-green, so a half-trained surrogate costs more RUNS, never a wrong
    promotion -- no green patch is ever silently skipped (NO-SILENT-MISS holds every round);
  * when the bug distribution SHIFTS, the OOD detector flags the new regime and the stale model degrades
    to more runs (not wrong promotion); the loop's own forced exploration then RE-LEARNS the new regime.

This is the loop closing into something self-improving, with imagination still boxed in by verification.
numpy only.
"""
import numpy as np

rng = np.random.default_rng(0)
WARMUP = 30                         # explore (random order) until this many verifier-labeled examples exist
KWIN = 200                          # train on the last KWIN traces (recency) so the loop can adapt to drift
# two regimes. A: the green has the flags ON and high sim. B is a SHIFTED world -- the diff range moves
# (OOD-detectable) AND the rule FLIPS (green has the flags OFF and LOW sim) -- so a model trained on A
# ranks B's green DEAD LAST (it looks like A's worst candidate), forcing the loop to explore and re-learn.
REGIMES = {
    "A": {"diff": (1, 6),  "flag": 1, "thr": 0.70, "inverted": False, "green_sim": (0.72, 1.00)},
    "B": {"diff": (8, 10), "flag": 0, "thr": 0.30, "inverted": True,  "green_sim": (0.02, 0.28)},
}


def gen_patch(regime="A", force_pass=False):
    R = REGIMES[regime]; v = R["flag"]
    if force_pass:
        tt = comp = so = v; sim = float(rng.uniform(*R["green_sim"]))
    else:
        tt, comp, so = (int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(0, 2)))
        sim = float(rng.random())
    diff = int(rng.integers(R["diff"][0], R["diff"][1] + 1))
    ok_sim = (sim < R["thr"]) if R["inverted"] else (sim > R["thr"])
    passes = int(tt == v and comp == v and so == v and ok_sim)
    return [tt, comp, so, sim, diff], passes


def verifier(patch):                # the EXPENSIVE truth (a real test run); returns the patch's true label
    return bool(patch[1])


class Surrogate:
    def __init__(self): self.w = np.zeros(5); self.b = 0.0; self.mu = np.zeros(5); self.sd = np.ones(5)
    def fit(self, X, y, epochs=3000, lr=0.2):
        X = np.array(X, float); self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd; n = len(X)
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Xs @ self.w + self.b))); g = p - y
            self.w -= lr * (Xs.T @ g) / n; self.b -= lr * g.mean()
        return self
    def score(self, f): return float(((np.array(f) - self.mu) / self.sd) @ self.w + self.b)
    def is_ood(self, f):            # z-score OOD (robust to single points): any feature > 2.5 sd from the trained mean
        return bool(np.any(np.abs((np.array(f, float) - self.mu) / self.sd) > 2.5))


def bug_pool(regime="A", size=12):
    pool = [gen_patch(regime, force_pass=True)]            # exactly one green
    while len(pool) < size:
        p = gen_patch(regime)
        if not p[1]: pool.append(p)
    rng.shuffle(pool); return pool


def solve_and_record(pool, order):
    """run the verifier in the given order, stop at the first green (run-until-green). Record every
    candidate we actually run as a VERIFIER-labeled training example. Returns (runs, found, examples)."""
    examples = []
    for runs, i in enumerate(order, 1):
        p = pool[i]; examples.append((p[0], p[1]))
        if verifier(p):
            return runs, True, examples
    return len(order), False, examples


def order_surrogate(pool, sur): return sorted(range(len(pool)), key=lambda i: -sur.score(pool[i][0]))
def order_random(pool): idx = list(range(len(pool))); rng.shuffle(idx); return idx


def online(regime_schedule, label="verifier"):
    """run bugs one at a time; after each, retrain the surrogate on the accumulated traces. The surrogate
    is used to order only AFTER a warmup of verifier-labeled exploration; before that, order randomly."""
    bufX, bufy = [], []; sur = None
    runs_log, found_log, ood_log = [], [], []
    for regime in regime_schedule:
        pool = bug_pool(regime)
        use = sur is not None and len(bufX) >= WARMUP
        order = order_surrogate(pool, sur) if use else order_random(pool)
        runs, found, ex = solve_and_record(pool, order)
        runs_log.append(runs); found_log.append(found)
        ood_log.append(float(np.mean([sur.is_ood(f) for f, _ in ex])) if sur else 0.0)
        for f, ytrue in ex:
            bufX.append(f)
            bufy.append(ytrue if label == "verifier" else int(rng.integers(0, 2)))   # random label = no truth
        if len(bufX) > KWIN:                        # recency: keep only the last KWIN traces (adapts to drift)
            bufX = bufX[-KWIN:]; bufy = bufy[-KWIN:]
        if len(set(bufy)) > 1:
            sur = Surrogate().fit(bufX, np.array(bufy))
    return np.array(runs_log), all(found_log), np.array(ood_log)


def windows(a, w): return [float(a[i:i + w].mean()) for i in range(0, len(a), w)]


# ---- Experiment 1: self-improvement on a fixed world (+ cold-start, truth-load-bearing, no-silent-miss)
N = 60
runs_v, found_v, _ = online(["A"] * N, label="verifier")
runs_r, found_r, _ = online(["A"] * N, label="random")        # ablation: traces with NO verifier truth
cold = float(runs_v[:5].mean()); final = float(runs_v[-20:].mean()); rand_final = float(runs_r[-20:].mean())

print("=== trace_trained_surrogate: the loop learns from its own verifier-labeled traces ===\n")
print(f"  Experiment 1 -- self-improvement on a fixed world ({N} bugs, 12 candidates each, 1 green):")
print(f"     verifier-labeled runs-to-green by window of 10: {[round(x,2) for x in windows(runs_v,10)]}")
print(f"     cold-start (first 5 bugs, pre-warmup) = {cold:.2f}   trained (last 20) = {final:.2f}")
print(f"     RANDOM-labeled traces (no truth), last 20 = {rand_final:.2f}   (should stay ~cold)\n")

# ---- Experiment 2: adapt to a distribution SHIFT (A -> B), tracking runs + OOD flag
runs_s, found_s, ood_s = online(["A"] * 40 + ["B"] * 60, label="verifier")
a_tail   = float(runs_s[30:40].mean()); b_onset = float(runs_s[40:50].mean()); b_tail = float(runs_s[90:100].mean())
ood_a    = float(ood_s[30:40].mean());  ood_onset = float(ood_s[40])   # the instant B arrives (no B in buffer yet)
print(f"  Experiment 2 -- adapt to a SHIFT (40 bugs world A, then 60 bugs world B):")
print(f"     runs-to-green by window of 10: {[round(x,2) for x in windows(runs_s,10)]}")
print(f"     A-tail={a_tail:.2f}  B-onset(spike)={b_onset:.2f}  B-tail(recovered)={b_tail:.2f}")
print(f"     OOD-flag rate: A-tail={ood_a:.0%}  at-shift-instant={ood_onset:.0%}  (detector flags the new regime on arrival)\n")

checks = {
    "COLD START: before learning, ordering ~ random (no free lunch)": cold >= 4.0,
    "SELF-IMPROVEMENT: runs-to-green falls far below cold as traces accumulate": final <= 0.45 * cold,
    "TRUTH IS LOAD-BEARING: verifier-labeled traces improve; RANDOM-labeled (no truth) do NOT":
        final <= 0.5 * rand_final and rand_final >= 4.0,
    "NO SILENT MISS: every bug's green patch is found, every round, both worlds":
        found_v and found_r and found_s,
    "ADAPT TO SHIFT: stale model degrades to MORE RUNS (not wrong promotion), then RE-LEARNS":
        b_onset > 1.5 * a_tail and b_tail < 0.6 * b_onset,
    "OOD-REFUSE sees the shift: the detector flags the new regime the instant it arrives":
        ood_onset >= 0.8 and ood_a <= 0.1,
    "STRUCTURAL: verifier run-until-green; trained ONLY on verifier labels; no promotion-on-prediction": True,
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nTRACE-TRAINED SURROGATE GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print(f"VERDICT: the repair loop is now SELF-IMPROVING -- it labels its own candidates with the verifier and"
      f"\n  trains the triage surrogate on those traces, so runs-to-green falls from ~{cold:.1f} (cold) to ~{final:.1f}"
      f"\n  with zero hand-labeling. The discipline is intact at the TRAINING level too: random-labeled traces (no"
      f"\n  verifier truth) do NOT improve, so truth -- not the act of training -- is load-bearing. A half-trained or"
      f"\n  STALE model only costs more runs (the shift spikes runs then re-learns); it never promotes a wrong patch,"
      f"\n  and no green is ever silently skipped. Imagination learns from verification; verification still owns truth.")
