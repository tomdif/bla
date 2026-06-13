#!/usr/bin/env python3
"""language_hypothesis_affordance_gate: the FIRST learned fusion seam. Language proposes a
TYPED AffordanceHypothesis (not raw latent influence); the Δachievable verifier owns truth.

The seam (per project_language_physics_fusion): language PROPOSES the finer causal key that the
alias-aware agent otherwise has to DISCOVER by active probing -> language saves probes. But the
verifier still owns the verdict: a wrong/shuffled hypothesis is rejected by an actual Δachievable
probe and the agent falls back to search, so language NEVER decides physical truth.

World = the hardened aliased episode: switches & decoys are VISUALLY IDENTICAL (all yellow),
separated only by the causal feature adj_wall (switch sits next to a wall/door). near_top is a
present-but-non-causal distractor (constant here). Goal behind 2 doors opened by the 2 switches.

AGENTS:
  no_lang            -- alias-aware feature search (baseline): probes to discover adj_wall, then transfers
  lang_verify        -- language proposes adj_wall=True->switch; VERIFY by 1 probe; accept iff Δ>0, else fall back
  lang_only          -- trust language, NO verifier (transfer immediately)
  oracle             -- knows the switches (upper bound)
SHUFFLED language = a decorrelated proposal (adj_wall=False -> "decoys are switches").

THE LOAD-BEARING CONTROL: shuffled language must DESTROY the advantage. Language may only help
when it is actually correlated with the episode.

PASS: lang_verify(correct) solves with FEWER actions than no_lang | lang_verify(SHUFFLED) gets NO
action savings (verifier rejects, falls back) | lang_only(correct) solves | lang_only(SHUFFLED)
FAILS (over-trusts, no verifier) | accepted hypotheses have actual Δachievable>0 | shuffled proposal
REJECTED by the verifier | oracle solves | no_lang baseline solves. Self-contained, stdlib only.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque

H = W = 11
FLOORC, WALLC, YELLOW, GOALC = 0, 1, 2, 4
FEATS = ("adj_wall", "near_top")
WALLS = {(r, 3) for r in range(H)} | {(r, 6) for r in range(H)}
AGENT, GOAL = (5, 0), (5, 10)
SWITCH_DOOR = {(5, 2): (5, 3), (1, 2): (5, 6)}                   # near-door switch (5,2) nearest agent -> gives Δ>0 first
DECOYS = [(2, 0), (8, 0)]                                        # adj_wall=F; same appearance as switches


@dataclass
class AffordanceHypothesis:                                      # the TYPED seam (auditable, rejectable)
    proposed_key: tuple                                          # (feature, value), e.g. ("adj_wall", True)
    role: str
    predicted_effect: str
    confidence: float
    provenance: str


def language_propose(correct):
    if correct:
        return AffordanceHypothesis(("adj_wall", True), "switch_candidate", "increase_reachability", 0.7,
                                    "yellow objects adjacent to a wall/door often act as switches")
    return AffordanceHypothesis(("adj_wall", False), "switch_candidate", "increase_reachability", 0.7,
                                "SHUFFLED: proposal decorrelated from the episode")


def adj_wall(cell): return any((cell[0] + d[0], cell[1] + d[1]) in WALLS for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
def cands_of():
    out = []
    for cell in list(SWITCH_DOOR) + DECOYS:
        out.append({"cell": cell, "color": "yellow",
                    "feat": {"adj_wall": adj_wall(cell), "near_top": cell[0] < 5}})
    return out
def reach_size(opened):
    passable = lambda n: (n not in WALLS) or (n in opened)
    seen = {AGENT}; q = deque([AGENT])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and passable(n):
                seen.add(n); q.append(n)
    return len(seen), GOAL in seen
def dist(c): return abs(c["cell"][0] - AGENT[0]) + abs(c["cell"][1] - AGENT[1])
def find_separating(evidence):
    for fk in FEATS:
        g = {}
        for feat, pos in evidence: g.setdefault(feat[fk], set()).add(pos)
        if len(g) > 1 and all(len(v) == 1 for v in g.values()): return fk
    return None


def run(mode, correct=True):
    cands = cands_of(); opened = set(); toggled = []; actions = [0]; evidence = []
    accepted = None; rejected = False; acc_delta = None

    def toggle(c):
        before = reach_size(opened)[0]
        if c["cell"] in SWITCH_DOOR: opened.add(SWITCH_DOOR[c["cell"]])
        d = reach_size(opened)[0] - before; toggled.append(c["cell"]); actions[0] += 1
        evidence.append((c["feat"], d > 0)); return d

    def search():                                               # alias-aware active disambiguation, then transfer
        un = [c for c in cands if c["cell"] not in toggled]
        if not un: return
        toggle(min(un, key=dist))                              # nearest
        un = [c for c in cands if c["cell"] not in toggled]
        pf = [e[0] for e in evidence]
        cc = [c for c in un if min(sum(c["feat"][k] != p[k] for k in FEATS) for p in pf) > 0]
        if cc:                                                  # min feature-diff -> isolate the causal variable
            toggle(min(cc, key=lambda c: (min(sum(c["feat"][k] != p[k] for k in FEATS) for p in pf), dist(c))))
        rk = find_separating([(e[0], e[1]) for e in evidence])
        val = next((e[0][rk] for e in evidence if e[1]), None) if rk else None
        if rk and val is not None:
            for c in cands:
                if c["cell"] not in toggled and c["feat"][rk] == val: toggle(c)

    if mode == "oracle":
        for c in cands:
            if c["cell"] in SWITCH_DOOR: toggle(c)
    else:
        hyp = language_propose(correct) if mode in ("lang_verify", "lang_only") else None
        if hyp:
            feat, val = hyp.proposed_key
            positives = [c for c in cands if c["feat"][feat] == val]
            if mode == "lang_only":
                accepted = (feat, val)                          # trust, NO verification
            elif positives:                                     # lang_verify: 1 probe, accept iff predicted effect fires
                d = toggle(min(positives, key=dist))
                if d > 0: accepted = (feat, val); acc_delta = d
                else: rejected = True                           # predicted increase but Δ=0 -> reject the proposal
        if accepted:
            feat, val = accepted
            for c in cands:
                if c["cell"] not in toggled and c["feat"][feat] == val: toggle(c)
        else:
            search()                                            # baseline (no_lang) OR fall back after rejection

    return {"mode": mode + ("" if correct else "/shuffled"), "solved": reach_size(opened)[1],
            "actions": actions[0], "accepted_key": accepted, "rejected": rejected, "acc_delta": acc_delta}


R = {"no_lang": run("no_lang"),
     "lang_verify(correct)": run("lang_verify", True), "lang_verify(SHUFFLED)": run("lang_verify", False),
     "lang_only(correct)": run("lang_only", True), "lang_only(SHUFFLED)": run("lang_only", False),
     "oracle": run("oracle")}
print("=== language_hypothesis_affordance_gate: typed proposals, Δachievable owns truth ===\n")
for name, r in R.items():
    print(f"  {name:22} solved={str(r['solved']):5} actions={r['actions']} accepted={r['accepted_key']} "
          f"rejected={r['rejected']} acc_delta={r['acc_delta']}")
print()

base = R["no_lang"]["actions"]
checks = {
    "no_lang baseline solves": R["no_lang"]["solved"],
    "lang_verify(correct) solves with FEWER actions than no_lang":
        R["lang_verify(correct)"]["solved"] and R["lang_verify(correct)"]["actions"] < base,
    "SHUFFLE KILLS ADVANTAGE: lang_verify(shuffled) gets NO action savings":
        R["lang_verify(SHUFFLED)"]["actions"] >= base,
    "accepted hypothesis had actual Δachievable>0": (R["lang_verify(correct)"]["acc_delta"] or 0) > 0,
    "shuffled proposal REJECTED by the verifier": R["lang_verify(SHUFFLED)"]["rejected"],
    "lang_only(correct) solves (language has real signal)": R["lang_only(correct)"]["solved"],
    "lang_only(SHUFFLED) FAILS (over-trusts, no verifier)": not R["lang_only(SHUFFLED)"]["solved"],
    "verifier makes lang ROBUST: lang_verify(shuffled) still solves (fell back)": R["lang_verify(SHUFFLED)"]["solved"],
    "oracle solves (upper bound)": R["oracle"]["solved"],
}
print("=== language-seam pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nLANGUAGE HYPOTHESIS AFFORDANCE GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print(f"VERDICT: language fused as TYPED hypotheses, NOT physical truth. Correct language saves a probe "
      f"({R['lang_verify(correct)']['actions']} vs {base} actions) by proposing the causal key;\n  the Δachievable "
      f"verifier confirms it (Δ={R['lang_verify(correct)']['acc_delta']}). SHUFFLED language is REJECTED by the "
      f"verifier and the agent falls back to search ({R['lang_verify(SHUFFLED)']['actions']} actions = NO advantage) "
      f"-- so the\n  benefit is real fusion, not decoration. Without the verifier, shuffled language is catastrophic "
      f"(lang_only/shuffled FAILS). Language proposes; physics verifies; the seam is auditable and shrinks as controls pass.")
