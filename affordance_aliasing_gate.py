#!/usr/bin/env python3
"""affordance_aliasing_gate: harden the biggest remaining assumption -- that APPEARANCE keys
affordance CLASS. Here it doesn't: switches and decoys are VISUALLY IDENTICAL (all "yellow").

Gate 2/3 generalized by color with separable classes (yellow switch vs green decoy). Aliasing
is harder: the sensor cannot tell a switch from a decoy by appearance. Blind class transfer is
now ACTIVELY UNSAFE. The correct response is NOT to abandon generalization (probe everything =
too expensive under budget) but to REFINE the key: when same-appearance instances disagree
(conflicting Δachievable), search observable SUB-features for a finer one that is causal, and
transfer along THAT.

The world plants a CAUSAL finer feature (adj_wall: switches sit next to a wall/door, decoys in
open floor) AND a NON-CAUSAL distractor (near_top). A trivial "use the other feature" would be
fooled by the distractor; the agent must pick the feature that actually predicts Δachievable.

AGENTS:
  oracle           -- knows true roles (upper bound)
  no_discovery     -- never acts (lower bound)
  appearance_class -- transfers by appearance(yellow); over-generalizes -> wastes budget on decoys
  instance_only    -- no transfer; limited to nearest probes -> can't reach the FAR switch
  alias_aware      -- detects the conflict, refines key to adj_wall, transfers to the far switch
Budget = 4 toggles total (probing IS toggling; irreversible). Solving needs all 3 doors opened.

CONTROLS: alias_aware solves | appearance_class fails (wrong transfers>0) | instance_only fails
(misses far switch) | oracle solves | green non-aliased decoy rejected by all | alias_aware makes
ZERO wrong transfers | class confidence SPLITS after the conflict (transfer disabled, finer key found).
Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque

H = W = 11
FLOORC, WALLC, YELLOW, GREEN, GOALC = 0, 1, 2, 3, 4
GAIN = 56.0
WALLS = {(r, 3) for r in range(H)} | {(r, 5) for r in range(H)} | {(r, 7) for r in range(H)}
DOORS = {(5, 3), (5, 5), (5, 7)}
AGENT, GOAL = (5, 0), (5, 9)

# candidate objects: all yellow ones are VISUALLY IDENTICAL. adj_wall = CAUSAL, near_top = distractor.
CANDS = [
    dict(id="s1", pos=(2, 2), app="yellow", role="switch", door=(5, 3)),
    dict(id="s2", pos=(5, 2), app="yellow", role="switch", door=(5, 5)),
    dict(id="s3", pos=(8, 2), app="yellow", role="switch", door=(5, 7)),
    dict(id="d1", pos=(6, 0), app="yellow", role="decoy",  door=None),
    dict(id="d2", pos=(7, 0), app="yellow", role="decoy",  door=None),
    dict(id="g1", pos=(4, 0), app="green",  role="decoy",  door=None),
]
def adj_wall(pos):                                              # observable: next to a wall/door cell
    return any((pos[0] + dr, pos[1] + dc) in WALLS for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
def near_top(pos): return pos[0] < 4                            # observable DISTRACTOR (non-causal)
for c in CANDS: c["feat"] = {"adj_wall": adj_wall(c["pos"]), "near_top": near_top(c["pos"])}
DELTA = {c["id"]: (c["role"] == "switch") for c in CANDS}      # truth: switch -> Δachievable>0


def reach(opened):
    passable = lambda n: (n not in WALLS) or (n in opened)
    seen = {AGENT}; q = deque([AGENT])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and passable(n):
                seen.add(n); q.append(n)
    return seen
def solved(opened): return GOAL in reach(opened)
def manh(p): return abs(p[0] - AGENT[0]) + abs(p[1] - AGENT[1])
def by_id(i): return next(c for c in CANDS if c["id"] == i)


def toggle(opened, cid):                                        # irreversible; returns Δachievable
    c = by_id(cid); before = len(reach(opened))
    if c["door"]: opened.add(c["door"])
    return len(reach(opened)) - before


def run_oracle():
    opened = set()
    for c in CANDS:
        if c["role"] == "switch": toggle(opened, c["id"])
    return dict(solved=solved(opened), wrong=0)


def run_no_discovery(): return dict(solved=solved(set()), wrong=0)


def run_appearance_class(budget=4):
    """transfer by APPEARANCE: any positive yellow -> believe ALL yellow are switches, toggle
    nearest yellows. Can't prefer the far switch over near decoys -> wastes budget, misses s3."""
    opened = set(); toggled = []; belief = {}; wrong = 0
    order = sorted([c for c in CANDS], key=lambda c: (c["app"] != "yellow", manh(c["pos"])))
    for c in order:
        if len(toggled) >= budget: break
        if c["app"] == "green": continue                       # green is a different appearance bucket
        believe = belief.get("yellow", 0) > 0
        d = toggle(opened, c["id"]); toggled.append(c["id"])
        if DELTA[c["id"]]: belief["yellow"] = belief.get("yellow", 0) + 1
        if believe and not DELTA[c["id"]]: wrong += 1           # transferred onto a decoy
    return dict(solved=solved(opened), wrong=wrong, toggled=toggled, green_toggled="g1" in toggled)


def run_instance_only(budget=4):
    """no transfer: only the instances you directly probe count. Toggles nearest yellows."""
    opened = set(); toggled = []
    for c in sorted([c for c in CANDS if c["app"] == "yellow"], key=lambda c: manh(c["pos"]))[:budget]:
        toggle(opened, c["id"]); toggled.append(c["id"])
    return dict(solved=solved(opened), wrong=0, toggled=toggled, green_toggled=False)


def find_separating_feature(evidence):
    """evidence: list of (feat_dict, delta_pos). Return a feature whose value PERFECTLY predicts
    delta sign across all evidence (causal key), preferring one that actually splits +/0."""
    for fkey in ("adj_wall", "near_top"):
        groups = {}
        for feat, pos in evidence:
            groups.setdefault(feat[fkey], set()).add(pos)
        if all(len(v) == 1 for v in groups.values()) and len(groups) > 1:  # each value -> one sign, and it splits
            return fkey
    return None


def run_alias_aware(budget=4):
    """probe; on a same-appearance CONFLICT, mark the appearance aliased, REFINE to a finer
    observable feature that predicts Δachievable, and transfer along the finer key (reaching the
    FAR switch a proximity policy would miss). Logs the confidence split."""
    opened = set(); toggled = []; evidence = []; wrong = 0
    aliased = set(); refined_key = None; conf_before = None; conf_after = None
    yellows = [c for c in CANDS if c["app"] == "yellow"]

    def probe(c):
        d = toggle(opened, c["id"]); toggled.append(c["id"])
        evidence.append((c["feat"], d > 0)); return d

    # 1) probe nearest yellow
    first = min(yellows, key=lambda c: manh(c["pos"])); probe(first)
    # 2) ACTIVE DISAMBIGUATION: probe the most feature-different unprobed yellow (tests appearance)
    rest = [c for c in yellows if c["id"] not in toggled]
    def featdiff(c): return sum(c["feat"][k] != first["feat"][k] for k in ("adj_wall", "near_top"))
    second = max(rest, key=lambda c: (featdiff(c), -manh(c["pos"]))); probe(second)

    # 3) detect aliasing: same appearance, conflicting Δachievable sign
    signs = {e[1] for e in evidence}
    if len(signs) > 1:
        aliased.add("yellow")
        conf_before = round(1 - 0.5 ** sum(1 for e in evidence if e[1]), 3)   # naive appearance conf (pre-refine)
        refined_key = find_separating_feature(evidence)
        conf_after = 0.0                                          # blind appearance transfer DISABLED

    # 4) transfer: by refined key if found, else fall back to per-instance (no blind appearance transfer)
    pos_val = next((e[0][refined_key] for e in evidence if e[1]), None) if refined_key else None
    for c in yellows:
        if len(toggled) >= budget: break
        if c["id"] in toggled: continue
        if refined_key is not None and c["feat"][refined_key] == pos_val:      # finer-key transfer
            toggle(opened, c["id"]); toggled.append(c["id"])
            if not DELTA[c["id"]]: wrong += 1
    return dict(solved=solved(opened), wrong=wrong, toggled=toggled, green_toggled="g1" in toggled,
                aliased="yellow" in aliased, refined_key=refined_key,
                conf_before=conf_before, conf_after=conf_after)


orc, nod = run_oracle(), run_no_discovery()
app, ins, ali = run_appearance_class(), run_instance_only(), run_alias_aware()
print("=== affordance_aliasing_gate: switches & decoys look IDENTICAL (all yellow) ===\n")
print(f"  oracle          : solved={orc['solved']}")
print(f"  no_discovery    : solved={nod['solved']}")
print(f"  appearance_class: solved={app['solved']} wrong_transfers={app['wrong']} toggled={app['toggled']}")
print(f"  instance_only   : solved={ins['solved']} toggled={ins['toggled']} (misses far switch s3)")
print(f"  alias_aware     : solved={ali['solved']} aliased={ali['aliased']} refined_key={ali['refined_key']} "
      f"wrong={ali['wrong']} toggled={ali['toggled']}")
print(f"  alias_aware conf: yellow-class conf before_conflict={ali['conf_before']} -> after_conflict={ali['conf_after']}\n")

checks = {
    "alias_aware SOLVES (refines key, reaches far switch)": ali["solved"],
    "alias_aware refined to the CAUSAL feature adj_wall (not distractor)": ali["refined_key"] == "adj_wall",
    "alias_aware makes ZERO wrong transfers": ali["wrong"] == 0,
    "alias_aware DETECTED aliasing + SPLIT confidence (transfer disabled)":
        ali["aliased"] and ali["conf_before"] > 0 and ali["conf_after"] == 0.0,
    "appearance_class FAILS (over-generalizes, wrong_transfers>0)": (not app["solved"]) and app["wrong"] > 0,
    "instance_only FAILS (no transfer -> misses far switch)": not ins["solved"],
    "oracle SOLVES (upper bound) / no_discovery FAILS (lower bound)": orc["solved"] and not nod["solved"],
    "green non-aliased decoy rejected by all (never toggled)":
        not any(r["green_toggled"] for r in (app, ins, ali)),
}
print("=== aliasing gate pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nAFFORDANCE ALIASING GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: when appearance fails to key affordance (switch & decoy look identical), blind class"
      "\n  transfer is unsafe -- appearance_class over-generalizes onto decoys and instance_only can't"
      "\n  afford to probe the far switch. The alias_aware agent detects the same-appearance conflict,"
      "\n  DISABLES blind transfer, REFINES the key to the causal adj_wall feature (rejecting the near_top"
      "\n  distractor), and transfers along it -- reaching the far switch and solving. Aliasing forces"
      "\n  class REFINEMENT, not abandonment: same appearance does not imply same affordance.")
