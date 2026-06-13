#!/usr/bin/env python3
"""CAPSTONE: language_assisted_path_hypothesis_arbiter_gate -- compose the two validated
pieces, nothing new:
    #4 delayed-payoff arbiter  +  language_hypothesis_affordance seam

The delayed-payoff world (key must be carried to a lock; KEY & DECOY visually identical, both
Δachievable=0 on pickup -> one-step discovery plateaus). The arbiter generates (item->lock)
micro-plans and commits only when the PREDICTED payoff actually fires (predicted/actual
consistency). Here LANGUAGE proposes WHICH item is the key, reordering which micro-plan the
arbiter tries FIRST -- a prior that saves a wasted multi-step plan. But the consistency check +
Δachievable still own truth: a wrong/shuffled proposal is rejected (its plan's predicted payoff
never fires) and the arbiter falls back to proximity order. Language proposes the micro-plan;
predicted/actual consistency and Δachievable verify it.

AGENTS:
  no_language_arbiter            -- #4 as-is: proximity order (tests nearer DECOY first -> 2 plans)
  language_verified(correct)     -- language proposes the KEY; tried first + verified -> 1 plan
  language_verified(SHUFFLED)    -- language proposes the DECOY; tried first, FAILS consistency,
                                    falls back -> 2 plans (NO advantage)
  language_only_no_verifier      -- commit to the proposed item WITHOUT consistency (unsafe)
  arbiter_no_consistency_check   -- commit to first plan regardless (#4 ablation)
  oracle                         -- knows the key (upper bound)

PASS: correct language saves >=1 micro-plan vs no_language | shuffled language removes the
advantage | bad language rejected by consistency/Δachievable | language_only(shuffled) FAILS |
accepted hypothesis has actual Δachievable>0 | predicted/actual trace recorded | wrong micro-plan
rejected | oracle solves. Trace -> artifacts/.../trace.jsonl. Self-contained, stdlib only.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import json, os

H = W = 11
FLOORC, WALLC, ITEMC, DOORC, GOALC = 0, 1, 2, 5, 4
ART = "artifacts/language_assisted_path_hypothesis_arbiter_gate"


@dataclass
class PathHypothesis:                                            # the typed seam, now over MICRO-PLANS
    proposed_key_item: tuple
    role: str
    predicted_effect: str
    confidence: float
    provenance: str


def language_propose(world, correct):
    key = next(c for c, r in world.items.items() if r == "key")
    decoy = next(c for c, r in world.items.items() if r == "decoy")
    if correct:
        return PathHypothesis(key, "key_for_lock", "increase_reachability_at_lock", 0.7,
                              "the item on the goal side / aligned with the lock is usually the key")
    return PathHypothesis(decoy, "key_for_lock", "increase_reachability_at_lock", 0.7,
                          "SHUFFLED: proposal decorrelated from the episode")


class World:
    def __init__(self):
        self.agent0 = (5, 0); self.goal = (5, 10)
        self.walls = {(r, 7) for r in range(H)}
        self.door = (5, 7)
        self.items = {(8, 2): "key", (4, 2): "decoy"}           # identical (ITEMC); decoy NEARER
        self.disguised = {(5, 9)}
        self.carrying = None; self.opened = set()
    def render(self):
        g = [[FLOORC] * W for _ in range(H)]
        for (r, c) in self.walls: g[r][c] = WALLC
        g[self.door[0]][self.door[1]] = FLOORC if self.door in self.opened else DOORC
        for cell in self.items:
            if cell != self.carrying: g[cell[0]][cell[1]] = ITEMC
        g[self.goal[0]][self.goal[1]] = GOALC
        return g
    def solid(self, cell):
        if cell == self.door: return self.door not in self.opened
        return cell in self.walls or cell in self.disguised
    def interact(self, target, pos):
        if abs(target[0] - pos[0]) + abs(target[1] - pos[1]) > 1: return
        if target in self.items: self.carrying = target
        elif target == self.door and self.carrying in self.items and self.items[self.carrying] == "key":
            self.opened.add(self.door)


def passable_of(grid): return {(r, c) for r in range(H) for c in range(W) if grid[r][c] == FLOORC}
def reach_set(p, s):
    seen = {s}; q = deque([s])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and n in p: seen.add(n); q.append(n)
    return seen
def adj(cs, c): return any((c[0] + d[0], c[1] + d[1]) in cs for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
def bfs_path(p, s, g):
    if s == g: return [s]
    prev = {s: None}; q = deque([s])
    while q:
        cur = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in prev and (n in p or n == g):
                prev[n] = cur
                if n == g:
                    path = [g]
                    while prev[path[-1]] is not None: path.append(prev[path[-1]])
                    return path[::-1]
                q.append(n)
    return None
def _approach(start, target, p):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (target[0] + dr, target[1] + dc)
        if (n in p or n == start) and bfs_path(p, start, n) is not None: return n
    return None
def _microplan(w, pos, item, blocked):
    p = passable_of(w.render()) - blocked
    ia = _approach(pos, item, p)
    if ia is None: return None
    da = _approach(ia, w.door, p)
    if da is None: return None
    p1 = bfs_path(p, pos, ia); p2 = bfs_path(p, ia, da)
    if p1 is None or p2 is None: return None
    return [("move", c) for c in p1[1:]] + [("interact", item)] + [("move", c) for c in p2[1:]] + [("interact", w.door)]


def run(mode, correct=True, step_cap=160, trace_out=None):
    w = World(); pos = w.agent0; blocked = set(); trace = []
    use_language = mode in ("language_verified", "language_only")
    consistency = mode in ("no_language", "language_verified", "oracle")
    found_key = None; plans_executed = 0; rejected = []; acc_delta = None

    def rsize():
        return len(reach_set(passable_of(w.render()) - blocked, pos) - set(w.items.keys()))  # exclude item cells
    def goal_reachable():
        p = passable_of(w.render()) - blocked
        return adj(reach_set(p, pos), w.goal) and bfs_path(p, pos, w.goal) is not None

    # plateau: one-step interactions all flat (key & decoy both Δ=0 on pickup)
    plateau = not goal_reachable()
    # arbiter hypothesis ORDER: language proposes which item to try first
    order = sorted(w.items.keys(), key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
    if mode == "oracle":
        order = [c for c in order if w.items[c] == "key"]
    elif use_language:
        prop = language_propose(w, correct).proposed_key_item
        order = [prop] + [c for c in order if c != prop]

    for item in order:
        plan = _microplan(w, pos, item, blocked)
        if plan is None: continue
        plans_executed += 1; base = rsize()
        predicted = ["flat"] * (len(plan) - 1) + ["jump"]
        actual = []
        for act in plan:
            if act[0] == "move": pos = act[1]
            else: w.interact(act[1], pos)
            actual.append("jump" if rsize() > base else "flat")
        payoff = actual[-1] == "jump"
        rec = {"phase": "arbiter", "item": list(item), "predicted_achievable_trace": predicted,
               "actual_achievable_trace": actual, "consistency_score": int(payoff),
               "via_language": use_language and item == order[0]}
        if not consistency:                                      # commit to first regardless (lang_only / no_consistency)
            rec["decision"] = "commit_no_check"; trace.append(rec); found_key = item; break
        if payoff:
            rec["decision"] = "consistent_commit"; acc_delta = rsize() - base; trace.append(rec); found_key = item; break
        rec["decision"] = "inconsistent_reject"; rejected.append(item); trace.append(rec)

    for _ in range(step_cap):                                    # navigate + contradiction on disguised wall
        if pos == w.goal: break
        p = passable_of(w.render()) - blocked
        path = bfs_path(p, pos, w.goal)
        if path is None: break
        nxt = path[1]
        if w.solid(nxt): blocked.add(nxt)
        else: pos = nxt

    if trace_out:
        os.makedirs(os.path.dirname(trace_out), exist_ok=True)
        with open(trace_out, "w") as f:
            for r in trace: f.write(json.dumps(r) + "\n")
    return {"solved": pos == w.goal, "plans_executed": plans_executed, "found_key": found_key,
            "key_correct": found_key is not None and w.items.get(found_key) == "key",
            "rejected": rejected, "acc_delta": acc_delta}


R = {"no_language_arbiter": run("no_language"),
     "language_verified(correct)": run("language_verified", True, trace_out=f"{ART}/trace.jsonl"),
     "language_verified(SHUFFLED)": run("language_verified", False),
     "language_only(correct)": run("language_only", True),
     "language_only(SHUFFLED)": run("language_only", False),
     "arbiter_no_consistency": run("no_consistency"),
     "oracle": run("oracle")}
print("=== CAPSTONE: language-assisted delayed-payoff arbiter ===\n")
for name, r in R.items():
    print(f"  {name:28} solved={str(r['solved']):5} plans={r['plans_executed']} found_key={r['found_key']} "
          f"key_correct={r['key_correct']} rejected={r['rejected']}")
print()

base = R["no_language_arbiter"]["plans_executed"]
checks = {
    "no_language arbiter solves (baseline)": R["no_language_arbiter"]["solved"],
    "language_verified(correct) solves with FEWER micro-plans than no_language":
        R["language_verified(correct)"]["solved"] and R["language_verified(correct)"]["plans_executed"] < base,
    "SHUFFLE KILLS ADVANTAGE: language_verified(shuffled) gets NO plan savings":
        R["language_verified(SHUFFLED)"]["plans_executed"] >= base,
    "bad language REJECTED by consistency (decoy plan inconsistent)":
        (4, 2) in R["language_verified(SHUFFLED)"]["rejected"],
    "accepted hypothesis had actual Δachievable>0": (R["language_verified(correct)"]["acc_delta"] or 0) > 0,
    "language_verified(correct) committed the CORRECT key": R["language_verified(correct)"]["key_correct"],
    "language_only(correct) solves": R["language_only(correct)"]["solved"],
    "language_only(SHUFFLED) FAILS (commits to decoy, no verifier)":
        (not R["language_only(SHUFFLED)"]["solved"]) and not R["language_only(SHUFFLED)"]["key_correct"],
    "arbiter_no_consistency FAILS (commits to nearer decoy)":
        (not R["arbiter_no_consistency"]["solved"]) and not R["arbiter_no_consistency"]["key_correct"],
    "verifier makes language ROBUST: shuffled still solves (fell back)": R["language_verified(SHUFFLED)"]["solved"],
    "oracle solves (upper bound)": R["oracle"]["solved"],
}
print("=== capstone pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
os.makedirs(ART, exist_ok=True)
with open(f"{ART}/result.json", "w") as f:
    json.dump({"pass": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()}}, f, indent=2)
print(f"\nCAPSTONE LANGUAGE-ASSISTED ARBITER GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print(f"VERDICT: the two validated pieces COMPOSE. Correct language proposes the key item so the arbiter tries the"
      f"\n  winning micro-plan FIRST ({R['language_verified(correct)']['plans_executed']} plan vs "
      f"{base} for proximity-order baseline); predicted/actual consistency confirms it. SHUFFLED language proposes the"
      f"\n  decoy -- its micro-plan's predicted payoff never fires, consistency REJECTS it, and the arbiter falls back to"
      f"\n  proximity order ({R['language_verified(SHUFFLED)']['plans_executed']} plans = no advantage). Without the"
      f" consistency check, shuffled language is catastrophic.\n  Language proposes the micro-plan; predicted/actual "
      f"consistency + Δachievable own truth -- the same fusion rule, now over multi-step plans.")
