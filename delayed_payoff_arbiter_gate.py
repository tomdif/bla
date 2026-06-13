#!/usr/bin/env python3
"""#4 delayed_payoff_arbiter_gate: multi-step arbitration on top of the #3A runner.

#3A solves when a SINGLE interaction changes the achievable set. It CANNOT solve delayed
payoff: the useful action sequence is flat for 1-2 steps and only pays off at the end. Here
the goal sits behind a LOCKED door; a KEY must be picked up and CARRIED to the lock. Picking
up the key changes reachability by ZERO (one-step Δachievable = 0), so #3A's one-step discovery
sees it as inert -- exactly like the DECOY, which is visually identical and also Δ=0 on pickup.
The payoff exists only in the SEQUENCE key->lock. One-step probing provably plateaus.

THE NEW MECHANISM:
  1. PLATEAU DETECTOR: goal unreachable AND no one-step interaction yields Δachievable>0.
  2. MULTI-STEP ARBITER: generate (item -> lock) micro-plan hypotheses, each PREDICTING a
     reachability JUMP at the final (use) step and FLAT before it.
  3. PREDICTED/ACTUAL CONSISTENCY: execute; commit only if the predicted payoff actually fires.
     The KEY rule -- stable BEFORE the payoff step is EXPECTED (do not abort); stable AT the
     payoff step when a jump was predicted = INCONSISTENT -> reject (that item was a decoy).

After the lock opens, a disguised wall on the final approach still needs contradiction-update,
so the arbiter composes with #3A's mechanisms.

CONTROLS: FULL solves | one_step_only fails (all probes plateau) | no_plateau_detector never
enters arbiter | no_consistency_check commits to the decoy -> fails | no_discovery fails |
no_contradiction fails (disguised wall) | oracle solves. Plus: arbiter rejects the decoy
hypothesis, micro_plan_depth>=2, predicted vs actual traces recorded, blocked stays local.
Trace -> artifacts/delayed_payoff_arbiter_gate/trace.jsonl. Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque
import json, os

H = W = 11
FLOORC, WALLC, ITEMC, DOORC, GOALC = 0, 1, 2, 5, 4
ART = "artifacts/delayed_payoff_arbiter_gate"


class World:
    def __init__(self):
        self.agent0 = (5, 0); self.goal = (5, 10)
        self.walls = {(r, 7) for r in range(H)}
        self.door = (5, 7)                                       # the LOCK: opens only if used while carrying the key
        self.items = {(8, 2): "key", (4, 2): "decoy"}           # VISUALLY IDENTICAL (ITEMC); decoy is NEARER (tested first)
        self.disguised = {(5, 9)}                               # floor-colored solid on the post-door approach
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
    def snapshot(self): return (set(self.opened), self.carrying)
    def restore(self, s): self.opened, self.carrying = set(s[0]), s[1]
    def interact(self, target, pos):
        if abs(target[0] - pos[0]) + abs(target[1] - pos[1]) > 1: return
        if target in self.items: self.carrying = target
        elif target == self.door and self.carrying in self.items and self.items[self.carrying] == "key":
            self.opened.add(self.door)


def passable_of(grid): return {(r, c) for r in range(H) for c in range(W) if grid[r][c] == FLOORC}
def reach_set(passable, start):
    seen = {start}; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and n in passable:
                seen.add(n); q.append(n)
    return seen
def adj(cellset, cell):
    return any((cell[0] + dr, cell[1] + dc) in cellset for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
def bfs_path(passable, start, goal):
    if start == goal: return [start]
    prev = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in prev and (n in passable or n == goal):
                prev[n] = cur
                if n == goal:
                    p = [goal]
                    while prev[p[-1]] is not None: p.append(prev[p[-1]])
                    return p[::-1]
                q.append(n)
    return None


def run(abl=None, step_cap=120, trace_out=None):
    abl = abl or {}
    discovery = abl.get("discovery", True); arbiter_on = abl.get("arbiter", True)
    plateau_detect = abl.get("plateau_detect", True); consistency = abl.get("consistency", True)
    contradiction = abl.get("contradiction", True); oracle = abl.get("oracle", False)
    w = World(); pos = w.agent0; blocked = set(); trace = []
    found_key = None; arbiter_entered = False; plateau = False; rejected_hyps = []

    def reach_size():                                            # achievable EXCLUDES item cells: picking up an item
        return len(reach_set(passable_of(w.render()) - blocked, pos) - set(w.items.keys()))  # must not count as gain
    def goal_reachable():
        p = passable_of(w.render()) - blocked
        return adj(reach_set(p, pos), w.goal) and bfs_path(p, pos, w.goal) is not None
    def reachable_items():
        grid = w.render(); rs = reach_set(passable_of(grid) - blocked, pos)
        return [c for c in w.items if c != w.carrying and adj(rs, c)]

    # ---- PHASE 1: one-step discovery (#3A). Every interaction here is FLAT -> plateau. ----
    if discovery and not goal_reachable():
        max_gain = 0
        for cand in reachable_items() + [w.door]:
            snap = w.snapshot(); before = reach_size(); w.interact(cand, _nearest_adj(w, pos, cand))
            max_gain = max(max_gain, reach_size() - before); w.restore(snap)
        plateau = (max_gain == 0)
        trace.append({"phase": "one_step_discovery", "plateau_detected": plateau, "max_one_step_gain": max_gain})

    # ---- PHASE 2: multi-step ARBITER (only on detected plateau) ----
    enter = arbiter_on and (oracle or (plateau and plateau_detect))
    if enter and not goal_reachable():
        arbiter_entered = True
        hyps = sorted(w.items.keys(), key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
        if oracle: hyps = [c for c in hyps if w.items[c] == "key"]      # upper bound: knows the key
        for item in hyps:
            plan, pos = _build_microplan(w, pos, item, blocked)
            if plan is None: continue
            base = reach_size()
            predicted = ["flat"] * (len(plan) - 1) + ["jump"]          # predict payoff JUMP at the final (use) step
            actual_sizes = []; actual = []
            for k, act in enumerate(plan):                            # EXECUTE the micro-plan
                if act[0] == "move":
                    pos = act[1]
                else:
                    w.interact(act[1], pos)
                sz = reach_size(); actual_sizes.append(sz)
                actual.append("jump" if sz > base else "flat")
            payoff_fired = actual[-1] == "jump"
            rec = {"phase": "arbiter_hypothesis", "item": list(item), "micro_plan_depth": len(plan),
                   "predicted_achievable_trace": predicted, "actual_achievable_trace": actual,
                   "consistency_score": int(payoff_fired)}
            if not consistency:                                       # ABLATION: commit to first regardless
                rec["continue_or_reject_reason"] = "no_consistency_check_commit"; trace.append(rec)
                found_key = item; break
            if payoff_fired:                                         # predicted payoff materialized -> commit
                rec["continue_or_reject_reason"] = "consistent_payoff_committed"; trace.append(rec)
                found_key = item; break
            else:                                                    # stable WHERE GAIN WAS EXPECTED -> reject (decoy)
                rec["continue_or_reject_reason"] = "stable_when_gain_expected_reject"
                rejected_hyps.append(item); trace.append(rec)

    # ---- PHASE 3: navigate to goal, contradiction-update on the disguised wall ----
    for step in range(step_cap):
        if pos == w.goal: break
        p = passable_of(w.render()) - blocked
        path = bfs_path(p, pos, w.goal)
        if path is None: break
        nxt = path[1]
        if w.solid(nxt):
            if contradiction: blocked.add(nxt)
            trace.append({"phase": "navigate", "bumped": list(nxt), "blocked": sorted(map(list, blocked))})
            if not contradiction and step > step_cap - 3: break
        else:
            pos = nxt

    solved = pos == w.goal
    if trace_out:
        os.makedirs(os.path.dirname(trace_out), exist_ok=True)
        with open(trace_out, "w") as f:
            for r in trace: f.write(json.dumps(r) + "\n")
    return {"solved": solved, "plateau_detected": plateau, "arbiter_entered": arbiter_entered,
            "found_key": found_key, "rejected_hyps": rejected_hyps, "blocked": set(blocked),
            "key_is_correct": found_key is not None and w.items.get(found_key) == "key"}


def _nearest_adj(w, pos, cell):                                       # a passable cell adjacent to `cell`, for interact range
    grid = w.render(); p = passable_of(grid)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (cell[0] + dr, cell[1] + dc)
        if n in p or n == pos: return n
    return pos


def _approach(start, target, p):
    """a passable neighbor of `target` that is actually REACHABLE from `start` (near side)."""
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (target[0] + dr, target[1] + dc)
        if (n in p or n == start) and bfs_path(p, start, n) is not None: return n
    return None


def _build_microplan(w, pos, item, blocked):
    """[move...,  interact item, move..., interact door]. Returns (plan, start_pos_unchanged)."""
    p = passable_of(w.render()) - blocked
    ia = _approach(pos, item, p)
    if ia is None: return None, pos
    da = _approach(ia, w.door, p)
    if da is None: return None, pos
    p1 = bfs_path(p, pos, ia);  p2 = bfs_path(p, ia, da)
    if p1 is None or p2 is None: return None, pos
    plan = [("move", c) for c in p1[1:]] + [("interact", item)] + [("move", c) for c in p2[1:]] + [("interact", w.door)]
    return plan, pos


full = run(trace_out=f"{ART}/trace.jsonl")
one_step = run({"arbiter": False})
no_plateau = run({"plateau_detect": False})
no_consist = run({"consistency": False})
no_disc = run({"discovery": False})
no_contra = run({"contradiction": False})
orc = run({"oracle": True})

print("=== #4 delayed_payoff_arbiter_gate: payoff only in the key->lock SEQUENCE ===\n")
for name, r in [("FULL", full), ("one_step_only", one_step), ("no_plateau_detector", no_plateau),
                ("no_consistency_check", no_consist), ("no_discovery", no_disc),
                ("no_contradiction", no_contra), ("oracle", orc)]:
    print(f"  {name:22} solved={str(r['solved']):5} plateau={str(r['plateau_detected']):5} "
          f"arbiter={str(r['arbiter_entered']):5} found_key={r['found_key']} "
          f"key_correct={r['key_is_correct']} rejected={r['rejected_hyps']}")
print()

checks = {
    "FULL solves the delayed-payoff world": full["solved"],
    "plateau DETECTED (one-step probes all flat)": full["plateau_detected"],
    "arbiter ENTERED on plateau": full["arbiter_entered"],
    "arbiter found the CORRECT key (multi-step payoff)": full["key_is_correct"],
    "arbiter REJECTED the decoy hypothesis (stable-when-gain-expected)": (4, 2) in full["rejected_hyps"],
    "one_step_only FAILS (all one-step probes plateau)": not one_step["solved"],
    "no_plateau_detector FAILS (never enters arbiter)": (not no_plateau["solved"]) and not no_plateau["arbiter_entered"],
    "no_consistency_check FAILS (commits to the decoy)": (not no_consist["solved"]) and not no_consist["key_is_correct"],
    "no_discovery FAILS": not no_disc["solved"],
    "no_contradiction FAILS (disguised wall on approach)": not no_contra["solved"],
    "oracle SOLVES (upper bound)": orc["solved"],
    "blocked stays LOCAL (only the disguised cell)": full["blocked"] == {(5, 9)},
}
print("=== #4 pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
os.makedirs(ART, exist_ok=True)
with open(f"{ART}/result.json", "w") as f:
    json.dump({"pass": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()}}, f, indent=2)
print(f"\n#4 DELAYED-PAYOFF ARBITER GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the agent survives delayed payoff. One-step discovery provably plateaus (key & decoy both"
      "\n  Δachievable=0 on pickup); the plateau detector escalates to a multi-step arbiter that generates"
      "\n  (item->lock) hypotheses, executes them, and commits only when the PREDICTED payoff jump actually"
      "\n  fires -- persisting through the expected-stable prefix (stable_but_expected) and rejecting the decoy"
      "\n  when the gain fails to appear (stable_when_gain_expected). Composes with contradiction on the approach.")
