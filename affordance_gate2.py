#!/usr/bin/env python3
"""Affordance discovery — GATE 2: color/object-CLASS generalization.

Gate 1 discovered a CELL. Gate 2 infers a CLASS: probe ONE yellow cell, discover
"yellow = switch", then ACT on other yellow cells WITHOUT probing them. That amortizes
discovery across a scene (1 probe -> N switches) instead of re-probing every cell.

LOAD-BEARING CONTROLS:
  (1) discovers the class from ONE probe;
  (2) generalization PAYS OFF: a 2-switch level solvable under a 1-PROBE budget only by
      generalizing -- a no-generalize agent must probe each switch and runs out;
  (3) generalize by CLASS, not SALIENCE: a same-salience GREEN decoy must NOT inherit the
      switch label. A salience-lumping agent treats green as a switch and FAILS (wastes its
      one action toggling the inert decoy); the class agent ignores green and solves it;
  (4) keep UNCERTAINTY when evidence is weak: confidence after 1 probe is moderate (<1),
      and an UNPROBED class stays UNKNOWN (never generalized as a switch from zero evidence).

Three agents differ ONLY in how they key beliefs: class -> color, no_gen -> cell,
salience -> "salient/inert" bucket. Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque

FLOOR, WALL, DOOR, GOAL = 0, 1, 2, 3
YELLOW, GREEN, NONE = "yellow", "green", None


def make_world():
    H, W = 11, 11
    g = [[FLOOR] * W for _ in range(H)]
    color = {}                                                  # (r,c)->color feature (observable)
    sw_door = {}                                                # switch cell -> the door it opens (hidden until probed)
    for r in range(H):
        g[r][4] = WALL; g[r][7] = WALL                          # two walls (serial gates)
    doorA, doorB = (5, 4), (5, 7)
    g[5][4] = DOOR; g[5][7] = DOOR
    agent = (5, 1); goal = (5, 9); g[5][9] = GOAL
    y1, y2 = (5, 2), (9, 1)                                     # YELLOW switches: y1 NEAREST (probed first), y2 FAR
    green = (3, 1)                                              # GREEN decoy: inert, and nearer than y2 (bait for salience)
    color[y1] = YELLOW; color[y2] = YELLOW; color[green] = GREEN
    sw_door[y1] = doorA; sw_door[y2] = doorB                    # the hidden truth
    feat = {y1: 0.7, y2: 0.7, green: 0.7}                       # interactiveness/salience -- SAME for all 3
    return g, agent, goal, doorA, doorB, [y1, y2], green, color, sw_door, feat


def reachable(g, agent, opened):
    H, W = len(g), len(g[0]); seen = {agent}; q = deque([agent])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen:
                cell = g[n[0]][n[1]]
                if cell in (FLOOR, GOAL) or (cell == DOOR and n in opened):
                    seen.add(n); q.append(n)
    return seen


def dist(g, agent, c, opened):
    H, W = len(g), len(g[0]); seen = {agent: 0}; q = deque([agent])
    while q:
        cur = q.popleft()
        if cur == c: return seen[cur]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            ok = g[n[0]][n[1]] in (FLOOR, GOAL) or (g[n[0]][n[1]] == DOOR and n in opened) or n == c
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and ok:
                seen[n] = seen[cur] + 1; q.append(n)
    return 999


def run(agent_kind, probe_budget=1, action_budget=1):
    g, agent, goal, doorA, doorB, switches, green, color, sw_door, feat = make_world()
    candidates = switches + [green]

    def key(c):
        if agent_kind == "class":   return ("color", color.get(c, NONE))
        if agent_kind == "no_gen":  return ("cell", c)
        return ("salient", feat.get(c, 0) > 0.5)               # salience: lumps all salient together

    belief = {}                                                # key -> dict(n_pos)
    opened = set(); probed = []
    # --- PROBE phase: discover (measure dReach), spend probe budget ---
    order = sorted(candidates, key=lambda c: dist(g, agent, c, opened))
    for c in order:
        if probe_budget <= 0: break
        probe_budget -= 1; probed.append(c)
        before = len(reachable(g, agent, opened))
        if c in sw_door: opened.add(sw_door[c])                # toggling a switch opens its door
        d = len(reachable(g, agent, opened)) - before
        k = key(c); belief.setdefault(k, {"n_pos": 0})
        if d > 0: belief[k]["n_pos"] += 1                      # positive switch evidence for this key

    def believed_switch(c):
        b = belief.get(key(c)); return bool(b and b["n_pos"] > 0)
    def confidence(c):
        b = belief.get(key(c)); return 0.0 if not b else round(1 - 0.5 ** b["n_pos"], 3)

    # --- EXPLOIT phase: ACT on believed-switches (no probing), spend action budget ---
    untoggled = [c for c in candidates if c not in probed and believed_switch(c)]
    untoggled.sort(key=lambda c: dist(g, agent, c, opened))    # nearest first
    for c in untoggled[:action_budget]:
        if c in sw_door: opened.add(sw_door[c])                # toggles its door (or no-op if not a switch)

    return {"solved": goal in reachable(g, agent, opened), "probed": probed,
            "belief_yellow_switch": believed_switch(switches[1]),   # generalized to the UNPROBED yellow?
            "belief_green_switch": believed_switch(green),          # wrongly generalized to decoy?
            "conf_yellow": confidence(switches[0]), "conf_green": confidence(green),
            "conf_unprobed_color": confidence((0, 0))}              # a never-seen color stays unknown


print("=== Affordance Gate 2: class generalization (probe budget=1, action budget=1) ===\n")
res = {k: run(k) for k in ("class", "no_gen", "salience")}
print(f"  {'agent':10} solved  belief:yellow2=switch  belief:green=switch  conf_yellow  conf_green")
for k, r in res.items():
    print(f"  {k:10} {str(r['solved']):5}   {str(r['belief_yellow_switch']):17}   "
          f"{str(r['belief_green_switch']):17}   {r['conf_yellow']:^11} {r['conf_green']:^9}")

c = res["class"]
passes = {
    "(1) class discovers switch class from ONE probe": c["conf_yellow"] > 0,
    "(2) class GENERALIZES to unprobed yellow2 (belief)": c["belief_yellow_switch"],
    "(2) generalization PAYS OFF: class solves, no_gen does NOT": c["solved"] and not res["no_gen"]["solved"],
    "(3) class does NOT generalize to green decoy": not c["belief_green_switch"],
    "(3) salience-lumping FAILS (treats green as switch & loses)": res["salience"]["belief_green_switch"] and not res["salience"]["solved"],
    "(4) keeps uncertainty: conf<1 after 1 probe": 0 < c["conf_yellow"] < 1,
    "(4) UNPROBED class stays unknown (conf 0)": c["conf_unprobed_color"] == 0.0,
}
print("\n=== Gate 2 pass criteria ===")
for k, v in passes.items():
    print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGATE 2: {'PASS' if all(passes.values()) else 'FAIL'}")
print("VERDICT: one probe -> a CLASS affordance that transfers to unprobed same-color cells"
      "\n  (amortized discovery, solves under a 1-probe budget a no-generalize agent can't),"
      "\n  generalizes by CLASS not salience (green decoy rejected; salience-lumping fails),"
      "\n  and keeps honest uncertainty (conf 0.5 from 1 probe; unprobed classes unknown).")
