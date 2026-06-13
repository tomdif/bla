#!/usr/bin/env python3
"""Affordance discovery — GATE 3: cost-sensitive CLASS-AWARE exploration.

Reconciliation: the cost/risk part of "exploration under uncertainty" already exists
(affordance_gate1.py: score = E[dReach] - lam*cost - mu*risk, trap ablation). The
genuinely NEW term here is class-TRANSFER value -- the piece that UNIFIES exploration
(gate1) with class generalization (gate2):

  score(c) = E[dReach(c)] - lam*cost(c) - mu*risk(c) + nu*class_transfer_value(c)

class_transfer_value(c) = (# unprobed cells sharing c's OBSERVABLE class) -- one probe
of a high-multiplicity class teaches about all its instances. The agent sees colors, so
it can count class members a priori.

LOAD-BEARING CONTROL (the new term must matter): a GREEN decoy is CLOSEST/CHEAPEST, and
all candidates share salience. A transfer-BLIND agent (nu=0) probes the cheap decoy,
learns nothing, and FAILS under a 1-probe budget. A transfer-AWARE agent (nu>0) probes a
YELLOW instead (its class has 3 unprobed members), discovers yellow=switch, generalizes
to the other two, opens all three doors, and SOLVES. Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque

FLOOR, WALL, DOOR, GOAL = 0, 1, 2, 3
YELLOW, GREEN = "yellow", "green"
GAIN = 56.0


def make_world():
    H, W = 11, 11
    g = [[FLOOR] * W for _ in range(H)]
    for r in range(H):
        for col in (4, 6, 8): g[r][col] = WALL                 # three serial walls
    doors = {(5, 4): "A", (5, 6): "B", (5, 8): "C"}
    for d in doors: g[d[0]][d[1]] = DOOR
    agent = (5, 1); goal = (5, 10); g[5][10] = GOAL            # behind ALL three doors
    yellows = [(3, 1), (7, 1), (9, 1)]                          # 3 YELLOW switches
    green = (5, 2)                                              # GREEN decoy: CLOSEST/cheapest, inert
    color = {**{y: YELLOW for y in yellows}, green: GREEN}
    sw_door = dict(zip(yellows, [(5, 4), (5, 6), (5, 8)]))      # each yellow opens one door
    feat = {**{y: dict(p_int=0.7, hazard=0.1) for y in yellows}, green: dict(p_int=0.7, hazard=0.1)}
    return g, agent, goal, yellows, green, color, sw_door, feat


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


def run(nu, lam=1.0, mu=10.0, probe_budget=1, action_budget=3):
    g, agent, goal, yellows, green, color, sw_door, feat = make_world()
    candidates = yellows + [green]
    opened = set(); probed = []; belief = {}                    # color -> n_pos

    def transfer_value(c):                                      # # UNPROBED cells sharing c's color
        return sum(1 for o in candidates if o not in probed and color[o] == color[c])

    def score(c):
        f = feat[c]
        return (f["p_int"] * GAIN - lam * dist(g, agent, c, opened)
                - mu * f["hazard"] + nu * transfer_value(c))

    # --- PROBE phase (cost+risk+TRANSFER-aware selection) ---
    while probe_budget > 0:
        un = [c for c in candidates if c not in probed]
        if not un: break
        c = max(un, key=score)
        probe_budget -= 1; probed.append(c)
        before = len(reachable(g, agent, opened))
        if c in sw_door: opened.add(sw_door[c])
        if len(reachable(g, agent, opened)) - before > 0:
            belief[color[c]] = belief.get(color[c], 0) + 1      # this COLOR class = switch

    # --- EXPLOIT phase: act on believed-switch CLASS members (generalization) ---
    believed = [c for c in candidates if c not in probed and belief.get(color[c], 0) > 0]
    for c in sorted(believed, key=lambda c: dist(g, agent, c, opened))[:action_budget]:
        if c in sw_door: opened.add(sw_door[c])
    probed_color = color[probed[0]] if probed else None
    return {"solved": goal in reachable(g, agent, opened), "probed_color": probed_color,
            "probed_decoy": any(color[c] == GREEN for c in probed),
            "green_is_switch_belief": belief.get(GREEN, 0) > 0}


print("=== Affordance Gate 3: cost-sensitive CLASS-AWARE exploration (probe budget=1) ===\n")
aware = run(nu=2.0); blind = run(nu=0.0)
print(f"  transfer-AWARE (nu>0): probed_color={aware['probed_color']}  solved={aware['solved']}")
print(f"  transfer-BLIND (nu=0): probed_color={blind['probed_color']}  solved={blind['solved']}\n")
passes = {
    "transfer-aware probes the HIGH-TRANSFER yellow class (not the cheap decoy)": aware["probed_color"] == YELLOW,
    "transfer-aware SOLVES under 1-probe budget (generalizes to all yellows)": aware["solved"],
    "transfer-aware did NOT waste the probe on the decoy": not aware["probed_decoy"],
    "transfer-aware does NOT believe green decoy is a switch": not aware["green_is_switch_belief"],
    "ABLATION: transfer-BLIND probes the cheap decoy and FAILS (nu load-bearing)":
        blind["probed_color"] == GREEN and not blind["solved"],
}
print("=== Gate 3 pass criteria ===")
for k, v in passes.items():
    print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGATE 3: {'PASS' if all(passes.values()) else 'FAIL'}")
print("VERDICT: the class-transfer-value term unifies exploration with generalization — the agent"
      "\n  spends its scarce probe where it teaches the MOST (the multi-instance class), not where it's"
      "\n  cheapest, amortizing one probe across the class. nu is load-bearing: without it the agent"
      "\n  wastes the probe on the nearby decoy and fails. Next: Gate 4 perception wiring.")
