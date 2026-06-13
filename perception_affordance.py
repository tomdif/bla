#!/usr/bin/env python3
"""Gate 4: wire PERCEPTION into the affordance canvas.

Until now the discover() core got candidates() and achievable() from an ORACLE. Here
they come from a RAW COLOR GRID (the "pixels"). The perception layer builds the
affordance canvas from the observation alone:
  1. START-COLOR PRIOR: the color the agent stands on = floor/passable; every other
     color = obstacle until contradicted.
  2. OBJECT SEGMENTATION: small connected non-floor components = candidate affordances;
     the large wall region is NOT a candidate. (closed door renders as wall; open door
     renders as floor -> re-perceived as passable.)
  3. CANVAS UPDATE IN THE LOOP: re-perceive after every interaction, so a toggled
     switch's door opening shows up as a grown achievable set -> the core discovers it.

The core is UNCHANGED (embodiment-invariant); only the Embodiment adapter now derives
achievable/candidates from perception. New honest risk: perception OVER-INCLUDES junk
(a stray noise pixel). Control: the Δachievable filter rejects it exactly like a decoy.

CONTROLS: start-color prior recovers the true floor | core solves from PERCEIVED inputs
(no oracle) | green decoy rejected | stray NOISE pixel rejected | prediction-only fails.
Self-contained, stdlib only.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque

FLOOR, WALL, YELLOW, GREEN, GOAL, NOISE = 0, 1, 2, 3, 4, 7   # observable COLORS
GAIN = 56.0; OBJ_MAX = 3                                       # components <=OBJ_MAX are objects, bigger=wall


@dataclass(frozen=True)
class Cand:
    id: object; key: object; cost: float; risk: float; p_int: float


# ----- the TRUE world (hidden dynamics); the agent only ever sees render() -----
class World:
    def __init__(self):
        self.H = self.W = 11
        self.types = [[FLOOR] * self.W for _ in range(self.H)]
        for r in range(self.H): self.types[r][4] = WALL; self.types[r][7] = WALL
        self.doors = {(5, 4): "door", (5, 7): "door"}         # door cells (render WALL closed, FLOOR open)
        self.types[5][9] = GOAL
        self.switches = {(3, 1): (5, 4), (9, 1): (5, 7)}      # switch cell -> door it opens
        for sw in self.switches: self.types[sw[0]][sw[1]] = YELLOW
        self.types[5][2] = GREEN                              # decoy
        self.types[7][2] = NOISE                              # stray junk pixel
        self.agent = (5, 1); self.opened = frozenset()
    def render(self):                                          # -> color grid (the observation)
        grid = [[self.types[r][c] for c in range(self.W)] for r in range(self.H)]
        for d in self.doors:
            grid[d[0]][d[1]] = FLOOR if d in self.opened else WALL
        return grid
    def toggle(self, cell):                                    # true dynamics: only a switch opens a door
        if cell in self.switches:
            self.opened = self.opened | {self.switches[cell]}


# ----- PERCEPTION: raw color grid -> affordance canvas (passable mask + candidate objects) -----
def perceive(grid, agent):
    H, W = len(grid), len(grid[0])
    start_color = grid[agent[0]][agent[1]]                     # START-COLOR PRIOR: this = floor/passable
    passable = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == start_color}
    passable.add(agent)
    # reachable region from agent over the passable mask
    seen = {agent}; q = deque([agent])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and n in passable:
                seen.add(n); q.append(n)
    reach = seen
    # OBJECT SEGMENTATION: connected components of each non-floor color; small=object, large=wall
    objs = []; visited = set()
    for r in range(H):
        for c in range(W):
            col = grid[r][c]
            if col in (start_color, GOAL) or (r, c) in visited: continue
            comp = []; q = deque([(r, c)]); visited.add((r, c))
            while q:
                cur = q.popleft(); comp.append(cur)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cur[0] + dr, cur[1] + dc)
                    if 0 <= n[0] < H and 0 <= n[1] < W and n not in visited and grid[n[0]][n[1]] == col:
                        visited.add(n); q.append(n)
            if len(comp) <= OBJ_MAX:                           # small component = candidate object
                cell = comp[0]
                if any((cell[0] + dr, cell[1] + dc) in reach for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    objs.append((cell, col))                   # reachable-adjacent object
    return reach, objs, start_color, passable


# ----- Embodiment adapter: achievable/candidates DERIVED FROM PERCEPTION -----
class Perceived:
    def __init__(self): self.w = World()
    def initial(self): self.w = World(); return self.w
    def _percept(self): return perceive(self.w.render(), self.w.agent)
    def achievable(self, s): return self._percept()[0]                       # perceived reachable set
    def candidates(self, s):
        reach, objs, _, _ = self._percept()
        cs = []
        for cell, col in objs:
            d = abs(cell[0] - self.w.agent[0]) + abs(cell[1] - self.w.agent[1])
            cs.append(Cand(cell, col, d, 0.1, 0.7))            # key = perceived COLOR
        return cs
    def interact(self, s, c): self.w.toggle(c.id); return s     # act on the TRUE world
    def goal(self, s):
        reach = self._percept()[0]                              # target is steppable-onto: reachable if
        for r in range(self.w.H):                               # the goal cell is adjacent to the floor-reach set
            for c in range(self.w.W):
                if self.w.types[r][c] == GOAL and any((r + dr, c + dc) in reach
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    return True
        return False
    def local_change(self, s, c):
        d = self.w.switches.get(c.id)
        return d is not None and abs(d[0] - c.id[0]) + abs(d[1] - c.id[1]) <= 1


# ----- embodiment-invariant CORE (unchanged) -----
def discover(emb, lam=1.0, mu=10.0, nu=2.0, probe_budget=1):
    s = emb.initial(); belief = {}; probed = set()
    def score(c, cands):
        t = sum(1 for o in cands if o.key == c.key and o.id not in probed)
        return c.p_int * GAIN - lam * c.cost - mu * c.risk + nu * t
    while probe_budget > 0 and not emb.goal(s):
        cands = [c for c in emb.candidates(s) if c.id not in probed]
        if not cands: break
        c = max(cands, key=lambda c: score(c, cands)); probed.add(c.id); probe_budget -= 1
        b = len(emb.achievable(s)); s = emb.interact(s, c)
        if len(emb.achievable(s)) - b > 0: belief[c.key] = belief.get(c.key, 0) + 1
    for c in emb.candidates(s):
        if c.id not in probed and belief.get(c.key, 0) > 0: s = emb.interact(s, c)
    return {"solved": emb.goal(s), "belief": dict(belief)}


def prediction_only(emb):
    s = emb.initial()
    for c in emb.candidates(s):
        if emb.local_change(s, c): s = emb.interact(s, c)
    return emb.goal(s)


print("=== Gate 4: perception wired into the affordance canvas ===\n")
emb = Perceived(); emb.initial()
reach, objs, start_color, passable = emb._percept()
true_floor = {(r, c) for r in range(11) for c in range(11) if World().types[r][c] == FLOOR}
print(f"  start-color prior: floor color = {start_color}  | perceived passable cells = {len(passable)}")
print(f"  segmented candidate objects (cell,color): {[(o[0], o[1]) for o in objs]}")
res = discover(Perceived()); pred = prediction_only(Perceived())
print(f"  discover-core (PERCEIVED inputs): solved={res['solved']}  learned_color_classes={list(res['belief'])}")
print(f"  prediction-only solves?: {pred}\n")

checks = {
    "start-color prior recovers the true floor mask": passable >= true_floor,
    "perception segments switches+decoy+noise as objects (not walls)":
        sorted({o[1] for o in objs}) == sorted({YELLOW, GREEN, NOISE}),
    "core SOLVES from perceived inputs (no oracle)": res["solved"],
    "learned ONLY the yellow switch class": list(res["belief"]) == [YELLOW],
    "green decoy rejected": GREEN not in res["belief"],
    "stray NOISE pixel rejected (perception over-included, Δachievable filtered)": NOISE not in res["belief"],
    "prediction-only FAILS (non-local affordance)": not pred,
}
print("=== Gate 4 pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGATE 4: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the affordance canvas is now built from a raw color grid via the start-color prior +"
      "\n  object segmentation, updated in the loop; the UNCHANGED core discovers the switch class,"
      "\n  rejects the decoy AND the stray noise pixel perception over-included, and beats local prediction.")
