#!/usr/bin/env python3
"""General world model: embodiment-invariant DISCOVER core + per-embodiment adapters.

The discovery mechanism validated across affordance_loop/gate1/gate2/gate3 is NOT
grid-specific. The invariant core is:
  - discover via ACHIEVABILITY change: interacting at c that grows |achievable(state)|
    is a decision-relevant affordance (invisible to local prediction).
  - select probes by  E[gain] - lam*cost - mu*risk + nu*class_transfer_value.
  - generalize by observable feature CLASS (key); keep confidence honest.
What is embodiment-SPECIFIC is narrow and lives behind the Embodiment interface:
  achievable(), candidates(), interact(), goal_achieved(), local_change().
"Tune per embodiment" = implement the adapter + set lam/mu/nu to the body's cost/risk
profile.

GENERALITY TEST (a control itself): the IDENTICAL `discover()` core -- no embodiment
branches -- must solve TWO structurally different bodies:
  A) GRID: discrete cells; a SWITCH opens a far DOOR; decoy cell is inert.
  B) REACH: continuous; grabbing a TOOL extends reach radius (non-local capability);
     decoy rock is inert.
Both have a prediction-INVISIBLE affordance + a decoy. If one core solves both with
controls passing, the discover-layer is embodiment-general. Self-contained, stdlib.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math

GAIN = 56.0


@dataclass(frozen=True)
class Cand:
    id: object
    key: object          # observable feature CLASS (for generalization)
    cost: float          # probe cost (e.g. distance / energy)
    risk: float          # irreversibility risk in [0,1]
    p_int: float         # interactiveness/salience prior


class Embodiment:                                  # interface (the ONLY thing that changes per body)
    def initial(self): ...
    def achievable(self, s) -> set: ...            # the value/reachability set
    def candidates(self, s) -> list: ...           # interactable elements (Cand)
    def interact(self, s, c: Cand): ...            # apply (may be irreversible) -> new state
    def goal(self, s) -> bool: ...
    def local_change(self, s, c: Cand) -> bool: ...# does interacting change the LOCAL scene near c?


# ============================ EMBODIMENT-INVARIANT CORE ============================
def discover(emb: Embodiment, lam=1.0, mu=10.0, nu=2.0, probe_budget=1, verbose=False):
    s = emb.initial(); belief = {}; probed = set()
    def score(c, cands):
        transfer = sum(1 for o in cands if o.key == c.key and o.id not in probed)
        return c.p_int * GAIN - lam * c.cost - mu * c.risk + nu * transfer
    # PROBE: cost/risk/transfer-aware selection; learn affordance CLASS from achievability change
    while probe_budget > 0 and not emb.goal(s):
        cands = [c for c in emb.candidates(s) if c.id not in probed]
        if not cands: break
        c = max(cands, key=lambda c: score(c, cands))
        probed.add(c.id); probe_budget -= 1
        before = len(emb.achievable(s)); s = emb.interact(s, c)
        d = len(emb.achievable(s)) - before
        if d > 0: belief[c.key] = belief.get(c.key, 0) + 1
        if verbose: print(f"    probed {c.id} (key={c.key}): dAchievable={d:+d}")
    # EXPLOIT: act on believed-class members WITHOUT probing (generalization)
    for c in emb.candidates(s):
        if c.id not in probed and belief.get(c.key, 0) > 0:
            s = emb.interact(s, c)
    return {"solved": emb.goal(s), "belief": dict(belief), "probed_keys": [c for c in belief]}


def prediction_only(emb: Embodiment):
    """a model that only acts on LOCALLY-visible effects never triggers a non-local affordance."""
    s = emb.initial()
    for c in emb.candidates(s):
        if emb.local_change(s, c):                 # only interacts with locally-visible effects
            s = emb.interact(s, c)
    return emb.goal(s)
# ==================================================================================


# ----------------------------- A) GRID embodiment -----------------------------
class Grid(Embodiment):
    def __init__(self):
        H = W = 11; self.g = [[0] * W for _ in range(H)]
        for r in range(H): self.g[r][4] = 1; self.g[r][7] = 1    # two walls
        self.g[5][4] = 2; self.g[5][7] = 2                       # two DOORs (2)
        self.goal_pos = (5, 9); self.g[5][9] = 3
        self.agent = (5, 1)
        self.switches = {(3, 1): (5, 4), (9, 1): (5, 7)}         # 2 switches -> their (far) doors
        self.decoy = (5, 2)
    def initial(self): return (self.agent, frozenset())
    def _reach(self, s):
        _, opened = s; seen = {self.agent}; q = deque([self.agent])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < 11 and 0 <= n[1] < 11 and n not in seen:
                    cell = self.g[n[0]][n[1]]
                    if cell in (0, 3) or (cell == 2 and n in opened):
                        seen.add(n); q.append(n)
        return seen
    def achievable(self, s): return self._reach(s)
    def _cost(self, c): return abs(c[0] - self.agent[0]) + abs(c[1] - self.agent[1])
    def candidates(self, s):
        cs = [Cand(sw, "yellow", self._cost(sw), 0.1, 0.7) for sw in self.switches]
        cs.append(Cand(self.decoy, "green", self._cost(self.decoy), 0.1, 0.7))
        return cs
    def interact(self, s, c):
        pos, opened = s
        return (pos, opened | {self.switches[c.id]}) if c.id in self.switches else s
    def goal(self, s): return self.goal_pos in self._reach(s)
    def local_change(self, s, c):                                # door is FAR from switch -> non-local
        if c.id not in self.switches: return False
        d = self.switches[c.id]; return abs(d[0] - c.id[0]) + abs(d[1] - c.id[1]) <= 1


# ----------------------------- B) REACH (continuous tool) embodiment -----------------------------
class Reach(Embodiment):
    """Agent at origin with a reach radius. TOOLS extend reach (a non-local capability);
    a heavy ROCK is inert. Targets sit at increasing radii; the GOAL target is far and only
    becomes achievable after grabbing enough tools. Structurally unlike the grid."""
    def __init__(self):
        self.targets = [1.0, 2.0, 3.0, 4.0, 5.0]                 # radii of target points
        self.goal_r = 5.0
        self.tools = [("tool_A", 2.0), ("tool_B", 6.0)]          # (id, distance-to-grab); each +2.0 reach
        self.rock = ("rock", 1.0)
    def initial(self): return (1.5, frozenset())                 # (reach_radius, grabbed)
    def achievable(self, s):
        reach, _ = s; return {i for i, r in enumerate(self.targets) if r <= reach}
    def candidates(self, s):
        cs = [Cand(t[0], "tool", t[1], 0.1, 0.7) for t in self.tools]
        cs.append(Cand(self.rock[0], "rock", self.rock[1], 0.1, 0.7))
        return cs
    def interact(self, s, c):
        reach, grabbed = s
        if c.id in grabbed: return s
        if c.key == "tool": return (reach + 2.0, grabbed | {c.id})  # extend reach (non-local capability)
        return (reach, grabbed | {c.id})                          # rock: inert
    def goal(self, s):
        reach, _ = s; return self.goal_r <= reach
    def local_change(self, s, c):                                 # reach extension is a GLOBAL capability,
        return False                                              # never a local scene change -> prediction blind


print("=== General world model: ONE discover() core, TWO embodiments ===\n")
for name, emb in (("GRID  (discrete switch/door)", Grid()), ("REACH (continuous tool/extend)", Reach())):
    res = discover(emb, verbose=False)
    pred = prediction_only(emb)
    print(f"[{name}]")
    print(f"  discover-core: solved={res['solved']}  learned_classes={res['probed_keys']}")
    print(f"  prediction-only solves?: {pred}")
    decoy_key = "green" if isinstance(emb, Grid) else "rock"
    print(f"  decoy class '{decoy_key}' flagged as affordance?: {decoy_key in res['belief']}\n")

print("=== generality controls (SAME core, no embodiment branches) ===")
g, r = Grid(), Reach()
gr, rr = discover(g), discover(r)
checks = {
    "GRID: core solves": gr["solved"],
    "GRID: rejects decoy (green not learned)": "green" not in gr["belief"],
    "GRID: prediction-only FAILS (non-local affordance)": not prediction_only(g),
    "REACH: core solves (SAME core)": rr["solved"],
    "REACH: rejects decoy (rock not learned)": "rock" not in rr["belief"],
    "REACH: prediction-only FAILS (non-local affordance)": not prediction_only(r),
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGENERAL WORLD MODEL: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the identical embodiment-invariant discover() core solves a discrete grid AND a"
      "\n  continuous reach-extension body, discovering each non-local affordance, rejecting each decoy,"
      "\n  where local prediction fails -- with NO embodiment-specific code in the core. Tune per body by"
      "\n  writing the adapter + setting lam/mu/nu to its cost/risk profile (e.g. high mu for irreversible robots).")
