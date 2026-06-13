"""Pluggable world adapters. An adapter owns PERCEPTION + DYNAMICS + MEASUREMENT for one modality;
the engine (proposers/governor/verifier/memory) is modality-agnostic and talks only to this
interface. Swap the adapter to point WMOS at a new world (grid game, reach body, real ARC, robot).

Contract:
    reset()                 -> None             begin / restart the episode
    observe()               -> dict             {candidates:[{id,label,features}], reachable:int,
                                                 solved:bool, view:{...for the cockpit}, scene:str}
    measure_delta(cid)      -> float            Δachievable of interacting with candidate cid (NO commit)
    apply(cid)              -> None             commit the interaction (real action feedback)
    solid(cell)            -> bool             (optional) physical truth for contradiction handling
"""
from collections import deque

_REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name; _REGISTRY[name] = cls
        return cls
    return deco


def get_adapter(name, **kw):
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


def list_adapters():
    return sorted(_REGISTRY)


class Adapter:
    name = "base"
    def reset(self): raise NotImplementedError
    def observe(self): raise NotImplementedError
    def measure_delta(self, cid): raise NotImplementedError
    def apply(self, cid): raise NotImplementedError
    def solid(self, cell): return False


# ----------------------------- discrete grid (mock ls20) -----------------------------
FLOOR, WALL, AGENT, GOAL, YELLOW, GREEN = 0, 1, 2, 3, 4, 5


@register("grid")
class GridAdapter(Adapter):
    """A locked door blocks the goal. A YELLOW switch (by the wall) opens it; an identical-looking
    YELLOW trap is inert; a GREEN object in the open is a decoy; a disguised floor-colored cell is solid."""
    def __init__(self, **kw):
        self.reset()
    def reset(self):
        self.H = self.W = 9
        self.agent = (4, 1); self.goal = (4, 7); self.door = (4, 4)
        self.walls = {(r, 4) for r in range(9)}
        self.objs = {"switch": ((1, 3), YELLOW, "switch"), "trap": ((7, 3), YELLOW, "inert"),
                     "decoy": ((6, 1), GREEN, "inert")}
        self.disguised = (4, 5); self.opened = set()
    def _grid(self):
        g = [[FLOOR] * self.W for _ in range(self.H)]
        for (r, c) in self.walls: g[r][c] = WALL
        if self.door in self.opened: g[self.door[0]][self.door[1]] = FLOOR
        for _id, (cell, col, _role) in self.objs.items(): g[cell[0]][cell[1]] = col
        g[self.goal[0]][self.goal[1]] = GOAL; g[self.agent[0]][self.agent[1]] = AGENT
        return g
    def _reach(self):
        g = self._grid()
        passable = {(r, c) for r in range(self.H) for c in range(self.W) if g[r][c] in (FLOOR, GOAL, AGENT)}
        seen = {self.agent}; q = deque([self.agent])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in passable and n not in seen: seen.add(n); q.append(n)
        return len(seen - {cell for cell, _, _ in self.objs.values()})
    def observe(self):
        g = self._grid()
        cands = []
        for cid, (cell, col, _role) in self.objs.items():
            adj_wall = any(0 <= cell[0] + dr < self.H and 0 <= cell[1] + dc < self.W and g[cell[0] + dr][cell[1] + dc] == WALL
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            label = {YELLOW: "yellow", GREEN: "green"}[col]
            cands.append({"id": cid, "label": f"{label} object at {cell}",
                          "features": {"color": label, "adj_wall": int(adj_wall), "signal": int(adj_wall),
                                       "dist": abs(cell[0] - self.agent[0]) + abs(cell[1] - self.agent[1]),
                                       "key": f"{label}|{'adj_wall' if adj_wall else 'open'}"}})
        scene = "A locked door blocks the exit. Interactables: " + "; ".join(c["label"] for c in cands) + "."
        return {"candidates": cands, "reachable": self._reach(), "solved": self.door in self.opened,
                "scene": scene, "view": {"grid": g, "disguised": self.disguised, "legend": _GRID_LEGEND}}
    def _snapshot(self): return set(self.opened)
    def _restore(self, s): self.opened = set(s)
    def measure_delta(self, cid):
        before = self._reach(); snap = self._snapshot()
        if cid == "switch": self.opened.add(self.door)          # only the real switch changes reachability
        delta = self._reach() - before; self._restore(snap)
        return float(delta)
    def apply(self, cid):
        if cid == "switch": self.opened.add(self.door)
    def solid(self, cell):
        if cell == self.door: return self.door not in self.opened
        return cell in self.walls or cell == self.disguised


_GRID_LEGEND = {0: ".", 1: "#", 2: "@", 3: "G", 4: "Y", 5: "g"}


# ----------------------------- continuous reach body (cross-modality) -----------------------------
@register("reach")
class ReachAdapter(Adapter):
    """A continuous body with a reach radius. Grabbing a TOOL extends reach (a non-local capability);
    a ROCK is inert. Targets sit at radii; the goal target becomes reachable after grabbing tools."""
    def __init__(self, **kw): self.reset()
    def reset(self):
        self.radii = [1.0, 2.0, 3.0, 4.0, 5.0]; self.goal_r = 5.0
        self.objs = {"tool_A": (2.0, "tool"), "tool_B": (6.0, "tool"), "rock": (1.0, "rock")}
        self.reach = 1.5; self.grabbed = set()
    def _ach(self, reach): return sum(1 for r in self.radii if r <= reach)
    def observe(self):
        cands = [{"id": cid, "label": f"{('tool' if k=='tool' else 'rock')} at distance {d}",
                  "features": {"color": k, "graspable": 1, "signal": 1, "dist": d, "key": f"{k}|graspable"}}
                 for cid, (d, k) in self.objs.items() if cid not in self.grabbed]
        return {"candidates": cands, "reachable": self._ach(self.reach), "solved": self.reach >= self.goal_r,
                "scene": "A reach body; targets sit at increasing radii. Interactables: "
                         + "; ".join(c["label"] for c in cands) + ".",
                "view": {"reach": round(self.reach, 2), "targets": self.radii, "goal_r": self.goal_r}}
    def measure_delta(self, cid):
        before = self._ach(self.reach); _d, k = self.objs[cid]
        after = self._ach(self.reach + (2.0 if k == "tool" else 0.0))
        return float(after - before)
    def apply(self, cid):
        _d, k = self.objs[cid]
        if k == "tool" and cid not in self.grabbed: self.reach += 2.0; self.grabbed.add(cid)


from . import ls20  # noqa: F401  (registers "ls20"; stdlib -- richer achievable + hierarchical sub-goals)
from . import ls20_real  # noqa: F401  (registers "ls20_real"; shape state grounded in real pixels)
from . import reach3d  # noqa: F401  (registers "reach3d"; 3D GeometryCanvas + reach affordance, stdlib)

# register adapters that need optional heavy imports (numpy) without breaking the base package
try:
    from . import arc  # noqa: F401  (registers "arc"; needs numpy + optional recorded frames)
except Exception:
    pass
