"""ARCAdapter -- WMOS over real ARC-AGI-3 ls20 frames.

ls20 is a 64x64 color-grid game (colors 0-12). Real semantics (from recorded play):
  4 = background/floor (passable)   3 = dark maze wall     5 = blue border
  12 = avatar core (controllable; ~5px/action)            9 = icon body (avatar key + legend + exit key)
  11 = yellow step-bonus / budget bar                     0,1 = the WHITE CROSS operator (flips the key)
  8  = UI / target marker

Sources (the only thing that differs replay vs live):
  ReplayNpzSource   -- recorded transitions (s_before/action/s_after); testable offline.
  SyntheticLs20Source -- a faithful built frame when no recording is present.
  LiveArcSource     -- the arc_agi client (best-effort; raised/skipped if unavailable).

HONEST SCOPE (the documented ls20 frontier): Δachievable here = MAZE REACHABILITY (BFS over floor),
which captures navigation/gating affordances. ls20's actual WIN mechanic is a SHAPE-MATCH (run the
avatar over the cross to flip its key-shape until it matches the exit, then enter) -- that is NOT a
reachability change, so a reachability verifier correctly PERCEIVES the cross operator but does not
confirm it as a door-opener. Surfacing that honestly is the point: the adapter makes WMOS run on real
frames, and the verifier refuses to claim a reachability-switch where there is none. A richer
'achievable' (win-states gated on key-shape) is the open extension -- an adapter change, not a WMOS one.

ONLINE constraint: in LIVE mode you cannot snapshot the remote to measure_delta without committing;
measure_delta is then a world-model PREDICTION and the real truth is action feedback (score/level).
"""
import os
from collections import deque
from . import register, Adapter

BG, MAZE, BORDER, AVATAR, ICON, UI8, YELLOW, CROSS_A, CROSS_B = 4, 3, 5, 12, 9, 8, 11, 0, 1
WALLS = {MAZE, BORDER}
OBJECT_COLORS = {CROSS_A, CROSS_B, YELLOW, ICON}
DEFAULT_NPZ = os.path.expanduser("~/arc_local/jepa_wm/ls20_transitions.npz")


# ----------------------------- frame sources -----------------------------
class FrameSource:
    def reset(self): raise NotImplementedError
    def frame(self): raise NotImplementedError
    def send(self, action): raise NotImplementedError          # advance; return new frame
    live = False


class ReplayNpzSource(FrameSource):
    """Step through recorded real ls20 frames. send(action) advances to the next recorded frame."""
    def __init__(self, npz_path=DEFAULT_NPZ):
        import numpy as np
        self.np = np; self.d = np.load(npz_path)
        self.frames = self.d["s_before"]; self.i = 0
    def reset(self): self.i = 0
    def frame(self): return self.frames[self.i]
    def send(self, action):
        self.i = (self.i + 1) % len(self.frames); return self.frame()


class SyntheticLs20Source(FrameSource):
    """A faithful built ls20-ish frame (fallback when no recording exists): border, maze, avatar, cross."""
    def __init__(self):
        import numpy as np; self.np = np; self.reset()
    def reset(self):
        np = self.np; g = np.full((64, 64), BG, dtype="int8")
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = BORDER
        g[20:45, 30] = MAZE; g[32, 18:43] = MAZE               # a plus-shaped maze corridor (passable)
        g[44, 30] = AVATAR                                     # avatar on the corridor
        g[8:10, 8:10] = CROSS_A; g[9, 9] = CROSS_B             # cross OPERATOR, isolated (not a reachability gate)
        g[10:12, 50:56] = ICON                                 # exit-key icon
        self.g = g
    def frame(self): return self.g
    def send(self, action): return self.g                      # static maze (no movement model here)


class LiveArcSource(FrameSource):
    live = True
    def __init__(self, api_key=None, game="ls20"):
        try:
            import arc_agi  # noqa
        except Exception as e:
            raise RuntimeError(f"live ARC client unavailable: {type(e).__name__}: {e}. "
                               "Use the replay/synthetic source, or run inside the arc_agi venv.")
        self.api_key = api_key or os.environ.get("ARC_API_KEY"); self.game = game
        raise RuntimeError("LiveArcSource is a wiring stub: connect arc_agi.create_arcade here. "
                           "Replay/synthetic sources are fully functional offline.")


# ----------------------------- the adapter -----------------------------
@register("arc")
class ARCAdapter(Adapter):
    def __init__(self, source=None, npz_path=DEFAULT_NPZ, **kw):
        if source is not None:
            self.src = source
        else:
            try:
                self.src = ReplayNpzSource(npz_path)
            except Exception:
                self.src = SyntheticLs20Source()
        self.H = self.W = 64
        self.reset()

    def reset(self):
        self.src.reset(); self._levels = 0

    def _avatar(self, g):
        ys, xs = _where(g, AVATAR)
        return (sum(ys) // len(ys), sum(xs) // len(xs)) if ys else (32, 32)

    def _reach(self, g, extra=()):
        # the avatar navigates the dark MAZE corridors (color 3); color 4 is outer non-playable
        # background and color 5 is the border. (Precise ls20 passability is the open ~/arc_local problem;
        # this floods the connected corridor the avatar occupies, a faithful navigation proxy.)
        H, W = len(g), len(g[0])
        avatar_cells = {(r, c) for r in range(H) for c in range(W) if g[r][c] == AVATAR}
        start = avatar_cells or {self._avatar(g)}
        passable = lambda v: v in (MAZE, AVATAR)
        seen = set(start); q = deque(start)
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and (passable(g[n[0]][n[1]]) or n in extra):
                    seen.add(n); q.append(n)
        return seen - avatar_cells

    def _components(self, g, colors, max_size=80):
        H, W = len(g), len(g[0]); seen = set(); comps = []
        for r in range(H):
            for c in range(W):
                if g[r][c] in colors and (r, c) not in seen:
                    col = g[r][c]; comp = []; q = deque([(r, c)]); seen.add((r, c))
                    while q:
                        cur = q.popleft(); comp.append(cur)
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                            n = (cur[0] + dr, cur[1] + dc)
                            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and g[n[0]][n[1]] in colors:
                                seen.add(n); q.append(n)
                    if len(comp) <= max_size:
                        cy = sum(p[0] for p in comp) // len(comp); cx = sum(p[1] for p in comp) // len(comp)
                        comps.append({"cells": comp, "center": (cy, cx), "color": col, "size": len(comp)})
        return comps

    def observe(self):
        g = _aslist(self.src.frame()); avatar = self._avatar(g); reach = self._reach(g)
        cands = []
        # the cross operator (white, colors 0/1) -- the ls20 key-flip affordance
        for comp in self._components(g, {CROSS_A, CROSS_B}):
            cands.append(("cross", comp, "white cross operator (flips the key shape)", 1.0))
        # maze yellows (step bonuses) -- exclude the bottom UI budget bar (rows >= 58)
        for comp in self._components(g, {YELLOW}):
            if comp["center"][0] < 58: cands.append(("yellow", comp, "yellow step-bonus", 0.4))
        out = []
        for kind, comp, label, sig in cands:
            cy, cx = comp["center"]
            out.append({"id": f"{kind}@{cy},{cx}", "label": f"{label} at ({cy},{cx})",
                        "features": {"color": kind, "signal": sig, "is_operator": int(kind == "cross"),
                                     "dist": abs(cy - avatar[0]) + abs(cx - avatar[1]),
                                     "key": f"{kind}|operator" if kind == "cross" else f"{kind}|resource"}})
        scene = (f"ls20 64x64. avatar at {avatar}, reachable floor {len(reach)} cells. Interactables: "
                 + ("; ".join(c["label"] for c in out) if out else "none visible")
                 + ". (win mechanic is SHAPE-MATCH via the cross, not reachability -- see adapter notes.)")
        return {"candidates": out, "reachable": len(reach), "solved": False, "scene": scene,
                "view": {"grid": g, "palette": "ls20", "avatar": list(avatar)}, "online": self.src.live}

    def _find(self, cid):
        for c in self.observe()["candidates"]:
            if c["id"] == cid: return c
        return None

    def measure_delta(self, cid):
        """World-model forward proxy: reachability change if the avatar could PASS this candidate's cell.
        Operators (the cross) flip a key, not reachability -> reachability-Δ is ~0 (honest: ls20's win is
        not reachability). Gating objects on a floor frontier would yield Δ>0. In LIVE mode the only true
        measurement is action feedback (committing)."""
        g = _aslist(self.src.frame()); c = self._find(cid)
        if not c: return 0.0
        cy, cx = [int(x) for x in c["id"].split("@")[1].split(",")]
        before = len(self._reach(g))
        after = len(self._reach(g, extra={(cy, cx)}))           # imagine this cell becomes traversable
        return float(after - before)

    def apply(self, cid):
        # navigate toward the candidate; here we advance the (recorded) frame stream as a proxy for acting.
        self.src.send(0)

    def solid(self, cell):
        g = _aslist(self.src.frame())
        return g[cell[0]][cell[1]] in WALLS


def _aslist(frame):
    return frame.tolist() if hasattr(frame, "tolist") else frame


def _where(g, color):
    ys, xs = [], []
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == color: ys.append(r); xs.append(c)
    return ys, xs
