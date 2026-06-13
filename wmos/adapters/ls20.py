"""Ls20Adapter -- the RICHER achievable signal for ls20's shape-match win, with HIERARCHICAL sub-goals.

ls20's win is compositional:  WIN = (key.shape == exit.shape) AND (avatar at exit).
A reachability verifier is blind to this (the cross flips a key shape, not what's reachable), so on the
arc adapter the cross is correctly REFUTED. Here the achievable signal is a hierarchical POTENTIAL over
sub-goals, so the cross's Δachievable = how much it advances the shape-match sub-goal -> it is CONFIRMED.

Sub-goal hierarchy (ordering enforced):
    WIN
    ├── shape_matched          (apply the cross operator; a multi-flip plan -- delayed payoff)
    └── at_exit  requires:[shape_matched]   (navigate; entering before matching does NOT win)

The cross cycles the key through a ring (KEYCYCLE, from the ~/arc_local ls20 findings); matching is a
plan of K flips, so measure_delta(cross) evaluates the SUB-GOAL plan, not a single step. A yellow tile
is a decoy: it advances no sub-goal of the win, so it is refuted. Faithful symbolic model (the real
shape-perception is the ~/arc_local open work); the point is the achievable signal + hierarchy.
"""
from . import register, Adapter
from ..goals import GoalHierarchy, SubGoal

KEYCYCLE = [0, 1, 2, 3]          # the operator ring: the cross advances key by +1 (mod 4)


@register("ls20")
class Ls20Adapter(Adapter):
    def __init__(self, target=2, **kw):
        self._target0 = target; self.reset()

    def reset(self):
        self.size = 11
        self.avatar = (5, 1); self.cross = (5, 5); self.exit = (5, 9); self.yellow = (2, 7)
        self.key = 0; self.target = self._target0          # need (target-key) mod 4 flips to match
        self.hierarchy = GoalHierarchy(
            subgoals=[
                SubGoal("shape_matched", satisfied=lambda s: s["key"] == s["target"],
                        progress=lambda s: 1 - _cyc(s["key"], s["target"]) / len(KEYCYCLE)),
                SubGoal("at_exit", satisfied=lambda s: s["avatar"] == s["exit"],
                        progress=lambda s: 1 - _manh(s["avatar"], s["exit"]) / (2 * self.size),
                        requires=["shape_matched"]),
            ],
            root_satisfied=lambda s: s["key"] == s["target"] and s["avatar"] == s["exit"])

    def _state(self):
        return {"avatar": self.avatar, "key": self.key, "target": self.target, "exit": self.exit}

    # ---- WMOS adapter interface ----
    def observe(self):
        st = self._state(); front = self.hierarchy.frontier(st)
        cands = [
            {"id": "cross", "label": f"white cross operator at {self.cross} (flips key shape)",
             "features": {"color": "cross", "signal": 1.0, "is_operator": 1,
                          "dist": _manh(self.avatar, self.cross), "key": "cross|operator"}},
            {"id": "yellow", "label": f"yellow tile at {self.yellow}",
             "features": {"color": "yellow", "signal": 0.3, "is_operator": 0,
                          "dist": _manh(self.avatar, self.yellow), "key": "yellow|resource"}},
        ]
        return {"candidates": cands, "reachable": round(self.hierarchy.value(st), 3),
                "solved": self.hierarchy.achieved(st),
                "scene": (f"ls20 shape-match. key={self.key} target={self.target} "
                          f"(matched={self.key == self.target}); avatar {self.avatar}; "
                          f"frontier sub-goal: {front.name if front else 'WIN'}."),
                "view": {"ls20_shape": {"key": self.key, "target": self.target,
                                        "avatar": list(self.avatar), "exit": list(self.exit)}},
                "goal_frontier": front.name if front else None}

    def measure_delta(self, cid):
        """RICHER achievable = ΔV of the hierarchical potential. For the cross, evaluate the shape sub-goal
        PLAN (flip until matched -- delayed payoff); a single flip can be cyclically non-monotone."""
        st = self._state(); before = self.hierarchy.value(st)
        if cid == "cross":
            k = self.key
            for _ in range(len(KEYCYCLE)):                 # the flip-until-matched plan
                if k == self.target: break
                k = (k + 1) % len(KEYCYCLE)
            return round(self.hierarchy.value({**st, "key": k}) - before, 3)
        return 0.0                                          # yellow decoy advances no win sub-goal

    def flat_reachability_delta(self, cid):
        """CONTROL: the old flat signal -- the cross changes no navigation reachability -> 0 (would refute)."""
        return 0.0

    def apply(self, cid):
        if cid == "cross": self.key = (self.key + 1) % len(KEYCYCLE)   # one real flip per interaction

    def goals(self):
        return self.hierarchy.snapshot(self._state())

    # ---- helpers for the full-solve path ----
    def flip_to_match(self):
        for _ in range(len(KEYCYCLE)):
            if self.key == self.target: break
            self.apply("cross")
    def go_to_exit(self):
        if self.key == self.target: self.avatar = self.exit          # navigation abstracted (corridor)


def _cyc(a, b):
    n = len(KEYCYCLE); d = (b - a) % n; return min(d, n - d)
def _manh(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
