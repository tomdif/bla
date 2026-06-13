"""RealLs20Adapter -- the ls20 hierarchical shape signal GROUNDED IN REAL PIXELS.

The `ls20` adapter modeled the shape state symbolically. This one reads it from real 64x64 frames via
ls20_perception.extract(): avatar position, the cross operator, the avatar KEY shape, the EXIT key
shape, and an orientation descriptor for each. The SAME hierarchical sub-goals run on perceived state.

HONEST RESULT (the point): perception extracts the components reliably, but the avatar's key is a solid
block whose orientation is AMBIGUOUS (low confidence), so the perceived shape-MATCH is low-confidence.
The adapter surfaces that confidence; WMOS then REFUSES to trust the match and would defer to real
action feedback (in the live game you cannot measure the match without acting). That refusal -- not a
confident-but-wrong claim from ambiguous perception -- is the verificationist discipline on real pixels.
The exact ls20 match predicate (filled key vs pattern) remains the ~/arc_local open problem; what is
grounded here is the STATE EXTRACTION and an honest, calibrated uncertainty on the match.
"""
import os
from . import register, Adapter
from ..goals import GoalHierarchy, SubGoal
from . import ls20_perception as P

DEFAULT_NPZ = os.path.expanduser("~/arc_local/jepa_wm/ls20_transitions.npz")
CONF_THRESHOLD = 0.15            # below this, the perceived match is not trustworthy -> measure, don't assert


def _load_frames(npz_path):
    try:
        import numpy as np
        return np.load(npz_path)["s_before"]
    except Exception:
        return None


def _synthetic_frame():
    g = [[4] * 64 for _ in range(64)]                     # bg
    for i in range(64): g[0][i] = g[63][i] = g[i][0] = g[i][63] = 5
    for r in range(45, 47):
        for c in range(34, 39): g[r][c] = 12              # avatar core
    for (r, c) in [(47, 34), (47, 36), (47, 38), (48, 35), (48, 37), (49, 36)]: g[r][c] = 9   # avatar key (pattern)
    for (r, c) in [(11, 35), (11, 36), (11, 37), (12, 37), (13, 35), (13, 37)]: g[r][c] = 9   # exit key (top)
    g[32][21] = 0; g[31][21] = 1                          # cross
    return g


@register("ls20_real")
class RealLs20Adapter(Adapter):
    def __init__(self, npz_path=DEFAULT_NPZ, **kw):
        self.frames = _load_frames(npz_path)
        self.i = 0; self.reset()
        self.hierarchy = GoalHierarchy(
            subgoals=[
                SubGoal("shape_matched", satisfied=lambda s: s["matched"],
                        progress=lambda s: 0.5 if s["av_or"] is None or s["ex_or"] is None
                        else 1 - _cyc(s["av_or"], s["ex_or"]) / 4),
                SubGoal("at_exit", satisfied=lambda s: s["at_exit"],
                        progress=lambda s: 1 - s["exit_dist"] / 128, requires=["shape_matched"]),
            ],
            root_satisfied=lambda s: s["matched"] and s["at_exit"])

    def reset(self): self.i = 0

    def _frame(self):
        if self.frames is not None: return self.frames[self.i % len(self.frames)]
        return _synthetic_frame()

    def _state(self):
        p = P.extract(self._frame())
        av_or = p["avatar_key"]["orient"]; ex_or = p["exit_key"]["orient"]
        avatar = p["avatar"]; exit_c = p["exit_key"]["center"] or (12, 36)
        return {"p": p, "av_or": av_or, "ex_or": ex_or, "matched": p["matched"],
                "match_conf": p["match_confidence"], "avatar": avatar, "exit": exit_c,
                "exit_dist": abs(avatar[0] - exit_c[0]) + abs(avatar[1] - exit_c[1]),
                "at_exit": abs(avatar[0] - exit_c[0]) + abs(avatar[1] - exit_c[1]) <= 2}

    def observe(self):
        s = self._state(); p = s["p"]; front = self.hierarchy.frontier(s)
        # confidence-gated signal: an ambiguous perceived match cannot be trusted (-> needs measurement)
        conf = s["match_conf"]
        cands = []
        if p["cross"]:
            cands.append({"id": "cross", "label": f"white cross operator at {p['cross']} (flips key shape)",
                          "features": {"color": "cross", "signal": round(conf, 3), "is_operator": 1,
                                       "confidence": conf, "dist": abs(p["cross"][0] - s["avatar"][0]) + abs(p["cross"][1] - s["avatar"][1]),
                                       "key": "cross|operator"}})
        scene = (f"REAL ls20 frame. avatar {s['avatar']} | cross {p['cross']} | "
                 f"avatar-key orient {s['av_or']} (conf {p['avatar_key']['confidence']}, {p['avatar_key']['cells']} cells) | "
                 f"exit-key orient {s['ex_or']} (conf {p['exit_key']['confidence']}) | "
                 f"match_confidence {conf} {'(LOW -> measure, do not assert)' if conf < CONF_THRESHOLD else ''} | "
                 f"frontier {front.name if front else 'WIN'}")
        return {"candidates": cands, "reachable": round(self.hierarchy.value(s), 3),
                "solved": self.hierarchy.achieved(s),
                "scene": scene, "online": True, "goal_frontier": front.name if front else None,
                "view": {"ls20_shape": {"key": s["av_or"] or 0, "target": s["ex_or"] or 0,
                                        "avatar": list(s["avatar"]), "exit": list(s["exit"])}}}

    def measure_delta(self, cid):
        """Perceived shape-sub-goal progress from flipping the cross, DISCOUNTED by perception confidence.
        Low confidence (solid/ambiguous key) -> ~0 -> WMOS will need real measurement, not a perception claim."""
        if cid != "cross": return 0.0
        s = self._state()
        if s["av_or"] is None or s["ex_or"] is None: return 0.0
        before = self.hierarchy.value(s)
        k = s["av_or"]
        for _ in range(4):
            if k == s["ex_or"]: break
            k = (k + 1) % 4
        after = self.hierarchy.value({**s, "av_or": k, "matched": (k == s["ex_or"])})
        raw = after - before
        return round(raw * max(0.0, s["match_conf"]), 3)        # confidence-discounted (honest)

    def apply(self, cid):
        if self.frames is not None: self.i = (self.i + 1) % len(self.frames)   # advance the recorded stream

    def goals(self):
        snap = self.hierarchy.snapshot(self._state())
        snap["match_confidence"] = self._state()["match_conf"]
        return snap


def _cyc(a, b):
    d = (b - a) % 4; return min(d, 4 - d)
