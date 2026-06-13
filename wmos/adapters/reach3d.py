"""Reach3DAdapter -- WMOS over a 3D scene through a GeometryCanvas. The achievable set is the
reachable objects (3D reach radius); grasping a reachable TOOL extends reach -> grows achievability.

The 2D-TRAP is built in: a reachable TOOL and a far (unreachable) look-alike share the SAME 2D
projection. A 2D-appearance agent can't tell them apart; the GeometryCanvas resolves depth (stereo)
and the verifier (a reach attempt = action feedback) owns truth. With monocular depth the canvas is
underdetermined and WMOS refuses to assert reachability (the geometry_canvas_gate discipline).
"""
from . import register, Adapter
from ..geometry import GeometryCanvas, dist3, project


@register("reach3d")
class Reach3DAdapter(Adapter):
    def __init__(self, stereo=True, **kw):
        self.stereo = stereo; self.reset()

    def reset(self):
        self.reach = 1.5
        self.objs = {
            "tool":      {"pos": (0.5, 0.0, 1.0), "type": "tool",  "dR": 1.5},   # reachable; grasp extends reach
            "lookalike": {"pos": (1.0, 0.0, 2.0), "type": "target", "dR": 0.0},  # SAME 2D projection, but far -> not reachable
            "decoy":     {"pos": (0.3, 0.3, 0.9), "type": "decoy", "dR": 0.0},   # reachable but inert
            "far":       {"pos": (1.4, 0.2, 2.6), "type": "target", "dR": 0.0},  # only reachable after the tool extends reach
        }
        self.grabbed = set()
        self.canvas = GeometryCanvas(reach=self.reach)

    def _reachable(self, reach):
        return {oid for oid, o in self.objs.items() if dist3(o["pos"]) <= reach}

    def _canvas(self):
        self.canvas.reach = self.reach
        return self.canvas.estimate({oid: o["pos"] for oid, o in self.objs.items()}, stereo=self.stereo)

    def observe(self):
        est = self._canvas()
        cands = []
        for oid, o in self.objs.items():
            if oid in self.grabbed: continue
            e = est[oid]
            reachable = e["reach_pred"]
            signal = 1.0 if reachable is True else (0.3 if reachable is False else 0.5)  # None(OOD)->0.5
            cands.append({"id": oid, "label": f"{o['type']} '{oid}' (depth {e['depth']}, reach_pred {reachable})",
                          "features": {"color": o["type"], "signal": signal, "depth": e["depth"],
                                       "confidence": 1.0 if not self.canvas else (0.9 if self.stereo else 0.3),
                                       "dist": round(dist3(o["pos"]), 2), "key": f"{o['type']}|graspable"}})
        return {"candidates": cands, "reachable": len(self._reachable(self.reach)),
                "solved": "far" in self._reachable(self.reach) and "tool" in self.grabbed,
                "online": not self.stereo,                       # monocular depth is a prediction, not committed truth
                "scene": (f"3D reach scene (camera=agent). reach radius {self.reach}. "
                          f"objects {len(self.objs)}; reachable now {len(self._reachable(self.reach))}. "
                          f"NB tool and look-alike share the 2D projection {tuple(round(v,2) for v in project(self.objs['tool']['pos']))} "
                          f"-- only depth distinguishes them."),
                "view": {"geometry": self.geometry()}}

    def measure_delta(self, cid):
        """Δachievable = new reachable objects if we grasp cid. A reachable TOOL extends reach; an
        unreachable look-alike or an inert decoy yields 0. (Action feedback = the real reach attempt.)"""
        o = self.objs.get(cid)
        if not o or dist3(o["pos"]) > self.reach: return 0.0     # can't grasp what you can't reach (the 2D trap)
        before = self._reachable(self.reach)
        after = self._reachable(self.reach + o["dR"])
        return float(len(after - before))

    def apply(self, cid):
        o = self.objs.get(cid)
        if o and cid not in self.grabbed and dist3(o["pos"]) <= self.reach and o["type"] == "tool":
            self.reach += o["dR"]; self.grabbed.add(cid)

    # ---- methods the harness 3D tools consume ----
    def geometry(self):
        est = self._canvas()
        return {"reach_radius": self.reach, "camera": (0, 0, 0), "stereo": self.stereo,
                "reachable": sorted(self._reachable(self.reach)), "grabbed": sorted(self.grabbed),
                "objects": [{"id": oid, "type": o["type"], "pose3d": est[oid]["pose3d"],
                             "depth": est[oid]["depth"], "depth_sigma": est[oid]["depth_sigma"],
                             "reach_pred": est[oid]["reach_pred"], "ood": est[oid]["ood"],
                             "reproj_err": est[oid]["reproj_err"], "uv": est[oid]["uv"]}
                            for oid, o in self.objs.items()]}

    def depth_explain(self, cid):
        if cid not in self.objs: return None
        e = self._canvas()[cid]
        mode = "stereo (2 views triangulated)" if self.stereo else "monocular (1 view -- depth underdetermined)"
        verdict = ("REFUSE: depth underdetermined, measure by reaching" if e["ood"]
                   else (f"reachable (est_dist {e['est_dist']} + sigma {e['depth_sigma']} <= reach {self.reach})"
                         if e["reach_pred"] else f"not reachable (est_dist {e['est_dist']} > reach {self.reach})"))
        return {"id": cid, "mode": mode, "depth": e["depth"], "depth_sigma": e["depth_sigma"],
                "est_dist": e["est_dist"], "reproj_err": e["reproj_err"], "uv": e["uv"],
                "ood": e["ood"], "reach_pred": e["reach_pred"], "verdict": verdict}

    def simulate_reach(self, cid):
        d = self.measure_delta(cid)
        e = self._canvas().get(cid, {})
        return {"id": cid, "imagined_delta_reachable": d, "depth": e.get("depth"),
                "reach_pred": e.get("reach_pred"), "note": "imagined (no commit); verify = a real reach attempt"}
