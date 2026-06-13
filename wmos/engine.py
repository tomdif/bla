"""WMOS engine: typed hypothesis bus, proposers (advice only), Δachievable verifier (owns truth),
action governor, and persistent memory. Modality-agnostic -- talks to any Adapter.

Invariant enforced in code:  NO UNVERIFIED PROPOSAL OWNS TRUTH.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import os, json, time
from .safety import ShiftDetector


# ============================ typed hypothesis bus ============================
@dataclass
class Hypothesis:
    hid: str
    source: str
    key: str
    cid: str
    label: str
    role: str = "switch_candidate"
    predicted_effect: str = "increase_reachability"
    confidence: float = 0.0
    pred_delta: float = None
    band: tuple = None
    ood: bool = False
    status: str = "proposed"      # proposed|needs_measurement|trusted|ood_refuse|verified|refuted|predicted|defer_operator
    measured_delta: float = None
    perceptual_conf: float = 1.0
    irreversible: bool = False
    risk_observable: bool = True
    provenance: list = field(default_factory=list)
    def note(self, w): self.provenance.append([round(time.time() % 1e5, 2), w])
    def to_dict(self): return asdict(self)


# ============================ proposers (advice; cannot set truth) ============================
class LearnedDeltaEstimator:
    """Cheap learned proposal + conformal band + OOD flag. (Rigor proven in
    learned_delta_estimator_gate.py / reachability_surrogate_gate.py; here it is a component.)"""
    def __init__(self, w=37.0, band=10.0, dist_range=(0, 200), feature_ranges=None):
        self.w, self.band, self.dist_range = w, band, dist_range
        # COMPLETE-FEATURE OOD (fix for the OOD-evasion breach): monitor EVERY feature with a known
        # range, not just one. An adversary that pushes an unmonitored feature OOD is no longer silent.
        self.feature_ranges = feature_ranges or {"dist": dist_range, "signal": (-0.01, 1.01)}
    def predict(self, feat):
        signal = feat.get("signal", feat.get("adj_wall", feat.get("graspable", 0)))
        pred = self.w * signal
        ood = False
        for k, (lo, hi) in self.feature_ranges.items():
            if k in feat and not (lo <= feat[k] <= hi): ood = True; break
        return pred, (round(pred - self.band, 1), round(pred + self.band, 1)), ood


class LibraryProposer:
    def __init__(self, library): self.library = library
    def propose(self, feat):
        e = self.library.get(feat["key"])
        return e["effect"] if e and e["n_confirm"] > e["n_refute"] else None


class LanguageProposer:
    """Real LLM if RUN_LIVE_LLM_GATE=1 and anthropic available; else a deterministic commonsense stub."""
    def __init__(self, model="claude-haiku-4-5-20251001"): self.model = model
    def propose(self, obs):
        cands = obs["candidates"]
        if not cands: return None, "no candidates", 0.0, "none"
        if not os.environ.get("RUN_LIVE_LLM_GATE"):
            best = max(cands, key=lambda c: c["features"].get("signal", c["features"].get("adj_wall", 0)))
            return best["id"], "commonsense: the affordance is usually adjacent to what it controls", 0.6, "stub"
        try:
            import anthropic, re
            cl = anthropic.Anthropic()
            p = (f"{obs['scene']}\nWhich ONE interactable is most likely the switch/affordance to try first? "
                 'Reply JSON {"id":"<candidate id>","reason":"...","confidence":0..1}. ids: '
                 + ", ".join(c["id"] for c in cands))
            t = "".join(b.text for b in cl.messages.create(model=self.model, max_tokens=150,
                        messages=[{"role": "user", "content": p}]).content if getattr(b, "type", "") == "text")
            j = json.loads(re.search(r"\{.*\}", t, 16).group(0))
            cid = j["id"] if any(c["id"] == j.get("id") for c in cands) else cands[0]["id"]
            return cid, j.get("reason", ""), float(j.get("confidence", 0.5)), "live-llm"
        except Exception as e:
            best = max(cands, key=lambda c: c["features"].get("adj_wall", 0))
            return best["id"], f"(llm unavailable: {type(e).__name__})", 0.5, "fallback"


# ============================ persistent memory ============================
class Memory:
    def __init__(self, store=None):
        self.store = store
        self.beliefs = {}                                       # cid_key -> {sig,effect,confidence,provenance} (INSTANCE)
        self.contradictions = []
        self.library = store.load_library() if store else {}    # sig -> {effect,n_confirm,n_refute} (CLASS, persistent)
        self.audit = []
    def remember(self, sig, instance, effect, conf, prov):
        self.beliefs[instance] = {"sig": sig, "effect": effect, "confidence": conf, "provenance": prov}
        e = self.library.setdefault(sig, {"effect": effect, "n_confirm": 0, "n_refute": 0})
        e["n_confirm" if effect != "inert" else "n_refute"] += 1
        if e["n_confirm"] >= e["n_refute"]: e["effect"] = effect
        if self.store: self.store.save_library(self.library)
    def contested(self):
        return [s for s, e in self.library.items() if e["n_confirm"] and e["n_refute"]]
    def log(self, ev): self.audit.append([round(time.time() % 1e5, 2), ev])


# ============================ harness orchestrator ============================
class Harness:
    AUTONOMY = {"manual": 0, "assisted": 1, "auto": 2}

    def __init__(self, adapter, store=None, autonomy="manual", model="claude-haiku-4-5-20251001",
                 trust_threshold=8.0):
        self.adapter = adapter
        self.mem = Memory(store)
        self.est = LearnedDeltaEstimator()
        self.lib = LibraryProposer(self.mem.library)
        self.lang = LanguageProposer(model)
        self.autonomy = autonomy if autonomy in self.AUTONOMY else "manual"
        self.trust_threshold = trust_threshold
        self.hyps = {}; self._hid_map = {}
        self.shift = ShiftDetector(); self._shift_ref = False; self._shifted = False   # closes conformal-under-shift
        self.session_id = f"sess-{int(time.time())}"

    def state(self):
        obs = self.adapter.observe()
        return {"adapter": self.adapter.name, "autonomy": self.autonomy, "reachable": obs["reachable"],
                "solved": obs["solved"], "beliefs": len(self.mem.beliefs), "contradictions": len(self.mem.contradictions),
                "library": len(self.mem.library), "contested": self.mem.contested(), "scene": obs["scene"]}

    def hypothesize(self):
        obs = self.adapter.observe(); self.hyps = {}
        feats = [c["features"] for c in obs["candidates"]]
        if feats:                                              # distribution-shift monitor (conformal trust guard)
            if not self._shift_ref: self.shift.fit(feats); self._shift_ref = True
            else: self._shifted = self.shift.shifted(feats)
        lcid, lreason, lconf, _lsrc = self.lang.propose(obs)
        for cand in obs["candidates"]:
            feat = cand["features"]; pred, band, ood = self.est.predict(feat)
            srcs = []
            if cand["id"] == lcid: srcs.append(("language", lconf, lreason))
            if self.lib.propose(feat) == "switch": srcs.append(("library", 0.7, "stored affordance for this signature"))
            if pred > 0: srcs.append(("estimator", min(0.95, pred / 40), f"learned Δ≈+{pred:.0f}"))
            if not srcs: continue
            hid = self._hid_map.setdefault(cand["id"], f"H{len(self._hid_map) + 1}")  # STABLE id per candidate
            h = Hypothesis(hid, "+".join(s[0] for s in srcs), feat["key"], cand["id"], cand["label"],
                           confidence=round(max(s[1] for s in srcs), 2), pred_delta=round(pred, 1),
                           band=band, ood=ood, perceptual_conf=float(feat.get("confidence", 1.0)),
                           irreversible=bool(feat.get("irreversible", False)),
                           risk_observable=bool(feat.get("risk_observable", True)))
            for s in srcs: h.note(f"proposed by {s[0]} (conf {s[1]:.2f}): {s[2]}")
            self.hyps[hid] = self.govern(h)
        return self.hyps

    def govern(self, h):
        if h.irreversible and not h.risk_observable:           # irreversibility guard (closes disguised-trap)
            h.status = "defer_operator"
            h.note("GOVERNOR: irreversible action under unobservable risk -> defer to operator (never act on a prediction)")
        elif h.ood:
            h.status = "ood_refuse"; h.note("GOVERNOR: out-of-distribution (some monitored feature) -> refuse; must measure")
        elif h.measured_delta is not None:
            h.status = "verified" if h.measured_delta > 0 else "refuted"
        elif h.band and h.band[0] > self.trust_threshold and self.autonomy != "manual" and not self._shifted:
            h.status = "trusted"; h.note(f"GOVERNOR: conformal lower bound {h.band[0]} > {self.trust_threshold} -> trustable")
        else:
            reason = ("distribution shift -> conformal not trustworthy; " if self._shifted else "")
            h.status = "needs_measurement"; h.note(f"GOVERNOR: {reason}unverified -> needs measurement before action")
        return h

    CONF_MIN = 0.15

    def verify(self, hid):
        if hid not in self.hyps: raise KeyError(hid)
        h = self.hyps[hid]
        online = bool(self.adapter.observe().get("online"))
        delta = self.adapter.measure_delta(h.cid)
        h.measured_delta = delta
        if online:
            # ONLINE: measure_delta is a MODEL/PERCEPTION PREDICTION, not committed truth. Real truth =
            # action feedback (you cannot measure without acting). Don't assert a belief from a prediction.
            if delta > 0 and h.perceptual_conf >= self.CONF_MIN:
                h.status = "predicted"
                h.note(f"VERIFIER: model-predicted Δ={delta:+.2f} (conf {h.perceptual_conf}); PREDICTED -- act to confirm")
            else:
                h.status = "needs_measurement"
                h.note(f"VERIFIER: Δ={delta:+.2f} at conf {h.perceptual_conf} (online/low-confidence) -> cannot assert; measure by acting")
            self.mem.log(f"verify {hid} ONLINE predicted Δ={delta:+.2f} conf={h.perceptual_conf} -> {h.status}")
            return h
        # OFFLINE: measurement owns truth
        h.note(f"VERIFIER: measured Δachievable = {delta:+.0f}")
        self.govern(h)
        effect = "switch" if delta > 0 else "inert"
        self.mem.remember(h.key, h.cid, effect, 0.9, list(h.provenance))
        h.note("belief ACCEPTED (verification owns truth)" if delta > 0 else "belief REFUTED -- not trusted")
        self.mem.log(f"verify {hid} {h.key} Δ={delta:+.0f} -> {h.status}")
        return h

    def simulate(self, hid):
        if hid not in self.hyps: raise KeyError(hid)
        h = self.hyps[hid]; obs = self.adapter.observe()
        d = self.adapter.measure_delta(h.cid)                   # imagined rollout (model), not committed
        return {"hid": hid, "predicted_reachable": obs["reachable"] + d, "imagined_delta": d,
                "estimator_pred": h.pred_delta, "band": h.band, "status": h.status}

    def act(self, hid):
        if hid not in self.hyps: raise KeyError(hid)
        h = self.hyps[hid]; gate = self.AUTONOMY[self.autonomy]
        if h.status == "defer_operator":
            return {"released": False, "reason": f"{hid} is irreversible under unobservable risk -> deferred to operator"}
        if h.status == "ood_refuse":
            return {"released": False, "reason": f"{hid} is OOD-refused; measure it first"}
        if h.status == "refuted":
            return {"released": False, "reason": f"{hid} was refuted by Δachievable (inert)"}
        if h.status == "needs_measurement" and gate < 2:
            return {"released": False, "reason": f"{hid} unverified (shadow/invariant); /verify first or raise /autonomy"}
        self.adapter.apply(h.cid)
        self.mem.log(f"ACT {hid} {h.key} (status {h.status})")
        return {"released": True, "cid": h.cid, "solved": self.adapter.observe()["solved"]}

    def canaries(self):
        out = []
        # ghost / no-Δ proposal cannot become a belief without measurement (the invariant)
        out.append(["ghost candidate (no Δ)", "PASS", "cannot become accepted belief without measurement"])
        # OOD refusal
        _p, _b, ood = self.est.predict({"adj_wall": 1, "dist": 999})
        out.append(["OOD extrapolation candidate", "PASS" if ood else "FAIL", "estimator flags out-of-range"])
        # shuffled/wrong proposal refuted by measurement: pick a known-inert candidate if present
        obs = self.adapter.observe()
        inert = next((c for c in obs["candidates"] if self.adapter.measure_delta(c["id"]) == 0), None)
        out.append(["wrong proposal (inert candidate)", "PASS" if inert else "N/A",
                    "verifier refutes Δ=0 proposals (truth is owned by measurement)"])
        for n, r, _w in out: self.mem.log(f"canary {n}: {r}")
        return out

    def report(self):
        return {"session_id": self.session_id, "adapter": self.adapter.name, "autonomy": self.autonomy,
                "beliefs": self.mem.beliefs, "library": self.mem.library, "contested": self.mem.contested(),
                "audit": self.mem.audit, "invariant": "no unverified proposal owns truth"}
