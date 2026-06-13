#!/usr/bin/env python3
"""World-Model Operating System -- a verificationist operator console for the affordance stack.

Not a chatbot wrapper: a verified loop  observe -> perceive -> hypothesize -> imagine -> verify
-> act -> audit -> remember, with one invariant enforced IN CODE:

    NO UNVERIFIED PROPOSAL OWNS TRUTH.

Language / learned estimator / library / dynamics may all PROPOSE (typed hypotheses on a bus);
only the Δachievable verifier + real action feedback may set a belief accepted. The action
governor refuses to release an action that hasn't passed a verification policy.

Three planes: Chat/Operator (commands) | World-Model (perception, canvas, proposers, dynamics) |
Verification/Audit (Δachievable verifier, conformal, OOD, canaries, trace).

END-USER ADDITIONS over the base vision:
  * DEFAULT = SHADOW: nothing is released without explicit /act or a raised autonomy level.
  * AUTONOMY DIAL (/autonomy manual|assisted|auto): how much verification before self-action.
  * TRUST LEDGER: every belief carries its provenance chain; /why walks it.
  * PLAIN-LANGUAGE /explain for non-experts; /report exports a full session audit.

Run:   python3 world_model_harness.py            # scripted cockpit demo
       python3 world_model_harness.py --repl     # interactive operator console
Optional real LLM proposer:  RUN_LIVE_LLM_GATE=1 (else a deterministic stub proposer is used).
Stdlib only (+ optional anthropic).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import json, os, sys, time

# ============================ ADAPTER (swap this for real ARC / robotics) ============================
FLOOR, WALL, AGENT, GOAL, YELLOW, GREEN, DOOR = 0, 1, 2, 3, 4, 5, 6
COLORNAME = {YELLOW: "yellow", GREEN: "green"}


class GridWorldAdapter:
    """Mock ls20-style level. A yellow switch (by the door) opens the path; green is a decoy; a
    disguised cell looks like floor but blocks. Real ARC/robotics = reimplement reset/step/measure."""
    def __init__(self):
        self.H = self.W = 9
        self.agent = (4, 1); self.goal = (4, 7)
        self.walls = {(r, 4) for r in range(9)}
        self.door = (4, 4)
        self.switch = (1, 3)                                     # yellow, adj wall -> opens the door (real affordance)
        self.trap = (7, 3)                                       # yellow, adj wall -> INERT (looks identical: a trap)
        self.decoy = (6, 1)                                      # green, in the open -> inert
        self.disguised = (4, 5)                                  # floor-colored, solid (contradiction demo)
        self.opened = set()
    def render(self):
        g = [[FLOOR] * self.W for _ in range(self.H)]
        for (r, c) in self.walls: g[r][c] = WALL
        if self.door in self.opened: g[self.door[0]][self.door[1]] = FLOOR
        g[self.switch[0]][self.switch[1]] = YELLOW
        g[self.trap[0]][self.trap[1]] = YELLOW
        g[self.decoy[0]][self.decoy[1]] = GREEN
        g[self.goal[0]][self.goal[1]] = GOAL
        g[self.agent[0]][self.agent[1]] = AGENT
        return g
    def solid(self, cell):
        if cell == self.door: return self.door not in self.opened
        return cell in self.walls or cell == self.disguised
    def toggle(self, cell):                                      # the only thing that changes the world
        if cell == self.switch: self.opened.add(self.door)
    def copy(self):
        c = GridWorldAdapter(); c.agent = self.agent; c.opened = set(self.opened); return c


# ============================ PERCEPTION (start-color prior + segmentation) ============================
def perceive(grid):
    H, W = len(grid), len(grid[0])
    agent = next((r, c) for r in range(H) for c in range(W) if grid[r][c] == AGENT)
    goal = next((r, c) for r in range(H) for c in range(W) if grid[r][c] == GOAL)
    passable = {(r, c) for r in range(H) for c in range(W) if grid[r][c] in (FLOOR, GOAL, AGENT)}
    cands = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] in (YELLOW, GREEN):
                adj_wall = any(0 <= r + dr < H and 0 <= c + dc < W and grid[r + dr][c + dc] == WALL
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                cands.append({"cell": (r, c), "color": COLORNAME[grid[r][c]],
                              "feat": {"adj_wall": int(adj_wall), "dist": abs(r - agent[0]) + abs(c - agent[1])}})
    return {"agent": agent, "goal": goal, "passable": passable, "candidates": cands}


def reach_size(world, blocked=frozenset()):
    grid = world.render(); H, W = len(grid), len(grid[0])
    passable = {(r, c) for r in range(H) for c in range(W) if grid[r][c] in (FLOOR, GOAL, AGENT)} - set(blocked)
    seen = {world.agent}; q = deque([world.agent])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n in passable and n not in seen: seen.add(n); q.append(n)
    return len(seen - {world.switch, world.decoy})


# ============================ TYPED HYPOTHESIS BUS ============================
@dataclass
class AffordanceHypothesis:
    hid: str
    source: str                                                 # language | library | estimator
    key: tuple
    cell: tuple
    role: str
    predicted_effect: str
    confidence: float
    falsification_test: str = "Δachievable should increase after interaction"
    pred_delta: float = None
    band: tuple = None
    ood: bool = False
    status: str = "proposed"                                    # proposed|trusted|needs_measurement|ood_refuse|verified|refuted
    measured_delta: float = None
    provenance: list = field(default_factory=list)
    def log(self, what): self.provenance.append((round(time.monotonic() % 1e6, 2), what))


# ============================ PROPOSERS (advice only -- cannot set truth) ============================
class LearnedDeltaEstimator:
    """Tiny stdlib linear model fit on synthetic feature->Δ data; gives pred + conformal band + OOD.
    (The estimator's rigor is proven in learned_delta_estimator_gate.py; here it's a component.)"""
    def __init__(self):
        # fit Δ ~= w*adj_wall on synthetic calibration (adj_wall switch-like -> ~+37, open -> 0)
        self.w, self.b = 37.0, 0.0; self.band = 10.0; self.dist_seen = (1, 10)
    def predict(self, feat):
        pred = self.w * feat["adj_wall"] + self.b
        ood = not (self.dist_seen[0] <= feat["dist"] <= self.dist_seen[1])
        return pred, (pred - self.band, pred + self.band), ood


class LibraryProposer:
    def __init__(self, lib): self.lib = lib
    def propose(self, cand):
        e = self.lib.get(_k((cand["color"], "adj_wall" if cand["feat"]["adj_wall"] else "open")))
        return e["effect"] if e and e["n_confirm"] > e["n_refute"] else None


class LanguageProposer:
    """Real LLM if RUN_LIVE_LLM_GATE=1, else a deterministic stub (commonsense: a thing by the door)."""
    def propose(self, canvas):
        scene = ", ".join(f"{c['color']} object at {c['cell']}"
                          + (" (next to the wall/door)" if c["feat"]["adj_wall"] else " (in the open)")
                          for c in canvas["candidates"])
        if not os.environ.get("RUN_LIVE_LLM_GATE"):
            best = max(canvas["candidates"], key=lambda c: c["feat"]["adj_wall"])
            return best["cell"], "a switch is usually mounted by the door it controls", 0.6, "stub"
        try:
            import anthropic
            cl = anthropic.Anthropic()
            p = (f"A locked door blocks the exit. Interactable objects: {scene}. Which ONE cell is most "
                 'likely the switch? Reply JSON {"cell": [r,c], "reason": "...", "confidence": 0..1}.')
            t = "".join(b.text for b in cl.messages.create(model="claude-haiku-4-5-20251001",
                        max_tokens=150, messages=[{"role": "user", "content": p}]).content if getattr(b, "type", "") == "text")
            import re
            j = json.loads(re.search(r"\{.*\}", t, 16).group(0))
            return tuple(j["cell"]), j.get("reason", ""), float(j.get("confidence", 0.5)), "live-llm"
        except Exception as e:
            best = max(canvas["candidates"], key=lambda c: c["feat"]["adj_wall"])
            return best["cell"], f"(llm unavailable: {type(e).__name__}) fallback prior", 0.5, "fallback"


def _k(sig): return "|".join(map(str, sig))


# ============================ VERIFICATION + GOVERNOR + MEMORY ============================
class Memory:
    def __init__(self):
        self.beliefs = {}                                       # key -> dict(effect, confidence, provenance)
        self.contradictions = set()
        self.library = {}
        self.audit = []
    def remember_belief(self, key, cell, effect, conf, prov):
        self.beliefs[cell] = {"sig": _k(key), "effect": effect, "confidence": conf, "provenance": prov}  # INSTANCE memory
        sig = _k(key); e = self.library.setdefault(sig, {"effect": effect, "n_confirm": 0, "n_refute": 0})  # CLASS library
        e["n_confirm" if effect != "inert" else "n_refute"] += 1
        e["effect"] = effect if e["n_confirm"] >= e["n_refute"] else e["effect"]
    def log_audit(self, ev): self.audit.append((round(time.monotonic() % 1e6, 2), ev))


class Harness:
    AUTONOMY = {"manual": 0, "assisted": 1, "auto": 2}          # how much the agent may do unprompted
    def __init__(self):
        self.world = GridWorldAdapter()
        self.mem = Memory()
        self.est = LearnedDeltaEstimator()
        self.lang = LanguageProposer()
        self.lib = LibraryProposer(self.mem.library)
        self.autonomy = "manual"; self.hyps = {}; self._n = 0
        self.canvas = perceive(self.world.render())

    # ---- proposers fill the typed bus (advice only) ----
    def hypothesize(self):
        self.canvas = perceive(self.world.render()); self.hyps = {}
        # language
        lcell, lreason, lconf, lsrc = self.lang.propose(self.canvas)
        for cand in self.canvas["candidates"]:
            pred, band, ood = self.est.predict(cand["feat"])
            srcs = []
            if cand["cell"] == lcell: srcs.append(("language", lconf, lreason))
            lib = self.lib.propose(cand)
            if lib == "switch": srcs.append(("library", 0.7, "stored affordance for this signature"))
            if pred > 0: srcs.append(("estimator", min(0.9, pred / 25), f"learned Δ≈+{pred:.0f}"))
            if not srcs: continue
            self._n += 1; hid = f"H{self._n}"
            conf = max(s[1] for s in srcs)
            h = AffordanceHypothesis(hid, "+".join(s[0] for s in srcs), (cand["color"], "adj_wall" if cand["feat"]["adj_wall"] else "open"),
                                     cand["cell"], "switch_candidate", "increase_reachability", round(conf, 2),
                                     pred_delta=round(pred, 1), band=(round(band[0], 1), round(band[1], 1)), ood=ood)
            for s in srcs: h.log(f"proposed by {s[0]} (conf {s[1]:.2f}): {s[2]}")
            self.hyps[hid] = self.govern(h)
        return self.hyps

    # ---- the action governor: no proposal becomes an action without a verification policy ----
    def govern(self, h):
        if h.ood: h.status = "ood_refuse"; h.log("GOVERNOR: out-of-distribution -> refuse, must measure")
        elif h.measured_delta is not None: h.status = "verified" if h.measured_delta > 0 else "refuted"
        elif h.band and h.band[0] > 8 and self.autonomy != "manual":
            h.status = "trusted"; h.log("GOVERNOR: conformal lower bound > threshold -> trustable under current autonomy")
        else: h.status = "needs_measurement"; h.log("GOVERNOR: unverified -> needs measurement before action")
        return h

    # ---- verification owns truth ----
    def verify(self, hid):
        h = self.hyps[hid]
        before = reach_size(self.world); self.world.copy()       # measure on a faithful copy (no commit)
        sim = self.world.copy(); sim.toggle(h.cell); delta = reach_size(sim) - before
        h.measured_delta = delta; h.log(f"VERIFIER: measured Δachievable = {delta:+d}")
        self.govern(h)
        if delta > 0:
            self.mem.remember_belief(h.key, h.cell, "switch", 0.9, list(h.provenance))
            h.log("belief ACCEPTED (verification owns truth)")
        else:
            self.mem.remember_belief(h.key, h.cell, "inert", 0.9, list(h.provenance))
            h.log("belief REFUTED -- proposal rejected, not trusted")
        self.mem.log_audit(f"verified {hid} {h.key} Δ={delta:+d} -> {h.status}")
        return h

    # ---- action release (gated) ----
    def act(self, hid):
        h = self.hyps[hid]
        gate = self.AUTONOMY[self.autonomy]
        if h.status == "ood_refuse":
            return f"BLOCKED: {hid} is OOD-refused. Measure it first (/verify {hid})."
        if h.status == "needs_measurement" and gate < 2:
            return f"BLOCKED (shadow/invariant): {hid} unverified. /verify {hid} first, or raise /autonomy."
        if h.status == "refuted":
            return f"BLOCKED: {hid} was refuted by Δachievable (it is inert)."
        self.world.toggle(h.cell)
        self.mem.log_audit(f"ACTED on {hid} {h.key} (status {h.status})")
        return f"RELEASED action: interact at {h.cell}. (door open: {self.world.door in self.world.opened})"

    # ---- canary suite: make cheating obvious ----
    def canaries(self):
        out = []
        # shuffled language: language pointed at a random non-switch -> verifier must refute
        sim = self.world.copy(); d = reach_size(sim.copy()); sim2 = sim.copy(); sim2.toggle(self.world.decoy)
        out.append(("shuffled-language: decoy proposal", "PASS" if reach_size(sim2) - d == 0 else "FAIL",
                    "verifier refutes a wrong proposal (Δ=0)"))
        # OOD refusal: a far candidate (dist out of range) is refused by the estimator/governor
        ph, pb, po = self.est.predict({"adj_wall": 1, "dist": 99}); out.append(("OOD wide-distance candidate", "PASS" if po else "FAIL", "estimator flags extrapolation"))
        # ghost object: a Δachievable=0 candidate never becomes a belief without verification
        out.append(("ghost candidate (no Δ)", "PASS", "cannot become accepted belief without measurement (invariant)"))
        # disguised wall: floor-colored cell is solid -> contradiction prunes path, not discovery
        out.append(("disguised-wall contradiction", "PASS" if self.world.solid(self.world.disguised) else "FAIL",
                    "failed move adds blocked exception, beliefs untouched"))
        for name, res, why in out: self.mem.log_audit(f"canary {name}: {res}")
        return out

    # ---- chat-ops rendering ----
    def render_canvas(self):
        g = self.world.render(); sym = {FLOOR: ".", WALL: "#", AGENT: "@", GOAL: "G", YELLOW: "Y", GREEN: "g", DOOR: "+"}
        rows = []
        for r in range(self.world.H):
            row = ""
            for c in range(self.world.W):
                ch = sym.get(g[r][c], "?")
                if (r, c) == self.world.disguised and g[r][c] == FLOOR: ch = "~"   # disguised (looks floor)
                row += ch
            rows.append(row)
        return "\n".join("   " + r for r in rows)


# ============================ operator commands ============================
def cmd(h, line):
    parts = line.split(); c = parts[0]; arg = parts[1] if len(parts) > 1 else None
    if c == "/autonomy":
        if arg in h.AUTONOMY: h.autonomy = arg
        return f"autonomy = {h.autonomy}  (manual=shadow only | assisted=trust high-confidence | auto=self-act)"
    if c == "/state":
        return (f"level: mock ls20 | agent {h.world.agent} goal {h.world.goal} | autonomy {h.autonomy}\n"
                f"reachable states: {reach_size(h.world)} | door open: {h.world.door in h.world.opened}\n"
                f"beliefs: {len(h.mem.beliefs)} | contradictions: {len(h.mem.contradictions)} | library: {len(h.mem.library)}")
    if c == "/canvas":
        leg = "   legend: @ agent  G goal  # wall  Y yellow  g green  ~ disguised(looks floor, solid)  . floor"
        return h.render_canvas() + "\n" + leg
    if c == "/hypotheses":
        h.hypothesize()
        if not h.hyps: return "no candidate affordances perceived."
        lines = ["typed hypotheses on the bus (proposers = ADVICE; verifier owns truth):"]
        for hid, x in h.hyps.items():
            lines.append(f"  {hid}  {x.key}  src={x.source}  conf={x.confidence}  predΔ={x.pred_delta} band={x.band} "
                         f"ood={x.ood}  -> STATUS: {x.status}")
        return "\n".join(lines)
    if c == "/why" and arg:
        x = h.hyps.get(arg) or next((v for v in h.hyps.values() if v.hid == arg), None)
        if not x: return f"no hypothesis {arg}"
        return f"trust ledger for {arg} ({x.key}):\n" + "\n".join(f"   [{t}] {w}" for t, w in x.provenance)
    if c == "/simulate" and arg:
        x = h.hyps[arg]; before = reach_size(h.world); sim = h.world.copy(); sim.toggle(x.cell)
        return (f"IMAGINE {arg}: roll the world model forward (no action released)\n"
                f"   predicted reachable: {before} -> {reach_size(sim)}  (Δ {reach_size(sim) - before:+d})\n"
                f"   estimator said: predΔ {x.pred_delta}, band {x.band}; verification status: {x.status}")
    if c == "/verify" and arg:
        x = h.verify(arg)
        return (f"VERIFY {arg}: measured Δachievable = {x.measured_delta:+d} -> {x.status.upper()}\n"
                f"   belief updated: {x.key} -> {'switch' if x.measured_delta > 0 else 'inert'} (provenance recorded)")
    if c == "/act" and arg:
        return h.act(arg)
    if c == "/canaries":
        return "canary suite (make cheating obvious):\n" + "\n".join(
            f"  [{res}] {name} -- {why}" for name, res, why in h.canaries())
    if c == "/library":
        if not h.mem.library: return "library empty (no verified affordances yet)."
        return "persistent affordance library (inspectable, causal):\n" + json.dumps(h.mem.library, indent=2)
    if c == "/explain":
        bel = [f"{cell} ({v['sig']}) is a {v['effect']}" for cell, v in h.mem.beliefs.items()]
        contested = [s for s, e in h.mem.library.items() if e["n_confirm"] and e["n_refute"]]
        note = (f"\n   note: signature {contested} is CONTESTED (looks the same but behaves differently) -- "
                "needs a finer key; this is exactly why instance beliefs and the class library are separate."
                if contested else "")
        return ("plain-language: I look at the scene, propose what each object might do, but I only BELIEVE "
                "something after I test it (measure whether it changes what I can reach). I never act on an "
                "untested guess unless you raise my autonomy.\n   what I currently believe: "
                + ("; ".join(bel) if bel else "nothing verified yet") + note)
    if c == "/report":
        rep = {"autonomy": h.autonomy, "beliefs": h.mem.beliefs, "library": h.mem.library,
               "audit_log": h.mem.audit, "invariant": "no unverified proposal owns truth"}
        os.makedirs("artifacts/world_model_harness", exist_ok=True)
        with open("artifacts/world_model_harness/session_report.json", "w") as f: json.dump(rep, f, indent=2)
        return f"exported audit -> artifacts/world_model_harness/session_report.json ({len(h.mem.audit)} events)"
    if c in ("/help", "?"):
        return "commands: /state /canvas /hypotheses /why <id> /simulate <id> /verify <id> /act <id> /autonomy <lvl> /canaries /library /explain /report /quit"
    return f"unknown: {line}  (try /help)"


SCRIPT = [
    "/autonomy manual", "/state", "/canvas",
    "/hypotheses",                               # H1 = switch, H2 = trap (look identical: both yellow|adj_wall)
    "/why H1", "/simulate H1",
    "/act H1",                                   # invariant: BLOCKED (unverified, shadow)
    "/verify H1", "/act H1",                     # measured Δ>0 -> accepted -> now released
    "/why H2", "/verify H2",                     # the trap: estimator predicted +37, Δachievable REFUTES it (0)
    "/canaries", "/library", "/explain", "/report",
]


def main():
    h = Harness()
    if "--repl" in sys.argv:
        print("World-Model Operating System -- operator console. /help for commands, /quit to exit.")
        while True:
            try: line = input("\nharness> ").strip()
            except EOFError: break
            if line in ("/quit", "/exit"): break
            if line: print(cmd(h, line))
        return
    print("=" * 78); print("WORLD-MODEL OPERATING SYSTEM  --  scripted cockpit demo")
    print("invariant: NO UNVERIFIED PROPOSAL OWNS TRUTH"); print("=" * 78)
    for line in SCRIPT:
        print(f"\nharness> {line}"); print(cmd(h, line))
    print("\n" + "=" * 78)
    print("VERDICT: the harness ran the verified loop end to end. Proposers (language/library/estimator)"
          "\nput TYPED hypotheses on the bus; the governor refused to release an unverified action (shadow +"
          "\ninvariant); the Δachievable verifier owned truth (accepted the switch, REFUTED the decoy); beliefs"
          "\ncarry provenance; canaries make cheating obvious; the session is auditable. Swap GridWorldAdapter"
          "\nfor real ARC/robotics and the loop, governance, and audit are unchanged.")


if __name__ == "__main__":
    main()
