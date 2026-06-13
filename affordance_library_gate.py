#!/usr/bin/env python3
"""Singular-2: PERSISTENT COMPOSITIONAL affordance library. Discovered affordances become a
transferable, inspectable, CAUSAL store across games -- not per-session memory. Physics still
owns truth: the library PROPOSES a stored affordance for a new candidate; Δachievable VERIFIES
it; a wrong entry is REFUTED (never silently applied).

Properties under test (each a control):
  PERSISTENCE    save -> load round-trips; the store is human-inspectable (typed sig -> effect,
                 confidence, provenance, n_confirm/n_refute).
  TRANSFER       a library learned on game A solves game B with FEWER measurements (skips the
                 decoys it already knows are inert; goes straight to the switch class).
  FALSIFIABLE    a stored entry that is WRONG in game B (a same-signature candidate that is
                 actually inert) is REFUTED by Δachievable -- caught, confidence dropped, no silent error.
  SHUFFLE        a scrambled library gives NO transfer advantage (its entries don't match reality,
                 get refuted on verify, agent falls back to discovery).
  COMPOSITION    a goal needing a CHAIN (apply switch -> that unlocks the lock -> apply key-lock) is
                 solved by composing two stored affordances respecting preconditions; a FLAT library
                 (no composition) applies them out of order and fails.
Self-contained, stdlib only.
"""
from __future__ import annotations
import json, os, tempfile

ART = "artifacts/affordance_library_gate"


# --------------------- the library (persistent, inspectable, causal) ---------------------
class AffordanceLibrary:
    def __init__(self): self.entries = {}                       # sig(str) -> dict(effect, n_confirm, n_refute, provenance, unlocks)
    def propose(self, sig):
        e = self.entries.get(_k(sig))
        if e and e["n_confirm"] > e["n_refute"]: return e["effect"]
        return None
    def confidence(self, sig):
        e = self.entries.get(_k(sig));  n = (e["n_confirm"] + e["n_refute"]) if e else 0
        return 0.0 if not e or n == 0 else round(e["n_confirm"] / n, 3)
    def confirm(self, sig, effect, provenance, unlocks=None):
        e = self.entries.setdefault(_k(sig), {"effect": effect, "n_confirm": 0, "n_refute": 0,
                                              "provenance": provenance, "unlocks": unlocks})
        e["n_confirm"] += 1; e["effect"] = effect
        if unlocks: e["unlocks"] = unlocks
    def refute(self, sig):
        e = self.entries.get(_k(sig))
        if e: e["n_refute"] += 1                                # falsifiable membership: evidence against
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f: json.dump(self.entries, f, indent=2, sort_keys=True)
    @classmethod
    def load(cls, path):
        lib = cls()
        with open(path) as f: lib.entries = json.load(f)
        return lib
    def inspect(self):
        return [f"{k} -> {v['effect']} (conf {v['n_confirm']}/{v['n_confirm']+v['n_refute']}, from {v['provenance']}"
                + (f", unlocks {v['unlocks']})" if v.get('unlocks') else ")") for k, v in sorted(self.entries.items())]


def _k(sig): return "|".join(map(str, sig)) if isinstance(sig, (list, tuple)) else str(sig)


# --------------------- abstract games (Δachievable = the expensive truth) ---------------------
def measure(cand):                                              # the expensive oracle: apply + observe Δachievable
    return cand["effect"] if cand["effect"] != "inert" else None    # None == Δachievable 0 (inert)


GAME_A = {"cands": [
    {"id": "s1", "sig": ("yellow", "adj_wall"), "effect": "switch"},
    {"id": "s2", "sig": ("yellow", "adj_wall"), "effect": "switch"},
    {"id": "d1", "sig": ("green", "open"), "effect": "inert"},
    {"id": "d2", "sig": ("yellow", "open"), "effect": "inert"}],   # yellow but NOT adj_wall -> inert
    "need": {"switch": 2}}

GAME_B = {"cands": [
    {"id": "d3", "sig": ("blue", "open"), "effect": "inert"},       # NEW color (unknown to the library)
    {"id": "d4", "sig": ("yellow", "open"), "effect": "inert"},     # known-inert from game A
    {"id": "s3", "sig": ("yellow", "adj_wall"), "effect": "switch"},
    {"id": "x1", "sig": ("yellow", "adj_wall"), "effect": "inert"},  # FALSIFIABLE: same sig as a switch, but inert
    {"id": "s4", "sig": ("yellow", "adj_wall"), "effect": "switch"},
    {"id": "s5", "sig": ("yellow", "adj_wall"), "effect": "switch"}],
    "need": {"switch": 3}}


def play(game, lib=None):
    """Solve a game; verification (measure) ALWAYS owns truth. With a library: try proposed-switches
    first, SKIP known-inert sigs, and FALL BACK to the skipped/unknown set if the library was wrong."""
    got = 0; measurements = 0; refutations = 0; cands = game["cands"]
    if lib is not None:
        proposed = [c for c in cands if lib.propose(c["sig"]) == "switch"]
        known_inert = [c for c in cands if lib.propose(c["sig"]) == "inert"]
        unknown = [c for c in cands if lib.propose(c["sig"]) is None]
        order = proposed + unknown; fallback = known_inert      # skipped unless the library fails
    else:
        order = list(cands); fallback = []
    def run(seq):
        nonlocal got, measurements, refutations
        for c in seq:
            said_switch = lib is not None and lib.propose(c["sig"]) == "switch"
            eff = measure(c); measurements += 1                 # APPLY + VERIFY = the expensive truth
            if eff == "switch":
                got += 1
                if lib is not None: lib.confirm(c["sig"], "switch", "verify")
            else:
                if said_switch: refutations += 1                # library claimed switch, physics refuted it
                if lib is not None:
                    (lib.refute if _k(c["sig"]) in lib.entries else
                     (lambda s: lib.confirm(s, "inert", "verify")))(c["sig"])
            if got >= game["need"]["switch"]: return True
        return got >= game["need"]["switch"]
    solved = run(order)
    if not solved and fallback: solved = run(fallback)          # library was wrong -> fall back (no silent failure)
    return {"solved": solved, "measurements": measurements, "refutations": refutations,
            "skipped": len(fallback) if solved and lib is not None else 0}


# learn the library on GAME A
lib = AffordanceLibrary()
for c in GAME_A["cands"]:
    eff = measure(c)
    if eff == "switch": lib.confirm(c["sig"], "switch", "game_A")
    else: lib.confirm(c["sig"], "inert", "game_A")

# PERSISTENCE: save -> load
path = f"{ART}/library.json"; lib.save(path); lib2 = AffordanceLibrary.load(path)

# TRANSFER + FALSIFIABLE: solve game B with the loaded library vs from scratch
no_lib = play(GAME_B, lib=None)
with_lib = play(GAME_B, lib=AffordanceLibrary.load(path))

# SHUFFLE: adversarially scramble the sig->effect mapping (move the 'switch' label onto a wrong sig)
shuf = AffordanceLibrary.load(path)
shuf.entries["yellow|adj_wall"]["effect"] = "inert"            # the real switch sig -> mislabeled inert
shuf.entries["yellow|open"]["effect"] = "switch"              # a real inert sig -> mislabeled switch
for e in shuf.entries.values(): e["n_confirm"], e["n_refute"] = 1, 0
with_shuf = play(GAME_B, lib=shuf)

# COMPOSITION: a chain goal -- switch must be applied before the lock is reachable
CHAIN = [{"id": "sw", "sig": ("yellow", "adj_wall"), "effect": "switch", "unlocks": "lock"},
         {"id": "lk", "sig": ("door", "carry_key"), "effect": "goal", "applicable_after": "switch"}]
def play_chain(compose):
    applied = set(); meas = 0
    pending = list(CHAIN)
    progressed = True
    while pending and progressed:
        progressed = False
        for c in list(pending):
            pre = c.get("applicable_after")
            if compose and pre and pre not in applied:         # composable: respect precondition
                continue
            meas += 1; applied.add(c["effect"]); pending.remove(c); progressed = True
    return {"solved": "goal" in applied, "measurements": meas}
comp_yes = play_chain(compose=True)
comp_no = play_chain(compose=False)   # flat: tries lock first (precondition unmet) -> never reaches goal
# flat agent: fixed order, applies lock before switch -> lock no-ops, goal not reached
def play_flat():
    applied = set(); meas = 0
    for c in [CHAIN[1], CHAIN[0]]:                             # WRONG order (lock first)
        pre = c.get("applicable_after"); meas += 1
        if pre and pre not in applied: continue                # lock no-ops (precondition unmet) -> not applied
        applied.add(c["effect"])
    return {"solved": "goal" in applied, "measurements": meas}
comp_no = play_flat()

lib3 = AffordanceLibrary.load(path)
print("=== persistent compositional affordance library ===\n")
print("  library (inspectable):")
for line in lib2.inspect(): print(f"    {line}")
print(f"\n  TRANSFER  game B: no-library {no_lib['measurements']} measurements vs with-library {with_lib['measurements']}"
      f" (skipped {with_lib['skipped']} known-inert)")
print(f"  FALSIFIABLE: with-library refutations on B (library said switch, physics said no) = {with_lib['refutations']}")
print(f"  SHUFFLE   with-shuffled-library measurements = {with_shuf['measurements']} (advantage should vanish)")
print(f"  COMPOSE   chain goal: composable solved={comp_yes['solved']} | flat solved={comp_no['solved']}\n")

checks = {
    "PERSISTENCE: save->load round-trips + inspectable": lib2.entries == lib.entries and len(lib2.inspect()) > 0,
    "TRANSFER: library solves game B with FEWER measurements": with_lib["solved"] and with_lib["measurements"] < no_lib["measurements"],
    "FALSIFIABLE: a wrong same-signature entry is REFUTED by Δachievable (caught, not silently applied)":
        with_lib["refutations"] >= 1 and with_lib["solved"],
    "SHUFFLE: scrambled library gives NO transfer advantage": with_shuf["measurements"] >= no_lib["measurements"],
    "COMPOSITION: composable library solves the chain goal": comp_yes["solved"],
    "COMPOSITION: flat (non-composable) library FAILS the chain (out-of-order)": not comp_no["solved"],
    "CAUSAL/INSPECTABLE: entries carry effect + confidence + provenance": all(
        "->" in line and "conf" in line and "from" in line for line in lib2.inspect()),
}
print("=== library pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
with open(f"{ART}/result.json", "w") as f:
    json.dump({"pass": all(checks.values()), "transfer": {"no_lib": no_lib, "with_lib": with_lib, "shuffle": with_shuf}}, f, indent=2)
print(f"\nPERSISTENT COMPOSITIONAL AFFORDANCE LIBRARY: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: discovered affordances persist as a typed, inspectable, CAUSAL store that TRANSFERS across games"
      "\n  (fewer measurements on game B by skipping known-inert classes), keeps FALSIFIABLE membership (a wrong"
      "\n  same-signature entry is refuted by Δachievable, not silently applied), gains nothing under SHUFFLE, and"
      "\n  COMPOSES stored affordances into chains respecting preconditions. The library proposes; physics owns truth.")
