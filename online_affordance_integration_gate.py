#!/usr/bin/env python3
"""#3A online_affordance_integration_gate: the SMALLEST stateful online loop --
prior + Δachievable discovery + class/refined-key memory + contradiction-update, stepping
over a HELD episode with belief state persisting across steps. NO plateau arbiter, NO
multi-step dynamics, NO path-hypothesis canvas yet (those are #3/#4).

The unit assumptions are all controls-passed already; the NEW risk is STATEFUL interaction:
belief staleness, probe/exploit scheduling, contradiction erasing a true discovery, three
memories overwriting each other. So the loop keeps THREE SEPARATE memories:
  class_beliefs        -- what KIND of thing tends to matter (color/appearance -> switch)
  refined_key / aliased-- this same-looking class split by a finer causal feature (adj_wall)
  blocked_exceptions   -- this PATH cell is not traversable (navigation contradiction)
None may overwrite another. The load-bearing control: after a failed move corrects the canvas,
the switch discovery and class belief MUST survive (contradiction prunes paths, not discoveries).

Held episode: agent must DISCOVER switches (open 2 serial doors) AND contradiction-update around
a disguised wall (floor-colored, solid) to reach the goal -- discovery + contradiction composed,
now run step-by-step with persistent state. Two variants: 'separable' (green decoy) and 'aliased'
(yellow decoys, needs key refinement). Ablations prove each memory is load-bearing.

Trace -> artifacts/online_affordance_integration_gate/trace.jsonl (first bugs are temporal).
Self-contained, stdlib only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import json, os

H = W = 11
FLOORC, WALLC, YELLOW, GREEN, GOALC = 0, 1, 2, 3, 4
GAIN, NU, LAM = 56.0, 2.0, 1.0
FEATS = ("adj_wall", "near_top")
ART = "artifacts/online_affordance_integration_gate"


class World:
    """TRUE state (hidden dynamics); the agent only ever sees render()."""
    def __init__(self, variant):
        self.variant = variant
        self.agent0 = (5, 0); self.goal = (5, 10)
        self.walls = {(r, 3) for r in range(H)} | {(r, 6) for r in range(H)}
        self.doors = {(5, 3), (5, 6)}
        self.switch_door = {(4, 2): (5, 3), (8, 2): (5, 6)}     # 2 switches (adj_wall=T), spaced so segmentation keeps them distinct
        self.decoys = {(10, 0): "green"} if variant == "separable" else {(4, 0): "yellow", (6, 0): "yellow"}
        self.disguised = {(5, 8)}                               # floor-colored, physically SOLID (the lie)
        self.opened = set()
    def render(self):
        g = [[FLOORC] * W for _ in range(H)]
        for (r, c) in self.walls: g[r][c] = WALLC
        for d in self.doors:
            if d in self.opened: g[d[0]][d[1]] = FLOORC         # opened door re-perceived as floor
        for sw in self.switch_door: g[sw[0]][sw[1]] = YELLOW
        for d, col in self.decoys.items(): g[d[0]][d[1]] = YELLOW if col == "yellow" else GREEN
        g[self.goal[0]][self.goal[1]] = GOALC                   # disguised stays FLOORC (looks passable)
        return g
    def solid(self, cell):
        if cell in self.doors: return cell not in self.opened
        return cell in self.walls or cell in self.disguised
    def toggle(self, cell):
        if cell in self.switch_door: self.opened.add(self.switch_door[cell])


# ---------- perception (validated in Gate 4 / perception_reach; reused) ----------
def passable_of(grid): return {(r, c) for r in range(H) for c in range(W) if grid[r][c] == FLOORC}
def reach_set(passable, start):
    seen = {start}; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and n in passable:
                seen.add(n); q.append(n)
    return seen
def adj_to(cellset, cell):
    return any((cell[0] + dr, cell[1] + dc) in cellset for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
def segment(grid):                                              # small non-floor components = candidate objects
    objs = []; vis = set()
    for r in range(H):
        for c in range(W):
            col = grid[r][c]
            if col in (FLOORC, GOALC) or (r, c) in vis: continue
            comp = []; q = deque([(r, c)]); vis.add((r, c))
            while q:
                cur = q.popleft(); comp.append(cur)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cur[0] + dr, cur[1] + dc)
                    if 0 <= n[0] < H and 0 <= n[1] < W and n not in vis and grid[n[0]][n[1]] == col:
                        vis.add(n); q.append(n)
            if len(comp) <= 3:
                objs.append((comp[0], col))
    return objs
def feat_of(grid, cell):
    return {"adj_wall": adj_to({(r, c) for r in range(H) for c in range(W) if grid[r][c] == WALLC}, cell),
            "near_top": cell[0] < 4}
def bfs_path(passable, start, goal):
    if start == goal: return [start]
    prev = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in prev and (n in passable or n == goal):
                prev[n] = cur
                if n == goal:
                    p = [goal]
                    while prev[p[-1]] is not None: p.append(prev[p[-1]])
                    return p[::-1]
                q.append(n)
    return None
def find_separating_feature(evidence):                          # evidence: [(feat, pos_bool)]
    for fk in FEATS:
        groups = {}
        for feat, pos in evidence: groups.setdefault(feat[fk], set()).add(pos)
        if len(groups) > 1 and all(len(v) == 1 for v in groups.values()): return fk
    return None


@dataclass
class OnlineState:
    pos: tuple
    toggle_budget: int
    class_beliefs: dict = field(default_factory=dict)           # MEMORY 1: appearance -> n_pos
    refined_key: object = None                                  # MEMORY 2: finer key for aliased class
    refined_val: object = None
    aliased: set = field(default_factory=set)
    instance_exceptions: dict = field(default_factory=dict)
    blocked_exceptions: set = field(default_factory=set)        # MEMORY 3: navigation contradictions
    evidence: list = field(default_factory=list)
    toggled: set = field(default_factory=set)
    rejected: set = field(default_factory=set)
    discovered: list = field(default_factory=list)
    failed_moves: list = field(default_factory=list)
    integrity_snapshot: object = None                           # (class_beliefs, discovered) at first contradiction


def run(variant="separable", abl=None, step_cap=80, trace_out=None):
    abl = abl or {}
    discovery = abl.get("discovery", True); contradiction = abl.get("contradiction", True)
    class_memory = abl.get("class_memory", True); alias_refine = abl.get("alias_refine", True)
    world = World(variant); st = OnlineState(pos=world.agent0, toggle_budget=3)
    trace = []

    def reach_now():
        return reach_set(passable_of(world.render()) - st.blocked_exceptions, st.pos)
    def goal_reachable():
        passable = passable_of(world.render()) - st.blocked_exceptions
        return adj_to(reach_set(passable, st.pos), world.goal) and bfs_path(passable, st.pos, world.goal) is not None
    def candidates():
        grid = world.render(); reach = reach_now(); out = []
        for cell, col in segment(grid):
            if cell not in st.toggled and adj_to(reach, cell):
                out.append({"cell": cell, "color": col, "feat": feat_of(grid, cell)})
        return out
    def transfer(cands, c): return sum(1 for o in cands if o["color"] == c["color"])
    def cost(c): return abs(c["cell"][0] - st.pos[0]) + abs(c["cell"][1] - st.pos[1])
    def score(cands, c): return GAIN * 0.7 - LAM * cost(c) + NU * transfer(cands, c)

    def believes_switch(c):
        if not class_memory: return False
        app = c["color"]
        if app in st.aliased:
            if not alias_refine: return st.class_beliefs.get(app, 0) > 0
            if st.refined_key is not None:
                return c["feat"][st.refined_key] == st.refined_val and c["cell"] not in st.rejected
            return False                                        # aliased + not yet refined -> no blind transfer
        return st.class_beliefs.get(app, 0) > 0 and c["cell"] not in st.rejected

    def choose_probe(cands):
        # active disambiguation (alias_refine): isolate a variable by probing min-feature-diff (>0)
        if alias_refine and st.evidence and st.refined_key is None:
            pf = [e["feat"] for e in st.evidence]
            cand = [c for c in cands if min(sum(c["feat"][k] != p[k] for k in FEATS) for p in pf) > 0]
            if cand:
                return min(cand, key=lambda c: (min(sum(c["feat"][k] != p[k] for k in FEATS) for p in pf), cost(c)))
        return max(cands, key=lambda c: score(cands, c))

    def apply_toggle(c, mode):
        before = len(reach_now()); world.toggle(c["cell"]); after = len(reach_now())
        delta = after - before; app = c["color"]
        st.toggled.add(c["cell"]); st.evidence.append({"app": app, "feat": c["feat"], "pos": delta > 0})
        if delta > 0:
            st.discovered.append(c["cell"]); st.class_beliefs[app] = st.class_beliefs.get(app, 0) + 1
        else:
            st.rejected.add(c["cell"]); st.instance_exceptions[c["cell"]] = "inert"
        for a in {e["app"] for e in st.evidence}:                # aliasing: same appearance, conflicting Δ sign
            signs = {e["pos"] for e in st.evidence if e["app"] == a}
            if len(signs) > 1:
                st.aliased.add(a)
                if alias_refine and st.refined_key is None:
                    rk = find_separating_feature([(e["feat"], e["pos"]) for e in st.evidence if e["app"] == a])
                    if rk:
                        st.refined_key = rk
                        st.refined_val = next(e["feat"][rk] for e in st.evidence if e["app"] == a and e["pos"])
        return delta, mode

    for step in range(step_cap):
        if not class_memory:                                    # ABLATION: object-memory does NOT persist across steps
            st.class_beliefs.clear(); st.rejected.clear(); st.evidence.clear()
            st.toggled.clear(); st.aliased.clear(); st.refined_key = None; st.discovered.clear()
        rec = {"step": step, "pos": st.pos, "budget": st.toggle_budget,
               "class_beliefs": dict(st.class_beliefs), "blocked": sorted(map(list, st.blocked_exceptions))}
        if st.pos == world.goal:
            rec["mode"] = "done"; trace.append(rec); break
        if goal_reachable():
            rec["mode"] = "navigate"
            passable = passable_of(world.render()) - st.blocked_exceptions
            path = bfs_path(passable, st.pos, world.goal); nxt = path[1]
            if world.solid(nxt):                                # failed move on a (disguised) solid cell
                st.failed_moves.append(nxt); rec["replan_reason"] = "failed_move"; rec["bumped"] = list(nxt)
                if contradiction:
                    if st.integrity_snapshot is None:           # snapshot for the load-bearing control
                        st.integrity_snapshot = (dict(st.class_beliefs), list(st.discovered))
                    st.blocked_exceptions.add(nxt)              # MEMORY 3 grows; MEMORIES 1/2 untouched
            else:
                st.pos = nxt; rec["moved_to"] = list(nxt)
        else:
            cands = candidates()
            believed = [c for c in cands if believes_switch(c)]
            if not discovery:
                rec["mode"] = "stop_no_discovery"; trace.append(rec); break
            if believed and st.toggle_budget > 0:
                c = min(believed, key=cost); st.toggle_budget -= 1
                d, _ = apply_toggle(c, "exploit")
                rec.update(mode="exploit", cand=list(c["cell"]), key=c["color"], delta=d)
            elif st.toggle_budget > 0 and cands:
                c = choose_probe(cands); st.toggle_budget -= 1
                d, _ = apply_toggle(c, "probe")
                rec.update(mode="probe", cand=list(c["cell"]), key=c["color"], delta=d,
                           refined_key=st.refined_key, aliased=sorted(st.aliased))
            else:
                rec["mode"] = "stuck_no_budget"; trace.append(rec); break
        trace.append(rec)

    solved = st.pos == world.goal
    # integrity control: contradiction did NOT erase discovery
    integ = True
    if st.integrity_snapshot is not None:
        snap_beliefs, snap_disc = st.integrity_snapshot
        integ = (st.class_beliefs.get(YELLOW, 0) >= snap_beliefs.get(YELLOW, 0)
                 and all(d in st.discovered for d in snap_disc))
    if trace_out:
        os.makedirs(os.path.dirname(trace_out), exist_ok=True)
        with open(trace_out, "w") as f:
            for r in trace: f.write(json.dumps(r) + "\n")
    return {"solved": solved, "blocked": set(st.blocked_exceptions), "class_beliefs": dict(st.class_beliefs),
            "discovered": list(st.discovered), "rejected": set(st.rejected), "integrity_ok": integ,
            "had_contradiction": st.integrity_snapshot is not None, "steps": len(trace)}


full_sep = run("separable", trace_out=f"{ART}/trace.jsonl")
no_disc = run("separable", {"discovery": False})
no_contra = run("separable", {"contradiction": False})
no_classmem = run("separable", {"class_memory": False})
full_ali = run("aliased")
no_alias = run("aliased", {"alias_refine": False})

print("=== #3A online_affordance_integration_gate: stateful runner over a held episode ===\n")
for name, r in [("FULL separable", full_sep), ("no_discovery", no_disc), ("no_contradiction", no_contra),
                ("no_class_memory", no_classmem), ("FULL aliased", full_ali), ("no_alias_refine (aliased)", no_alias)]:
    print(f"  {name:26} solved={str(r['solved']):5} blocked={sorted(map(tuple,r['blocked']))} "
          f"beliefs={r['class_beliefs']} integrity_ok={r['integrity_ok']} steps={r['steps']}")
print()

checks = {
    "FULL runner solves held episode (separable)": full_sep["solved"],
    "FULL runner solves aliased variant (key refinement online)": full_ali["solved"],
    "CONTRADICTION DID NOT ERASE DISCOVERY (load-bearing)": full_sep["had_contradiction"] and full_sep["integrity_ok"],
    "class belief persisted across steps (yellow believed)": full_sep["class_beliefs"].get(YELLOW, 0) > 0,
    "only LOCAL blocked exception added (just the disguised cell)": full_sep["blocked"] == {(5, 8)},
    "no_discovery FAILS (doors never open)": not no_disc["solved"],
    "no_contradiction FAILS (loops on disguised wall)": not no_contra["solved"],
    "no_class_memory FAILS (belief doesn't persist -> can't accumulate)": not no_classmem["solved"],
    "no_alias_refine FAILS on aliased (blind transfer wastes budget)": not no_alias["solved"],
    "green/decoy rejected (not in discovered switches)": (10, 0) not in full_sep["discovered"],
}
print("=== #3A pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
result = {"checks": checks, "pass": all(checks.values()),
          "runs": {"full_sep": full_sep["solved"], "full_ali": full_ali["solved"],
                   "no_disc": no_disc["solved"], "no_contra": no_contra["solved"],
                   "no_classmem": no_classmem["solved"], "no_alias": no_alias["solved"]}}
def _json_safe(o):
    if isinstance(o, dict): return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (set, tuple, list)): return [_json_safe(x) for x in o]
    return o
with open(f"{ART}/result.json", "w") as f: json.dump(_json_safe(result), f, indent=2)
print(f"\n#3A ONLINE INTEGRATION GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the affordance stack runs ONLINE across steps with three SEPARATE persistent memories "
      "(class belief / refined key / blocked path).\n  The runner discovers switches, generalizes by class, "
      "refines the key when appearance aliases, and contradiction-updates the disguised wall -- all step-by-step "
      "with state intact.\n  The load-bearing result: contradiction prunes the PATH, never the DISCOVERY. "
      "Each memory is load-bearing (ablate any -> fail). trace.jsonl written for temporal debugging.")
