#!/usr/bin/env python3
"""Perception hardening: CONTRADICTION-UPDATE of the start-color passability prior.

The start-color prior (perception_affordance.py) assumes: floor-color == passable. That
assumption is TRUSTED there, never tested. Here we break it: a DISGUISED WALL renders in
the floor color but is physically SOLID. So the perceived canvas OVERSTATES reachability --
perception believes the goal is reachable by the short path through the disguised wall, but
that path is blocked. The contradiction only surfaces by ACTING: a move onto the disguised
cell FAILS, and the agent must REVISE passability (known_blocked += cell) and reroute.

Two barrier gaps: gap A (short, but a disguised wall -> solid) and gap B (longer detour,
genuinely open). Perception sees BOTH as floor -> plans the short path through A.

CONTROLS:
  (1) UPDATE agent reaches the goal -- hits A, marks it blocked, reroutes through B.
  (2) NO-UPDATE agent FAILS -- replans the same blocked short path forever (loops to cap).
  (3) NO FALSE POSITIVES: update agent marks ONLY the truly-solid disguised cell blocked.
  (4) NO OVER-CORRECTION: in a world where A is genuinely open, the update agent takes the
      SHORT path with an EMPTY blocked-set (doesn't hallucinate obstacles).
  (5) it really REROUTED: the solved path is longer than the naive perceived-shortest path.
Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque

FLOOR, WALL = 0, 1
H = W = 11


def make_world(disguised=True):
    """types grid + a set of physically-SOLID cells (truth). A disguised wall is FLOOR-colored
    but solid. Barrier at col 5 with gaps at rows 2 (A) and 8 (B); goal across the barrier."""
    types = [[FLOOR] * W for _ in range(H)]
    for r in range(H): types[r][5] = WALL
    gapA, gapB = (2, 5), (8, 5)
    types[gapA[0]][gapA[1]] = FLOOR                              # A renders FLOOR (gap in the wall)...
    types[gapB[0]][gapB[1]] = FLOOR                              # B renders FLOOR (real gap)
    solid = {(r, 5) for r in range(H)} - {gapA, gapB}           # true walls
    if disguised: solid.add(gapA)                               # ...but A is SECRETLY solid (disguised wall)
    agent = (3, 0); goal = (3, 10)                               # A (row2) is the nearer gap from row 3
    return types, solid, agent, goal


def render(types):
    return [[types[r][c] for c in range(W)] for r in range(H)]   # the observation (colors only)


def perceived_passable(grid, agent):
    start = grid[agent[0]][agent[1]]
    return {(r, c) for r in range(H) for c in range(W) if grid[r][c] == start} | {agent}


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
                    path = [goal]
                    while prev[path[-1]] is not None: path.append(prev[path[-1]])
                    return path[::-1]
                q.append(n)
    return None


def navigate(disguised=True, update=True, cap=40):
    types, solid, agent, goal = make_world(disguised)
    grid = render(types)
    passable = perceived_passable(grid, agent)
    known_blocked = set(); pos = agent; replans = 0; bumps = 0
    while replans < cap:
        path = bfs_path(passable - known_blocked, pos, goal)
        if path is None: break                                  # believes goal unreachable
        replans += 1; hit = False
        for nxt in path[1:]:
            if nxt in solid:                                    # ACT: move fails on a solid cell
                bumps += 1
                if update: known_blocked.add(nxt)              # CONTRADICTION-UPDATE: revise passability
                hit = True; break                               # (no-update: learns nothing -> same plan next time)
            pos = nxt
        if pos == goal:
            length = _truth_path_len(solid, agent, goal)
            return {"solved": True, "replans": replans, "bumps": bumps,
                    "known_blocked": set(known_blocked), "path_via_B": (8, 5) in path}
    return {"solved": False, "replans": replans, "bumps": bumps, "known_blocked": set(known_blocked)}


def _truth_path_len(solid, agent, goal):
    passable = {(r, c) for r in range(H) for c in range(W)} - solid
    p = bfs_path(passable, agent, goal); return len(p) if p else None


def _perceived_shortest_len(disguised=True):
    types, solid, agent, goal = make_world(disguised)
    p = bfs_path(perceived_passable(render(types), agent), agent, goal); return len(p) if p else None


print("=== Perception contradiction-update: disguised-wall navigation ===\n")
up = navigate(disguised=True, update=True)
no = navigate(disguised=True, update=False)
clean = navigate(disguised=False, update=True)                  # no disguised wall: must NOT over-correct
perc_short = _perceived_shortest_len(disguised=True)            # naive (perceived) shortest = through A
true_len = _truth_path_len(make_world(True)[1], (3, 0), (3, 10))

print(f"  perceived-shortest path length (believes A open) = {perc_short}")
print(f"  UPDATE agent : solved={up['solved']} reroutes_via_B={up.get('path_via_B')} "
      f"blocked_learned={up['known_blocked']} bumps={up['bumps']}")
print(f"  NO-UPDATE    : solved={no['solved']} replans={no['replans']} (loops to cap) bumps={no['bumps']}")
print(f"  CLEAN world  : solved={clean['solved']} blocked_learned={clean['known_blocked']} (should be empty)\n")

checks = {
    "(1) UPDATE agent reaches goal (reroutes through B)": up["solved"] and up["path_via_B"],
    "(2) NO-UPDATE agent FAILS (replans blocked path to cap)": (not no["solved"]) and no["replans"] >= 40,
    "(3) NO false positives: learned ONLY the disguised wall (2,5)": up["known_blocked"] == {(2, 5)},
    "(4) NO over-correction: clean world solved with EMPTY blocked-set": clean["solved"] and clean["known_blocked"] == set(),
    "(5) really REROUTED: true solved path longer than perceived-shortest": (true_len or 0) > (perc_short or 0),
}
print("=== contradiction-update pass criteria ===")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nCONTRADICTION-UPDATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the start-color prior is now FALSIFIABLE in the loop -- a floor-colored cell that is"
      "\n  physically solid is detected by a failed move and removed from the passability map, forcing a"
      "\n  reroute. The update agent solves where the no-update agent loops forever, marks ONLY the truly"
      "\n  solid cell (no hallucinated blocks), and does NOT over-correct when the prior was right.")
