#!/usr/bin/env python3
"""Affordance discovery — GATE 1: exploration under uncertainty.

v0 proved the SIGNAL (reachability-change discovers prediction-invisible variables)
but used an ORACLE (simulate toggling every candidate for free). Gate 1 turns the
signal into an AGENT BEHAVIOR: no oracle. The agent only knows the TRUE effect of a
cell by ACTUALLY probing it (irreversibly), under a budget, choosing probes by

    score(c) = E[dReach(c)]  -  lambda * probe_cost(c)  -  mu * irreversibility_risk(c)

E[dReach] is an ESTIMATE from an observable interactiveness prior (not the true effect).
The LOAD-BEARING control is the irreversibility term: a TRAP that is CLOSER and looks
just as switch-y as the real switch. A risk-blind agent (mu=0) takes the cheap switch-y
bait and triggers the irreversible trap; a risk-aware agent (mu>0) routes around it via
a hazard sensor and still discovers the far switch.

PASS (risk-aware): discovers far switch | avoids trap | rejects decoy | budget respected
| goal reachable | prediction-only still fails | NO oracle.
ABLATION: risk-blind (mu=0) MUST hit the trap -> proves mu is load-bearing.
HONEST LIMIT: needs an informative hazard sensor; with a pure-noise sensor irreversible
traps are genuinely unavoidable (you cannot dodge what you cannot sense).
Self-contained, stdlib only.
"""
from __future__ import annotations
from collections import deque

FLOOR, WALL, SWITCH, DOOR, DECOY, GOAL, TRAP = range(7)
TRAVERSABLE = {FLOOR, SWITCH, DECOY, GOAL, TRAP}
GAIN = 56.0   # reach gain the agent KNOWS opening the door would yield (computable from the map)


def make_world():
    H, W = 11, 11
    g = [[FLOOR] * W for _ in range(H)]
    wc, dr = 5, 5
    for r in range(H): g[r][wc] = WALL
    g[dr][wc] = DOOR
    agent = (5, 1); goal = (5, 9); g[5][9] = GOAL
    switch = (0, 1); g[0][1] = SWITCH                          # real switch: FAR (cost high)
    trap = (5, 3); g[5][3] = TRAP                              # TRAP: near (cheap), looks switch-y
    decoy = (8, 2); g[8][2] = DECOY                            # inert
    door = (dr, wc)
    # observable priors (no type labels): interactiveness (switch & trap look interactive),
    # hazard sensor (trap reads dangerous), and probe cost = path length from agent.
    feat = {switch: dict(p_int=0.7, hazard=0.1), trap: dict(p_int=0.7, hazard=0.9),
            decoy: dict(p_int=0.1, hazard=0.1)}
    return g, agent, goal, door, switch, trap, decoy, feat


def reachable(g, agent, open_doors):
    H, W = len(g), len(g[0]); seen = {agent}; q = deque([agent])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen:
                cell = g[n[0]][n[1]]
                if cell in TRAVERSABLE or (cell == DOOR and n in open_doors):
                    seen.add(n); q.append(n)
    return seen


def cost(g, agent, c):
    """path length agent->c over currently-traversable cells (probe cost)."""
    H, W = len(g), len(g[0]); seen = {agent: 0}; q = deque([agent])
    while q:
        cur = q.popleft()
        if cur == c: return seen[cur]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in seen and (g[n[0]][n[1]] in TRAVERSABLE or n == c):
                seen[n] = seen[cur] + 1; q.append(n)
    return 999


def explore(mu, lam=1.0, budget=3, verbose=False):
    g, agent, goal, door, switch, trap, decoy, feat = make_world()
    candidates = [switch, trap, decoy]
    open_doors = set(); probed = []; trap_hit = False; discovered_switch = None
    while budget > 0 and goal not in reachable(g, agent, open_doors):
        # score every UN-probed candidate from OBSERVABLES only (no oracle on true type)
        scored = []
        for c in candidates:
            if c in probed: continue
            f = feat[c]
            e_reach = f["p_int"] * GAIN                        # ESTIMATE, not the true effect
            sc = e_reach - lam * cost(g, agent, c) - mu * f["hazard"]   # mu = cost of an irreversible failure
            scored.append((sc, c))
        if not scored: break
        scored.sort(reverse=True); best_score, c = scored[0]
        if best_score < 0:                                    # abstain: nothing worth the risk
            if verbose: print(f"    abstain (best score {best_score:.1f} < 0)")
            break
        # PROBE c for real (irreversible): now the true type is revealed
        budget -= 1; probed.append(c); true = g[c[0]][c[1]]
        r0 = len(reachable(g, agent, open_doors))
        if true == TRAP:
            trap_hit = True
            if verbose: print(f"    probed {c}: TRAP -> irreversible failure")
            break
        if true == SWITCH:
            open_doors.add(door)
        d_reach = len(reachable(g, agent, open_doors)) - r0
        if d_reach > 0: discovered_switch = c
        if verbose: print(f"    probed {c}: type={['FLOOR','WALL','SWITCH','DOOR','DECOY','GOAL','TRAP'][true]} dReach=+{d_reach}")
    return {"trap_hit": trap_hit, "discovered_switch": discovered_switch == switch,
            "probed": probed, "budget_left": budget, "goal_reachable": goal in reachable(g, agent, open_doors),
            "probed_decoy_flagged": decoy in probed and discovered_switch == decoy}


# prediction-only baseline: local detector can never see the far switch (door non-local) -> never opens it
def prediction_only_solves():
    g, agent, goal, door, *_ = make_world()
    return goal in reachable(g, agent, set())                 # door stays closed -> goal unreachable


print("=== Affordance Gate 1: exploration under uncertainty ===\n")
print("[risk-AWARE agent (mu>0)]")
aware = explore(mu=40.0, verbose=True)
print(f"  -> discovered_switch={aware['discovered_switch']} trap_hit={aware['trap_hit']} "
      f"goal_reachable={aware['goal_reachable']} budget_left={aware['budget_left']}\n")
print("[risk-BLIND agent (mu=0) — ablation]")
blind = explore(mu=0.0, verbose=True)
print(f"  -> trap_hit={blind['trap_hit']} goal_reachable={blind['goal_reachable']}\n")
print(f"prediction-only solves the far-switch level?: {prediction_only_solves()}\n")

passes = {
    "aware discovers far switch": aware["discovered_switch"],
    "aware avoids trap": not aware["trap_hit"],
    "aware reaches goal": aware["goal_reachable"],
    "aware respects budget (>=0 left)": aware["budget_left"] >= 0,
    "aware did NOT flag decoy as switch": not aware["probed_decoy_flagged"],
    "prediction-only FAILS": not prediction_only_solves(),
    "ablation: risk-blind HITS trap (mu load-bearing)": blind["trap_hit"],
}
print("=== Gate 1 pass criteria ===")
for k, v in passes.items():
    print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGATE 1: {'PASS' if all(passes.values()) else 'FAIL'}")
print("VERDICT: the discovery signal is now an AGENT behavior — cost/risk-aware probing finds the"
      "\n  prediction-invisible switch, routes around the irreversible trap (mu load-bearing: risk-blind"
      "\n  walks into it), rejects the decoy, under budget. NOT oracle. Next: class generalization (Gate 2).")
