# WMOS — World-Model Operating System

A verificationist operator console for affordance world models. Not a chatbot wrapper: a verified
loop with one invariant enforced **in code**:

> **No unverified proposal owns truth.**

Proposers (a **language** model, a **learned Δ-estimator**, a **persistent library**) put *typed*
hypotheses on a bus — advice only. A **Δachievable verifier** and real action feedback are the only
things that can mark a belief accepted. An **action governor** refuses to release an action that
hasn't passed a verification policy. Everything is auditable.

```
observe → perceive → hypothesize → imagine → verify → act → audit → remember
language proposes · world model predicts · affordance/value verifies · action feedback owns truth
```

## Install & run

Zero required dependencies (stdlib only). Optional `anthropic` enables the live-LLM proposer.

```bash
python3 -m wmos                  # operator console (grid world, shadow autonomy)
python3 -m wmos --demo           # scripted cockpit demo
python3 -m wmos --serve          # web cockpit at http://127.0.0.1:8765
python3 -m wmos --adapter reach  # point at a different world (continuous reach body)
# or, after `pip install -e .`:
wmos --serve
```

The live-LLM proposer is opt-in: `RUN_LIVE_LLM_GATE=1 python3 -m wmos` (needs `ANTHROPIC_API_KEY`).
Without it, a deterministic commonsense stub proposer is used.

## Console commands

```
/state /canvas /hypotheses /why <id> /simulate <id> /verify <id> /act <id>
/autonomy manual|assisted|auto /canaries /library /explain /report /reset /adapters /help
```

## Autonomy dial

| level      | behavior                                                              |
|------------|----------------------------------------------------------------------|
| `manual`   | shadow only — nothing is released without an explicit `/act` (default)|
| `assisted` | high-confidence, in-band proposals become *trustable* (still gated)   |
| `auto`     | the governor may release trusted actions without an explicit verify   |

## Architecture (layers)

```
wmos/
├── adapters/       pluggable worlds (grid, reach; add your own via @register)
├── engine.py       hypothesis bus · proposers · verifier · governor · memory · Harness
├── persistence.py  durable library + session audit under ~/.wmos
├── config.py       defaults ← ~/.wmos/config.json ← env
├── cli.py          robust operator console
└── server.py       web cockpit (stdlib http.server, zero deps)
```

## Real ARC-AGI-3 (ls20)

```bash
python3 -m wmos --adapter arc        # runs on REAL recorded ls20 frames (64x64), or a synthetic fallback
```

The `arc` adapter perceives the real game (avatar = color 12, maze corridors = 3, the white-cross
operator = 0/1, yellows = 11) and maps them to WMOS candidates. It's honest about the state of the
art: Δachievable here is **maze reachability**, which captures navigation/gating affordances — but
ls20's actual **win mechanic is a shape-match** (run the avatar over the cross to flip its key until
it matches the exit). So the estimator/language confidently propose the cross, and the reachability
verifier correctly **refuses to confirm it as a door-opener** (Δreach = 0). That refusal — no false
affordance — *is* the point. A richer "achievable" (win-states gated on key-shape) is the open
extension, and it's an **adapter change, not a WMOS one**. Sources: recorded frames
(`~/arc_local/.../ls20_transitions.npz`), a synthetic fallback, and a live-client wiring stub.

## Richer achievable + hierarchical sub-goals (ls20 shape-match)

```bash
python3 -m wmos --adapter ls20    # /goals shows the decomposition; the cross is now CONFIRMED
```

The `arc` adapter is honest that reachability can't see ls20's win (the cross flips a *key shape*, not
what's reachable) — so the cross is refuted there. The `ls20` adapter supplies the **richer achievable
signal**: a hierarchical potential over sub-goals.

```
WIN = key.shape == exit.shape  AND  avatar at exit
 ├── shape_matched          (apply the cross operator — a multi-flip plan, delayed payoff)
 └── at_exit  requires: [shape_matched]   (navigate; entering before matching does NOT win)
```

Now `measure_delta(cross)` is the *shape sub-goal* progress (+1.1), not reachability (0), so the
verifier **confirms** the cross where the flat signal refuted it — and a yellow decoy still measures 0
and is refuted. `wmos.goals.GoalHierarchy` is generic (frontier-finding, ordering, value, achievement);
the adapter supplies the predicates. The `/goals` command shows the live decomposition and frontier.
This is the *adapter change, not a WMOS change* the arc adapter pointed to — and it composes the
delayed-payoff arbiter (matching is K flips) with hierarchical ordering (match before exit).

## Add your own world

```python
from wmos.adapters import register, Adapter

@register("my_world")
class MyWorld(Adapter):
    def reset(self): ...
    def observe(self):       # {candidates:[{id,label,features}], reachable:int, solved:bool, scene:str, view:{...}}
        ...
    def measure_delta(self, cid):  # Δachievable of interacting with cid (no commit) — the expensive truth
        ...
    def apply(self, cid):    # commit the interaction
        ...
```

`python3 -m wmos --adapter my_world`. The loop, governor, verifier, memory, and audit are unchanged —
real ARC / robotics is an adapter, not a rewrite.

## Guarantees (tested)

`python3 -m pytest tests/test_wmos.py` covers: the invariant (unverified actions blocked),
verification-owns-truth (switch accepted, identical-looking trap refuted), OOD refusal, the autonomy
dial, library persistence across sessions, the reach adapter, and that no console command crashes.
