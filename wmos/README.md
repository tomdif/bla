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
