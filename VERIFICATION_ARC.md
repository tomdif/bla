# The Verification Arc — proposers propose, verifiers own truth

A self-contained body of work (the `phase-e-*` tags) demonstrating one architecture across multiple
verifier domains. The single invariant, unbroken across every layer and domain:

> **Everything proposes — language, an LLM, a learned surrogate, a trace-trainer, a cross-domain library.
> Only the domain's real verifier promotes. Truth is the deep verdict, never the shallow one.**

Every harness is **controls-first**: it ships falsifiable checks (including a *verifier-ablation* control
that shows what breaks when the verifier is removed). A gate prints `... : PASS` only if every control holds.

## How to run

```bash
python3 <gate>.py                      # deterministic stub mode — no API, reproducible, used by CI
RUN_LIVE_LLM_GATE=1 python3 <gate>.py  # opt-in: a real LLM (anthropic) is the proposer; needs ANTHROPIC_API_KEY
```
LLM gates fall back to the stub if the key is absent or the model returns nothing. No secrets are in source.

## Verifier domains and their native "gaming" attack

Each domain offers a cheap way to fake success; in each, *truth* had to be the deep check that catches it:

| domain | real verifier | gaming attack (passes the shallow check) | the guard that catches it |
|---|---|---|---|
| code | `python -m pytest` / `unittest` (subprocess) | special-case the failing input | a held-out test |
| math | `lean` / `lake env lean` + `#print axioms` | `by sorry` (it *typechecks*) | the axiom audit (`sorryAx`) |
| data | real `sqlite3` query execution | hardcode the visible rows (`WHERE name IN (...)`) | a held-out dataset |

## The harnesses

### Code domain
- **`code_repair_harness.py`** — the technique loop on a real failing `unittest`; canned patch pool; a gaming patch is rejected by a held-out test. `phase-e-code-repair-v0`
- **`code_repair_v1.py`** — Stage 1: a **real LLM** patch-proposer wired into the real test verifier. `phase-e-code-repair-v1`
- **`code_repair_v2.py`** — Stage 2: the same loop on a **real multi-file package** (`wmos`), verified by its own 38-test suite; real regression protection. `phase-e-code-repair-v2`
- **`patch_outcome_surrogate.py`** — Stage 3: a learned surrogate **triages** candidates (cheap imagination) — it only orders; run-until-green means a green patch is never silently skipped. `phase-e-patch-surrogate-v0`
- **`trace_trained_surrogate.py`** — Stage 4: the surrogate **learns from the loop's own traces** (self-improving); trains only on verifier labels, so it can't bootstrap a delusion; adapts under distribution shift. `phase-e-trace-surrogate-v0`
- **`proposer_from_traces.py`** — Stage 5: the **proposer itself** learns what to propose per bug-signature; gaming-guarded; an unseen signature triggers exploration. `phase-e-proposer-traces-v0`
- **`swebench_adapter.py`** — the loop on a **real external GitHub repo** (`mahmoud/boltons`), verified by its own pytest suite (the SWE-bench shape). `phase-e-swebench-adapter-v0`

### Math domain (Lean 4)
- **`lean_repair_v1.py`** — the loop with the **real Lean type checker** as verifier; `sorry` typechecks but the axiom audit rejects it. `phase-e-lean-repair-v1`
- **`lean_proposer_from_traces.py`** — Stages 3–5 on Lean: a trace-trained **tactic proposer** over a real-Lean truth table. `phase-e-lean-proposer-traces-v0`
- **`lean_repair_real_v1.py`** — the loop on a **real Mathlib-backed research repo** (`collatz-proven`): inject `sorry` into a real proven lemma, re-derive a real proof verified by `lake env lean`; original restored byte-for-byte. `phase-e-lean-repair-real-v1`

### Data domain (SQL)
- **`sql_repair_v1.py`** — the loop with a **real SQLite database** as verifier; a hardcoded query passes the visible rows but is rejected by held-out data. `phase-e-sql-repair-v1`

### Cross-domain
- **`cross_domain_library.py`** — one technique store keyed by abstract signature spanning **code + Lean + SQL**; transfer is *proposed, never assumed* — each domain's real verifier must re-confirm; spurious transfers are rejected. `phase-e-cross-domain-library-3domain-v0`

### Earlier scaffolding (same invariant, other modalities)
- **`technique_discovery_gate.py`** — the researcher loop: techniques propose, verifiers promote, a cumulative `TechniqueCard` library. `phase-e-technique-discovery-v0`
- **`rh_proof_harness.py`** — the loop as a manipulable RH obstruction atlas (a gatekeeper, *not* a proof); the circularity audit is load-bearing. `phase-e-rh-proof-harness-v0`
- **`geometry_canvas_gate.py`** — 3D latents earn their place only via action-verified controls. `phase-e-geometry-canvas-gate-v0`
- **`adversarial_gate.py`, `adversarial_full_gate.py`** — red-team of the invariant and the trust model / TCB. `phase-e-adversarial-*-v0`
- **`wmos/`** — the World-Model Operating System the early stages run on (engine, adapters, safety, techniques).

## Status

All gates pass in deterministic stub mode; the LLM gates (`code_repair_v1/v2`, `swebench_adapter`,
`lean_repair_v1`, `lean_repair_real_v1`, `sql_repair_v1`) have additionally been run live and succeed.
The honest failures encountered along the way (a wrong LLM proof rejected by Lean, a write-race against a
real test subprocess, shuffle-null leaks, gaming guards that only fired once `sorry` could win a tie) are
themselves evidence the verifiers were load-bearing, not decorative.
