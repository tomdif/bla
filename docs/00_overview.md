# B.L.A. design overview

Five pillars, each its own doc:

1. [JEPA (System 1)](01_jepa.md) — patch JEPA + spatiotemporal JEPA +
   temporal predictor + SIGReg.
2. [D.C.A. (System 2)](02_dca.md) — SSM scratchpad + rectified-flow
   diffusion + frozen lexical decoder + differentiable RAM read.
3. [Bus + Router + Veto + Prefetch](03_bus.md) — connective tissue.
4. [Tensor RAM](04_tensor_ram.md) — non-parametric fact storage.
5. [Planning + Diagnostics](05_planning_diagnostics.md) — CEM, navigate
   env, linear probe, RAM attention.

## What this scaffold proves and what it doesn't

**Proves (or makes provable, given data):**
- The blueprint is implementable as one coherent training graph: every
  named component is here and gradient flows end-to-end across them.
- Each component can overfit a controlled toy task (single-batch
  prediction, multi-step rollout, RAM-aware diffusion, veto round-trip,
  bus alignment).
- The CEM planning loop closes against the temporal predictor — the
  measurable promise the blueprint's "true reasoning engine" claim
  rests on.

**Doesn't prove:**
- Anything about training-curriculum claims (e.g. "CPU has zero factual
  trivia"). That's a corpus-design question, not an architecture
  question.
- That the trained system is competitive with a Transformer monolith
  at any task. There is no benchmark in this repo where we have
  numbers to defend, only scaffolding to run them.
- That O(1) infinite context works. The SSM scratchpad has finite
  state-dim capacity; "effectively infinite" is an aspiration of SSMs
  broadly.

The point of this repo: a reproducible substrate for testing whether
the B.L.A. blueprint is a real architecture or a vibes paper. The next
work — picking benchmarks, running real curricula — is research, not
plumbing.

## Phase entry points

| Phase | Script | What it trains |
| --- | --- | --- |
| 1A | `scripts/phase1a_train_jepa.py` | Patch JEPA on still images |
| 1B | `scripts/phase1b_train_dca.py` | DCA diffusion on synthetic latents |
| 1C | `scripts/phase1c_populate_ram.py` | Tensor RAM index population |
| 1D | `scripts/phase1d_train_bus.py` | Latent bus alignment via InfoNCE |
| 1E | `scripts/phase1e_train_temporal.py` | Temporal predictor + multi-step rollout |
| 1F | `scripts/phase1f_navigate_plan.py` | CEM planning on navigate-to-target |
| utils | `scripts/load_codebook.py` | Load pretrained tokenizer embeddings into the lexical decoder |
