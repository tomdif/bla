# B.L.A. Roadmap

The multi-phase plan from the current scaffold to a validated
implementation of the B.L.A. vision (`VISION.md`). Each phase is bounded
in scope, ends with a falsifiable gate, and has an explicit
decision point at which the project either advances, pivots, or stops.

The roadmap is sequenced so that **early phases produce evidence that
makes later phases more credible** — and so that a failure at any phase
isolates which architectural pillar is responsible, instead of
invalidating the whole vision.

## Overview

Total: 10 phases. Phase 0 is done. Phases 1–4 ("foundation") are doable
on local + small cloud compute. Phases 5–8 ("scale") require real
compute budget. Phase 9 ("world model") is the embodied-perception
integration. Phase 10 ("hardening") is post-research engineering.

| Phase | Pillar | Time | Compute | Status |
| --- | --- | --- | --- | --- |
| 0 | Substrate | done | $20 | ✅ |
| 1 | Cognitive headroom | 1–2 weeks | $50 | next |
| 2 | Verification layer | 3–4 weeks | $100 | |
| 3 | Hybrid memory | 3–4 weeks | $100 | |
| 4 | Text proof-of-concept | 4–6 weeks | $500 | |
| 5 | Compute economy (RL router) | 3–5 weeks | $300 | |
| 6 | Scale procedural core (500M, 3 GPU) | 1–3 weeks | $500–$2K | timed: ready to launch |
| 7 | Scale hybrid memory | 4–6 weeks | $2K | |
| 8 | Certified cognitive throughput benchmark | 4–8 weeks | $5K | |
| 9 | World model (System 1 at scale) | 8–16 weeks | $50K–$200K | |
| 10 | Hardening | post-research | — | |

**Decision points** at the end of phases 1, 4, 6, and 8. Anything below
gate threshold triggers an explicit go / pivot / stop call before
proceeding.

---

## Phase 0 — Substrate (DONE)

**Goal.** A composable Python scaffold for every named B.L.A.
component, validated end-to-end on a toy task.

**Delivered.** Six packages (`system1_jepa`, `system2_dca`, `latent_bus`,
`tensor_ram`, `diagnostics`, `tests`); 6×B200 DDP + AMP infrastructure
verified; BC + linear policy hits 100% on the navigate environment in
1000 training steps. Full session log in `SESSION_2026_05_09.md`.

**Gate (passed).** End-to-end pipeline trains and reaches a non-trivial
success rate without architectural failure.

---

## Phase 1 — Cognitive headroom

**Goal.** Build a task that BC alone *cannot* solve, then exercise the
cognitive components (world model rollouts + memory + temporal
predictor) on it. Without this gate, every subsequent claim about B.L.A.
adding value over imitation is unfalsifiable.

**Deliverables.**

1. **Multi-target navigate env.** Extension of `system1_jepa/navigate_env.py`:
   3 targets in a fixed sequence (red → green → blue). State must
   include "which targets visited so far"; reward only on visiting the
   *correct next* target. Memory becomes load-bearing.
2. **BC baseline on multi-target.** Confirm BC plateaus below 60%
   (failure mode: forgets which target was last visited). This number
   is the score B.L.A. must beat.
3. **Real state-space scratchpad.** Replace the Python for-loop in
   `system2_dca/ssm.py` with `mamba-ssm` (or `flash-linear-attention`
   if mamba-ssm doesn't install cleanly).
4. **End-to-end pipeline run.** First time the full graph trains: image
   → encoder → memory query → SSM scratchpad → temporal predictor →
   action. With reward signal but no verification yet.
5. **Comparison plot.** BC vs B.L.A. on multi-target: success rate over
   training steps, with per-component ablations (no SSM, no memory, no
   predictor).

**Gate (pass = continue).**

- B.L.A. ≥ BC + 20 percentage points on multi-target success rate.
- Or: ablation pinpoints which subsystem is missing.

**Decision point.** If B.L.A. = BC, the architecture is decoration on
this task. Either redesign the task to make memory + planning
load-bearing, or rethink the architectural assumptions before
proceeding to Phase 2. *Do not advance with an unfalsified claim.*

**Estimate.** 1–2 weeks. ~$50 of compute.

---

## Phase 2 — Verification layer

**Goal.** Outputs become *commitment objects*, not raw tensors or token
sequences. The router learns a real action space. Uncertainty is
calibrated.

**Deliverables.**

1. **`verification/` package.**
   - `CommitmentObject` dataclass — claim, evidence, reasoning_trace,
     proofs_run, tests_run, simulations_run, counterexamples_searched,
     uncertainty, reproducibility_packet.
   - `Certifier` ABC with three concrete implementations:
     `TestRunner` (Python sandbox + property tests), `ProofChecker`
     (Z3 / SymPy stub for now, Lean later), `SimulatorAgreement` (run
     the world model forward N times under noise, check the action
     sequence still reaches goal).
2. **Router action space expanded** from 2 ({wake, sleep}) to 7
   ({answer, retrieve, simulate, prove, search, ask, defer}) in
   `latent_bus/router.py`. Each action is a callable that consumes
   compute budget and produces certifier-checkable output.
3. **One closed-loop demo.** Multi-target navigate, but with a
   commitment object emitted per episode. The certifier is
   `SimulatorAgreement`: the action plan is run forward 10× under noise
   and the certifier signs off only if ≥ 8 trajectories reach the goal.
4. **Calibration audit.** Plot `commitment.uncertainty` vs actual error
   rate on held-out episodes. Compute Brier score and Expected
   Calibration Error.

**Gate.**

- Brier score ≤ 0.1 on held-out commitment objects.
- ECE ≤ 0.05.
- Certified episodes have ≥ 95% success at deployment.

**Decision point.** If uncertainty is uncalibrated, the verification
layer is theater. Either fix calibration (likely needs RL with proper
reward signal) before proceeding, or admit B.L.A.'s "verified outputs"
claim is currently aspirational.

**Estimate.** 3–4 weeks. ~$100 compute.

---

## Phase 3 — Hybrid memory

**Goal.** Four-layer memory stack. Provenance and freshness are
mandatory fields, not optional.

**Deliverables.**

1. **Symbolic memory.** `system2_dca/symbolic_memory.py` —
   `MemoriaSymbolicStore` wrapping the existing `memoria` repo
   (knowledge graph + spectral retrieval + cross-encoder rerank,
   already at 95% R@5 on LongMemEval). Bypass the LLM extractor for
   structured ingestion.
2. **Executable memory.** `system2_dca/executable_memory.py` —
   typed registry of:
   - Python sandbox (subprocess with `RestrictedPython` or `nsjail`)
   - SymPy for math
   - Z3 for SAT/SMT
   - The world-model simulator from Phase 1
   Each entry is callable with arg/return type contracts.
3. **Episodic memory.** `system2_dca/episodic_memory.py` — time-indexed
   (observation, action, outcome, certifier_result) tuples. Indexed by
   recency + salience. Reuses memoria's SQLite schema, separate table.
4. **Wikidata-100K ingestion.** `scripts/ingest_wikidata.py` — pull
   100,000 entities + their direct triples from Wikidata into the
   symbolic store. Verify retrieval precision on held-out factual
   queries.
5. **Memory-augmented closed-book QA test.** Comparison: parametric
   answer vs symbolic-retrieval-augmented answer on simple factual
   questions. Provenance audit: for every answer, can we cite the
   retrieved triples?

**Gate.**

- Retrieval-augmented closed-book QA accuracy ≥ parametric + 30 percentage points on held-out factual questions.
- Provenance correctness ≥ 95% (cited triples actually support the answer).

**Decision point.** None. Memory is foundation; if it doesn't help, the
problem is the integration with downstream modules, not the memory itself.
Iterate within Phase 3 until the gate is met.

**Estimate.** 3–4 weeks. ~$100 compute.

---

## Phase 4 — Text proof-of-concept

**Goal.** Latent diffusion on real text, retrieval-grounded, with
commitment objects. First time B.L.A. produces text, not just navigate
actions.

**Deliverables.**

1. **Tokenizer wiring.** Frozen GPT-2 BPE input embeddings as the
   `DeterministicLexicalDecoder` codebook. Add tokenizer to the data
   path. Train-time text tokenization → diffusion canvas.
2. **SEDD baseline.** Reproduce a small SEDD (or CDLM) on a 100M-token
   subset of OpenWebText. This is a published recipe; run it before
   B.L.A. extensions to make sure the substrate is competitive at all.
3. **B.L.A. text recipe.** Add to the SEDD baseline: bus alignment with
   the JEPA encoder (not used for text — used for cross-modal grounding
   when present), RAM read at every diffusion step, commitment object
   on output.
4. **Closed-book QA evaluation.** TruthfulQA (small subset, 100
   questions). Measure hallucination rate vs SEDD baseline vs
   B.L.A.+retrieval.

**Gate.**

- B.L.A. + retrieval ≤ 50% of the SEDD baseline's hallucination rate
  at equivalent compute on TruthfulQA.
- Output commitment objects pass calibration audit (Brier ≤ 0.15 in
  open-domain — looser than Phase 2 because text is harder).

**Decision point.** If B.L.A. + retrieval doesn't reduce hallucination,
either (a) retrieval isn't being used by the diffusion (instrument and
fix), or (b) the diffusion-retrieval coupling is wrong (rethink). Don't
proceed to scaling.

**Estimate.** 4–6 weeks. ~$500 compute (small text models on a single
B200 are tractable).

---

## Phase 5 — Compute economy

**Goal.** Router learns to allocate test-time compute. The "every joule
routed to the right substrate" claim becomes empirical.

**Deliverables.**

1. **Multi-difficulty task mix.** Easy (translate "hello" to French) +
   medium (3-step arithmetic) + hard (proof obligation, code with
   property tests). Each task has a ground-truth difficulty label and a
   compute-budget upper bound.
2. **Router RL training loop.** Reward = task accuracy − λ × FLOPs
   used. Routed actions are: shallow exit / single retrieve + answer /
   deep recurrence with verification. λ scheduled to gradually
   penalize compute as accuracy stabilizes.
3. **Calibration of allocation.** Plot allocated compute vs ground
   truth difficulty. Plot accuracy vs compute used per task.

**Gate.**

- Trained router uses ≥ 10× compute on hard tasks vs easy tasks.
- Accuracy on easy tasks does not drop more than 5 percentage points
  vs always-deep baseline.
- Total compute on the mix is ≤ 30% of always-deep baseline.

**Decision point.** RL training collapsing into "always deep" or
"always shallow" indicates the reward signal is wrong. Iterate within
Phase 5; do not advance to scaling without a working router.

**Estimate.** 3–5 weeks. ~$300 compute.

---

## Phase 6 — Scale procedural core

**Goal.** A compact procedural reasoner trained on a synthetic-logic
curriculum. The "asymmetric scaling" claim becomes testable for the
first time.

**Architectural decisions made during prep (2026-05-10):**

- **Target size: 500M params** (down from the original 1B-3B band).
  Beating 1.5B GPT-2 with a 500M B.L.A. core is a *stronger*
  asymmetric-scaling result (3× ratio vs ~1.5×) and iterates 4× faster,
  which we'll need given recipe debugging. Can scale up to 1B if 500M
  is undertrained.
- **Standard transformer with PyTorch SDPA / FlashAttention** instead
  of an SSM scratchpad. (a) `mamba-ssm` requires CUDA dev headers and
  compile-from-source which our pod image doesn't ship; (b) the
  asymmetric-scaling thesis is about parameter efficiency vs frontier
  *transformers* — testing it with a transformer keeps the comparison
  clean; (c) for 1-2K-token sequences on B200, attention is fine
  memory-wise. SSM scratchpad becomes a Phase 9 concern (long video
  sequences are its natural home).
- **3 GPUs instead of 6.** Pod availability constraint. FSDP on 3
  B200s handles 500M-1B comfortably; wallclock is ~2× slower than
  6-GPU baseline, total cost is roughly the same in $.

**Deliverables.**

1. **Synthetic curriculum.** Mixture of:
   - Code execution traces — synthetic Python programs run in a
     sandbox, captured stdout. (`curriculum_python.py`)
   - Math derivations — MetaMathQA from HuggingFace, with synthetic
     fallback. (`curriculum_math.py`)
   - Formal logic problems — FOLIO from HuggingFace (auth-gated, falls
     back to a synthetic propositional generator). (`curriculum_logic.py`)
   - AST manipulation (synthetic, generated) — TODO
   - Game-theory / planning (synthetic Sokoban, BabyAI) — TODO
   The curriculum explicitly *excludes* factual content (no Wikipedia,
   no Common Crawl). All facts must be retrieved from memory at
   inference. **Status: smoke at 5K examples validated; full 30K-100K
   curriculum is one `scripts/build_curriculum.py` call away.**
2. **500M reasoner training.** FSDP across 3 B200s, bf16, standard
   transformer w/ RoPE + SDPA + RMSNorm + tied input/output embeddings
   + GPT-style scaled init. ~30K-50K steps of curriculum pretraining.
   **Status: model + training script validated on micro config (16M
   params, 100 steps, loss 10.83 → 9.81 with correct random-init
   baseline = log(50257)).** Code: `system2_dca/procedural_core.py`,
   `scripts/phase6_train.py`.
3. **Comparison.** B.L.A. 500M + memory vs GPT-2 1.5B parametric on:
   - Math (GSM8K, MATH)
   - Code (HumanEval, MBPP)
   - Proof (MiniF2F-test)
   - Long-horizon planning (custom)

**Gate.**

- B.L.A. 500M + memory ≥ GPT-2 1.5B on at least 2 of the 4 task
  categories, *with 3× fewer parameters*.
- B.L.A. on retrieved-fact tasks ≥ GPT-2 + RAG (shows the architecture
  is not just "RAG with extra steps").

**Decision point.** Make-or-break gate for the asymmetric scaling
thesis. If 500M B.L.A. doesn't beat 1.5B GPT-2 on procedural tasks
where memory matters, options are: (a) scale to 1B B.L.A. and re-run
(more time, marginal cost); (b) revisit the curriculum (might be too
narrow); (c) admit the thesis isn't supported at this scale.

**Estimate (timed 2026-05-10 on 3×B200 FSDP, actual 500M config):**

  * Throughput: **67K tokens/sec** at steady state
  * Step time: **0.66 sec/step** at batch=16, seq=1024
  * Loss trajectory: 9.90 (step 25) → 4.40 (step 200) — strong gradient signal

Single-run wallclock + cost (3 × $4/hr ≈ $12/hr):

| Schedule | Wallclock | Cost |
| --- | --- | --- |
| 30K steps | 5.5 hrs | ~$66 |
| 50K steps | 9.2 hrs | ~$110 |
| 100K steps | 18 hrs | ~$220 |

**Full Phase 6 program** (curriculum sweeps + 5-10 iterations + eval
cycles + GPT-2 1.5B comparison): **1-3 weeks calendar time, $500-$2K
compute.** Originally estimated $15-50K based on production-LLM
schedules (billions of tokens, dozens of iterations); the
procedural-only curriculum is much cheaper because the model isn't
trying to absorb the whole web. Real budget commitment but not the
multi-month / multi-team scope the original number implied.

---

## Phase 7 — Scale hybrid memory

**Goal.** 10⁸–10⁹ entries across vector + symbolic + episodic +
executable layers. Adversarial-robustness audit.

**Deliverables.**

1. **Wikidata full** (~10⁸ entities) ingested into symbolic memory.
2. **Common Crawl filtered subset** (~10⁹ documents → vector embeddings)
   in the vector layer.
3. **arXiv abstracts** (~2M) for technical retrieval.
4. **StackExchange** (~10⁷ Q&A pairs) for episodic-style memory.
5. **Code corpus** — Python stdlib + top 1000 PyPI packages → executable
   memory entries with type signatures.
6. **Embedder upgrade audit.** Memoria's default
   `all-MiniLM-L6-v2` is fast and good at small scale (95% R@5 on
   LongMemEval, 96.5% on Phase 3 — see `PHASE_3_DECISION.md`). At
   10⁶+ entries, embedder choice starts to matter for retrieval
   precision. Benchmark candidates:
     * `all-mpnet-base-v2` (stronger general-purpose SBERT)
     * `bge-large-en-v1.5` (BAAI; SOTA on retrieval benchmarks)
     * `e5-large-v2` (Microsoft; strong on factual retrieval)
     * a domain-specialized variant (e.g. SciNCL for arXiv,
       CodeBERT for code corpus)
   Goal: pick the embedder that minimizes precision degradation
   from 10⁶ to 10⁹ entries while staying within latency budget.
7. **Adversarial-robustness audit.**
   - Insert 1000 plausible-but-wrong "facts" into symbolic memory.
   - Measure: how often does the verification layer catch them?
   - Insert poisoned retrieval candidates with typos / contradictions.
   - Measure: contamination rate of downstream answers.
8. **Chroma Context-1 evaluation as retrieval handler.** Once the
   public agent harness ships, drop Context-1 in as a replacement
   for memoria's three-pass pipeline on multi-hop / context-pruning
   tasks. Context-1 is a 20B agentic search model with explicit
   prune_chunks (0.94 accuracy on irrelevant-passage discard) and
   query decomposition — designed for the kind of long-horizon
   retrieval where simple top-k breaks down. Specific test:
   single-fact QA (where memoria already excels) vs multi-hop QA
   where the answer requires combining 3+ retrieved facts. Goal:
   know when to dispatch the router's RETRIEVE action to Context-1
   vs memoria, based on the question's structural complexity.

**Gate.**

- Retrieval precision degrades < 5 percentage points from 10⁶ to 10⁹
  entries.
- Adversarial poisoned facts detected by verification ≥ 80% of the time.
- Latency p95 < 100ms for retrieval at 10⁹ scale.

**Decision point.** None — this is engineering, not research. If
adversarial robustness is below threshold, it indicates the
verification layer needs strengthening, which loops back to Phase 2.

**Estimate.** 4–6 weeks. ~$2K compute (mostly storage and indexing,
not training).

---

## Phase 8 — Certified Cognitive Throughput benchmark

**Goal.** Define the metric. Run the comparison. Publish the number.

**Deliverables.**

1. **CCT score definition.**
   - Verified-decisions-per-joule × verified-decisions-per-second ×
     verified-decisions-per-dollar, normalized across 4 task categories.
   - Verification is task-specific: tests pass / proof checks /
     simulator agrees / sources cite.
2. **Task suite.** Math (MATH benchmark) + Code (HumanEval+ with hidden
   tests) + Factual QA (TruthfulQA + custom retrieval-only set) +
   Planning (custom multi-step environments).
3. **Baselines.** (a) frontier LLM (Llama / DeepSeek / GPT-5.x),
   (b) LLM + RAG, (c) LLM + tools, (d) **Chroma Context-1 + Chroma
   DB** as a 20B-parameter retrieval-specialist comparison. The
   Context-1 baseline is the right adversarial test for the
   asymmetric-scaling thesis: if a 20× larger model trained
   specifically for retrieval beats B.L.A.'s 1B reasoner + memoria
   + verification stack on the retrieval-heavy categories, the
   thesis weakens. If B.L.A. gets within striking distance with
   20× fewer reasoning-side parameters, that's a strong asymmetric-
   scaling result.
4. **Comparison runs.** Each system on each task category. Report CCT
   per system per category.

**Gate.**

- B.L.A. higher CCT than at least 2 of the 3 baselines on at least 2 of
  the 4 task categories.
- Each comparison reproducible via the reproducibility_packet emitted
  with every commitment object.

**Decision point.** Final research-program gate. If B.L.A. doesn't beat
*any* baseline on *any* category, the asymmetric-scaling thesis is
empirically unsupported at this scale and the program halts.

**Estimate.** 4–8 weeks. ~$5K compute.

---

## Phase 9 — World model + SSM scratchpad

**Note (2026-05-10):** This phase now also absorbs the deferred SSM
scratchpad work. Phase 1 used a Python-loop reference SSM (fine at
T=14 episodes); Phase 6 uses standard transformer attention (fine at
2K sequences on B200). Phase 9 is the right place for real `mamba-ssm`
or `flash-linear-attention` because (a) embodied-perception sequences
can be much longer (video clips at 60Hz), and (b) the SSM state-space
scratchpad is the *blueprint's* claim, fitting System 1 better than
System 2.

**Goal.** A V-JEPA-2-class System 1 perception model integrated with the
action layer + a real SSM scratchpad for long-horizon recurrence.

**Deliverables.**

1. **Spatiotemporal JEPA at scale.** Use the existing
   `system1_jepa/spatiotemporal.py` substrate, scaled to ~5B params,
   trained on a real video corpus (DROID + Ego4D subset).
2. **Action-conditioned rollouts.** Predictor takes (history, action
   sequence) → future-state distribution.
3. **Counterfactual prediction benchmark.** "Given this scene, predict
   the outcome of action X vs action Y." Compare to V-JEPA-2 published
   numbers.
4. **Integration with planning.** CEM/MPPI on top of the world model
   for embodied tasks (PushT, MetaWorld).
5. **Real SSM kernel.** Deferred from Phase 1/Phase 6: integrate
   `mamba-ssm` (preferred, requires CUDA dev image) or
   `flash-linear-attention` (fallback, pure-Torch). Replace the
   Python-loop `CausalSSMScratchpad` reference impl with the fused
   kernel. Validate that the O(1) per-step claim holds on long video
   sequences. Drop into the spatiotemporal predictor's working memory.

**Gate.**

- Counterfactual prediction quality matches published V-JEPA-2 on
  shared benchmark categories.
- Embodied planning success on PushT ≥ 70% (LeWM published 98% as the
  ceiling).

**Estimate.** 8–16 weeks. **$50K–$200K compute.** This is the second
phase requiring real budget. Sequenced after Phase 8 because the
text-world-memory loop (Phases 4–6) is the cheaper validation path.

---

## Phase 10 — Hardening

Post-research engineering. Multi-user serving, security hardening,
monitoring, SLAs, recovery. Out of scope until Phases 1–9 produce a
system worth deploying.

---

## Risk register

What could derail the program, ranked by probability:

1. **Multi-target navigate solved by BC anyway** (Phase 1). Probability:
   medium. Mitigation: design 2-3 backup harder tasks (partial
   observability, delayed-effect actions) before starting.
2. **Calibration is uncalibrated** (Phase 2). High probability initially.
   Mitigation: known fix is RL with proper proper-scoring-rule reward
   (Brier or log-loss). Budget extra time.
3. **Memory retrieval doesn't help downstream** (Phase 3). Low —
   memoria already shows 95% R@5. Risk is integration, not the memory.
4. **Diffusion + retrieval coupling fails** (Phase 4). Medium. The
   recipe is open research. Mitigation: SEDD baseline first to isolate
   "diffusion works at all" from "retrieval helps."
5. **Router collapses to always-deep / always-shallow** (Phase 5).
   Medium-high. Known failure mode in PonderNet-style RL. Mitigation:
   curriculum + reward shaping research, budget extra weeks.
6. **1B B.L.A. doesn't beat 1.5B GPT-2** (Phase 6). Medium. If this
   fails, the *whole asymmetric-scaling thesis* is on the line.
   Mitigation: Phases 1–5 must produce strong evidence first; otherwise
   don't burn $50K to find out.
7. **Memory at 10⁹ scale is too slow** (Phase 7). Low. Standard ANN
   techniques handle this; risk is integration.
8. **CCT metric is uninterpretable** (Phase 8). Medium. Verifying
   "verified decisions" across modalities is genuinely hard. Mitigation:
   per-category subscores published alongside the aggregate.
9. **World model training fails to converge** (Phase 9). Medium-high.
   V-JEPA-2 is itself recent research; reproducing it at our scale is
   non-trivial.

## Living-document principle

This roadmap is updated whenever architectural reality shifts during
phase prep or execution. Recent updates:

- **2026-05-10 (Phase 6 prep)** — scoped target down 1B → 500M;
  adopted standard transformer + SDPA instead of mamba-ssm scratchpad
  (mamba-ssm needs CUDA dev image we don't have, and the asymmetric-
  scaling thesis tests cleaner with a standard transformer); 3-GPU
  pod constraint accepted at ~2× wallclock; SSM kernel work absorbed
  into Phase 9.
- **2026-05-10 (Phase 6 timing measured)** — actual 500M config on
  3×B200 FSDP: 67K tokens/sec, 0.66 s/step, loss 11→4.4 in 200 steps.
  Single-run cost is $66-$220 (30K-100K steps); full Phase 6 program
  is $500-$2K. Original $15-50K estimate revised down ~30× because
  the procedural-only curriculum doesn't need production-LLM scale.
  Calendar time revised 4-8 wk → 1-3 wk.
- **2026-05-10 (Phase 4b)** — split into 4a (CPU-tractable RAG with
  GPT-2) and 4b (BERT-MLM as non-AR baseline). Full SEDD reproduction
  parked for Phase 6+ scale.
- **2026-05-10 (Phase 7)** — added embedder upgrade audit and
  Chroma Context-1 evaluation as separate deliverables.
- **2026-05-10 (Phase 8)** — added Chroma Context-1 + Chroma DB as
  20B retrieval-specialist baseline for the asymmetric-scaling
  comparison.

If a phase produces a finding that adjusts a downstream phase's spec,
the change goes here, not buried in a decision doc.

## How we work this plan

- **One phase active at a time.** Multi-tasking across phases dilutes
  signal and makes failure attribution hard.
- **Each phase ends with a decision document.** Plain Markdown:
  what we built, what we measured, did we hit the gate, what next.
- **Decision points are real.** "Pivot" is a valid outcome at every
  decision point. Don't burn compute past a failed gate.
- **Compute budget per phase is locked in advance.** Phase 6 doesn't
  start without a $30K-$100K budget approval. Phase 9 doesn't start
  without $50K-$200K.
- **Falsifiability over enthusiasm.** Every claim in `VISION.md` should
  be testable by exactly one phase. If a claim isn't tested by any
  phase, either add a phase or drop the claim.

## Sequence

```
Phase 0 (DONE) → 1 → 2 → 3 → 4 → [decision] → 5 → 6 → [decision] → 7 → 8 → [decision] → 9 → 10
                ↑                  ↑                   ↑
                gate: B.L.A. > BC  gate: hallucination  gate: 1B beats 1.5B GPT-2
                                   reduces by retrieval
```

Total estimated time, optimistic: **52 weeks of focused research work**.
Total compute, optimistic: **$90K–$310K**. Both estimates roughly double
when accounting for failed runs and iteration cycles.

## Success criterion for the whole program

The program succeeds if Phase 8 produces a comparison plot where B.L.A.
beats at least one of {LLM, LLM+RAG, LLM+tools} on at least 2 of the 4
task categories on certified cognitive throughput, with reproducibility
packets that allow third-party verification.

The program *partially succeeds* if Phases 1–4 hit their gates but
Phases 6+ are unsupported by compute budget. In that case we have a
small-scale validated architecture and a clear scaling path documented;
that is itself a credible research contribution.

The program *fails informatively* if Phase 1 doesn't beat BC on the
multi-target task. That tells us something specific is wrong with the
architecture and is worth more than a vague affirmation.
