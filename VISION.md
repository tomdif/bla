# B.L.A. — Bicameral Latent Architecture

**A research program for asymmetric cognitive scaling.**

## Thesis

The next scaling law is not parameter count. It is **certified cognitive
throughput** — verified, world-updating decisions per joule, per second,
per dollar.

A monolithic autoregressive transformer entangles perception, reasoning,
memory, syntax, and facts into a single weight matrix. Adding parameters
scales all of those axes uniformly, whether the bottleneck is missing
knowledge, missing compute, missing planning, or missing verification.
Frontier transformer-style systems already work around this — sparse
mixtures-of-experts activate a fraction of total parameters per token,
retrieval-augmented systems externalize knowledge — but each of those
fixes is a patch on a fundamentally entangled architecture.

B.L.A. is not "smaller than a transformer." Its advantage is **functional
disentanglement**: each cognitive substrate scales on its own axis, with
its own metric, against its own ceiling. Every joule of compute is routed
to the substrate that needs it.

## The wager: Asymmetric Cognitive Scaling

Six axes, six metrics. None is parameter count.

| Axis | What scales | How it's measured |
| --- | --- | --- |
| **Perception** | Latent world-model capacity | Counterfactual physical-prediction quality per watt |
| **Procedural reasoning** | Recurrence depth, search, tool use, proof-state manipulation | Verified solutions per second on long-horizon problems |
| **Memory** | Hybrid stack capacity, retrieval precision, freshness | Successful retrieval-augmented decisions per query |
| **Action** | Tool repertoire, sandbox safety, rollback fidelity | Successful goal-state changes with verified outcomes |
| **Verification** | Certifier coverage, counterexample search depth | Fraction of outputs accompanied by valid certificates |
| **Test-time compute** | Adaptive ponder budget per query | Calibrated thinking time vs task difficulty |

The system wins by allocating across these axes, not by maximizing any
one. A 10B procedural core paired with a 100B grounded world model and
typed external memory is not a smaller transformer — it is a different
machine with different ceilings.

## Architecture

### System 1 — Grounded World Model (Perception)

A multimodal joint-embedding predictive architecture (JEPA-class) trained
on video, action, proprioception, audio, and touch. Optimized for:

- multi-horizon counterfactual prediction
- object-centric latent slots
- action-conditioned rollouts
- causal affordances ("what happens if I do X?")
- uncertainty maps over predictions
- foveation and active perception

System 1 scales until additional parameters no longer improve
counterfactual prediction per watt. The empirical ceiling is set by
sensor bandwidth, simulator quality, and the prediction horizon — not by
pretraining-corpus token count.

### System 2 — Procedural Reasoner (Logic Core)

A compact, recurrent reasoning model trained primarily on:

- formal logic and proof obligations
- code execution traces
- abstract syntax trees
- mathematical derivations
- planning, game theory, search trajectories
- verifier feedback (compiler errors, proof failures, test outcomes)

The reasoner is intentionally smaller than its world model and its
external memory. The bet: procedural intelligence benefits more from
recurrence, search, proof checking, tool execution, memory access, and
self-verification than from unlimited dense parameter growth. We do not
claim the reasoner becomes a "lossless logic engine" at any parameter
count. We claim that procedural workloads route their compute into
test-time iteration, not into more weights.

### Latent Workspace (Working Memory)

A typed working memory, not a single hidden state. At minimum:

- **Object slots** — entities perceived or hypothesized, with attributes
- **Causal graph** — relations between slots, updated as evidence arrives
- **Scratchpad** — fixed-size SSM state for fast read/write of
  intermediate conclusions
- **Proof state** — current obligations, available lemmas, applied tactics
- **Simulator state** — running predictive rollouts of plans
- **Task stack** — goal hierarchy, pending subgoals, deferred branches

The workspace runs at O(1) per-step compute via state-space recurrence.
We do not claim "infinite context"; we claim that the workspace's
representational capacity is the bound on context, and that capacity
is engineered rather than implicit in attention quadratic memory.

### External Memory — Four Layers

Knowledge is not stored in weights. It lives in a hybrid stack:

1. **Vector memory** — semantic retrieval over continuous embeddings.
   ANN-indexed (HNSW / IVF), differentiable read-path for query learning.
2. **Symbolic memory** — typed entity / relation / law / equation /
   API-signature store. Graph-structured. Provenance-bearing. Source-cited.
3. **Episodic memory** — time-indexed agent experience: observations,
   actions, outcomes, certifier results. Indexed by recency and salience.
4. **Executable memory** — registry of tools, sandboxes, simulators,
   theorem provers, test runners, verified programs. Memory entries that
   can *run*, not just be read.

Each entry carries provenance, freshness, and confidence. Updates are
typed transactions. Contradictions trigger explicit resolution, not
silent overwrite.

The retrieval contract is not "fetch a vector." It is **"retrieve a
stateful, typed, provenance-bearing cognitive object."**

### Entropy Router — Budget Allocator

At every reasoning step the router chooses an action, with a compute
budget attached:

- **Answer** — emit current best response
- **Retrieve** — query memory (vector / symbolic / episodic / executable)
- **Simulate** — run the world model forward under candidate actions
- **Prove** — invoke the verification layer on a sub-claim
- **Search** — counterexample search against the current draft
- **Ask** — emit a clarifying query to the user or environment
- **Defer** — escalate to human, longer-running tool, or batch later

The router is not a confidence threshold. It is a learned policy over
this action space, trained with `Reward = Task Success − λ · FLOPs`. The
research bet is that this policy is learnable and generalizes; this is
not yet established at production scale.

### Verification Layer — Where Truth Is Certified

Truth is not stored. It is certified.

The verification layer turns answers into **commitment objects**:

```
CommitmentObject:
  claim                # the answer
  evidence             # retrieved passages, facts, citations
  reasoning_trace      # summary of the latent rollout
  proofs_run           # proof-checker outcomes (Lean / Coq / Z3)
  tests_run            # unit / property / integration test results
  simulations_run      # world-model rollouts and disagreements
  counterexamples_searched
                       # adversarial probes that failed
  uncertainty          # calibrated bound on the claim
  consensus            # if multi-agent: agreement structure
  reproducibility_packet
                       # everything needed to redo the computation
```

Outputs without commitment objects are draft-quality. Outputs with
commitment objects are deployable. Stop conditions are task-specific:

- **Code** — tests pass under the specified inputs
- **Proofs** — proof checks under a chosen formal system
- **Physics / planning** — simulator agrees across perturbations
- **Factual** — sources are consistent and uncertainty is below threshold
- **Multi-agent** — independent agents converge

This pillar is what separates B.L.A. from "a smarter sampler." Without
it, the architecture is a structurally cleaner hallucination engine.
With it, the architecture is a cognitive operating system.

### Action Layer

Real-world or virtual actions flow through:

- **Sandboxes** — code, browser, API, robotics — with rollback semantics
- **Causal outcome prediction** — the world model previews each action
- **Veto loop** — the verification layer can reject before execution
- **Audit log** — every action with cause, expected outcome, and result

The action layer is the place where the system commits *to the world*
rather than to a token stream.

## What we are NOT claiming

A research vision is more credible when its boundaries are clear.

- **Not** "the procedural core has a 100B–150B parameter ceiling at which
  Turing-complete logic becomes lossless." Neural networks do not become
  exact logic engines at any parameter count. The reasoner is compact
  because procedural workloads route their compute into recurrence and
  verification, not into weights.
- **Not** "10B reasoner + 10TB memory equals a 1T transformer." That
  comparison requires an empirical study we have not run. The defensible
  claim is that the architecture has *better scaling on the axes that
  matter for verified reasoning*. The crossover point is unknown.
- **Not** "infinite context." Fixed-size recurrent state has finite
  representational capacity. The capacity is engineered explicitly. Past
  it, information is lost — same problem as fixed windows, just visible
  in a different place.
- **Not** "cannot hallucinate." Retrieval-augmented systems still
  hallucinate when retrieval fails or is adversarially poisoned. The
  claim is that the verification layer makes hallucinations *visible* in
  the commitment object's uncertainty field — not that they are impossible.
- **Not** "catastrophic forgetting is solved." External memory updates
  do not require touching the procedural core's weights. But the
  procedural core itself still has training dynamics; updating it on new
  tasks may interfere with old ones. The architecture mitigates this; it
  does not eliminate it.
- **Not** "absolute-zero-entropy stopping." Entropy goes to zero only in
  closed formal systems. For open-world tasks the stop is a
  *task-specific certification*: tests, proofs, simulator agreement,
  source consistency, calibrated uncertainty bound, consensus across
  agents, or counterexample-search exhaustion under a defined budget.

## How we know if it works

The thesis is testable. A research program is honest when its claims can
be falsified.

1. **Allocation efficiency.** On a held-out task suite, does the trained
   B.L.A. system produce more verified decisions per joule than (a) a
   frontier LLM, (b) the same LLM + retrieval, (c) the same LLM + tools?
   If not, asymmetric scaling is decorative.
2. **Hallucination visibility.** When B.L.A. is wrong, does the
   commitment object's uncertainty field correlate with the error? If
   the uncertainty estimate is uncalibrated, the verification layer is
   theater.
3. **Compute economy.** Does the entropy router actually save FLOPs on
   easy tasks while spending them on hard ones, while preserving
   accuracy? If the router collapses to "always run the deepest path,"
   it is a uniform-compute system in disguise.
4. **Memory scaling.** Does retrieval precision degrade gracefully as
   the four-layer memory grows from 10⁶ to 10⁹ entries, or does
   contention break the system? Stale facts and contradictions are
   adversarial conditions; we test under them.
5. **Knowledge update integrity.** When a fact in symbolic memory is
   updated, do downstream answers reflect the change without retraining
   the procedural core? When the change is deliberately wrong (poisoned
   update), does the verification layer detect the inconsistency?

Pass all five and the thesis is empirically supported. Fail any of
them and the relevant axis needs rethinking — not the whole vision.

## Where we are today

The current scaffold is the **substrate** for this research program, not
the program's output.

- All six subsystems exist as composable Python modules with passing
  unit tests.
- Distributed training infrastructure (DDP + AMP + checkpointing) is
  validated on 6×B200.
- The simplest end-to-end pipeline (encoder → linear policy →
  navigate-to-target) reaches 100% success in 1000 training steps —
  evidence the substrate is wired correctly.
- The cognitive components (world-model rollouts, RAM read/write,
  bus alignment, veto loop, prefetch) exist but have not yet been
  exercised together on a task that *requires* them.

## Where we go next — six concrete deliverables

In priority order. None requires more than weeks of work; together they
turn the substrate into a falsifiable research vehicle.

1. **A task BC alone cannot solve.** Multi-target navigate, partial
   observability, or delayed-effect actions. Without this gate, every
   downstream B.L.A. claim is unfalsifiable.
2. **Real state-space scratchpad** via mamba-ssm. Validates the O(1)
   per-step claim on something concrete instead of a Python loop.
3. **Verification layer skeleton.** `CommitmentObject` schema, three
   concrete certifiers (`TestRunner`, `ProofChecker`, `SimulatorAgreement`),
   wired to the router as the `prove` action.
4. **Hybrid memory v0.** Symbolic store (Wikidata subset) + executable
   store (Python sandbox + Z3 + simulator) on top of the existing
   vector store. Provenance and freshness fields are mandatory.
5. **End-to-end demo with a commitment object.** One closed-loop
   example: prompt → router decides `simulate` → world model rolls
   forward → certifier checks → commitment object out. Even at toy
   scale, this is the first time the architecture runs as designed.
6. **Calibration audit.** Plot uncertainty estimates against actual
   error rates on held-out tasks. If uncalibrated, the verification
   layer needs work before any further scaling.

Past these six, scaling becomes the question — text tokenizer wiring,
real corpora, larger models, better certifiers. Until them, scaling is
premature.

## The bet, in one sentence

> Smarter system = better allocation across perception, reasoning,
> memory, action, verification, and time. Every joule routed to the
> right cognitive substrate.
