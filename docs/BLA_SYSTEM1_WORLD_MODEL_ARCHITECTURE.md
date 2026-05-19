# BLA System-1: Object-File World Model — Architecture Spec

**Status:** v0 architecture spec, locked 2026-05-19 (updated after
D4 close).
**Cross-references:** `docs/00_overview.md`, `docs/01_jepa.md`,
`docs/BLA_SYSTEM1_JEPA_ARC_SUMMARY.md`, all `docs/phases/PHASE_*.md`.

## 1. Executive Summary

BLA System-1 is an **object-file world model and planner substrate**
built on Object-File JEPA (OF-JEPA v0) plus a small action-conditioned
predictor and a goal-progress value head.

**Core thesis:**

> OF-JEPA v0 is a *temporally batched* object-file world model and
> planner substrate. It is **not yet** a live streaming object-file
> memory, but it has a clear ladder to become one: batched encode →
> rolling window → stateful encode_step.

**Doctrine claim (locked after Phase Scale-1, 2026-05-19):**

> **Across Lift, PickPlaceCan, NutAssemblySquare, AND ToolHang,
> `demo_no_cem` is the highest-mean and lowest-variance mode. This
> validates Recipe E as the default for contact-sensitive expert-
> demonstration regimes — across grasp-and-lift, grasp-and-place,
> grasp-and-insert, and long-horizon multi-stage tasks. The
> transferable object is the demonstration manifold, not CEM
> exploration around it.**

**Router validation:** at `bla/routing/recipe_router.py` commit
`668e02c`, the deployed router correctly predicts the winning
recipe on 6 of 7 tasks (85.7%) — clearing the ≥80% strong-pass
threshold from the Scale-1 precommit.

**Scale-1 locked statement:**

> Scale-1 validates the BLA recipe router as a deployment decision
> system. Across seven tasks, the router selects the empirically
> winning recipe with 85.7% accuracy, clearing the precommitted
> ≥80% threshold. Recipe E, `demo_no_cem`, is now the default for
> contact-sensitive expert-demo regimes, validated across Lift,
> PickPlaceCan, NutAssemblySquare, and ToolHang.

The architecture has been validated on robosuite Stack (push) across
Phases 14–18 (FSM-prior regime) and **cross-task transferred to three
contact-sensitive tasks** (demo-prior regime) — Lift (Phase 18κ R3),
PickPlaceCan (Phase D3-main), and NutAssemblySquare (Phase D4). The
legibility of the underlying object files was established in Phase D1
(2026-05-19) under the temporal-batching contract.

What this doc covers:

- §2 Canonical System-1 (OF-JEPA v0)
- §3 Runtime contract (batched / rolling / stateful)
- §4 Planner stack and locked recipes A–E
- §5 Deployment regime map
- §6 Durable architectural lessons
- §7 Known failure modes and caveats
- §8 Evidence table by phase
- §9 Demo artifacts (Demo A = legibility figure-of-merit)
- §10 Next roadmap

What this doc does **not** cover:

- BLA System-2 (planner-internal reasoning, dialog) — separate spec
- BLA bus, DCA, tensor-RAM — separate specs in `docs/03_*.md` etc.
- Long-horizon dreamer-style imagination — future work, see §10

## 2. Canonical System-1: OF-JEPA v0

### 2.1 What OF-JEPA v0 is

**Object-File JEPA** is an object-centric self-supervised video
representation learner. Each scene is encoded as a small set of
**slots** — typically 6 in our robosuite configuration — each of
which behaves as a persistent **object file**: a stable address for
"one entity in the world."

Two coupled systems (Phase 7B/C/D, 8A, 8C):

1. **Predictive dynamics** — JEPA-style latent prediction at fixed
   temporal stride. Trained to predict future state from past state
   without reconstructing pixels.
2. **Identity binding** — persistent learned `slot_proto` prototypes
   + Sinkhorn matching across the temporal axis. Identity is the
   binding mechanism, not a content feature of the slot.

### 2.2 Slot composition

Each slot is split into two halves:

```
slot = [ id_dim | state_dim ]
```

- `id_dim` is the persistent address — stable across time within a
  single encode pass. The predictor does **not** write to this half.
- `state_dim` is the dynamics-carrying half. The action-conditioned
  predictor writes here.

`slot_to_pos_aux` is a small auxiliary head that decodes each slot's
2D position from the full slot vector (`id_dim + state_dim`).
Trained jointly with the encoder against ground-truth entity
positions via Hungarian matching on the matched subset.

### 2.3 What "object file" means here

A slot is an object file in the Pylyshyn sense: a persistent
**index** that picks out the same entity over time, independent of
its features. Phase 8C established that **identity is an address**,
not a content feature. The 328× reduction in switch_rate from v0
versus the baseline (Phase 8C decision doc) is the empirical
foundation for this claim.

### 2.4 Identity-conditioned evaluation is primary

For object-file architectures, **anonymous metrics are gameable**
(Phase 9 v0/v1 lesson). Always evaluate identity-conditioned:
- `id_h_mse` = MSE conditioned on the same persistent identity
- `id_v_mse` = visibility-conditioned variant

Anonymous Hungarian re-matching at every step makes a model that
shuffles slots look fine. Identity-conditioned evaluation does not.

## 3. Runtime Contract: Batched vs Rolling vs Stateful

This is the most important practical section for anyone building
on top of System-1.

### 3.1 The three runtime modes

```text
v0 offline:
  full batched encode_video(T)
  validated, but cubeA decode err drifts from ~2.7 cm to ~8.4 cm
  over a 25-step rollout.

v1 near-live:                              ← current deployment runtime
  rolling-window encode_video(K), K=5 preferred initial setting
  cubeA decode err: 1.5 cm flat at K=5 (3× BETTER than batched).
  No monotonic drift — old frames are forgotten.

v2 live:
  stateful encode_step API, future work
  state = of_jepa.init_state()
  for frame in stream:
    object_files, state = of_jepa.encode_step(frame, state)
  Lower per-step compute than v1; less urgent now that v1 is
  already a runtime upgrade over v0.
```

**Locked deployment statement (D1b, 2026-05-19):**

> **Rolling temporal windows are the current deployment runtime
> for OF-JEPA v0. K=5 is the recommended initial setting.**

### 3.2 What the planner uses

The Phase 15–18 planner stack uses a frame-by-frame encode pattern
(`encode_frame(model, image)`, T=1) and **still works correctly**.
The planner does not depend on absolute per-frame decoded positions:
it does **relative trajectory scoring** — comparing predicted
trajectories under candidate action sequences. Relative differences
are stable even when absolute decode is not.

This is why Phase 14–18 results are not invalidated by the legibility
finding in Phase D1.

### 3.3 What single-frame encoding does NOT support

Three failure modes (Phase D1):

1. **Decoded positions are nearly constant across frames** under
   per-frame `encode_video(T=1)` calls (0.002 cm drift observed
   across 30 frames where real env motion was 11 cm).
2. **Slot identity is non-deterministic across calls** — the same
   reset env produced slot index 0, 4, 5 for cubeA across three runs.
3. **Long-horizon autoregressive pure-latent rollout drifts**
   (~5–7 cm cubeA motion predicted under zero actions over 25 steps).

### 3.4 Choose the runtime by the use case

| Use case | Correct mode |
|---|---|
| Planning (CEM scoring, value-head ranking) | Frame-by-frame (validated) |
| Per-frame legibility / visualization | **Batched encode** (Demo A) |
| Approximate live operation | Rolling window (D1b, next) |
| True streaming object-file memory | Stateful `encode_step` (D3, future) |

## 4. Planner Stack and Locked Recipes A–E

### 4.1 The planning stack (Phase 17 / 18)

```text
encoder  : OF-JEPA v0 (slot states from frame)
predictor: ActionConditionedOFJEPA.predict_state_delta
           (single-step latent dynamics, action-conditioned)
prior    : scripted FSM (Stack push) OR robomimic demo replay (Lift)
search   : light CEM (K=32, 1 iter, σ=0.12, σ_floor=0.05)
           with per-dim sigma masking on structured channels
score    : combined_sum = λ·predictor_score + (1−λ)·value_head_score
           where the value head reads either engineered geometry,
           supervised adapter output, or learned end2end latent
```

### 4.2 Recipes (current locked set)

| Recipe | Substrate | Value-head input | Prior | Search | When to use |
|---|---|---|---|---|---|
| **A** | OF-JEPA v0 | engineered 10-D geo (sim-true features) | scripted FSM | light CEM | sim-only, sim-true features available |
| **B** | OF-JEPA v0 | supervised adapter(slot, goal) → 10-D geo | scripted FSM | light CEM | BLA-native; cross-task transfer |
| **C** | OF-JEPA v0 | end2end adapter+VH (joint) | scripted FSM | light CEM | OOD-favored; higher variance |
| **D** | OF-JEPA v0 | pretrain+ft schedule (geo-aux then end2end) | scripted FSM | light CEM | best in-dist of schedule variants |
| **E** | OF-JEPA v0 | n/a (no value head used) | **expert demo replay** | **no CEM** | expert demo prior; contact-sensitive |

A/B/C/D are validated for **FSM-prior** regimes (Stack push). E is
validated for **demo-prior** regimes across **four independent
contact-sensitive task families** (Lift / PickPlaceCan /
NutAssemblySquare / ToolHang) as of Phase Scale-1 close, 2026-05-19.
The doctrine is no longer hedged — Recipe E is the deployment
default for the demo-prior regime.

**Recipe E's selection mechanism scales via NN retrieval** (Phase
DR1, 2026-05-19): replace "replay one fixed demo" with "retrieve the
closest useful demo from a bank". On PickPlaceCan, top-1 retrieval
over a 24-demo bank hits 35% success vs 1% for fixed-5-demo
cycling — a 35× scaling improvement. CEM around the retrieved demo
still hurts (10% success), confirming search-budget-zero extends to
retrieved demos.

Recipe E has two engineering variants:
- **E1 (Lift)**: demo replay on random fresh env reset (works when
  the task's initial-state distribution is narrow enough that some
  demos succeed on fresh resets — Lift has 2-4 of 50 working).
- **E2 (PickPlaceCan)**: demo replay with env-state-matched init
  (`env.sim.set_state_from_flattened(demo.states[0])`) — needed
  when the init distribution is wider than demo-bank coverage
  (PickPlaceCan has 0 of 50 succeeding on fresh reset, 5 of 20 on
  matched init).

### 4.3 Why Recipe E exists

Phase 18κ R3 (3-seed + rerun, n=4 runs total) established that when
the prior is an expert demonstration manifold, action-space Gaussian
CEM degrades performance regardless of value head:

```
Mode                       imp_mean (n=4)
demo_no_cem                 0.308   ⭐ rank 1 on 3/4 runs, lowest var
phase17_locked              0.208
combined_sum_supervised     0.192
combined_sum_geo            0.183
combined_sum_end2end        0.167
naive_cem                   0.000   (floor)
```

The mechanism: expert demos already encode contact-sensitive
behavior on a narrow manifold; Gaussian CEM perturbations push
candidates off-manifold; no learned scorer can reliably re-rank
them back. See `feedback_search_budget_zero_around_expert_demos.md`.

## 5. Deployment Regime Map

```text
                          ┌──────────────────────────┐
                          │   Prior characterization │
                          └─────────────┬────────────┘
                                        │
                ┌───────────────────────┴────────────────────────┐
                │                                                 │
        ┌───────▼────────┐                              ┌─────────▼────────┐
        │ Scripted FSM   │                              │   Expert demo    │
        │ prior          │                              │   prior          │
        │ (noisy enough  │                              │   (narrow,       │
        │  for CEM to    │                              │   contact-       │
        │  refine)       │                              │   sensitive)     │
        └───────┬────────┘                              └─────────┬────────┘
                │                                                 │
       ┌────────▼─────────┐                              ┌────────▼─────────┐
       │ Recipes A / B    │                              │   Recipe E       │
       │   (+ C/D for     │                              │  demo_no_cem     │
       │    OOD variants) │                              │  no perturb      │
       └──────────────────┘                              └──────────────────┘
```

**Locked applicability rule (strengthened by D4 close, 2026-05-19):**

> **In contact-sensitive demo-prior regimes, the demonstration manifold
> is the policy. Do not add action-space CEM by default.**

Recipe E formal statement:

```
Recipe E — demo_no_cem

Use for:
  grasp-and-lift              (validated: Lift)
  grasp-and-place             (validated: PickPlaceCan)
  grasp-and-insert            (validated: NutAssemblySquare)
  contact-sensitive expert-demo regimes (generally)

Avoid:
  CEM exploration around expert demos unless a trust-region
  audit proves it helps in your specific regime.
```

The empirical basis: across three independent task families,
`demo_no_cem` is **both the highest-mean AND the lowest-variance**
nontrivial mode. The variance σ_imp actually *decreases* with each
new task (0.054 on Lift → 0.043 on PickPlaceCan → 0.022 on Square)
as the recipe's regime is mapped more precisely.

**Cross-task evidence (out-of-sample, 2026-05-19):**

| Task | Constraint structure | Δ(demo_no_cem − phase17_locked) imp | Δ success | σ_imp(demo) | Reference |
|---|---|---:|---:|---:|---|
| Lift (Phase 18κ R3) | grasp-and-lift | +0.10 (4-run agg) | +10pp | 0.054 | `PHASE_18K_REGIME3_LIFT_DECISION.md` |
| PickPlaceCan (Phase D3-main) | grasp-and-place | **+0.564** | **+56.7pp** | 0.043 | `PHASE_D3_MAIN_DECISION.md` |
| NutAssemblySquare (Phase D4) | grasp-and-insert | +0.400 | +38.6pp | 0.022 | `PHASE_D4_DECISION.md` |
| **ToolHang (Phase Scale-1)** | **long-horizon grasp + hang** | **+0.533** | **+53.3pp** | **0.051** | `PHASE_SCALE1_DECISION.md` |

Four independent contact-sensitive task families. Zero falsification
triggers across precommit predictions. demo_no_cem is BOTH the highest-
performing AND lowest-variance mode on every task.

**Sibling caveat:** same CLI seed is not necessarily the same run in
robosuite/MuJoCo/demo pipelines unless all RNG sources are audited
and pinned (Phase 18κ R3 addendum finding).

## 6. Durable Architectural Lessons

These are the lessons that survived multiple phases and that future
experiments should respect.

### 6.1 Object-file substrate lessons (Phases 7–9)

- **Prediction vs assignment**: object-centric world models need
  TWO coupled systems — predictive dynamics (JEPA) and identity
  binding (assignment). One monolithic slot cannot do both.
  (Phases 7B/C/D, 8A, 8C confirm from four directions.)
- **Identity is an address, not a feature**: object identity should
  be a persistent memory address used to bind observations, NOT a
  content feature decoded from a slot. (Phase 8C OF-JEPA v0:
  switch_rate 0.689 → 0.002, 328× reduction.)
- **Identity-conditioned metrics are primary**: anonymous Hungarian
  rematching is gameable; report identity-conditioned. (Phase 9
  v0/v1 lesson.)
- **Slot persistence requires LayerNorm**: slot persistence +
  additive predictor across long horizons blows up without inter-
  frame LayerNorm. (Phase 7 v1 lesson.)
- **Joint metric vs single axis**: switch_rate alone is gameable by
  constant-slot baselines; always report joint (mse, switch).

### 6.2 Planner stack lessons (Phases 14–18)

- **Value head complementary to dynamics**: a small episode-level
  goal-progress value head is COMPLEMENTARY to a one-step latent
  predictor; combined_sum (50/50 mix) wins where value_only barely
  does and combined_max fails (scale-brittle without z-norm).
  (Phase 18η-multi: +0.061 absolute, +24% relative.)
- **Rank vs candidate quality are orthogonal**: predictor rank
  quality and candidate quality are independent axes. "Trust
  region" is where candidate quality is preserved, not where the
  predictor thinks it knows best. (Phase 18γ.)
- **Less search when score is anti-correlated**: if the learned
  ranker is anti-correlated with end-effect on the current
  distribution, MORE search makes things worse. Light CEM beats
  heavy CEM at 8.3% of the compute. (Phase 18β.)
- **Engineered aux loss is a useful inductive bias in-distribution,
  destructive OOD**: supervising the adapter on engineered geometry
  beats end2end at 720 samples in-dist; the same recipe inverts
  under distribution shift (end2end > geo > sup OOD). Geo-MSE aux
  is a higher-bias / lower-variance estimator. (Phase 18κ R2.)
- **Value-relevant subspace IS recoverable from slots**: Phase 18λ
  overturns the strong reading of 18θ — a supervised adapter
  recovers goal-relative subspace cleanly (goal_xy 0.89, push_dir
  0.55, cube_xy 0.61 Spearman). The earlier "frozen slots aren't
  enough for value" lesson stands only against bigger MLPs on raw
  slots, not against a structured adapter.

### 6.3 Lessons about *the search itself* (Phase 18κ R3)

- **CEM must preserve semantic action channels**: Gaussian noise on
  every action dim corrupts structured channels (gripper open/close
  bit, mode flags). Use per-dim sigma masking. Sigma=0 on the
  gripper bit unblocked demo-replay-as-prior for Lift.
- **Search budget around expert demos = 0**: the transferable object
  across tasks is the demonstration manifold, not CEM exploration
  around it. CEM may occasionally match a demo but is not reliably
  additive. (Phase 18κ R3, n=4 runs, locked applicability rule.)

### 6.4 Runtime / legibility lesson (Phase D1, 2026-05-19)

- **OF-JEPA v0 is temporally batched, not yet a live stateful
  object-file model**: identity binding lives WITHIN a single
  multi-frame `encode_video(T)` call. Independent single-frame
  calls do not preserve temporal binding and can produce unstable
  slot indices or near-constant decoded positions. For legibility
  demos, use full-rollout batched encode; for live operation, the
  architecture needs rolling-window or a stateful `encode_step` API.

## 7. Known Failure Modes and Caveats

### 7.1 What does NOT work in the current System-1

| Failure mode | Cause | Mitigation |
|---|---|---|
| Single-frame `encode_video(T=1)` for per-frame legibility | Identity binding is across-temporal-axis only | Use batched encode or rolling window |
| Long autoregressive pure-latent rollouts (>10 steps) | Phase 17 predictor was trained single-step | Keep imagination short, or train Dreamer-style multi-step loss |
| CEM around expert demos | Off-manifold Gaussian samples | Use Recipe E (no CEM); future: time-warp / low-D residual search |
| Frozen slot features into a generic MLP value head | Slots don't directly expose scalar goal-relative value | Use a supervised geometry adapter as the readout layer (Phase 18λ) |
| Discontinuous perturbations cleanly localizing to one slot | Encoder reroutes content across slots | Stateful encode_step + persistence penalty (future) |

### 7.2 What does NOT change under these failures

- All Phase 14–18 planner gates remain valid (relative scoring).
- The locked recipes A–E continue to work in their declared regimes.
- Identity binding within a single multi-frame encode pass is
  validated end-to-end (Phases 7–9).

### 7.3 Variance caveats

- **Same CLI seed ≠ same run** in robosuite/MuJoCo/demo pipelines
  unless all RNG sources are audited and pinned (Phase 18κ R3
  addendum). Treat seed-level numbers as run-level aggregates.
- **CEM-with-value-head modes have wild run-to-run variance** on
  Lift (range 0.067–0.300 for the same `--seed 2`). `demo_no_cem`
  has the lowest variance (most reliable) and the highest mean.

## 8. Evidence Table by Phase

Selected milestones; full per-phase docs in `docs/phases/PHASE_*.md`.

| Phase | Question | Result |
|---|---|---|
| 7 (MOVi-A) | Does slot_delta beat dense JEPA on a clean object dataset? | YES, joint (mse, switch) shows the gap |
| 7B/C/D | Can the predictor side alone or encoder side alone fix slot persistence? | NO — both are needed (Phase 7 verdict) |
| 8A/v2 | Identity-contrastive λ sweep | λ≈1.0 stable; foundation for 8C |
| 8C | Persistent prototype + Sinkhorn matching | switch_rate 0.689→0.002 (328×) |
| 8D | Hard-MOVi stress | OF-JEPA v0 holds up at stride=4, 8+ entities |
| 9 / 9B | Lifecycle (MOVi-D) with visibility | OF-JEPA v1 visibility-gated improves id_h further |
| BLA integration | Adapt OF-JEPA into BLA System-1 API | done |
| 13 | CLEVRER external benchmark | OF-JEPA generalizes to CLEVRER |
| 14 | Action-conditioned predictor (robosuite) | which-object-changed accuracy + relations work |
| 14.5 | Joint gates A/B/C scripted v3 | passing |
| 14.6 | Action-conditioning generalization | confirmed |
| 15 | CEM-on-predictor planning | Stack push working |
| 16 | Policy-prior MPC (BC-warm-started CEM) | scripted prior > BC prior |
| 17 | Focused-contact predictor fine-tune | locked predictor for downstream |
| 18d | Phase 17 multi-seed confirmation | robust |
| 18β | Heavy CEM vs light CEM | light wins at 8.3% compute |
| 18η-multi | Goal-progress value head + combined_sum | +0.061 vs locked, 3/3 seeds |
| 18λ | Object-file geometry adapter | mean Spearman 0.50 in-dist |
| 18λ-v2 | End2end adapter+VH | competitive; OOD-favored |
| 18κ R2 | OOD goal-distance shift | aux loss is distribution-dependent |
| 18κ R3 | Cross-task to Lift | Recipe E (demo_no_cem) dominates; A/B/C/D do not extend |
| 18ν | Scheduled aux loss | pretrain+ft locked as Recipe D |
| **D1 (2026-05-19)** | **OF-JEPA legibility?** | **PASS under batched encode; runtime ladder defined** |
| **D3 pilot** | **Does the regime map predict on a NEW task?** | **YES — PickPlaceCan pilot Δ=+0.60 (n=5)** |
| **D3-main (2026-05-19)** | **Cross-task doctrine at scale?** | **STRONG PASS — PickPlaceCan 3 seeds × n=30, all 4 gates clear, Δ=+0.564 / +56.7pp success, demo_no_cem also LOWEST variance** |
| **D4 (2026-05-19)** | **Second external task (precise insertion)?** | **STRONG PASS — NutAssemblySquare 3 seeds × n=30, all 4 gates clear, Δ=+0.400 / +38.6pp success, demo_no_cem σ_imp=0.022 (lowest variance yet)** |
| **Scale-1 (2026-05-19)** | **Router accuracy on a 7-task suite?** | **STRONG PASS — 6/7 (85.7%) router matches, clears ≥80% threshold. ToolHang 3 seeds × n=30: Δ=+0.533 / +53.3pp. 4th cross-task demo-prior validation.** |
| **D1b (2026-05-19)** | **Does rolling-window encode work?** | **STRONG PASS — rolling K=5 cubeA decode err 1.5 cm vs batched 4.7 cm (3× BETTER). v1 is a runtime upgrade over v0, not a compromise. v2 stateful encode_step less urgent.** |
| **DR1 (2026-05-19)** | **Does NN demo retrieval scale Recipe E?** | **STRONG PASS — top-1 retrieval over 24-demo bank: 0.346 / 35% on PickPlaceCan (3 seeds × n=30) vs fixed-5-cycle 0.014 / 1% (a 35× scaling improvement). Matches oracle within seed noise. CEM-around-retrieval still hurts (10% success).** |

## 9. Demo Artifacts

### 9.1 Legibility Demo: Batched Object-File Tracking

**Demo A** — `scripts/demo_counterfactual_rollouts.py` (commit
`fde6bae`). Headline figure of OF-JEPA v0's legibility.

```text
Demo A — PASS / headline
Batched encode_video(T) exposes stable object-file trajectories.
Identity-bound slots track distinct objects over time.
```

Working recipe:

```text
collect full rollout
→ encode_video(T)
→ bind slots once across the temporal window (Hungarian at t=0)
→ decode trajectories from identity-bound slots (slot_to_pos_aux)
→ visualize object-file tracks
```

Result: 3 entities (cubeA / cubeB / eef) → 3 Hungarian-matched
slots; mean decode error 4.8 cm / 3.3 cm / 6.6 cm. Decoded
trajectories visibly follow ground truth. Figure at
`/workspace/demos/demo_A_trajectory_tracks.png`.

### 9.2 Diagnostic demos (B and C)

Run from the same script; not promoted to headline artifact.

```text
Demo B — PARTIAL / diagnostic
Teleport perturbation does not cleanly spike the originally bound slot.
Discontinuous content can be rerouted to other slots.

Demo C — PARTIAL / weak signal
Counterfactual +x vs +y rollouts are differentiated, but only weakly at H=5.
Predictor is goal/action-responsive at training horizon, but not visually
dramatic.
```

**Demo B teaches**: the encoder handles discontinuities by
redistributing content across slots rather than preserving the
originally assigned object file. Future work: stateful `encode_step`,
explicit persistence penalty across discontinuities, surprise head
over identity-conditioned file state.

**Demo C teaches**: the action predictor is useful for planner
scoring, but short-horizon latent counterfactuals are not visually
dramatic enough for a public-facing rollout demo yet. Matches the
broader BLA arc — planner effects emerged through CEM/value scoring,
not through long pure imagination.

## 10. Next Roadmap

### 10.1 Immediate next steps (D-track)

- **D1b — Rolling-window legibility demo**: `--mode rolling_window`
  in the demo script. Each step encodes the last K=5–8 frames,
  uses the final slot states. Moves toward live operation without
  touching OF-JEPA internals. Task #160.
- **D2 — this doc**: written. Now used as the source of truth for
  future phases.
- **D3 — Cross-task doctrine validation**: pick PickPlace / Door /
  NutAssembly (or a small selection), state the regime-map
  predictions before running, then run. This converts BLA from
  "experiment stack" to "doctrine that makes predictions."

### 10.2 Architecture work behind D1b

After D1b, the candidate ladder for live operation is:

```text
v2 runtime: stateful encode_step
  state = of_jepa.init_state()
  for frame in stream:
    object_files, state = of_jepa.encode_step(frame, state)
```

This requires changing the OF-JEPA internal contract — `slot_proto`
state must be carried across calls rather than reset each call. Not
worth doing until the rolling-window version proves insufficient for
downstream demos / live operation.

### 10.3 Demo C's open problems → future model work

- **Multi-step latent rollout reliability**: Phase 17 was trained
  single-step. A Dreamer-style multi-step latent rollout loss in a
  future phase (call it Phase 19) would unlock longer counterfactual
  imagination demos.
- **Discontinuity-robust object-file tracking** (Demo B):
  - explicit persistence penalty: cost slot reassignment more
  - surprise head: train an identity-conditioned anomaly head
  - stateful encode_step: live persistent identity binding

### 10.4 Hard reframes that the doc protects against

- Don't write a section called "BLA dreams the future in pixels" —
  the arc has affirmatively NOT taken that path, and Phase 18κ R3
  shows generative imagination is not the bottleneck for
  manipulation. (See answer to "should we incorporate Sana-WM"
  internal exchange: adjacent, not symbiotic at this phase.)
- Don't claim "BLA is a streaming object-file memory" — it isn't
  yet. The runtime ladder is honest about this.
- Don't promote Demo B or C as headline; they remain diagnostic.

## Appendix A — Key file pointers

| Concept | Location |
|---|---|
| OF-JEPA v0 model | `system1_jepa/of_jepa/` (object_file_memory, predictor, metrics) |
| Action-conditioned wrapper | `scripts/slot_jepa_robosuite_train.py` |
| Hungarian probe | `system1_jepa/identity_probe.py` |
| Geometry adapter | `system1_jepa/geometry_adapter.py` (Phase 18λ) |
| Value head | `system1_jepa/value_head.py` (Phase 18η) |
| Locked predictor ckpt | `/workspace/phase17/model_action_finetuned.pt` |
| Locked adapter ckpt | `/workspace/phase18l2_seed0/supervised_adapter.pt` |
| Planning entrypoint (light CEM + scripted prior) | `scripts/phase16_policy_prior_mpc.py` |
| Lift task variant (Recipe E setup) | `scripts/phase18k_r3_lift.py`, `scripts/phase18k_r3_full.py` |
| Legibility demo | `scripts/demo_counterfactual_rollouts.py` |

## Appendix B — Source memory entries

The doctrine memory entries this spec consolidates:

- `bla-locked-planning-recipe`
- `search-budget-zero-around-expert-demos`
- `of-jepa-legibility-requires-temporal-window`
- `cem-preserves-semantic-channels`
- `value-head-complementary-to-dynamics`
- `engineered-aux-loss-useful-inductive-bias`
- `aux-loss-distribution-dependent`
- `less-search-when-score-anti-correlated`
- `rank-vs-candidate-quality-orthogonal`
- `value-relevant-subspace-recoverable-from-slots`
- `frozen-slots-not-enough-for-value`
- `identity-as-address`
- `prediction-vs-assignment`
- `identity-conditioned-metrics`
- `slot-persistence-layernorm`
- `joint-metric-vs-single-axis`
- `proxy-vs-end-effect-gate`
- `data-vs-architecture`
- `per-state-vs-per-episode-metrics`
- `diagnostic-vs-name`

If any of these memory entries are updated, this doc's corresponding
section should be re-checked.
