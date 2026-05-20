# BLA × Visual Foundation Models — Integration Spec

**Date:** 2026-05-20.
**Status:** v0 architecture spec. Locks the integration order
before any work on Cosmos / V-JEPA 2 / SANA-WM.
**Parents:**
- `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` (BLA architecture)
- `docs/BLA_SCALING_ROADMAP.md` (scaling priority)
- `docs/BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md` (real-world bridge)

## Core thesis

> **Sit on top of a visual world model. Do not replace BLA with one.**

Visual foundation models (Cosmos, V-JEPA 2, SANA-WM) are good at
pixels: rich features, scene synthesis, video imagination. They
are NOT good at the questions BLA answers:

```
Which object is this?
What recipe should I deploy?
Should I retrieve a demo or search?
What happened to object file #3?
```

The right integration architecture keeps BLA's object-file +
recipe-router + planner core intact, and uses visual foundation
models as **a richer perception/data layer** beneath it.

## §1 The 4-layer integration architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4 — Planner / policy                                      │
│   demo retrieval (E2_FAST / E2_STABLE) or light CEM             │
│   execute physical behavior; log outcome; update bank           │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3 — Recipe Router                                         │
│   object files + task descriptor + demo-bank stats → recipe     │
│   FSM-prior → A/B/C/D; demo-prior → E2; OOD → C/D               │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2 — OF-JEPA object files                                  │
│   visual features → persistent object files                     │
│   identity-as-address, rolling-window K=5 runtime               │
│   identity-conditioned tracking + slot_to_pos_aux               │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1 — Visual foundation (NEW; pluggable)                    │
│   Cosmos / V-JEPA 2 / SANA-WM features OR generated videos      │
│   Role: richer pixel features + synthetic variations            │
└─────────────────────────────────────────────────────────────────┘
```

The integration is **bottom-up**: Layer 1 feeds richer features
or augmented data to Layer 2 without changing Layer 2's API. All
of Layer 3 and Layer 4 stay frozen.

## §2 What each foundation model is candidate-for

```
Cosmos (NVIDIA)
  ├── role: synthetic-data / visual-augmentation layer
  ├── strengths: scene synthesis, camera-motion variation,
  │              physics-plausible video generation
  └── fits BLA at: data augmentation for Layer 2 training +
                    Layer-2 robustness stress testing

V-JEPA 2 (Meta)
  ├── role: visual feature backbone (BLA Layer 1 encoder swap)
  ├── strengths: self-supervised video features, identity-stable
  │              over short clips
  └── fits BLA at: drop-in feature replacement under OF-JEPA;
                    closest in spirit to BLA's encoder

SANA-WM (NVIDIA)
  ├── role: long-video augmentation, minute-scale variation
  ├── strengths: long-horizon scene generation, camera motion
  └── fits BLA at: later phase — long-horizon stress + multi-
                    minute occlusion scenarios
```

## §3 Phase V1 — feature replacement / augmentation test

**Don't start with full video generation.** Start with whether a
visual foundation improves Layer 2 without breaking it.

```
Updated 2026-05-20 after V0 feasibility memo. Compare 4 encoders,
holding Layers 2/3/4 fixed:
  A. current Phase-14 OF-JEPA visual encoder (baseline)
  B. V-JEPA 2 ViT-L/16 fpc64-256          (primary candidate)
  C. Cosmos-Tokenizer CV4x8x8, encoder only  (secondary)
  D. V-JEPA 2 ViT-L fpc16 SSv2 finetune    (native 16-frame variant
                                            for natural K=5 fit)

SANA-WM removed from V1 (V0: wrong fit — camera-only conditioning,
no robotics demos, CC-BY-NC-SA weight license).
```

**Metrics (must all be reportable per encoder):**

```
M1. rolling-window K=5 object tracking error (Phase D1b protocol)
    target: ≤ 1.5× current OF-JEPA's 1.5cm error

M2. identity-conditioned position error (id_h_mse, identity_probe)
    target: ≤ current OF-JEPA's id_h on MOVi-style stress

M3. demo retrieval success rate (DR1/DR3 PickPlaceCan protocol)
    target: ≥ current E2_FAST 0.369 / 37%

M4. recipe_router input stability (same task description → same recipe)
    target: 100%; this is logic, not perception, but the router
    output should not change as Layer 1 swaps

M5. BLA-Forge-style visual perturbation robustness
    measured under Phase V2's perturbation suite (see §4 below)
```

**Pre-committed V1 gate:**

> Foundation-model features improve M1 or M5 by ≥ 20% relative
> WITHOUT degrading M2 (identity stability) by more than 10%
> relative, AND without changing M3 (demo retrieval) by more
> than 5pp absolute.

That gate frames the test as **"does the foundation model help
BLA without breaking what makes BLA work?"** — not "is the
foundation model better at pixels."

## §4 Phase V2 — synthetic BLA-Forge edge cases

After V1 establishes a working encoder layer, generate stress
data to test simulator → real bridge robustness:

```
Synthetic perturbation suite (Cosmos-generated or augmented):
  - lighting changes (4 brightness × 4 color temperature)
  - camera angle changes (±15° pitch/yaw/roll)
  - object color/texture changes (random + adversarial)
  - occlusions (3 occluder shapes × 5 trajectories)
  - table/background changes (10 textures)
  - minor distractor objects (1-3 added irrelevant objects)
  - failed grasps/pushes (existing demos rendered with these
    perturbations applied)
```

**For each perturbation set, ask 3 specific questions:**

```
Q1. Does OF-JEPA rolling-window K=5 tracking survive?
    Metric: per-frame cubeA decode err stays within 2× of
    nominal-condition error.

Q2. Does demo retrieval still pick the right demo?
    Metric: top-1 retrieval match rate ≥ 80% on the perturbation
    set (compared to 100% on nominal state-matched protocols).

Q3. Does the recipe_router choose the same recipe?
    Metric: 100% — router is deterministic logic over the task
    descriptor; perturbations shouldn't change its output unless
    they change the task descriptor (which they shouldn't for
    appearance-only perturbations).
```

The intent is to **front-load BLA-Forge's sim-to-real risks** in
the cheaper synthetic regime before hardware exists.

## §4.5 Locked admission tests (after V0/V1-G0/V1a-G0/V1a-M2)

Any future foundation encoder considered for BLA's Layer 1 must
pass BOTH of the following before any larger compute commitment:

```
G0a — positional / window stability:
  Encode the same content at shifted temporal windows; cosine
  per token ≥ 0.95 on overlapping positions.
  Catches: RoPE-style position-bound representations.
  Failed by: V-JEPA 2 ViT-L (Phase V1-G0).

G0b — content-change / object-identity stability:
  Encode a rollout with REAL object motion (≥ 0.5 m displacement
  over the rollout). Find high-temporal-variance latent cells.
  Their consecutive-frame cosine mean should be ≥ 0.70 (smooth
  evolution as objects move through the cell), with no severe
  inversions (cosine min ≥ 0).
  Catches: reconstruction-bottleneck features that encode current
  local pixel content rather than persistent object identity.
  Failed by: Cosmos-Tokenizer CV4x8x8 (Phase V1a-M2).

PASS REQUIRES BOTH. G0a alone is insufficient — it catches one
mode of instability but misses the deeper object-centric question.
```

Locked rule, 2026-05-20: **No Layer-1 encoder swap proceeds to
the full M1-M5 pipeline without first passing G0a AND G0b.**

## §5 What NOT to do (locked anti-patterns)

```
1. Don't make the visual world model the planner.
   The doctrine claim (Phase D3/D4/Scale-1) is that planning =
   recipe-router + demo retrieval, NOT generative imagination.
   Sana-WM-style minute-scale dreaming is irrelevant to that claim.

2. Don't replace object files with video tokens.
   Phase 8C's "identity is an address" finding (switch_rate
   0.689 → 0.002) is the architectural foundation. Video tokens
   don't have addressable identity.

3. Don't evaluate success by "pretty video."
   The integration gates (§3 M1-M5) are object-file fidelity
   and recipe-router accuracy. Visual quality is a means, not
   an end.

4. Don't let a generative model hallucinate objects that the
   planner treats as real without identity-conditioned verification.
   Phase 8C identity-conditioned metrics (id_h_mse) are the
   safety check. Any feature-backbone swap must preserve
   identity-conditioned performance.

5. Don't scale model size (Layer 2) just because Layer 1 changed.
   Per scaling roadmap §6, capacity scaling is LAST. Layer-1
   swap is a Layer-1 experiment.

6. Don't deprecate Phase D1b's runtime ladder.
   v1 rolling-window K=5 is the deployment runtime. Layer 1
   swap doesn't change that. v2 stateful encode_step remains
   deferred.

7. (Locked 2026-05-20 after V1a-M2)
   Don't try to fix a non-object-centric encoder by stacking
   SlotAttention on top.
   Object-centricity must be IN the encoder's training objective.
   If the foundation was trained for reconstruction or generic
   SSL, its per-cell or per-token features encode current local
   appearance, not persistent object identity. A downstream slot
   layer cannot retrofit identity onto chaotic per-cell features
   (V1a-M2: Cosmos active-cell cosines flip from 0.99 to -0.76
   when one object enters and another exits the same cell).
```

## §5.5 Locked final role map (2026-05-20)

After 5 negative findings on 3 candidates (V-JEPA 2, Cosmos, SANA-WM),
the visual-foundation track has produced a sharp role assignment:

```
V-JEPA 2
  per-token encoder swap: ❌ failed (V1-G0)
  retrieval key in sim:   ❌ failed (V1b)
  remaining role:         clip-summary real-world / context only

Cosmos-Tokenizer
  G0a positional stability:  ✅ passed
  G0b content stability:     ❌ failed (V1a-M2)
  per-token encoder swap:    ❌ failed
  remaining role:            data augmentation / perturbation gen

SANA-WM
  not viable per V0 (camera-only conditioning, CC-BY-NC-SA)
  remaining role:            none

OF-JEPA (Phases 7–9)
  CANONICAL. Its training objective is object-centric and
  identity-aligned. Foundation models do not replace it.
```

## §5.6 The biggest architectural conclusion (locked)

> **Object-centricity has to be trained into the encoder.**
> **It is not recovered by placing SlotAttention on top of**
> **reconstruction or generic SSL features.**

This protects the BLA direction from "just use a bigger foundation
model." Bigger features do not automatically provide the right
state. Identity-as-address (Phase 8C's switch_rate 0.689 → 0.002)
is a property of the **training objective**, not the **architecture
size**.

## §6 The eventual hybrid (where this leads)

```
Visual foundation (Cosmos / V-JEPA 2 / SANA-WM)
  generates / extracts realistic scene features
                              ↓
OF-JEPA
  extracts stable object files from those features
                              ↓
Recipe Router
  chooses correct deployment mode per task descriptor
                              ↓
Demo Retrieval (E2_FAST / E2_STABLE)
  executes contact-sensitive task
                              ↓
BLA-Forge
  tests and logs real outcomes
```

The combination delivers:
- large-scale visual diversity (Layer 1)
- explicit object memory (Layer 2)
- predictive deployment recipes (Layer 3)
- contact-sensitive policy execution (Layer 4)
- real-world physical feedback (BLA-Forge)

That's stronger than any single layer alone.

## §7 Phase sequence (integration roadmap)

```
V0 — feasibility (≤ 1 day)
     Check: do Cosmos / V-JEPA 2 / SANA-WM expose feature
     interfaces or only end-to-end APIs?
     If only end-to-end, integration is limited to data
     augmentation (V2-only path).

V1 — encoder-layer swap test (this spec §3)
     A/B/C/(D) encoder comparison on existing simulator tasks.
     Gate: foundation helps without breaking identity.

V2 — synthetic perturbation suite (this spec §4)
     Cosmos-generated edge cases for BLA-Forge readiness.
     Gate: retrieval + tracking + router survive perturbations.

V3 — pre-BLA-Forge synthetic deployment validation
     Use V1's winning encoder + V2's perturbation suite to
     simulate a BLA-Forge "real-world" condition entirely in
     synthesis. Pre-commit BLA-Forge gates G1-G4
     (`BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md`) against this
     synthetic pre-run.

V4 — physical BLA-Forge integration
     Real hardware deployment with foundation-model encoder
     in the perception pipeline. Compare against the V3
     synthetic predictions.
```

**Critical ordering:**

- V1 / V2 / V3 can all run in parallel with BLA-Forge BF-0
  (hardware bring-up). The synthetic work doesn't block the
  physical work, and vice versa.
- V4 only begins when BLA-Forge BF-2 (pick/place) clears.

## §8 Open questions to resolve during V0

```
Q1. Are Cosmos / V-JEPA 2 / SANA-WM feature interfaces public?
    If only inference APIs are public, V1's encoder swap may
    be infeasible without distillation.

Q2. What's the feature dimensionality and frame rate of each?
    OF-JEPA expects 128×128 RGB at robosuite frame rate. If
    foundation models require different input sizes, an adapter
    is needed.

Q3. Does V-JEPA 2 produce features stable enough across short
    clips to feed OF-JEPA's identity binding?
    V-JEPA 2's training objective is short-clip predictive;
    it's the closest match in spirit to OF-JEPA.

Q4. Can Cosmos generate physically-plausible robosuite-like
    rollouts?
    If yes, V2's perturbation suite gets cheaper.
    If no, V2 needs hand-curated perturbations or sim-side
    rendering with domain randomization instead.

Q5. Does SANA-WM's minute-scale generation help anything BLA
    cares about, given Phase D3/D4's "search-budget-zero around
    demos" doctrine?
    Honest answer: probably not at v0. SANA-WM is generative
    imagination, which the doctrine explicitly de-prioritizes.
    Keep SANA-WM as V3+ if at all.
```

## §9 What this changes about prior commitments

| Prior doctrine | Status after V0–V4 |
|---|---|
| Recipe map (4-task validated) | unchanged |
| Recipe E (E2_FAST / E2_STABLE) | unchanged |
| Rolling-window K=5 runtime | unchanged |
| recipe_router 85.7% accuracy | unchanged (router is logic) |
| "Search budget = 0 around demos" | unchanged |
| BLA-Forge phase sequence | unchanged (V1-V4 run alongside) |

The integration adds a **new perception layer** beneath BLA.
Nothing about BLA's deployed decision system changes.

## §10 Locked

V1 is the next concrete technical move. V0 feasibility check
should precede any compute commitment. Until V0 returns, no
Layer-1 work proceeds.

## Files

- This spec: `docs/BLA_VISUAL_FOUNDATION_INTEGRATION.md`
- Architecture: `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md`
- Scaling: `docs/BLA_SCALING_ROADMAP.md`
- Real-world testbed: `docs/BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md`
- DR-arc decisions: `docs/phases/PHASE_DR{1,2,3}_DECISION.md`
