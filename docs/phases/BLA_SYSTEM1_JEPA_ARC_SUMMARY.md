# BLA System-1 JEPA Arc — Master Summary

**Span:** 2026-05-09 → 2026-05-17.
**Final architecture:** Object-File JEPA v0 (OF-JEPA v0).
**Decision verdict:** the canonical BLA System-1 perceptual substrate.

This document is the single-page narrative of the JEPA arc. Detailed
phase-by-phase decision docs live in `docs/phases/PHASE_*_JEPA_DECISION.md`.

---

## The arc in one sentence

> We started looking for a better slot world model on multi-target
> navigate, discovered slot-delta wins anonymous spatial state but
> trades off identity, falsified four content-side identity fixes
> in a row, and finally found that **object identity is structurally
> an address, not a feature**: persistent learned memory cells that
> query observations (rather than exchangeable slots produced from
> observations) solve identity + state jointly. We then
> over-engineered an occlusion lifecycle and self-corrected: a
> metric we chose for evaluation had been misleading us.

---

## Phase-by-phase trajectory

### Phase 2 – 6 (JEPA track) — slot-delta as spatial state memory

Slot architecture with sparse-delta update rule (`next = slot + change_mask · tanh(delta)`). On the multi-target navigate stress matrix:

| | Result |
|---|---|
| Slot-count sweet spot (Phase 5E) | 8–24 slots |
| Soft-Gaussian sub-pixel rendering (Phase 6) | mechanism survives |
| Identity-aware Hungarian probe (Phase 5C) | the right anonymous eval |

Headline: slot-delta beats dense JEPA on visible position MSE by 5–12% under stress matrices, robustly across seeds.

### Phase 7 v1 (Kubric/MOVi-A) — the tradeoff

Moved from synthetic navigate to rendered video. Four modes × 3 seeds:

| Mode | vis_pos_mse | hidden_pos_mse | switch_rate |
|---|---|---|---|
| slot_delta | **1.0e-5** | **1.6e-5** | 0.705 (bad) |
| slot_dense_update | 4.4e-4 | 3.7e-4 | **0.376** |
| dense_jepa | 2.1e-4 | 2.3e-4 | 0.467 |
| copy (untrained) | 1.9e-3 | 1.9e-3 | **0.155** (gameable!) |

The `copy` baseline winning switch rate revealed the **joint metric** lesson: switch rate alone is gameable by constant-slot architectures. Must read jointly with position MSE.

slot_delta won precision by 20-40× but lost identity stability. The mental model "slot_delta dominates" was false.

### Phase 7B → 8A — four content-side fixes, all falsified

Each phase tried to add identity stability *to the slot's content*. All failed under the joint metric:

| Phase | Intervention | Result |
|---|---|---|
| 7B | predictor-side EMA on id half | switch 0.711 → 0.717 (noise) |
| 7C | encoder consistency loss + GT-supervised aux head | id_drift drops, switch unchanged |
| 7D | id-subspace Hungarian readout | sw_id ≈ sw_full (no subspace) |
| 8A | identity-contrastive loss, λ ∈ {0.1, 0.3, 1.0, 3.0} | at λ=0.1 no effect; at λ ≥ 0.3 collapse |

**The pattern:** anything that pushed identity-stability into the slot's content vector either had no effect or collapsed the content. The slot can't be both a stable address and a dynamic state simultaneously.

[[feedback-prediction-vs-assignment]] — memory entry from this arc:
> *Prediction learns state. Assignment learns identity. Do not expect one loss to learn both.*

### Phase 8C — OF-JEPA v0 (the architectural break)

Restructured the architecture along the prediction-vs-assignment split:

```
exchangeable slots                 →  persistent learned id_proto
                                       (one per file, in model parameters)

observation → slot                 →  memory queries observation
(SlotAttention)                       (id_proto cross-attends to frame proposals)

dense slot update                  →  id_key EMA (slow) + state_value sparse delta (fast)
                                       with LayerNorm between frames

no explicit binding mechanism      →  differentiable Sinkhorn assignment
                                       (memory ↔ proposals)
```

Result on MOVi-A:

| Metric | slot_delta | OF-JEPA v0 |
|---|---|---|
| switch_rate ↓ | 0.689 | **0.0021** (328× better) |
| hidden_pos_mse | 2.20e-5 | 3.24e-5 (1.47× worse) |
| slot_diversity | 5.21 | **1.05** (near-perfect binding) |
| cos_gap | ~0 | **0.45** (id subspace structured) |

The Phase 7 tradeoff **dissolved**. Both axes won simultaneously.

[[feedback-identity-as-address]] — memory entry:
> *Identity is not a feature to be decoded from a slot. Identity is an address used to bind observations into persistent memory.*

### Phase 8D — stress test (≥8 entities + JEPA stride k=4)

Under harder conditions, OF-JEPA didn't merely hold — it widened its lead:

| Metric | slot_delta | OF-JEPA v0 |
|---|---|---|
| switch_rate | 0.788 | **0.024** (33× better) |
| hidden_pos_mse | 2.70e-5 | **2.22e-5** (slightly *better*) |

Strengthened thesis: stable identity binding doesn't just add an identity capability — it **improves predictive state quality when the JEPA target is non-trivial**.

### Phase 9 (MOVi-D) — misdiagnosis under anonymous metric

Tested OF-JEPA on HDRI/GSO MOVi-D (11–23 entities per scene). Anonymous Hungarian metric reported hpm = 4.2e-2 (vs slot_delta's 1.25e-5) and we wrote PHASE_9_JEPA_DECISION.md marking it PARTIAL/FAIL on occlusion.

### Phase 9B — built the wrong fix, found the metric error

Built OF-JEPA v1: null-Sinkhorn dustbin + transition model + visibility BCE — the "obvious" fix for occlusion. But we also added the right metric: **identity-conditioned hidden MSE** (per-file modal-entity comparison, can't be gamed by anonymous rematching).

Under the right metric, v0 was already passing:

| Mode | id_visible_mse | id_hidden_mse | id_h/v |
|---|---|---|---|
| **OF-JEPA v0** | 2.5e-5 | 4.2e-5 | **1.51** |
| slot_delta | 1.5e-5 | 1.3e-5 | 1.10 |
| OF-JEPA v1 | 8.8e-3 | 1.3e-2 | 2.68 (300× regression!) |

v1's "fix" improved the wrong metric (anonymous hpm) while regressing the right one (identity-conditioned hpm).

[[feedback-identity-conditioned-metrics]] — memory entry:
> *For object-file architectures, identity-conditioned metrics are primary. Anonymous Hungarian rematching is a secondary diagnostic that systematically rewards non-persistent shuffling architectures and misdiagnoses real object-file systems.*

Phase 9 verdict retroactively corrected to ✅ (OF-JEPA v0 was already passing on the right metric).

### BLA integration (locked 2026-05-16)

Wrapped the locked architecture in a four-method substrate API:

```python
substrate = OFJEPAObjectFiles(image_size=128, n_files=12, slot_dim=128)
substrate.reset_episode(batch_size=1)
ofb = substrate.observe(frame_t)                # → ObjectFileBatch
tokens = per_file_project(ofb, latent_bus)       # → [B, N_files, d_core]
```

10/10 smoke tests pass. Per-file projection through latent_bus preserves object-file structure to System-2 (no pooling).

---

## Locked canonical architecture (OF-JEPA v0)

```
ConvNeXt-T (proposal encoder)
   ↓ [T, n_patches, proposal_dim]
ObjectFileMemory (canonical)
   ├── id_proto    (persistent learned addresses, parameter)
   ├── Sinkhorn matching (cosine sim + temperature 0.1, 20 iters)
   ├── id_key      ← EMA toward matched proposal id (α=0.05, LayerNormed inter-frame)
   ├── state_value ← + change_mask · tanh(delta) (LayerNormed inter-frame)
   └── slot_to_pos_aux (eval-only)

→ ObjectFileBatch  [id_keys, state_values, confidences, frame_idx]
→ per_file_project(ofb, latent_bus)  → System-2 tokens
```

What's **NOT** in the canonical architecture (all shelved/falsified):
- Visibility belief / null-Sinkhorn / transition model (v1) — Phase 9B falsified
- Encoder identity consistency loss — Phase 7C falsified
- Identity contrastive loss — Phase 8A falsified
- Predictor-side id_dyn_split — Phase 7B falsified
- Exchangeable SlotAttention init — replaced by persistent id_proto

The falsified code lives in `system1_jepa/_attic/` with decision-doc citations.

---

## Memory entries durable from this arc

1. **[[feedback-identity-as-address]]** — the architectural principle.
2. **[[feedback-prediction-vs-assignment]]** — why content-side identity fixes fail.
3. **[[feedback-identity-conditioned-metrics]]** — anonymous Hungarian misdiagnoses object-file systems.
4. **[[feedback-joint-metric-vs-single-axis]]** — single-axis gates are gameable.
5. **[[feedback-slot-persistence-layernorm]]** — inter-frame LayerNorm prevents recurrence blowup.

---

## What this arc demonstrates

For object-centric world models, the locked design rule:

> *Persistent memory addresses + memory-anchored differentiable
> assignment + slow EMA on id_key + sparse delta on state_value +
> identity-conditioned eval primary, anonymous Hungarian secondary.*

The path from here, per Phase 10–20 roadmap:
- **Phase 10**: lock OF-JEPA v0 as reusable System-1 module (this commit + the refactor)
- **Phase 11**: CLEVRER external benchmark
- **Phase 12**: relation graph over object files
- **Phase 13**: action-conditioned dynamics
- **Phase 14**: reachability geometry
- **Phase 15**: behavioral transfer with recurrent policy
- **Phase 16**: scale perception (MOVi-C/E → Kubric → real video)
- **Phase 17**: hierarchical memory (fast/medium/slow object files)
- **Phase 18**: uncertainty + multi-hypothesis identity (may resurrect v1-style lifecycle)
- **Phase 19**: System-2 query interface
- **Phase 20**: full world-model loop
