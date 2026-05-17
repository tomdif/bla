# BLA — Current Status

Last updated: 2026-05-16

## What works (committed + tested)

### System-2 cognition (BLA roadmap phases 0–5)

| Phase | Capability | Headline |
|---|---|---|
| 0 | Substrate + BC baseline | BC = 100% on single-target navigate |
| 1 | Multi-target navigation | BLA = 59% vs BC = 17% (+42pp) — [details](docs/phases/PHASE_1_DECISION.md) |
| 2 | Verification + 7-action router + calibration | Brier ≤ 0.1, ECE 0.045 — [details](docs/phases/PHASE_2_DECISION.md) |
| 3 | Hybrid memory (symbolic/executable/episodic) | precision@5 = 0.965, MRR = 0.95 — [details](docs/phases/PHASE_3_DECISION.md) |
| 4a | RAG on GPT-2 small | hallucinations 70.8% → 4.9% — [details](docs/phases/PHASE_4_DECISION.md) |
| 4b | RAG on BERT non-AR | parametric 42% → RAG 98% — [details](docs/phases/PHASE_4b_DECISION.md) |
| 5 | RL-trained compute router | 75.9× compute split, −2.2pp easy-task drop — [details](docs/phases/PHASE_5_DECISION.md) |

### System-1 perception (JEPA arc — Phases 7-9B JEPA + integration)

The architectural journey, all in `docs/phases/`:

| Phase | Status | Headline |
|---|---|---|
| 2-6 (JEPA) | ✅ | `slot_delta` strong spatial state memory under stress |
| 7 v1 | ⚠ | slot_delta / slot_dense_update tradeoff identified |
| 7B-D | ❌ | 4× slot-content interventions falsified |
| 8A | ❌ | identity-contrastive loss collapses content |
| 8C | ✅ | **OF-JEPA v0** on MOVi-A: switch 0.002, joint gate passed |
| 8D | ✅ | OF-JEPA wins BOTH axes under stress (≥8 entities, stride=4) |
| 9 | ✅ | (corrected) OF-JEPA on MOVi-D: id_h/v 1.51, switch 11× better |
| 9B | ❌ | v1 visibility-gating is a regression; v0 was already passing |
| Integration | ✅ | OF-JEPA v0 as System-1 substrate; per-file bridge to latent_bus |

**Locked canonical architecture (OF-JEPA v0):**
- ConvNeXt-T proposal encoder
- Persistent learned `id_proto` (identity addresses, one per file)
- Memory-anchored Sinkhorn matching (memory queries observations)
- Slow EMA on `id_key` + sparse-delta on `state_value` with inter-frame LayerNorm
- Identity-conditioned position MSE as the primary eval metric (anonymous Hungarian is secondary)

**Substrate API** (`system1_jepa/of_jepa_api.py`):

```python
substrate = OFJEPAObjectFiles(image_size=128, n_files=12, slot_dim=128)
substrate.reset_episode(batch_size=1)
ofb = substrate.observe(frame_t)           # → ObjectFileBatch
tokens = per_file_project(ofb, latent_bus)  # → [B, N_files, d_core]
```

## What doesn't exist yet

1. **BLA Phase 6 (1B procedural core, make-or-break)** — not started. Needs 6×B200 + $30-100K compute.
2. **Object birth/death lifecycle** (OF-JEPA v1) — shelved per Phase 9B; revive when streaming data with true entries/exits.
3. **External benchmarks** — no CCT, CLEVRER, V-JEPA-2-class embodied test.
4. **End-to-end pipeline** — System-1 (OF-JEPA), latent_bus, System-2 (DCA + memory) exist independently. No script wires the full perception→planning→action loop together yet.

## Durable architectural principles (memory entries)

- [Identity is an address, not a feature](https://...memory/feedback_identity_as_address.md) — Phase 8C
- [Prediction vs assignment need separate signals](https://...memory/feedback_prediction_vs_assignment.md) — Phase 7-8 arc
- [Identity-conditioned metrics are primary](https://...memory/feedback_identity_conditioned_metrics.md) — Phase 9B
- [Joint metric vs single axis](https://...memory/feedback_joint_metric_vs_single_axis.md) — Phase 7v1
- [Slot persistence requires LayerNorm](https://...memory/feedback_slot_persistence_layernorm.md) — Phase 7B
- [Calibrate before hash](https://...memory/threshold_calibration_retrospective.md) — Phase 2 / 8C v1

## Repo layout

```
system1_jepa/        # JEPA + OF-JEPA implementation
  of_jepa.py         # canonical architecture (v0 + shelved v1)
  of_jepa_api.py     # substrate API: observe/predict/read/metrics
  identity_probe.py  # Hungarian + identity-conditioned probes
  movi_data.py       # MOVi-A/D loader
  convnext_encoder.py # ConvNeXt-T → patch tokens
  (legacy: model.py, spatiotemporal.py, slot.py, slot_predictor.py — kept for reproducibility)

system2_dca/         # System-2: 1B procedural core, decoder, memories
latent_bus/          # System-1/2 bridge

scripts/             # Phase orchestrators (many are single-use legacy)
tests/               # Unit + smoke tests (test_of_jepa_api.py, test_identity_probe.py, test_id_dyn_split.py)

artifacts/           # Phase run results (committed JSONs + decision artifacts)
docs/phases/         # Decision docs, one per phase
runs/                # Local training logs (gitignored)

README.md, ROADMAP.md, VISION.md, STATUS.md (this file)
```
