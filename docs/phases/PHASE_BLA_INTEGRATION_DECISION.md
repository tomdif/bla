# BLA System-1 Integration Decision document

**Date:** 2026-05-16.
**Status:** ✅ **OF-JEPA v0 wired into BLA as the System-1 substrate. Smoke test 10/10 pass; bridge confirms object-file structure flows through to System-2 unaltered.**

> Locked the Phase 7→9B JEPA arc into a stable substrate API. The
> four-method interface (`observe / predict / read / metrics`) lets
> System-2, planning, and verification consume OF-JEPA's object-file
> state without knowing the internal architecture. The bridge to
> the existing `latent_bus` projects EACH file independently — the
> per-file [id_key, state_value] structure reaches System-2 as a
> sequence of tokens, not as a pooled vector.

## What landed

**`system1_jepa/of_jepa_api.py`** — `OFJEPAObjectFiles` substrate class
+ `ObjectFileBatch` data type + `per_file_project` bridge adapter.

Four-method API (matches the user-specified locked interface):

```python
substrate = OFJEPAObjectFiles(image_size=128, n_files=12, slot_dim=128, device="cuda")
substrate.reset_episode(batch_size=1)

# Per-frame update.
ofb: ObjectFileBatch = substrate.observe(frame_t)
#   ofb.id_keys      [B, N_files, id_dim]      persistent identity addresses
#   ofb.state_values [B, N_files, state_dim]   dynamic content
#   ofb.confidences  [B, N_files]              Sinkhorn match confidence
#   ofb.frame_idx    int                       which frame this state reflects
#   ofb.full_slot   [B, N_files, slot_dim]    legacy concat view

# Forward prediction. v0: k=0 only (returns current). v1: k>0 uses transition_model.
ofb_pred = substrate.predict(k_steps=0)

# System-2 readable structured-state dump.
state_dict = substrate.read("all")  # or "id_keys" / "state_values" / "active"

# Identity-conditioned diagnostics (populated externally by evaluator).
metrics = substrate.metrics()
```

Bridge adapter (`per_file_project`) projects each file separately:

```python
from latent_bus.bus import TokenlessLatentBus
bus = TokenlessLatentBus(d_jepa=128, d_core=4096)
tokens_for_system2 = per_file_project(ofb, bus)
# tokens_for_system2.shape = [B, N_files, d_core]   <-- per-file tokens, NOT pooled
```

**The architectural commitment:** System-2 sees object-file STRUCTURE
(per-file tokens with persistent identity addresses), not a pooled
scene summary. This preserves the prediction-vs-assignment
decomposition all the way through the stack.

## Smoke test results

10/10 tests pass on CPU in 8 seconds:

- `observe()` advances frame_idx and updates memory correctly
- `predict(k=0)` returns current state with confidence=1
- `predict(k>0)` on v0 raises NotImplementedError (correctly surfaces
  the locked-architecture limitation)
- `predict(k>0)` on v1 (still shelved per Phase 9B) uses
  transition_model — code path verified for when v1 is revived
- `read()` supports all four query types
- `metrics()` cache roundtrip works
- `ObjectFileBatch.full_slot` concatenates id_keys + state_values correctly
- **`per_file_project()` produces [B, N_files, d_core] with non-degenerate per-file variance** — System-2 receives object-file structure
- `reset_episode()` clears memory bindings

## Design choices locked

1. **Substrate is stateful.** It carries persistent memory across
   `observe()` calls within an episode. `reset_episode()` clears.
   System-2 doesn't need to manage memory — that's the substrate's job.

2. **predict() on v0 is k=0 only.** OF-JEPA v0 has no transition_model
   (per Phase 9B's locked architecture — v1's transition is shelved).
   Calling `predict(k > 0)` raises `NotImplementedError` with a
   pointer to revive v1 OR supply an action-conditioned predictor.
   This surfaces the limitation honestly instead of silently
   returning bad predictions.

3. **read() returns a serializable dict.** System-2 can pickle,
   send over network, or render to JSON. Tensors come down to CPU
   automatically.

4. **Bridge is per-file, not pooled.** Each object file is projected
   independently through the latent_bus. This is the architectural
   choice that distinguishes the OF-JEPA bridge from the old
   slot-based bridge in `latent_bus/bridge.py` (which pools to a
   single vector via `dca_plan.mean(dim=1)`). The new path preserves
   identity addresses for System-2 to address by.

5. **metrics() is read-only from the substrate's side.** External
   evaluators populate it via `cache_metrics()`. Decouples metric
   computation (often heavy: needs a full eval pass) from inference
   (cheap: just runs the model forward).

## What this DOES NOT integrate

Deliberately narrow per user direction:

- The existing `latent_bus/bridge.py` `VetoLoop` still uses the old
  slot-based JEPA predictor API. It is NOT rewritten. Replace it
  only when a downstream task (planning, verification) actually
  needs OF-JEPA's object-file structure.
- The old `BLAJEPAModel` from `system1_jepa/model.py` is untouched.
- No System-2 training script is modified. Integration happens at
  inference time via the substrate API.
- The shelved OF-JEPA v1 lifecycle (`ObjectFileMemoryV1`) remains
  available but isn't surfaced through the API as the default.

## Locked locked-architecture summary (for the canonical doc)

Final BLA System-1 perception substrate (post Phase 7→9B):

```
OFJEPAObjectFiles (substrate)
├── ProposalEncoder    (ConvNeXt-T → patch tokens)
├── ObjectFileMemory   (v0 — current canonical)
│   ├── id_proto       (persistent learned addresses, N_files × id_dim)
│   ├── proposal_id_proj  (proposal → id-key space)
│   ├── Sinkhorn matching with cosine sim + LayerNorm
│   ├── delta_head + change_head (sparse state_value update)
│   └── slot_to_pos_aux (eval-only)
└── (shelved v1 components — null Sinkhorn, transition, visibility — not active)

ObjectFileBatch  (per-frame dataclass)
per_file_project(ofb, bus)  (bridge adapter, per-file independent projection)
```

Three durable architectural principles, locked to memory:
- [[feedback-identity-as-address]]
- [[feedback-prediction-vs-assignment]]
- [[feedback-identity-conditioned-metrics]]

## What stays open

| Item | Status | Trigger |
|---|---|---|
| Replace `latent_bus/bridge.py` VetoLoop with OF-JEPA-aware version | not started | when a downstream task needs it |
| CLEVRER external benchmark | pending | when ready for external validation |
| MOVi-E with camera motion | pending | when v1 lifecycle becomes load-bearing |
| Revive v1 lifecycle | shelved | when streaming/birth-death regime is the test |
| Phase 8C v1 (#102) — full lifecycle | shelved | superseded by Phase 9B finding |

## Reproducibility

Code committed in this commit. Smoke test:

```bash
python3 -m pytest tests/test_of_jepa_api.py -v
```

10/10 tests pass on CPU in ~8 seconds with small-config substrate
(image_size=64, n_files=8, slot_dim=32).
