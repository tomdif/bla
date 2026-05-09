# Project B.L.A.

Runnable Phase 1 scaffold for the Bicameral Latent Architecture blueprint.
The code intentionally avoids next-token cross-entropy training and a
monolithic autoregressive model. Five subsystems, all individually trainable:

- **`system1_jepa`** — V-JEPA-style sensorimotor encoder with EMA target,
  random-patch masking, action-conditioned predictor at target positions.
  Anti-collapse via SIGReg (Cramér-Wold Epps-Pulley / LeWM moment-fit, ported
  from cser-jepa-v2 — strictly stronger than VICReg). Frame-level temporal
  predictor with multi-step rollout supervision on top, trained on a synthetic
  moving-patch sequence generator.
- **`system2_dca`** — non-autoregressive cognitive engine: bidirectional
  state-space scratchpad, 1D rectified-flow latent diffusion with adaLN-Zero
  DiT blocks and memory cross-attention, frozen fp32 lexical codebook decoder.
  Differentiable RAM read inside the forward pass via `RAMReader` —
  the model learns *what* to fetch, RAM contents stay non-parametric.
- **`latent_bus`** — tokenless `d_jepa ↔ d_core` projection bus, InfoNCE-style
  alignment objective, entropy router with predictor-variance gating,
  veto loop with differentiable loss for plan rejection, and an
  `AsyncPrefetcher` that uses JEPA's continuous latent stream to pre-load
  RAM facts before the DCA wakes up.
- **`tensor_ram`** — both a NumPy/FAISS `FaissTensorRAM` (production retrieval)
  and a torch-side `DifferentiableTensorRAM` (training-time differentiable
  query). int8 + int4 (uniform 4-bit, *not* FP4/NF4) row-wise quantization.

Mixed precision: `tiny()` configs default to `float32` so smoke loops
overfit a single batch reliably; blueprint configs default to `bfloat16`,
with the JEPA target encoder and the DCA codebook held in `float32` to keep
EMA updates and argmax decoding numerically safe.

The blueprint-scale jobs are not launched automatically. A 10M × 4096 RAM
matrix is roughly 153 GiB in float32 or 76 GiB in bfloat16 before index
overhead, so `phase1c_populate_ram.py` refuses payloads above 8 GiB unless
`--force-large` is provided.

## Quick start

```bash
cd /Users/thomasdifiore/bla
python3 scripts/smoke_forward.py
python3 -m pytest
```

## Phase 1 entry points

```bash
python3 scripts/phase1a_train_jepa.py     --steps 50 --overfit
python3 scripts/phase1b_train_dca.py      --steps 50 --overfit
python3 scripts/phase1c_populate_ram.py   --num-vectors 4096 --output runs/ram_smoke
python3 scripts/phase1d_train_bus.py      --steps 50
python3 scripts/phase1e_train_temporal.py --steps 100
```

`phase1e` is the multi-step rollout trainer: a small `TemporalPredictor`
takes pooled JEPA frame embeddings + an action chunk, predicts the next
frame's pooled embedding, and is supervised against ground-truth pooled
embeddings of the future frames. Rollout loss should drop ~20× in 5 steps
on the moving-patch task — that's the smoke signal that temporal
self-consistency is being learned.

The `--overfit` flag in 1a/1b reuses one fixed batch every step; the loop
is healthy when prediction loss + velocity loss decrease monotonically.
The default tiny dimensions run on CPU. Move to blueprint dimensions by
replacing `JEPAConfig.tiny()` / `DCAConfig.tiny()` / `TemporalConfig.tiny()`
with hand-tuned configs and pointing at real datasets.

## Blueprint mapping

| Blueprint pillar | Code |
| --- | --- |
| Tokenless Latent Bus | `latent_bus/bus.py::TokenlessLatentBus` + InfoNCE |
| Neural CPU (D.C.A.) | `system2_dca/diffusion.py::LatentDiffusionEngine` |
| Continuous Tensor RAM | `tensor_ram/torch_ram.py::DifferentiableTensorRAM` |
| Differentiable RAM read | `system2_dca/ram_reader.py::RAMReader` |
| State-Space Scratchpad | `system2_dca/ssm.py::WorkingMemory` |
| Latent canvas / iterative denoising | `LatentDiffusionEngine.sample` |
| JEPA-prior diffusion warm start | `LatentDiffusionEngine.sample(prior=…, t_start<1)` |
| Bidirectional self-correction | full self-attention DiT, no causal mask |
| Frozen Lexical Decoder | `system2_dca/decoder.py::DeterministicLexicalDecoder` |
| Entropy router (System 1 ↔ 2 gate) | `latent_bus/router.py::EntropyRouter` |
| Async prefetch (Pillar §4) | `latent_bus/prefetch.py::AsyncPrefetcher` |
| Veto loop (Pillar §5) | `latent_bus/bridge.py::veto_loss` |
| Multi-step rollout self-consistency | `system1_jepa/temporal.py::multistep_rollout_loss` |

What is *not* yet here, deliberately: the Phase-3 training curriculum
(synthetic-logic-only CPU pretraining, RL for compute economy on the
router, value/reward heads on the temporal predictor), and a real video
dataloader. Those are training-curriculum deliverables; the architectural
substrate to run them is now in place.
