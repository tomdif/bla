# Project B.L.A.

This repository is a runnable Phase 1 scaffold for the Bicameral Latent
Architecture blueprint. It intentionally avoids next-token cross-entropy
training and a monolithic autoregressive model. The code is split into:

- `system1_jepa`: V-JEPA-style sensorimotor encoder, EMA target encoder,
  action-conditioned latent predictor, Smooth L1 plus VICReg losses.
- `system2_dca`: non-autoregressive cognitive engine with a bidirectional
  state-space scratchpad, 1D latent diffusion CPU, and frozen deterministic
  lexical codebook decoder.
- `latent_bus`: 1024-to-4096 tokenless projection bus, InfoNCE-style alignment
  objective, entropy router, and veto loop utilities.
- `tensor_ram`: FAISS-backed MIPS RAM abstraction with small local defaults and
  quantized frozen embedding helpers.

The blueprint-scale jobs are not launched automatically. A 10M x 4096 RAM
matrix is roughly 153 GiB in float32 or 76 GiB in bfloat16 before index
overhead, so the scripts default to smoke-sized runs unless `--force-large` is
provided.

## Quick Start

```bash
cd /Users/thomasdifiore/bla
python3 scripts/smoke_forward.py
python3 -m pytest
```

## Phase 1 Entry Points

```bash
python3 scripts/phase1a_train_jepa.py --steps 1
python3 scripts/phase1b_train_dca.py --steps 1
python3 scripts/phase1c_populate_ram.py --num-vectors 4096 --output runs/ram_smoke
```

These scripts use tiny dimensions by default so they run on CPU. Move to
blueprint dimensions by passing larger config values and real datasets.
The RAM script defaults to a NumPy MIPS backend for portability; pass
`--backend faiss` in an environment with a working FAISS build.
