# proofworldlean — portable Mathlib kernel for proofworld

Self-contained Lake project that puts **Mathlib** behind `proofworld`'s verifier (used by
`proofworld/research_mathlib.py`). It pins Mathlib to a rev whose toolchain (`v4.30.0-rc2`) has a public olean
cache, so setup is a download — **no full compile**.

## Setup on a fresh checkout
```bash
cd proofworld/lean
lake exe cache get     # downloads ~8000 prebuilt Mathlib oleans (minutes, no compile)
lake build             # builds the stub lib ProofWorldLean (fast: ~12s with cache)
```
This requires `elan` (it will use the pinned `lean-toolchain`, `v4.30.0-rc2`). `.lake/` is gitignored — only the
pins (`lakefile.toml`, `lean-toolchain`, `lake-manifest.json`) and `ProofWorldLean.lean` are tracked, which is what
makes the project reproducible.

## How proofworld uses it
`proofworld/research_mathlib.py` auto-discovers this project (it checks for built Mathlib oleans + an installed
toolchain) and runs `lake env lean` from here, so the tactic-dreamer/LLM can use `nlinarith`, `ring`, etc. If this
project isn't built, it falls back to borrowing a built Mathlib from another local Lean project; override either
with `PROOFWORLD_MATHLIB_PROJECT=/path/to/project`.

`ProofWorldLean.lean` is a thin preamble lib (`import Mathlib.Tactic.Linarith` + the demo's recursive `oddSum`).
