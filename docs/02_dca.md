# Pillar 2 — D.C.A. (System 2)

**Code:** `system2_dca/`. Three modules:

1. **`ssm.py::WorkingMemory`** — bidirectional diagonal SSM scratchpad.
   Folds query + retrieved facts into a single fixed-size hidden state.
   Reference implementation is a Python for-loop scan; swap for
   `mamba-ssm` / parallel scan at scale.
2. **`diffusion.py::LatentDiffusionEngine`** — 1D rectified-flow over the
   latent canvas. Full self-attention (no causal mask), adaLN-Zero
   conditioning, cross-attention from canvas tokens to memory tokens.
3. **`decoder.py::DeterministicLexicalDecoder`** — frozen fp32 codebook,
   argmax decode (temperature 0). Random by default; `from_embeddings()`
   loads a real tokenizer table.

## Why rectified flow, not DDPM

The interpolant is `x_t = (1 - t) * x0 + t * noise`. Training target is
the velocity `v* = noise - x0`. Sampling integrates `dx/dt = v(x, t)`
from `t = 1` to `t = 0` with explicit Euler. We default to 16 steps; for
warm-started sampling (JEPA prior at `t_start < 1`) you can integrate
fewer steps.

This was our biggest correctness fix in the early rounds: the original
scaffold mixed the rectified-flow interpolant with an ε-prediction
target, which has no consistent fixed point and collapses to constant
output. If you change either the interpolant or the target, change both.

## Why adaLN-Zero

DiT blocks zero-initialize the conditioning linear so the layer is
identity at init. This stabilizes training but has a quiet consequence:
**gradients to upstream modules (memory, query, bus) are exactly zero in
the first step** because the diffusion's `out_proj` (also zero-init) zeros
the output entirely. By step 2 the out_proj has trained a little and
upstream gradient flows. The integration test asserts this over multiple
steps, not on step 0.

## Differentiable RAM read (the load-bearing claim)

`ram_reader.py::RAMReader` lives inside `DCAEngine.forward`. The model
takes a query, projects into RAM-key space (multi-head), retrieves
soft-top-k from `DifferentiableTensorRAM`, and value-projects back into
working space. Keys are buffer (frozen, no gradients into the index);
queries flow gradients through `query @ keys.T` and the softmax-weighted
reconstruction.

**Sparsity:** `sparsity_weight > 0` adds an entropy-of-weights penalty;
`hard=True` returns straight-through one-hot retrieval at top-1. The
default reader hedges across top-k; honoring "physical separation of
facts and logic" requires turning sparsity on.

## Memory injection paths

Two paths from `WorkingMemory` to the diffusion canvas:
- **Scalar conditioning** — `memory_global(memory)` adds to the per-block
  shift/scale generator (cond vector).
- **Cross-attention** — `memory_to_tokens(memory)` reshapes into a small
  set of memory tokens; each DiT block cross-attends from canvas to those
  tokens.

The second is what makes the working memory load-bearing instead of a
scalar bottleneck.
