# Pillar 4 — Tensor RAM

**Code:** `tensor_ram/`. Two implementations behind the same conceptual
interface.

## DifferentiableTensorRAM (`torch_ram.py`)

Torch-side, all-in-memory. Keys held as a frozen buffer; queries flow
gradients. Brute-force `query @ keys.T` for top-k MIPS. This is what the
DCA's `RAMReader` queries during training. Use up to ~100k entries; for
larger you pay quadratic memory unless you swap in a top-k that
detaches indices.

## FaissTensorRAM (`faiss_ram.py`)

NumPy + optional FAISS backend. This is the *deployment* RAM: pre-built
index, sub-linear retrieval, save/load to disk. The smoke pipeline uses
it via `weighted_retrieve()`; the differentiable training path uses
`DifferentiableTensorRAM`.

## Quantization (`quantization.py`)

`quantize_int8` and `quantize_int4` (uniform 4-bit, **not** FP4/NF4 —
those use nonuniform code points; this is integer quantization). Both
return a `QuantizedEmbeddings` object with `dequantize()` round-trip.

## Sizing

| n_vectors | d_ram | fp32 | bf16 |
| --- | --- | --- | --- |
| 1M | 1024 | 3.8 GiB | 1.9 GiB |
| 10M | 4096 | 153 GiB | 76 GiB |

`phase1c_populate_ram.py` refuses payloads above 8 GiB without
`--force-large` to make the failure mode obvious instead of running OOM
two hours into a session.
