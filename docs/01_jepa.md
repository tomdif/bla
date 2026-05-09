# Pillar 1 — JEPA (System 1)

**Code:** `system1_jepa/`. Two coexisting JEPA implementations:

1. **Patch JEPA** (`model.py`, `predictor.py`, `vit.py`) — single-image, masks
   patches within one frame, predicts masked patches at target positions
   given context patches + action vector. Use this for still-image
   pretraining.
2. **Spatiotemporal JEPA** (`spatiotemporal.py`) — V-JEPA-2-style. Encoder
   over [B, T, C, H, W] → [B, T, N_patches, D]. Tube masking (same patch
   positions masked across all frames). Predictor at masked (t, p)
   positions. Use this for clip pretraining.

Both share:
- ViT patch embed (`Conv2d`) + sincos-2d position embeddings
- EMA target encoder, weights held in fp32 to keep EMA mul+add updates
  numerically safe even when the trained branch runs bf16
- Action conditioning broadcast onto every input token
- Smooth L1 prediction loss + SIGReg anti-collapse regularizer (`sigreg.py`)

## Why SIGReg, not VICReg

VICReg (which we shipped first) penalizes per-axis variance and feature
covariance. It misses **oblique** low-rank subspaces — distributions with
healthy per-axis std but compressed mass along a particular direction.

SIGReg uses the Cramér–Wold theorem: a distribution is Gaussian iff every
1D projection is Gaussian. We draw K random unit-norm projections and
apply a univariate normality test (Epps–Pulley, 16 directions; or LeWM
moment-fit, 1024 directions) to each. Catches what VICReg misses.

`losses.py::collapse_regularizer` is the entry point;
`JEPAConfig.sigreg_variant ∈ {"epps_pulley", "lewm"}`.

## Why temporal predictor + multi-step rollout

The `TemporalPredictor` (`temporal.py`) operates on pooled per-frame
embeddings, separate from the per-patch JEPA. It takes [B, T, D] history
+ [B, K, action_dim] action chunk and predicts the next-frame embedding.
`multistep_rollout_loss` slides the predicted z back into the context
and supervises each horizon — trains self-consistency under recursion,
which is what CEM exploits at inference.

The reward + value heads exist (zero-initialized) so the planner can
score sequences without extra architectural changes when reward becomes
available.

## When to use which

| Goal | Module |
| --- | --- |
| Pretrain on still images | `BLAJEPAModel` (patch JEPA) |
| Pretrain on video clips | `SpatiotemporalJEPA` |
| World-model rollout for planning | `TemporalPredictor` on top of either |
| Anti-collapse regularizer | `sigreg_epps_pulley` (cheap) / `sigreg_lewm` (stronger) |

## Things that bit us, kept here so they don't bite again

- **EMA in bf16 underflows.** `(1 - tau) ≈ 0.004` is at the edge of bf16
  resolution. Target encoder must be fp32 even when context is bf16.
- **VICReg invariance term double-counts the prediction loss.** We use
  variance/covariance only (now SIGReg) — never the invariance term.
- **No causal mask on patch tokens.** Patches are spatial; a causal mask
  is a bug.
- **Random-image scripts are decorative.** JEPA on `randn` learns
  nothing. Use `make_image_loader(source="cifar10")` or `"synthetic"`
  for anything real.
