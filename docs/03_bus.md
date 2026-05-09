# Pillar 3 — Latent Bus + Router + Veto + Prefetch

**Code:** `latent_bus/`. The connective tissue between System 1 and System 2.

## TokenlessLatentBus (`bus.py`)

`d_jepa ↔ d_core` projection MLPs. JEPA's continuous patch features are
projected up to D.C.A.'s working space without an intermediate text
representation. Aligned by `contrastive_infonce` over pooled features —
training script in `scripts/phase1d_train_bus.py`.

**Per-token vs pooled.** The bus operates per-token; the alignment loss
mean-pools. This is intentional — alignment trains the *manifold*, but
downstream consumers (DCA RAMReader, prefetch) take pooled vectors
because the working memory is pooled by construction.

## EntropyRouter (`router.py`)

Wake/sleep gate: variance across multiple JEPA predictor heads is the
"predictive entropy"; when it crosses `threshold_tau`, D.C.A. wakes.
Threshold is a non-trainable buffer because the comparison is
non-differentiable. To train the threshold, attach a smooth surrogate
elsewhere; this module just applies the cut.

## VetoLoop + veto_loss (`bridge.py`)

Pillar §5 of the blueprint, in code:
- DCA emits a plan in core space.
- `bus.forward_down` projects it back into JEPA action space.
- JEPA's predictor simulates the predicted future under that action.
- `veto_loss` computes (simulated_future − safety_target)² so the DCA
  has a gradient channel to revise its plan.

The default `safety_target = zeros` is a "no disruption" placeholder.
Real applications should learn a safety classifier and use its output
here.

## AsyncPrefetcher (`prefetch.py`)

Pillar §4 of the blueprint, in code: while JEPA runs at System-1 rate,
its pooled latents are projected up and continuously query the
`DifferentiableTensorRAM`. The most recent retrieval is cached. When the
entropy router wakes the DCA, the relevant facts are already in cache —
wake-up retrieval latency is zero.
