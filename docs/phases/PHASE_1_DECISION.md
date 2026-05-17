# Phase 1 — Decision document

**Date:** 2026-05-10.
**Status:** ✅ **GATE PASSED. Advancing to Phase 2.**

## What was built

1. **Multi-target navigate env** (`system1_jepa/navigate_env.py::MultiTargetNavigateEnv`) —
   N indistinguishable target patches on canvas, agent must visit all of them
   within `max_steps`. Visited-state is hidden from observation by design.
   Reward: shaped distance-to-nearest-unvisited + bonus on hitting unvisited.
2. **BC baseline** (`scripts/bc_multitarget.py`) — encoder + linear policy +
   expert imitation. The same architecture that hit 100% on single-target
   navigate.
3. **B.L.A. recurrent** (`scripts/bla_multitarget.py`) — encoder per frame,
   SSM scratchpad over the episode history, linear policy reading
   most-recent scratchpad output.
4. **Causal SSM fix** (`system2_dca/ssm.py`) — `DiagonalSSMLayer.causal`
   option + `CausalSSMScratchpad` class. Bidirectional was the default; for
   policy / online-decision contexts it leaks future context at training
   time and tanks deployment performance.

## What was measured

Configuration: image=16×16, patch=2, n_targets=2, max_steps=14, d=128,
encoder_depth=3, ssm_layers=2, 1500 training steps, batch=16 episodes/step.
All on CPU.

| Method | Final BC loss | Success rate | Mean targets visited |
| --- | --- | --- | --- |
| Expert (oracle) | — | 100.0% | 2.00 |
| BC baseline (no memory) | 0.98 | 17.2% | 1.02 |
| B.L.A. with bidirectional SSM | 0.48 | 10.9% | 1.02 |
| **B.L.A. with causal SSM** | **0.17** | **59.4%** | **1.54** |

Delta vs gate: **+42.2pp** (gate required ≥ +20pp).

## Diagnostic findings

1. **BC fails as predicted.** Imitation loss plateaus around 1.0 because
   the regression target (expert action) depends on hidden visited-state.
   The agent visits one target, then oscillates — it can't tell which
   target is "the next one" from the observation alone.
2. **Bidirectional SSM is worse than BC.** Lower training loss (0.48 vs
   0.98) but lower deployment success (10.9% vs 17.2%) — classic
   distribution-shift symptom from training-time future leak. The model
   learned to depend on backward-pass features that aren't available
   at decision time.
3. **Causal SSM crushes BC.** Same encoder, same head, same training
   recipe — only the scratchpad's directionality changed. Loss drops
   from BC's 0.98 to 0.17, and deployment success from 17.2% to 59.4%.
4. **The model is still learning.** Final eval was at training loss 0.17
   and dropping; success rate 59.4% and rising over the last 750 steps.
   With another 1500 steps (or larger model + GPU compute) we'd expect
   70-80%+.

## Caveats

- **CPU-only, small scale.** Image=16, n_targets=2, T=14. The result
  generalizes in principle but should be replicated at the full
  blueprint scale (image=32+, n_targets=3+, T=24+) once GPU is back.
- **No per-component ablation beyond bidirectional vs causal.** The
  roadmap deliverable mentioned "no SSM / no memory / no predictor"
  ablations. In this minimal setup, "no SSM" *is* BC (already covered),
  and there's no temporal predictor used. Real ablation comes when we
  add the rest of the cognitive stack in Phase 2.
- **Real mamba-ssm not integrated.** Phase 1.3 was deferred. The Python
  for-loop scan is fast enough at T=14 to make the comparison; the real
  kernel is a Phase 6 prerequisite, not a Phase 1 prerequisite.
  Tracked separately.

## What this proves and doesn't prove

**Proves:** memory-bearing components add measurable value over
imitation-only on a task where memory is provably necessary. The
B.L.A. scaffold composes correctly — encoder + scratchpad + policy
trains end-to-end and outperforms BC by a wide margin.

**Does not prove:** any of the larger asymmetric-scaling claims in
`VISION.md`. The verification layer hasn't been built. The hybrid
memory hasn't been built. The compute-economy router hasn't been
trained. This phase only tests *one* prediction: that having memory
helps when the task requires memory. It does. Onward.

## Decision

**Advance to Phase 2 (verification layer).** Phase 1 gate met. Real
architectural lesson (causal SSM) folded into the codebase. Foundation
ready for the next pillar.

## Logged for memory

- BC on multi-target plateaus at ~17% with mean 1 target visited (out of 2)
- B.L.A. + causal SSM hits 59% in 1500 CPU-steps
- **Bidirectional SSM is wrong for policy contexts; use `CausalSSMScratchpad`**
- The training-loss vs deployment-success gap is the canary — when training
  loss drops faster than deployment success climbs, suspect future-leak.
