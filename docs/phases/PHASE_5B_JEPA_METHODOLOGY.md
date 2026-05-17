# Phase 5B (JEPA Track) — Methodology finding

**Date:** 2026-05-16.
**Status:** ⚠ **METHODOLOGY ISSUE — both probe protocols are wrong in different ways.**

> Phase 5B-attempt-1 used within-episode probe splitting: dynamic-slot
> mode scored 0 MSE everywhere because the probe memorized
> (state-signature → targets) on near-constant per-episode state.
> Phase 5B-attempt-2 used episode-held-out splitting: every mode scored
> ~185 MSE because a linear probe on flattened slots can't generalize
> across episodes (slot-permutation problem). The right metric is
> somewhere in between; neither attempt answers the dynamic-vs-fixed
> question cleanly.

## Two attempts, two failure modes

### Attempt 1: within-episode probe (Phase 2-5A protocol)

```
train probe on visible frames from ALL collected episodes
test  probe on hidden  frames from SAME episodes
```

Reward signal: the probe learns to decode targets when the encoder
state contains enough scene-specific information. With dynamic slots
(N-K of N slots frozen), 75% of the slot state is constant within an
episode. The probe memorizes (episode-signature → targets) and
scores 0 MSE on both visible and hidden frames trivially.

| nt | fixed-16 | fixed-32 | fixed-64 | **dyn-64/16** | **dyn-64/24** | **dyn-64/32** |
|---|---|---|---|---|---|---|
| 8 | 13.31 | 4.71 | 2.93 | **0.00** | **0.00** | **0.00** |
| 16 | 8.90 | 7.79 | 5.02 | **0.00** | **0.00** | **0.00** |
| 32 | 13.39 | 6.19 | 2.96 | **0.00** | **0.00** | **0.00** |

The 0.00 entries are probe overfit, not perfect memory.

### Attempt 2: held-out-episode probe

```
split episodes into train (80%) and test (20%)
train probe on visible frames from TRAIN episodes only
test  probe on hidden  frames from TEST  episodes
```

Reward signal: probe must extract generalizable structure (e.g.
"target position is in slot K") that transfers to unseen episodes.
The slot system is permutation-equivariant — slot index 7 might
encode "target A" in one episode and "distractor C" in another — so
a linear probe trained on flat slot states can't generalize.

Result: every mode hits ~180-190 MSE, indistinguishable from
predicting the mean target position.

| nt | fixed-16 | fixed-32 | fixed-64 | dyn-64/16 | dyn-64/24 | dyn-64/32 | dense_jepa | copy |
|---|---|---|---|---|---|---|---|---|
| 8 | 185.34 | 194.27 | 185.19 | 187.41 | 187.08 | 185.94 | 181.62 | 179.91 |
| 16 | 188.15 | 189.34 | 183.27 | 184.82 | 184.59 | 185.22 | 179.83 | 178.37 |
| 32 | 189.35 | 187.52 | 183.63 | 187.17 | 187.14 | 187.25 | 182.01 | 180.30 |

The dynamic-vs-fixed comparison is now at parity (1.00-1.02× ratio
at every cell). But that's because BOTH have been knocked down to
chance — the probe can't decode targets in either.

## What this means for the earlier phases

The Phase 2/3/4A/4B/5A probe used the within-episode protocol. With
that protocol:

- The fixed-N modes' "40-64× margin over dense_jepa" measures
  *within-episode discrimination* — i.e. "given encoder state, can a
  probe distinguish this scene's targets from the constant-mean
  prediction?". Slot_delta clearly wins this comparison.
- The within-episode protocol rewards encoders that preserve
  scene-specific state through occlusion. That IS the falsifiable
  memory claim of the architecture.
- The within-episode protocol does NOT measure whether slot bindings
  transfer across scenes. That's a stronger claim (permutation-
  invariant generalization) which the architecture was never
  explicitly designed for and which slot-attention is known to make
  hard.

So the earlier claims should be qualified:

| Earlier claim | Honest restatement |
|---|---|
| "slot_delta hidden MSE 3.87 at J=20" (Phase 2B) | within-episode probe |
| "60×+ margin over dense_jepa" (Phase 3) | within-episode discrimination |
| "bounded forgetting plateau" (Phase 3, 4A, 4B, 5A) | within-episode preservation |

None of those need to be retracted; they need a footnote.

## What this means for Phase 5B

The original question was: "does dynamic slot allocation match fixed
slot allocation in memory quality?"

Within-episode probe rewards both, but dynamic looks artificially
perfect (0 MSE) because of the constant-state shortcut. Cross-episode
probe penalizes both equally and tells us nothing.

The clean answer would need a probe protocol that:
1. Doesn't reward constant-per-episode state shortcuts.
2. Tolerates slot-permutation by being permutation-equivariant.

Options:
- **Permutation-invariant readout** (set transformer or attention-
  based decoder over slots, not flat).
- **Hungarian-matching MSE**: best-match assignment between predicted
  and true target positions, ignoring slot ordering.
- **Within-episode probe + a constant-state penalty**: subtract a
  baseline trained to predict targets from the *time-averaged* slot
  state across the episode. If a model's probe MSE stays low after
  removing the constant component, real per-step memory is happening.

## Decision

**Phase 5B is INCONCLUSIVE on the dynamic-vs-fixed question.** Banking:

- The methodology issue (both probes wrong) as a finding.
- The fixed-N data from attempt 1 as a reproduction of Phase 5A.
- The cross-episode null result from attempt 2 as a flag on linear
  probes for slot systems generally.

The Phase 2/3/4A/4B/5A claims **stand with a footnote** that they
measure within-episode preservation, not cross-episode generalizable
decoding. That is still the architecture's intended claim, just
narrower than I wrote it.

## What's next

Phase 5C should pause the capacity question and address the
methodology. Build a permutation-invariant readout (Hungarian-match
MSE is the cheapest) and re-test the Phase-5B dynamic comparison +
spot-check Phase-4B / 5A under the better metric.

This is more important right now than CLEVRER realism — without a
trustworthy metric, scaling perception is premature.

Concrete plan for Phase 5C:

```
1. Hungarian-match readout:
   for each scene, pair each predicted (slot → xy) with the closest
   ground-truth target_xy (min-cost matching); report mean of best
   pair MSEs

2. Re-run a small Phase-4B-equivalent sweep with the new readout:
   slot_delta vs dense_jepa_flatten vs copy, 3 seeds, J=20/40/80

3. If slot_delta still wins by a large margin under Hungarian-match
   on held-out episodes, the architecture's claim is durable
4. If margin shrinks: the architecture-level story shrinks
   accordingly, but is still defensible at the within-episode level
```

This is the right shape of fix — keeps the existing data + adds a
metric that's robust to both failure modes we just identified.

## Artifacts

```
artifacts/phase5b_attempt1/   raw rows from within-episode probe
artifacts/phase5b_attempt2/   raw rows from held-out-episode probe
                              (same setup; both kept for the record)
```
