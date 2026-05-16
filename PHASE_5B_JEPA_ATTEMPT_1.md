# Phase 5B (JEPA Track) — Attempt 1 (INCONCLUSIVE on dynamic, valid on fixed-N)

**Date:** 2026-05-16.
**Status:** ⚠ **INCONCLUSIVE — probe overfits when most slots are frozen.**

> The dynamic-slot mode (top-K active gate with K active slots from a
> pool of 64) collapsed the linear probe to MSE 0.0 on both visible AND
> hidden frames. The (N-K)/N = 75% frozen slots carry an
> episode-specific binding signature that lets a 262K-parameter linear
> probe perfectly memorize the (state → target_xy) mapping from only
> 80 visible training samples. This is a probe-protocol artifact, not
> a Phase-5B result on the architecture.

## Fixed-N results (still valid — reproduce Phase 5A)

The fixed-N control rows confirm the Phase-5A capacity curve:

| nt | fixed-16 | fixed-32 | fixed-64 | dense_jepa | copy |
|---|---|---|---|---|---|
| 8 | 13.31 | 4.71 | **2.93** | 153.23 | 155.14 |
| 16 | 8.90 | 7.79 | **5.02** | 156.77 | 159.35 |
| 32 | 13.39 | 6.19 | **2.96** | 157.02 | 160.47 |

(hidden MSE at J=80, n_distractors=10, 5 seeds. Phase 5A had 3 seeds.)

This is consistent with Phase 5A: doubling slots roughly halves MSE;
fixed-64 maintains MSE ~3 across all entity counts; the gap to
dense_jepa stays at 40-65×.

## Dynamic-N results (invalid — probe overfitting)

For all `dynamic_64 / target_active ∈ {16, 24, 32}` cells across all
3 entity counts at J=80:

```
visible_mse = 0.0
hidden_mse  = 0.0
```

That is not "perfect memory". That is "the probe memorized the
visible-frame state-to-targets mapping for each training episode,
then the same near-constant state appeared in the hidden frames of
the same episodes, so the probe scored 0 by reading off the
memorized table."

## Root cause

The current probe protocol:

```
for each rollout episode:
  collect (state_t, target_xy_t, is_visible_t) for every frame
train probe on (state, target) pairs from visible frames in ALL episodes
test  probe on (state, target) pairs from hidden  frames in ALL episodes
```

When the encoder produces a near-constant state per episode (e.g.
because most slots are frozen by the dynamic gate), every frame in a
given episode shares essentially the same state vector with shared
ground-truth targets. The probe trivially overfits to a
state-signature → target lookup, and the hidden-frame test is a
within-distribution generalization rather than a memory test.

This works fine when slot state changes across the episode (Phase
2/3/4 / fixed-N here) but breaks when the dynamic gate makes most
of the slot state inert.

## What this rules out

This attempt **does NOT** disprove the dynamic-slot hypothesis. The
mechanism may work perfectly well; the probe protocol is the problem.

Fixed-N comparison confirms the Phase 5A curve holds at 5 seeds.

## Fix for Phase 5B-attempt-2

Hold out a fraction of episodes for the probe test set:

```
collect rollouts from N_total episodes
split episodes into N_train (visible-frame samples for probe fitting)
                 and N_test  (hidden-frame samples for probe eval)
no episode appears in both — eliminates the memorization shortcut
```

Roughly N_train = 80% / N_test = 20%, with N_total scaled so each
side has enough samples. Probe MSE on held-out test episodes is the
true memory metric.

Implementation: ~30 lines in `_collect_probe_rollouts` and
`_train_linear_probe`. Same eval matrix; just rerun.

## Artifacts

```
artifacts/phase5b_attempt1/
  manifest.json
  raw_all.jsonl          300 per-(sub-run × J) rows from 5 seeds
                          dynamic cells have visible_mse=hidden_mse=0
```

The fixed-N rows are valid Phase-5A reproductions; the dynamic rows
are flagged inconclusive.

## Decision

**Phase 5B-attempt-1 INCONCLUSIVE.** Do not lock as pass or fail; log
as a probe-protocol artifact. Fix the probe protocol and rerun
Phase 5B-attempt-2 with held-out episodes. The Phase 2/3/4A/4B/5A
claims are unaffected.
