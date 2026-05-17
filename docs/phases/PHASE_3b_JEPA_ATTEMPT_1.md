# Phase 3b (JEPA Track) — Attempt 1 (INCONCLUSIVE)

**Date:** 2026-05-16.
**Status:** ⚠ **INCONCLUSIVE — experimental-design flaw, not a representational failure.**

> Both slot_delta and dense_jepa BC policies hit 0/128 success at
> J=20, 40, and 80. The cause is not the encoder. It's that the env's
> `expert_action` depends on `env.visited` — which targets the agent
> has already collected — and **visited state is hidden from
> observation**. No single-step policy can learn this from any
> encoder.

## What was tested

The plan from the Phase-4B decision doc:

> Freeze slot_delta encoder + dense_jepa encoder, train identical
> policy heads with replay-buffer BC + DAGGER-lite, compare success
> at J=20, 40, 80.

Implementation: `scripts/slot_jepa_phase3b_bc.py`.

- Encoder: frozen from Phase 4B `final.pt` for the (seed=0, K=5,
  n_targets=3, n_distractors=5) sub-run.
- Policy: 2-hidden-layer MLP from encoded state → 2-D action.
- BC: 500 episodes of DAGGER-lite (expert mixing linearly from 1.0
  → 0.0). Replay buffer 50K (state, expert_action) pairs.
- Same env stress flags as Phase 4B (moving distractors + partial
  observability + perceptual noise + colour/background randomization).
- Eval: 128 episodes per J at J=20, 40, 80.

## Result

```
slot_delta:          J=20: 0/128  J=40: 0/128  J=80: 0/128
dense_jepa_flatten:  J=20: 0/128  J=40: 0/128  J=80: 0/128
```

BC loss converged to ~2.0 (slot_delta) and ~3.0 (dense_jepa). Both
policies output something; neither solves the env.

## Why this is not a representational failure

The multi-target navigate env's expert policy is:

```python
def expert_action(self):
    dists = self._distances_to_targets()
    masked = dists.masked_fill(self.visited, float("inf"))
    target_idx = masked.argmin(dim=1)
    return target_position - agent_position
```

`self.visited` is a hidden bool vector tracking which targets have
already been collected. **A single-frame observation never carries
this information** — it shows all targets in identical colour, and
the agent must remember which ones it already touched.

A single-step policy `π(obs) → action` therefore cannot match the
expert *even with a perfect encoder*. The information needed
(visit history) simply isn't in the policy's input.

The previous Phase-2 BC attempt failed for the same reason; we
diagnosed it then as "BC distribution shift", but the deeper cause
is the same hidden-state issue. DAGGER-lite doesn't fix it because
DAGGER also passes only the current observation to the policy.

## What this *does* show

- The probe-level evidence from Phase 3 and 4 remains valid: the
  slot representation encodes target positions well, including
  through occlusion. That's the representational claim.
- A *single-step* policy on top of that representation is the wrong
  policy class for this env. The encoder isn't the bottleneck.

## Fixes for Phase 3b-attempt-2

Three honest options, in increasing difficulty:

1. **Add visited history to the policy input.** Concatenate a small
   action-history vector (last K actions, or last K rewards) to the
   state. The policy then has the info needed to infer visited
   targets. Cheapest fix; ~30 lines of code.
2. **Recurrent policy.** Replace the 2-layer MLP with an LSTM / GRU
   that maintains its own hidden state across the episode. Standard
   approach for partially observed MDPs. ~50 lines.
3. **Change the success metric to be observation-computable.** E.g.,
   "reach any target" instead of "visit all 3 in any order". Loses
   the memory-test angle of the env. Not recommended.

Option 1 or 2 should be Phase 3b-attempt-2. Option 1 is simpler
and would already be a strong test of whether the slot representation
helps once the visit-state bottleneck is removed.

## Artifacts

```
artifacts/phase3b_attempt1/
  slot_delta_bc_eval.json    final JSON with all-zero success
  slot_delta.log             full BC training trace
  dense_jepa_bc_eval.json
  dense_jepa.log
```

The training traces (BC loss curves) ARE informative: slot_delta loss
plateaued ~30% lower than dense_jepa, which is a weak hint that the
slot state was easier to map to expert actions. But the policy-class
mismatch dominates that signal.

## Decision

**Phase 3b-attempt-1 INCONCLUSIVE.** Do not lock as a pass or fail —
log as a failed experimental design, move to attempt-2 with a
history-aware policy. The Phase 2/3/4A/4B representational claims
are unaffected.
