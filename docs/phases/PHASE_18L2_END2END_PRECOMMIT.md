# Phase 18λ-v2 — End-to-end adapter + value head (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18λ-multi established that:
- The supervised geometry adapter robustly recovers the value-
  relevant subspace from OF-JEPA slots (mean Spearman 0.506 across
  3 seeds).
- The adapter value head reaches 85% of the engineered-geo planner
  on average, with high seed-batch variance.

The bottleneck is *not* the representation: the geometric subspace
is recovered. The bottleneck is **value calibration / planner
integration** — the supervised loss (slot → 10-dim engineered geo)
doesn't directly optimize for episode-level value-prediction
quality.

Phase 18λ-v2 replaces the intermediate-geometry MSE objective with
**end-to-end MSE on episode improvement**. The adapter is no longer
forced to reconstruct engineered features; it's free to find a
10-dim latent that maximizes value prediction.

## The question

> Does end-to-end adapter+value training (loss directly on episode
> improvement, no intermediate geometry MSE) produce a planner that
> matches the engineered-geo recipe (`combined_sum_geo`) or beats
> the supervised-adapter baseline (`combined_sum_supervised` ≈
> 0.207 mean)?

## Three heads, one run

Each seed trains **three value heads** on the same cached rollouts
(720 train / 180 val):

| Head | State input | Training loss | Source |
|---|---|---|---|
| **geo** | engineered 10-dim geometry from simulator | MSE on episode_imp | Phase 18η reference |
| **supervised** | adapter(slot, goal) output, where adapter is trained slot→engineered_geo with MSE | MSE on episode_imp (with frozen adapter) | Phase 18λ recipe |
| **end2end** | adapter(slot, goal) output, where adapter+VH is trained jointly | MSE on episode_imp (joint) | NEW — Phase 18λ-v2 |

All three have **10-dim state input** (latent_dim = 10 = engineered
geo dim) for fair architectural comparison. Only the training
objective changes.

This decomposition isolates four failure modes:

- *Representation problem*: would show as supervised < geo on
  per-seed held-out Spearman + decile gap.
- *Adapter bottleneck (geometry-MSE objective wrong)*: would show
  as supervised plateaus but end2end exceeds. **This is the
  hypothesis under test.**
- *Value objective problem*: end2end also fails to clear geo. The
  10-dim bottleneck might be wrong.
- *Planner integration problem*: even end2end stays below geo at
  the planner level despite stronger held-out signal.

## Pre-committed gates

```
G1. mean end2end VH Spearman across 3 seeds >= 0.25
       (matches supervised gate; verify end2end produces
        rankable signal at least as good as supervised)

G2. mean end2end VH top/bot ratio across 3 seeds >= 2.0×
       (matches supervised; elite-pick quality is intact)

G3. mean(combined_sum_end2end) across 3 seeds >= 0.90 × mean(combined_sum_geo)
       (closer to geo than supervised's 85%; if end2end can't
        clear this, end-to-end training didn't help)

G4. mean(combined_sum_end2end) >= mean(combined_sum_supervised)
       (the end2end must beat the supervised baseline — that's
        the real test of "joint training vs intermediate geo")

Stretch (not gated):
G5. end2end beats phase17_locked on at least 2/3 seeds
       (the value-combined recipe robustness gate — kept off
        the main pass criterion because the recipe itself has
        shown seed-batch sensitivity)
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3 + G4** | end2end matches engineered-geo planning. The adapter no longer needs hand-engineered features. BLA System-1 → System-2 → planner is the locked architecture. Move to multi-seed lock + cross-task. |
| G1 + G2 + G4 (G3 fail at e.g. 87%) | end2end better than supervised but still slightly below geo. Continued progress; consider wider latent (32 or 64-dim) in v3. |
| G4 only | end2end beats supervised but still below 90% of geo. Intermediate progress; investigate joint training stability. |
| 0/4 | End-to-end training does not help. The bottleneck is elsewhere — possibly the 10-dim latent dim, possibly the linear-MSE objective, possibly the dataset size. |

## Architecture

```python
End2EndAdapterValue:
  adapter = ObjectFileGeometryAdapter(
    slot_dim=768, goal_dim=2, out_dim=10,        # SAME 10-dim bottleneck as geo
    hidden=256, n_hidden=3,
  )
  value_head = GoalProgressValueHead(
    state_dim=10, action_dim=7, plan_horizon=10,
    hidden=256, n_hidden=3,
  )
  
  forward(slot, goal, plan):
    latent = adapter(slot, goal)        # [B, 10]
    return value_head(latent, goal, plan)  # [B]

loss = MSE(end2end_output, episode_imp_label)
```

Total params: adapter (~250K) + VH (~150K) = ~400K.
Training data: 720 samples, 2000 steps, AdamW lr=3e-4. Same val
split as Phase 18λ.

## Setup

- Reuses Phase 18λ-multi caches (`/workspace/phase18l_seed{0,1,2}/
  rollout_cache.npz`) — no new collection.
- All three heads trained from scratch per seed (geo head, supervised
  adapter+VH, end2end joint).
- Same 6-mode eval per seed (gt, locked, geo, supervised, end2end,
  naive).

## Implementation sketch

New module additions:
- `system1_jepa/geometry_adapter.py`:
  - `End2EndAdapterValue` (composes adapter + value head)
  - `train_end2end_supervised` (joint MSE on episode_imp)

New script: `scripts/phase18l2_end2end.py`:
- Per-seed loop: load cache, train 3 heads, eval 6 modes, save
  summary.
- Multi-seed aggregate: per-seed summaries → aggregate.json.

Pod runbook (3 seeds in parallel on GPUs 0/1/2):

```bash
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$SEED \
  nohup python3 -u scripts/phase18l2_end2end.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-cache /workspace/phase18l_seed${SEED}/rollout_cache.npz \
    --n-eval-episodes 30 \
    --out /workspace/phase18l2_seed${SEED} \
    --seed $SEED &
done
```

(Seed 0's cache is at `/workspace/phase18l_main/rollout_cache.npz`
in the original 18λ run; for consistency the 18λ-multi
re-collection at `/workspace/phase18l_seed0/` doesn't exist. We can
use either — the 18λ-main cache for seed 0.)

Estimated cost (cache reused):
- 3 head trainings × ~3 min = ~10 min
- Eval 6 modes × 30 ep × ~15s = ~45 min
- **Total per seed: ~55 min** in parallel = ~55 min wall.

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18L2_END2END_DECISION.md`
- Artifacts: `artifacts/phase18l2_multi/{aggregate.json,
  summary_seed{0,1,2}.json, decile_*.json}`

## Sibling memory

- `[[bla-locked-planning-recipe]]` — `combined_sum_geo` recipe
  remains locked unless 18λ-v2 supplies G1+G2+G3+G4
- `[[value-relevant-subspace-recoverable-from-slots]]` — 18λ-multi
  lesson; 18λ-v2 tests whether end-to-end training closes the
  remaining gap
- `[[value-head-complementary-to-dynamics]]` — Phase 18η lesson;
  same combined_sum framing applies
- `[[frozen-slots-not-enough-for-value]]` — Phase 18θ; reframed by
  18λ-multi result that slot features DO contain the relevant
  subspace
