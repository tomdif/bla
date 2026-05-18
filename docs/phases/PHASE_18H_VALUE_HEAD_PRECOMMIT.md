# Phase 18η — Goal-progress value head (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — design, training data, and gates
locked before any training run.**

## Why this phase exists

Phase 18γ's three-variable decomposition shows the BLA System-1
planning stack's binding constraint is **episode-level goal
compounding**, not local rank quality or candidate quality. D5
proved local rank can be positive (+0.015) while episode quality
fails (0.154 vs D2's 0.244). The current one-step OF-JEPA predictor
scores `(state, action_sequence) → immediate cube motion`. It does
NOT score `does this local choice compound toward the goal over the
remaining MPC trajectory?`.

Phase 18η adds that capability as a **value head**: a small network
that takes (state, goal, action_sequence) and predicts the
full-episode realized improvement — what a Q-value-ish quantity
would tell us before committing to an action plan.

## The question

> Does adding a goal-progress / multi-step value head on top of
> the existing OF-JEPA dynamics predictor improve episode-level
> planning performance over the Phase 18β locked recipe
> (`scripted_prior + light CEM`, improvement = 0.244)?

## Architecture

```
inputs:
  - OF-JEPA slot features at the replan boundary
    (frozen Phase 17 encoder, B × n_slots × slot_dim)
    OR fallback: 10-dim geometric features (cube_xy, eef_xyz,
    cube_z, goal_xy, push_dir) per Phase 18β
  - goal_xy (B × 2)
  - candidate action sequence (B × plan_horizon × action_dim)
                                  = B × 10 × 7

value head V(s, g, a):
  - Concatenate flattened slot features + goal_xy + flattened actions
  - 2-3 hidden layers, 256-dim, ReLU
  - Linear output → scalar predicted episode improvement
  - NO output activation (regression target ∈ [0, 1])

frozen substrate:
  - OF-JEPA encoder (Phase 17)
  - One-step OF-JEPA dynamics predictor (Phase 17)
  - Goal-conditioned scripted prior (Phase 18β/locked recipe)

new trainable:
  - value head only
```

**Primary input choice: slot features.** The BLA System-1 thesis is
that object-file slots are the right state representation for
planning; 18η is a clean test of whether they carry the goal-progress
signal a value head needs. If slot-features V doesn't beat baseline,
fall back to 10-dim geometric features as a sanity check (rules out
"slots are wrong representation" vs "value-head approach is wrong").

## Training data

**Source**: Run `scripted_prior_light_cem` (Phase 18β locked recipe)
for ~300 episodes with same setup as Phase 18β:

```
30 ep × 10 seed-perturbed configs ≈ 300 episodes total
total_actions=15, replan_every=5, plan_horizon=10
CEM K=32, 1 iter, σ=0.12
```

**Sample collection**: At each replan boundary of each episode,
record:

```python
{
    "slot_features": ...,          # B × n_slots × slot_dim
    "geometric_features": ...,     # B × 10  (fallback input)
    "goal_xy": ...,                # B × 2
    "action_plan": ...,            # B × plan_horizon × action_dim
    "label_episode_imp": ...,      # scalar, FULL-EPISODE realized
                                    # improvement of this episode
}
```

Per episode: 3 replan boundaries (executed actions 0, 5, 10), each
gets the SAME `label_episode_imp` (the whole episode's final
improvement).

Total training data: ~300 episodes × 3 replans ≈ 900 examples.
Same label across replans within an episode is *intentionally
correlated training data* — this teaches the value head "the
state-and-plan at this replan boundary co-occurred with an episode
that achieved improvement X." Replans differ in state/plan, share
the trajectory outcome.

**Validation split**: 80/20 within-episode (random episode held-out
on each fold), use full-episode improvement as the regression target.

## Loss

```
L = MSE(V(s, g, a), label_episode_imp)
```

Plain regression. Episode improvements ∈ [0, ~0.5] in the locked
recipe (mean ~0.24). Simple MSE first; if calibration is poor, can
move to Huber.

## Training hyperparameters (locked)

- 2000 steps, AdamW lr=3e-4, weight_decay=1e-4
- Batch size 64
- Hidden 256, 3 hidden layers, dropout 0.0
- Gradient clip 1.0
- Seed 0 (with seed 1, 2 multi-seed in Phase 18η-multi if Phase 18η
  passes)

## Evaluation

Same Phase 18β / 16 MPC framing, 30 episodes:

| Mode | Scoring function |
|---|---|
| `oracle_gt_closed_loop` | (no scoring, oracle baseline) |
| `phase17_locked` (current best) | OF-JEPA one-step predictor |
| `value_head_only` | New value head |
| `combined_max` | `max(predictor_score, value_head_score)` |
| `combined_sum` | `λ * predictor + (1-λ) * value_head`, λ=0.5 |
| `naive_cem` | (no scoring, μ=0 floor) |

For combined modes, both scores normalized to z-scores before
combination.

## Pre-committed gates

```
G1. value_head_only improvement ≥ 0.244 + 0.02 (= 0.264)
       (matches or beats locked recipe by ≥ 2pp absolute, ~8% relative)
       → Pass means slot-based value head replaces the one-step
         predictor as the primary scoring function.

G2. best_combined_mode improvement ≥ 0.244 + 0.04 (= 0.284)
       (combined head beats locked recipe by ≥ 4pp absolute, ~16%)
       → Pass means value head adds complementary signal to the
         existing predictor.

G3. value head's per-replan top_vs_bot_gap > +0.05 on a held-out
    18γ-style audit (D2 distribution, 20 states, M=64)
       → Sanity check that the value head IS what it claims to be:
         a positive-rank-quality predictor, not just a noisy regressor.

Diagnostic (not gated):
G4.  value_head_only improvement vs phase17_locked per-seed:
       compute the gap distribution across seeds 0, 1, 2 if
       multi-seed is run.
G5.  Pearson correlation of value head's predicted-improvement
       vs realized episode improvement on held-out 60 evals.
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **G1 + G2 + G3** | **Value head replaces or augments predictor.** Move to multi-seed (Phase 18η-multi) before locking. |
| G1 + G3 | Value head wins alone but combination doesn't help. The value head captures everything the predictor did + more. Drop the one-step predictor from planning. |
| G2 + G3 | Value head is complementary but weak alone. Keep both as combined. |
| G3 only | Value head IS calibrated (positive rank) but doesn't help episode-level. Suggests the audit's per-replan signal isn't load-bearing — the failure is elsewhere (maybe action capacity, not scoring). |
| 0/3 | Value head approach is wrong direction. Reconsider; possibly the failure isn't scoring at all and is purely in the candidate prior (which would point back to a deeper learned policy). |

## What this phase is NOT

- Not a new planner architecture (CEM, MPC, prior all unchanged)
- Not a re-training of OF-JEPA (frozen)
- Not a new candidate distribution (uses locked recipe)
- Not yet multi-seed (single-seed Phase 18η; multi-seed Phase
  18η-multi only if 18η passes)

## Implementation sketch

Files to write:

- `system1_jepa/value_head.py` — `GoalProgressValueHead` module +
  `train_value_head_supervised` + `combine_scores` helper.
- `scripts/phase18h_value_head.py` — main script:
  1. Load Phase 17 model + scripted prior setup.
  2. Collect ~300 locked-recipe rollouts, log per-replan-boundary
     (slot_features, geometric_features, goal_xy, action_plan,
     label_episode_imp).
  3. Train value head (2000 steps).
  4. Eval 6 modes × 30 eps × seed-0, compute G1/G2/G3 gates.
- `tests/test_value_head.py` — shape, training-loss-decreases on
  synthetic data, score combination.

Reuse:
- Phase 18β `phase18b_policy_distill.py` for episode collection +
  feature extraction pattern
- Phase 16 `cem_with_prior`
- Phase 17 OF-JEPA loading

Pod budget:
- Collection: 300 episodes × ~25s each = ~2 hours
- Training: ~5 min
- Eval: 6 modes × 30 ep × ~15s = ~45 min
- Total: ~3 hours single-GPU seed-0

## Reproducibility

Precommit: this file.
Decision doc: `docs/phases/PHASE_18H_VALUE_HEAD_DECISION.md` (to be
written after run).
Artifacts: `artifacts/phase18h/{summary.json, per_episode_*.jsonl,
audit.json}`.

Pod run:

```bash
python3 scripts/phase18h_value_head.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 300 \
    --train-steps 2000 \
    --eval-episodes 30 \
    --out /workspace/phase18h \
    --seed 0
```

## Sibling memory

This phase implements the lever locked in
`[[bla-next-architectural-lever]]`. The locked planning recipe
remains the data-source and baseline per
`[[bla-locked-planning-recipe]]`. Evaluation discipline follows
`[[per-state-vs-per-episode-metrics]]` (episode is the gate,
per-replan is diagnostic) and
`[[rank-vs-candidate-quality-orthogonal]]` (don't confuse the three
axes).
