# Phase 18δ (Phase 17 multi-seed confirmation) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 17 was a major positive: model-based stack exceeded oracle
(0.235 > 0.224) at seed 0. But it's the first time in the BLA arc
that the planning stack beats its baseline, and the margin is meaningful
but not enormous (+5%). Before building Phase 18β (policy distillation)
on top of this finding, we need to know the result is stable across
seeds, not a single-seed artifact.

## The question

> Does Phase 17's "OF-JEPA + competent prior + CEM-refinement exceeds
> oracle" finding hold across 3 seeds, not just seed 0?

## Setup

Identical to Phase 17, repeated at seeds {0, 1, 2}:
- Same 200-episode goal-directed-push data (reused; cache is seed-
  independent for collection RNG vs training RNG)
- Same v3 broad-scripted cache
- Same 50/50 mixed training (1500 steps, lr=1e-4)
- Same Phase 16 MPC eval (30 episodes, 4 CEM iters, K=128)
- Modes: gt_closed_loop, naive_cem, scripted_prior_cem

Seed 0 is already complete (Phase 17). Phase 18δ runs seeds 1, 2.

Per-seed cost: ~25 min training + ~50 min eval = ~75 min. Plus
collection (~10 min, but data already collected and reused). 2 new
seeds = ~150 min wall time. Then aggregation.

## Pre-committed gates

```
G1. mean improvement(scripted_prior_cem) across 3 seeds >= mean
    improvement(gt_closed_loop) across 3 seeds
       (planner-stack mean beats oracle mean)

G2. mean improvement(scripted_prior_cem) >= 0.18
       (matches the Phase 17 absolute threshold, unchanged)

G3. mean dir_score(scripted_prior_cem) >= 0.90 × oracle dir_score
       (within 10% of oracle directional quality, or better)

G4 (diagnostic, not gated). mean pred-actual corr > 0
       (predictor calibration trending positive, vs Phase 16's -0.387)
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **3/3 main + G4 positive** | **Phase 17's planner-beats-oracle is robust.** Lock the architectural conclusion; move to Phase 18β (policy distillation). |
| 2/3 main | Partial confirmation. Phase 17 was probably real but margin is fragile. Inspect which gate failed; possibly need to widen perturbation in training data. |
| 1/3 | Phase 17 was lucky seed. The model-based stack is competitive with oracle but not reliably above it. Reframe as "matches oracle" not "beats." |
| 0/3 | Phase 17 was a fluke. Strong negative; reconsider the architectural conclusion. |

## "At least 2/3 seeds positive vs oracle"

Additional soft constraint: at least 2 of 3 seeds should have
scripted_prior_cem improvement >= oracle improvement (or within seed-
noise). This catches the case where mean beats oracle only because
one seed had an outlier.

## What this does NOT establish

- Cross-task transfer (Phase 18α).
- End-to-end policy distillation (Phase 18β).
- Longer-horizon planning (Phase 18γ).

## Reproducibility — pod runbook

```bash
# Seed 0 already done. Run seeds 1 + 2:
for SEED in 1 2; do
  python3 scripts/phase17_finetune.py \
    --train-caches /workspace/robosuite_local/stack_scripted,\
/workspace/robosuite_local/stack_goal_directed \
    --train-mix 0.5,0.5 --max-steps 1500 --jepa-stride 4 --seed $SEED \
    --model-out /workspace/phase18d/model_seed${SEED}.pt
  python3 scripts/phase16_policy_prior_mpc.py \
    --model-action /workspace/phase18d/model_seed${SEED}.pt \
    --seed $SEED --jepa-stride 4 \
    --total-actions 15 --replan-every 5 --plan-horizon 10 \
    --cem-iters 3 --main-K 128 \
    --modes gt_closed_loop,naive_cem,scripted_prior_cem \
    --n-episodes 30 --oracle-sanity-n 30 \
    --oracle-min-improvement 0.10 --oracle-min-contact 0.60 \
    --bc-episodes 0 \
    --out /workspace/phase18d/eval_seed${SEED}
done

# Aggregate (Phase 17's seed-0 results + new seed-1 + seed-2):
python3 scripts/phase18d_aggregate.py \
  --seed-0 /workspace/phase17_eval \
  --seed-1 /workspace/phase18d/eval_seed1 \
  --seed-2 /workspace/phase18d/eval_seed2
```
