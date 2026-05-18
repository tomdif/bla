# Phase 18η-multi — value head multi-seed confirmation (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18η at seed 0 showed `combined_sum` beats the locked recipe
by +0.050 (0.318 vs 0.268), with success 0.30 → 0.467, dir_score
0.27 → 0.525. That's a substantial single-seed win, but the BLA arc's
pattern (Phase 17 → 18d) has been: single-seed positive followed by
multi-seed confirmation before locking the architectural claim.

This phase repeats Phase 18η at seeds 1, 2. Seed 0 is already
complete (commit 5ab8ea5).

## The question

> Does `combined_sum` (predictor + value head) beat `phase17_locked`
> in mean across 3 seeds, with consistent per-seed direction?

## Setup

Identical to Phase 18η, at seeds 1 and 2. Same:
- 300 rollout episodes per seed (scripted_prior_light_cem)
- 2000 training steps
- Value head architecture (10-dim geometric features + goal + plan,
  3-hidden 256-dim MLP)
- Phase 16/18β MPC eval framing
- 6 modes × 30 eval episodes
- combined-sum λ = 0.5

Per-seed cost: ~12 min collection + ~2 min training + ~70 min eval
= ~85 min. Two new seeds in parallel on separate GPUs = ~90 min
wall time total.

## Pre-committed gates

```
G1. mean improvement(combined_sum) across 3 seeds
        >= mean improvement(phase17_locked) across 3 seeds + 0.02
        (the +0.02 absolute margin matches the Phase 18η G1 precommit)

G2. mean improvement(combined_sum) >= 0.25 absolute
        (sanity floor: a positive result needs more than just
         beating-by-noise above locked)

G3. Consistency: at least 2 of 3 seeds have combined_sum >=
        phase17_locked (within -0.01 tolerance, same Phase 18d rule)

G4 (diagnostic, not gated).
        mean held-out Spearman across 3 seeds > 0.20 (Phase 18η
        seed 0 was 0.319; expect similar magnitude)
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **3/3 main + G4 positive** | combined_sum is the new locked planning recipe. Update [[bla-locked-planning-recipe]]. Move to Phase 18θ (slot-feature value head). |
| 2/3 main | Real win but fragile. Inspect which gate failed; consider increasing rollout count to ~500/seed. |
| 1/3 | Phase 18η was probably a lucky seed. The value head approach is marginal, not transformative. Investigate why seed 0 was an outlier. |
| 0/3 | Phase 18η was a fluke. Reframe as "value head is data-bound, needs more rollouts and/or richer state representation (Phase 18θ slot features)." |

## What this phase is NOT

- Not a new architecture (value head unchanged)
- Not a new candidate distribution (locked recipe unchanged)
- Not cross-task transfer
- Not the slot-feature value head — that's Phase 18θ

## Reproducibility — pod runbook

Seed 0 is already done (`/workspace/phase18h_main/`).

Run seeds 1, 2 (in parallel on separate GPUs):

```bash
# Seed 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
nohup python3 -u scripts/phase18h_value_head.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 300 --train-steps 2000 \
    --n-eval-episodes 30 \
    --out /workspace/phase18h_seed1 \
    --seed 1 \
    > /workspace/phase18h_seed1/log.txt 2>&1 &

# Seed 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u scripts/phase18h_value_head.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 300 --train-steps 2000 \
    --n-eval-episodes 30 \
    --out /workspace/phase18h_seed2 \
    --seed 2 \
    > /workspace/phase18h_seed2/log.txt 2>&1 &
```

Aggregator script (to be written): `scripts/phase18h_multi_aggregate.py`,
patterned after `scripts/phase18d_aggregate.py`. Reads three
summary.json files, reports per-seed and mean ± std for each mode,
evaluates Phase 18η-multi gates.

## Artifacts

- Per-seed pod: `/workspace/phase18h_seed{1,2}/{summary.json,
  per_episode_*.jsonl, rollout_cache.npz, value_head.pt, log.txt}`
- Per-seed decile diagnostics computed locally after pull
- Repo: `artifacts/phase18h_multi/{aggregate.json, summary_seed1.json,
  summary_seed2.json, decile_seed{0,1,2}.json}`
- Decision doc: `docs/phases/PHASE_18H_MULTISEED_DECISION.md`
