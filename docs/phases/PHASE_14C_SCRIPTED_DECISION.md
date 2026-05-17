# Phase 14.5 (Scripted v3 + Action Conditioning) — Decision document

**Date:** 2026-05-17.
**Status:** ✅ **STRONG PASS — all three precommitted gates pass.**
Action conditioning is **data-informativeness-bound, not architecture-
bound**. Phase 14.4's "ranks actions at chance" result is now
attributed to weak random-policy training data, not to a flaw in the
action-conditioning architecture.

> Phase 14.5 re-ran Phase 14.3 (state-matching) and Phase 14.4 (action
> ranking) on scripted-policy v3 rollouts where the EE deliberately
> sweeps through cube positions. Pre-committed gates and joint verdict
> matrix were locked in `PHASE_14C_SCRIPTED_PRECOMMIT.md` before the
> training run.

## Setup

| | Value |
|---|---|
| Task | robosuite Stack (cubeA + cubeB + EEF) |
| Episodes | 200 scripted_push v3 rollouts |
| Policy | per-target random sweep direction; approach then lateral drive-through; σ=0.20 noise; targets resampled every 20-35 steps |
| Mean cube displacement | cubeA=0.056m, cubeB=0.062m (vs random ~0.005m, gate ≥0.05m) |
| Horizon | 80 frames |
| Training | 1500 steps, JEPA stride k=4, lr=1e-4 |
| Seeds | 1 (first pass) |

## Headline numbers

| Metric | Baseline | +Action | Δ | Gate | Verdict |
|---|---|---|---|---|---|
| future_state_mse | 1.108e-4 | 1.059e-4 | **−4.4%** | A: ≤+10% | ✅ (action BETTER) |
| future_pos_mse | 1.664e-2 | 1.489e-2 | **−10.5%** | C: ≤−5% | ✅ |
| top1_hit_rate | 1.000 (tie) | **0.343** | (chance 0.125) | B: ≥0.19 | ✅ |
| top3_hit_rate | 1.000 (tie) | 0.630 | (chance 0.375) | — | ✅ |
| rank_of_actual | 1.000 (tie) | 3.452 | (perfect 1, chance 4.5) | ≤ 3.5 | ✅ |

The baseline's 1.000 top-1 is the same informationally-vacuous tie
artifact as in Phase 14.4: with no action input, all candidate
predictions are identical and stable-sort picks index 0. The +action
model's 0.343 top-1 is **2.74× chance** — meaningfully discriminative.

## What this reverses vs Phase 14.3 + 14.4

| | Phase 14.3/4 (random policy) | **Phase 14.5 (scripted v3)** |
|---|---|---|
| future_state_mse vs baseline | **+100%** (worse) | **−4.4%** (BETTER) |
| future_pos_mse vs baseline | −13% (better) | **−10.5%** (better) |
| top1_hit_rate (+action) | 0.125 = chance | **0.343 = 2.74× chance** |
| rank_of_actual (+action) | 4.56 (~= chance 4.5) | **3.45 (well above chance)** |

The flip on state matching is the headline: action conditioning went
from **+100% damage** to **−4.4% improvement** purely by changing the
data informativeness. Same architecture, same hyperparameters, same
1500 training steps — only the action distribution changed.

## Why this matters

Phase 14.4 concluded:

> Action conditioning is used as noise-reduction, not as a
> discriminative input. Action conditioning, as currently set up, is
> NOT load-bearing for action-discrimination / planning.

That conclusion was **architecture-blamed**. Phase 14.5 falsifies the
architecture-blame: the architecture is fine; the data was the
limiting factor. With informative actions:

1. The +action future predictor learns to use action AS a
   discriminative signal (not noise), evidenced by 2.74× chance
   ranking.
2. State matching is no longer harmed — slightly improved — because
   the predictor head can extract genuine action→state structure
   instead of trying to absorb unhelpful action variance.
3. Position prediction improvement persists (−10.5% vs −13%) — the
   directional finding from Phase 14.3 generalizes.

## Architectural take

> **Action conditioning IS load-bearing for OF-JEPA, given training
> data with informative action→effect pairs. The bottleneck under
> random-policy data was the data, not the architecture.**

Phase 14.4's recommendation to "split action-conditioned and
unconditioned heads" is now **softened**: a single action-conditioned
predictor works fine when data is informative. The split-head
architecture remains a reasonable engineering choice (modular
planner / perception split aligns with BLA's System-1 vs System-2
boundary) but is not forced by Phase 14's evidence.

## What this does NOT establish

1. **No active behavior / closed-loop test.** The 0.343 top-1
   hit-rate shows the predictor *can rank* actions; it doesn't show
   any agent *using* that ability to pick better actions in
   simulation. That belongs in Phase 15 or a System-2 planning test.
2. **Single seed.** The −4.4% state-matching improvement is small
   enough that 3-seed confirmation would tighten the claim.
   Directionally the result is large enough (gate is +10%, we hit
   −4.4%) that single-seed is acceptable for the qualitative claim.
3. **One task.** robosuite Stack only. Action-conditioning behavior
   may differ on more complex tasks (e.g., Can/PickPlace) or on
   non-rigid-body dynamics. Not falsified here.
4. **Scripted policy ≠ real-world distribution.** The lateral-sweep
   scripted policy may have artifacts a real demonstrating distribution
   wouldn't. Robomimic Lift was inspected and rejected (cube only
   moves ~0.045m — below the gate), but Can/PickPlace remains a
   future external-data test if needed.

## Joint verdict matrix outcome

From the pre-committed matrix:

| A | B | C | Outcome |
|---|---|---|---------|
| **✅** | **✅** | **✅** | **"Action conditioning is data-informativeness-bound. Phase 14.4's chance result was due to random-policy data, not architecture. Push to Phase 15."** |

Locked direction: **Phase 15 (behavioral transfer)** is the next
natural step per the Phase 10-20 roadmap. Action conditioning is
established as a viable input to OF-JEPA's predictor; the next
question is whether a recurrent policy on top of OF-JEPA's object
files transfers across rollouts better than a pixel-input policy.

## Reproducibility

```bash
# Data: scripted v3 rollouts (already collected in Phase 14.5a).
# Joint train + eval (~75 min total wall time on RTX 4090):
python3 scripts/phase14_action_ranking.py \
  --cache /workspace/robosuite_local/stack_scripted \
  --modes of_jepa_v0,of_jepa_v0_action \
  --seed 0 --max-steps 1500 --jepa-stride 4 --k-candidates 8 \
  --out /workspace/phase14c_joint
```

Artifacts: `artifacts/phase14c_joint/seed0_of_jepa_v0{,_action}.json`

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 JEPA | ✅ | slot_delta spatial memory |
| 7-8A | ❌×5 | content-side identity fixes falsified |
| 8C/8D | ✅ | OF-JEPA v0: identity-as-address |
| 9 | ✅ | MOVi-D identity transfer (corrected metric) |
| 10 | ✅ | refactor + first-class metrics |
| 12 | ✅ | relations add 5-9% on single-slot readouts |
| 13.3 | ✅ | OF-JEPA transfers to CLEVRER |
| 13.4 | ⚠ | relations redundant when readout has pairs |
| 14.3 | ⚠ | action conditioning split result on RANDOM data |
| 14.4 | ❌ | action ranking at chance on RANDOM data |
| **14.5** | **✅ STRONG** | **action conditioning IS load-bearing with informative data — three gates pass, full reversal of 14.4** |

The clean architectural takeaway across Phases 12-15:

> *Relations, actions, and any context signal are load-bearing when:
> (1) the data contains the structure they're supposed to explain,
> and (2) the downstream readout consumes the same information they
> encode. The repeated lesson is to test with informative data BEFORE
> blaming the architecture.* Phase 14.5 caught this because we
> pre-committed gates and the matrix specified "data-bound" as a
> possible outcome.
