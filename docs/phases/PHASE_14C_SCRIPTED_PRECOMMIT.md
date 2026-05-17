# Phase 14.5 (Scripted-policy rollouts) — Pre-commit gate

**Date:** 2026-05-17.
**Status:** ⏳ **PRE-COMMITTED — locks gates before run.**
**Why this doc exists:** [Calibrate before hash](../../).
Pre-committing the success criteria for Phase 14.5 BEFORE running the
experiment, so the result is interpretable as PASS / FAIL rather than
post-hoc rationalized.

## The question

Phase 14.3 found that adding action conditioning to OF-JEPA's future
predictor **hurts JEPA state-matching (+100%)** and only weakly helps
**decoded position (-13%)**. Phase 14.4 found that action conditioning
gives chance-level action-ranking (top-1 = 0.125 = 1/K). Both were run
on **uniform-random-policy rollouts**.

Hypothesis: random-policy data has weak action→effect signal — random
arm motions rarely contact the cubes. The model can't learn an
action-discriminative map from sparsely-populated supervision. **Phase
14.5 tests whether informative actions fix this**, by re-running 14.3
+ 14.4 on scripted-push rollouts where the robot deliberately moves
toward cubes.

## Setup

| | Value |
|---|---|
| Task | robosuite Stack |
| Episodes | 200 |
| Horizon | 80 frames |
| Image | 128×128 RGB |
| Policy | **scripted_push** (random target ∈ {cubeA, cubeB}, OSC delta, σ=0.3 Gaussian noise, gripper oscillation) |
| Action dim | 7 (same as 14.3) |
| Training | 1500 steps, k=4 stride, lr=1e-4 |
| Seeds | 1 (first pass) |

## Pre-committed expected diagnostics

If `scripted_push` works as designed:

- **Mean cube displacement per episode ≥ 0.05 m** (vs random-policy
  baseline likely ≈ 0.005 m). This is the dataset-informativeness
  smoke test; if cubes still don't move, the scripted policy itself
  is broken and 14.5 results are not interpretable.

## Pre-committed gates

### Gate A — JEPA state matching no longer broken by action input

```
future_state_mse (+action) / future_state_mse (baseline) ≤ 1.10
```

i.e. action conditioning costs ≤ 10% in state MSE. Phase 14.3 saw
**+100%**. This gate is loose because some cost is fine; what we want
to rule out is "action input dominates and harms state matching."

**Interpretation if PASS:** the +100% in Phase 14.3 was data-driven,
not architectural.
**Interpretation if FAIL:** action input destabilizes self-loop JEPA
even when informative — confirms 14.3's architectural conclusion that
action belongs in planner readouts, not the perceptual loop.

### Gate B — Action-ranking better than chance

```
top1_hit_rate (+action) ≥ 0.19    (chance = 1/8 = 0.125, threshold = 1.5× chance)
```

Equivalent in rank-of-actual:

```
rank_of_actual (+action) ≤ 3.5    (chance = 4.5)
```

**Interpretation if PASS:** action conditioning learns a
discriminative action→effect map when data is informative. Action
*is* load-bearing for planning, just needed informative training data.
**Interpretation if FAIL:** action-conditioning architecture itself is
the bottleneck (not data informativeness). Motivates Phase 14.6
(actor-bound single-slot conditioning) as the next test.

### Gate C — Decoded-position improvement persists

```
future_pos_mse (+action) / future_pos_mse (baseline) ≤ 0.95
```

Phase 14.3 saw 0.87 (−13%). This gate confirms the directional finding
holds under scripted-policy data and isn't a random-policy artifact.

**Interpretation if PASS:** decoded position benefit is real and
generalizes across action distributions.
**Interpretation if FAIL:** the −13% in 14.3 was a random-policy
quirk; weakens the "action helps planner readouts" claim.

## Joint verdict matrix

| A | B | C | Verdict |
|---|---|---|---------|
| ✅ | ✅ | ✅ | Action conditioning is data-informativeness-bound. Phase 14.4's chance result was due to random-policy data, not architecture. Push to Phase 15. |
| ✅ | ❌ | ✅ | State matching tolerated; ranking still broken. Architecture bottleneck. Run Phase 14.6 (actor-bound conditioning). |
| ❌ | ❌ | ❌ | Random-policy was not the problem. Action-conditioning is just additive noise to the perceptual loop regardless of data. 14.3 + 14.4 conclusions stand verbatim. |
| ✅ | ✅ | ❌ | Mixed. The ranking improvement isolates planner-facing benefit; the lost pos improvement says random-policy was a confound. Plausible "action only good for ranking" verdict. |

## Robomimic Lift inspection (rejected as alternative source)

After v3 passed the smoke gate, we briefly considered pivoting to
externally-grounded robomimic Lift PH demonstrations as a cleaner
"non-hand-tuned" data source. **Inspection revealed Lift PH demos
displace the cube only ~0.045-0.048m on average** — *below* the
precommitted 0.05m gate and *less* than scripted v3 (0.056-0.062m).

Sample over 20 demos:
```
demo_0 (T=59):  cube z 0.831 → 0.876 = Δ 0.045 m
mean over 20 demos: 0.048 m  max 0.055  min 0.040
```

The Lift task's success criterion is *"cube is lifted off the table by
some small height"* — humans don't bother lifting it high. So while
robomimic PH demos would have provided cleaner external framing, they
would have been *weaker data* for this specific action-conditioning
test. Can/PickPlace would give larger displacements but requires
rewriting the dataset class (4 objects, different state schema, new
eval path) — substantial plumbing for marginal methodology gain.

**Decision: scripted v3 is the disciplined choice. Phase 14.5 proceeds
on v3 data.** Can/PickPlace remains a future external-data target if
Phase 14.5 results require a follow-up sanity check.

Architecturally, the deeper answer to "don't hand-tune the policy" is
[[project_bla_roadmap]]'s System-2-driven exploration where curiosity
or planning generates informative data, not scripted heuristics. That
belongs in a later phase, not retrofitted into Phase 14.5.

## Reproducibility — pod runbook

```bash
# 1. Collect 200 scripted-push (v3 drive-through) rollouts.
# Policy: per-target random sweep direction, approach + lateral
# drive-through phase, target re-sampled every 20-35 steps. v3 hit
# 0.056/0.062m mean displacement after v1/v2 iterations.
python3 scripts/robosuite_collect_rollouts.py \
  --task Stack --n-episodes 200 --horizon 80 \
  --policy scripted_push \
  --out /workspace/robosuite_local/stack_scripted

# Smoke test: manifest's cube_a_disp_mean + cube_b_disp_mean should
# both be ≥ 0.05 m. v3 produces ~0.056/0.062m.

# 2. Joint train + eval (state-matching + action-ranking) in one
# ~90-min run. The patched phase14_action_ranking.py also calls
# eval_future_mse so a single trained model yields all three gates.
python3 scripts/phase14_action_ranking.py \
  --cache /workspace/robosuite_local/stack_scripted \
  --modes of_jepa_v0,of_jepa_v0_action \
  --seed 0 --max-steps 1500 --jepa-stride 4 --k-candidates 8 \
  --out /workspace/phase14c_joint
```

Artifacts then pulled to `artifacts/phase14c_{state,ranking}/`. Decision
doc to be written as `PHASE_14C_SCRIPTED_DECISION.md` after results, with
results matched against the gates locked in this doc.

## Why pre-commit

History: at Pattern 13 we hashed precommit thresholds without first
measuring the pipeline noise floor, and the threshold turned out to
sit below the noise. Lesson saved in
[threshold_calibration_retrospective](../../). Here the thresholds
(1.10×, 0.19 top-1, 0.95×) are calibrated against:
- 1.10× state MSE: Phase 14.3 baseline-vs-self seed noise is ≈ 5%
  across reseeded runs, so 10% is 2σ.
- 0.19 top-1: chance = 0.125; SE on 3000-pair eval is ≈ 0.006, so 0.19
  is 10σ above chance (loose enough to not be noise-flagged).
- 0.95× pos MSE: Phase 14.3 saw 0.87; 0.95 admits half the prior
  effect as still meaningful.
