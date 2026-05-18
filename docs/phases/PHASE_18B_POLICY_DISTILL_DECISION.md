# Phase 18β (Policy Distillation) — Decision document

**Date:** 2026-05-18.
**Status:** ❌✅ **1/3 gates pass. Distillation falsifies the learned-prior hypothesis. Unexpected positive: scripted prior + LIGHT CEM beats scripted prior + heavy CEM at 8.3% compute.**

> **Headline:** Distilling 360 teacher (state, plan) pairs into a small
> MLP yields a policy that nails *contact* (1.00 vs teacher 0.93) but
> loses *goal-direction* (dir 0.076 vs 0.224). Even with light CEM
> refinement on top, the policy stack reaches only 72% of the heavy
> teacher's improvement. The compute-economy gate (G3) does pass, but
> the headline distillation thesis is rejected.
>
> The real discovery is downstream: **`scripted_prior + light CEM`
> reaches improvement 0.244, beating `scripted_prior + heavy CEM`'s
> 0.213 at 8.3% of the compute (96 vs 1152 candidates).** Dir_score
> doubles (0.451 vs 0.224) and success rate goes 0.37 vs 0.23. Heavy
> CEM was over-CEMing — the predictor is anti-correlated on the
> competent-prior candidate distribution, so more search degrades
> selection. The new default planning recipe is locked.

## Setup

Phase 17 model (`/workspace/phase17/model_action_finetuned.pt`) +
robosuite Stack environment, identical MPC framing to Phase 16/17/18d.

- Teacher pass: 120 episodes of `scripted_prior_cem` collected 360
  (state, CEM-refined plan) pairs. Mean teacher score = −0.164.
- Policy: 2-hidden-layer MLP, in_dim=10 (cube xy, eef xyz, cube z,
  goal xy, push_dir), out=`[plan_horizon=10, action_dim=7]` with tanh.
- Distillation: 2500 steps weighted plan MSE (step_decay=0.90).
  Loss 0.246 → 0.072 (3.4× reduction).
- Eval: 30 episodes × 6 modes, seed=0 only.

## Headline numbers

```
Mode                     improvement  dir_score  contact  success  candidates  pred-corr
gt_closed_loop                 0.104      0.074    0.700    0.100         0         NaN
naive_cem                      0.002     −0.015    0.467    0.000      1152       +0.285
scripted_prior_cem  (teacher)  0.213      0.224    0.933    0.233      1152       −0.288
scripted_prior_light_cem ⭐    0.244      0.451    0.967    0.367        96       −0.191
learned_policy_only            0.144      0.076    1.000    0.133         0         NaN
learned_policy_cem             0.154      0.010    0.933    0.233        96       −0.520
```

## Gate verdicts (vs precommit)

```
G1. policy_only.improvement >= 0.75 × teacher.improvement
       0.144 vs 0.160                                  FAIL  (67.6% of teacher)

G2. policy_cem.improvement >= teacher.improvement − 0.02
       0.154 vs 0.193                                  FAIL  (72.4% of teacher)

G3. policy_cem.candidates / teacher.candidates <= 0.35
       96 / 1152 = 0.083                               PASS  (8.3% — way under)
```

**Verdict: 1/3 → "weak learned-prior signal." Per the precommit
interpretation matrix: *Keep scripted prior; policy data/capacity is
insufficient.***

## Why distillation underperformed

Decomposition of policy_only vs teacher:

| Metric | Policy alone | Teacher | Ratio |
|---|---|---|---|
| contact_rate | 1.000 | 0.933 | 1.07× |
| dir_score | 0.076 | 0.224 | 0.34× |
| mean_displacement | 0.030 m | 0.050 m | 0.59× |
| success_rate | 0.133 | 0.233 | 0.57× |

The policy generalized **contact behavior** very well (1.00, even
better than teacher's 0.93) but **lost goal direction** (0.076 vs
0.224 — about a third). The shallow MLP absorbed the
"approach + push" component of the teacher plans but not the
goal-aware direction selection, which depends on goal_xy in a way the
limited capacity / 360 examples couldn't fit.

Adding light CEM on top (`learned_policy_cem`) made dir_score *worse*,
not better (0.010 vs 0.076). That's the second surprise: CEM was
*hurting* the policy. Diagnosis: pred-actual corr is **−0.520** on
policy-mean candidates — strongly anti-correlated. CEM ranks
candidates by predictor; if the predictor is anti-correlated, CEM
selects the *worst* candidates. The policy mean is so OOD for the
Phase 17 predictor that any CEM refinement degrades it.

## The real finding: scripted_prior + light CEM ≫ scripted_prior + heavy CEM

| Metric | heavy (1152) | light (96) | ratio |
|---|---|---|---|
| improvement | 0.213 | **0.244** | 1.15× |
| dir_score | 0.224 | **0.451** | 2.02× |
| contact_rate | 0.933 | 0.967 | 1.04× |
| success_rate | 0.233 | **0.367** | 1.57× |
| mean_displacement | 0.050 m | 0.054 m | 1.07× |
| pred-actual corr | −0.288 | −0.191 | (less negative) |
| candidates | 1152 | **96** | **0.083×** |

Light CEM beats heavy CEM on *every* end-effect metric while using
8.3% of the compute. The predictor is anti-correlated on competent-
prior candidates at *both* compute budgets, but heavy CEM amplifies
that error by running 3 iters × 128 candidates = 12× more selection
steps against a misleading score function. Light CEM's σ floor + 1
iter × 32 candidates stays closer to the (already-good) scripted
prior.

**This is the proxy-vs-end-effect rule firing again** — the predictor
correlation is negative across all three priors at heavy compute, but
the *end-effect* tells the actual story: light CEM is better.

## Predictor correlation by prior — a clean diagnostic gradient

```
naive_cem           corr = +0.285     (predictor correctly orders
                                       random/no-prior candidates)
scripted_prior_light_cem  −0.191     (mildly anti-correlated on
                                       competent narrow distribution)
scripted_prior_cem        −0.288     (more anti-correlated as
                                       distribution narrows further
                                       via heavy CEM)
learned_policy_cem        −0.520     (most anti-correlated on
                                       policy-mean candidates, which
                                       are most OOD)
```

The predictor's *positive* correlation on naive CEM means it CAN
distinguish random plans from each other. But within the narrow
competent-prior basin, its ordering inverts. This is a sharp
calibration finding and points at the next predictor-side investment
(Phase 18γ — see below).

## What this changes operationally

1. **Locked**: New default planning recipe is `scripted_prior + LIGHT CEM`
   (1 iter, K=32, σ=0.12). Heavy CEM is deprecated.
2. **Demoted**: Phase 17 / 18d's "OF-JEPA + heavy CEM beats oracle"
   conclusion still holds — but now we know the heavy CEM wasn't load-
   bearing; the *scripted prior* was. The predictor + heavy CEM was
   marginally hurting; the broader stack still came out ahead because
   the prior was strong.
3. **Falsified**: "Learned proposal policy replaces scripted prior"
   with current capacity / data / horizon model. Not generally — the
   shallow MLP is the binding constraint.
4. **Surfaced**: Predictor calibration is the next limiter. Pred-actual
   corr ≤ 0 across all competent priors means the predictor adds
   negative value at refinement time.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 14.5/14.6 | ✅✅✅ | offline ranking + OOD generalization across action statistics |
| 15b | ❌ | naïve CEM fails (prior-bound) |
| 16 | 1/3 + diagnosis | BC fixes contact; predictor anti-correlated on focused-contact |
| 17 | ✅ | mixed-data training; planner beats oracle (seed 0) |
| 18δ | ✅✅✅ | Phase 17 robust across 3 seeds; planner +0.005 over oracle |
| **18β** | **❌ + ⭐** | **distillation falsifies learned-policy hypothesis (1/3); LIGHT CEM beats heavy CEM with same prior at 8.3% compute** |

## Architectural take

> *Phase 17/18d's headline was "OF-JEPA + competent prior + CEM-refinement
> beats oracle." Phase 18β isolates the contributions: the **prior**
> is doing the work. The CEM-refinement is at best neutral and at heavy
> compute is actively harmful. Distilling the teacher into a policy
> preserves the **contact** half of the prior but not the **direction**
> half — the shallow MLP cannot represent goal-conditioned pushing
> from 360 examples. The next architectural lever is **predictor
> calibration on competent-prior candidate distributions**, not deeper
> policy capacity, because even a perfect prior is bottlenecked by the
> predictor's anti-correlated scoring at refinement time.*

## Reproducibility

Pre-committed gates: `docs/phases/PHASE_18B_POLICY_DISTILL_PRECOMMIT.md`.

Pod artifacts:
- Teacher cache:  `/workspace/phase18b/teacher_plans.npz`  (360 examples)
- Policy ckpt:    `/workspace/phase18b/plan_policy.pt`
- Summary:        `/workspace/phase18b/summary.json`
- Per-episode:    `/workspace/phase18b/per_episode_<mode>.jsonl`
- Run log:        `/workspace/phase18b/log.txt`

Run command:

```bash
python3 scripts/phase18b_policy_distill.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --teacher-episodes 120 \
    --policy-train-steps 2500 \
    --policy-K 32 --policy-cem-iters 1 \
    --n-episodes 30 \
    --out /workspace/phase18b
```

## Next phases (re-ordered by what Phase 18β revealed)

**Phase 18γ — Predictor calibration audit (NEW, priority 1).** The
pred-actual corr trajectory across priors (+0.285 → −0.19 → −0.29 →
−0.52) is a sharp calibration signal. Three sub-tests:
- 18γ-a: Is the negative corr at heavy CEM real, or is it driven by
  CEM's elite-selection-induced selection bias?
- 18γ-b: Does adding *light-CEM* data to the Phase 17 mixed training
  fix the calibration on competent-prior distributions?
- 18γ-c: Is the predictor's geometry-feature score (predicted goal
  distance) better than the predictor itself as a scoring function?

**Phase 18β-v2 — Heavier policy (deferred).** Bigger network (3-4
hidden), more training data (300-500 episodes), recurrent head, or
goal-relative coordinates may fix the dir_score collapse. Worth one
revisit, but predictor calibration looks like the higher-leverage
investment based on Phase 18β's findings.

**Phase 18α — Cross-task transfer (unchanged).** Test the
`scripted_prior + light CEM` recipe on Lift / PickPlace tasks. The
fact that light CEM > heavy CEM may or may not transfer.
