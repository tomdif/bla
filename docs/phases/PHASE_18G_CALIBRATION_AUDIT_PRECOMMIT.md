# Phase 18γ — Predictor calibration audit (Precommit)

**Date:** 2026-05-18.
**Status:** ⏳ **PRE-COMMITTED — gates locked before run.**

## Why this phase exists

Phase 18β surfaced a clean correlation gradient across candidate
distributions:

```
naive_cem               pred-actual corr =  +0.285
scripted_prior_light    pred-actual corr =  -0.191
scripted_prior_heavy    pred-actual corr =  -0.288
learned_policy_cem      pred-actual corr =  -0.520
```

Heavier search amplified ranker error (heavy CEM lost to light CEM
at 12× compute). This is not a "make the planner better" question;
it's a *where is the predictor trustworthy* question. Phase 18γ
maps the trust region before any new constructive work (trust-region
CEM, residual prior, predictor retrain) is justified.

## The question

> Where does the action-conditioned predictor produce reliable
> rankings, and where does CEM start exploiting model error?

## Candidate distributions to scan

For each, sample `M = 256` candidate plans per state at `N = 60`
states drawn from MPC replan boundaries (10 episodes × ~6 replans
each):

```
D1. scripted_prior + tiny noise            (σ = 0.02)
D2. scripted_prior + light CEM elites      (terminal σ ≈ 0.12)
D3. scripted_prior + heavy CEM elites      (terminal σ ≈ 0.05 after 3 iters)
D4. learned_policy_only mean + tiny noise  (σ = 0.02)
D5. learned_policy + light CEM elites
D6. naive Gaussian CEM elites              (σ = 0.5 → 0.2 after iters)
```

D1/D4 isolate "what does the predictor say about the prior itself,
before any search bias." D2/D3/D5/D6 isolate "what does CEM do to the
distribution that's scored."

For ground truth, *execute* each candidate plan from the same state
(via state save/restore in robosuite, just like Phase 16's env-clone
rollout) for `replan_every = 5` actions and record actual goal-
distance improvement.

## Per-distribution metrics (all logged)

```
pred_actual_corr        Pearson, candidate scores vs realized improvements
pred_actual_rank_corr   Spearman, the key calibration metric
top_k_precision_at_5    fraction of true-top-5 in predicted-top-5
top_1_realized_imp      mean realized improvement of predicted-best
mean_realized_imp       mean realized improvement of all M candidates
predicted_best_minus_mean   "search gain" claimed by predictor
realized_best_minus_mean    "search gain" actually achievable
contact_rate            fraction of executed plans that made contact
dir_score               mean directional alignment with goal
ood_dist_l2             mean L2 distance to nearest training action
                        sequence (proxy for "off-manifold")
calibration_by_decile   list[10] of (mean_pred, mean_actual) per
                        score decile — for calibration-curve plot
```

## The decisive plot

```
x-axis: distribution_breadth_proxy
        = mean L2 between candidate plans within distribution
        (small for D1/D4, larger for D6/naive)
        OR ood_dist_l2 — both reported

y-axis: pred_actual_rank_corr
```

Expected thesis (write before run):

> Predictor rank correlation is positive on the scripted-prior
> manifold (D1, D2) and degrades to negative as the candidate
> distribution moves off-manifold (D3 heavy CEM, D5 policy + CEM,
> D6 naive). D2 light CEM is the rightmost point where rank corr
> remains usable.

This thesis is **pre-registered** so I cannot retro-fit.

## Pre-committed gates

```
G1. Scripted_prior_light (D2) rank correlation > +0.10.
       (predictor is usefully calibrated near the deployment prior)

G2. Heavy_CEM (D3) rank correlation < D2 rank correlation by ≥ 0.10.
       (the calibration degrades measurably under heavy search)

G3. The calibration audit *predicts* the Phase 18β planner result:
       D2 mean(top_1_realized_imp) > D3 mean(top_1_realized_imp).
       (light CEM's "best per state" exceeds heavy CEM's "best per
        state" as measured on the audit's own ground truth)

G4 (diagnostic, not gated).
       Per-decile calibration curve for D2 vs D3.
       Visualize whether ranker monotonicity inverts inside heavy CEM
       elite regions.
```

## Verdict matrix

| Pass | Verdict |
|---|---------|
| **3/3 + thesis confirmed** | Predictor's trust region is the scripted-prior light-CEM manifold. Move to Phase 18δ (trust-region CEM). |
| 2/3, thesis directionally right | Calibration story is real but boundary is fuzzier than expected. Refine the breadth measure; consider mixed-distribution training (Phase 18ζ) before 18δ. |
| 1/3 | Either D2 isn't a positive-corr regime (calibration is worse than thought, predictor needs retrain) or the calibration trajectory is noisier than seed-0 18β suggested. Rerun across seeds. |
| 0/3 | Predictor is uncalibrated everywhere; the Phase 17/18d positives were carried *entirely* by the scripted prior. Drop the predictor as a scoring function; pursue model-free / geometric scoring. |

## What this phase is NOT

- Not a "build a better planner" phase. Pure measurement.
- Not a predictor retraining phase (that's Phase 18ζ if 18γ
  warrants it).
- Not a learned-policy revisit (deferred until calibration story
  is mapped).
- Not multi-seed yet. Single-seed mapping first; if results are
  clean, multi-seed confirmation; if marginal, expand to 3 seeds.

## After 18γ — locked next-phase preference

User preference recorded 2026-05-18:

> **Trust-region CEM first** (Phase 18δ), because Phase 18β already
> showed less search is better when correlation degrades.

Other options deprioritized but kept on the shelf:
- 18ε: residual learned prior (policy predicts residual around
  scripted prior, not full sequence) — keep if 18γ shows calibration
  is fine but policy capacity is the limiter
- 18ζ: predictor fine-tune on heavy/light CEM candidates — keep if
  18γ shows ranker is fixable on the broader distribution

## Implementation sketch

New script: `scripts/phase18g_calibration_audit.py`.

```bash
python3 scripts/phase18g_calibration_audit.py \
  --model-action /workspace/phase17/model_action_finetuned.pt \
  --policy-ckpt /workspace/phase18b/plan_policy.pt \
  --n-states 60 --n-candidates 256 \
  --replan-eval-horizon 5 \
  --distributions D1,D2,D3,D4,D5,D6 \
  --out /workspace/phase18g \
  --seed 0
```

Reuse modules:
- `scripts/phase15_planning.py` — env build, encode_frame, predict_score_seq
- `scripts/phase16_policy_prior_mpc.py` — state_features, rollout_scripted_prior,
  cem_with_prior (with returned elites)
- `scripts/phase18b_policy_distill.py` — load_policy, rollout_policy_prior

New helpers:
- `sample_candidates(distribution_id, env, model, policy, state, …)`
  → returns `(plans [M, H, A], scores [M])`
- `execute_plan(env, plan)` → returns realized improvement via
  `env.sim.get_state()` / `env.sim.set_state()` env-clone (Phase 16 pattern)
- `calibration_metrics(scores, realized)` → dict of all per-distribution
  metrics above

## Run sequencing (locked)

```
Step 1.  M=64, N=20  pilot         (~30 min single-GPU)
Step 2.  M=128, N=40 main          (~2 hr single-GPU) — the
                                     numbers that go in the decision doc
Step 3.  M=256, N=60 multi-seed    (only if Step 2 is noisy/borderline;
                                     parallelize across 5× RTX 4090)
```

### Step 1 pilot — correctness, not statistics

The pilot is a *correctness check*, NOT a statistical readout. Do not
interpret the rank-correlations as final. Just verify:

```
[ ] all 6 distributions produce nonempty candidate sets
[ ] contact / displacement ranges look plausible
[ ] predicted scores and realized outcomes are aligned in shape
[ ] no NaNs or degenerate constant predictions
[ ] top-k and per-decile calibration tables populate
[ ] execute_plan env-clone restore works across all distributions
```

If any of these fail, fix and re-pilot before Step 2.

### Step 2 main — the actual Phase 18γ result

M=128, N=40 at seed-0. This produces the numbers that go in
`PHASE_18G_CALIBRATION_AUDIT_DECISION.md`. Gates G1/G2/G3 evaluated
against these.

### Step 3 expansion — only if needed

If Step 2's per-distribution rank-corr standard error is wide enough
that the G1/G2 calls are within noise (rough threshold: |corr_se| >
0.05 on D2 or D3), expand to M=256/N=60 across seeds {0, 1, 2} on the
5 idle GPUs. Otherwise the audit is complete after Step 2.

## Reproducibility

- Precommit: this file.
- Decision doc: `docs/phases/PHASE_18G_CALIBRATION_AUDIT_DECISION.md`
- Artifacts:    `artifacts/phase18g/{summary.json, candidate_audit.npz,
                calibration_curves.png}`
- Script:       `scripts/phase18g_calibration_audit.py`
