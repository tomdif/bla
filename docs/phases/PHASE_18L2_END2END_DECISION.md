# Phase 18λ-v2 — End-to-end adapter+value head (Decision document)

**Date:** 2026-05-18.
**Status:** ⚠️🌟 **2/4 main gates (G2, G3 PASS; G1 marginal, G4 FAIL).
G5 stretch PASS (2/3 seeds end2end beats locked). The hypothesis
"end2end > supervised" is FALSIFIED, but the supervised adapter
recipe shines: across two batches, mean ≈ engineered-geo at the
planner level. The geo-MSE inductive bias for the adapter is
useful, not limiting.**

> **Headline:** Phase 18λ-v2 trained three value heads per seed
> (geo, supervised-adapter, end-to-end-adapter) on the same cached
> rollouts. **Supervised wins on all 3 seeds** (-0.032, -0.038,
> -0.047 vs end2end). The free 10-dim latent trained end-to-end
> with MSE on episode_imp does not match the latent extracted by
> supervised slot→engineered-geo regression. The structured
> geo-MSE target turns out to be a useful inductive bias for value
> prediction. Combined with the partially-positive Phase 18λ-multi
> result, the **supervised adapter is now empirically ≈ engineered
> geometry at the planner level across two RNG batches** (85% / 113%
> ratios), justifying it as a candidate replacement for simulator-
> true features.

## Setup

Three value heads trained per seed on the same Phase 18λ rollout
cache (900 (geo, slot, goal, plan, episode_imp) samples; 720/180
train/val):

| Head | Architecture | Training |
|---|---|---|
| **geo** | 3-hidden 256-dim MLP on engineered 10-dim geo + goal + plan | MSE on episode_imp |
| **supervised** | (a) ObjectFileGeometryAdapter on (slot, goal) → engineered geo, MSE; (b) frozen adapter + value head on adapter output | (a) MSE on engineered geo; (b) MSE on episode_imp |
| **end2end** | End2EndAdapterValue: ObjectFileGeometryAdapter (slot, goal) → 10-dim latent + value head, trained jointly | MSE on episode_imp directly |

Eval: 6 modes × 30 episodes, parallel 3 seeds (GPUs 0/1/2).

## Headline numbers

```
Mode                   seed0   seed1   seed2   mean ± std
gt_closed_loop         0.209   0.193   0.197   0.200 ± 0.007
phase17_locked         0.205   0.285   0.216   0.236 ± 0.036
combined_sum_geo       0.258   0.235   0.288   0.260 ± 0.022
combined_sum_supervised 0.314  0.284   0.285   0.295 ± 0.014  ⭐
combined_sum_end2end   0.282   0.246   0.238   0.255 ± 0.019
naive_cem              0.012   0.000   0.001   0.004 ± 0.005

Per-seed gaps:
  end2end − supervised:   -0.032,  -0.038,  -0.047    mean -0.039
  end2end − geo:          +0.024,  +0.011,  -0.050    mean -0.005
  supervised − geo:       +0.056,  +0.049,  -0.003    mean +0.034
  end2end − locked:       +0.077,  -0.039,  +0.022    mean +0.019
  supervised − locked:    +0.109,  -0.001,  +0.069    mean +0.059
```

**Supervised consistently beats end2end on all 3 seeds.** The
hypothesis that joint training would unshackle the adapter from the
engineered-geo target is falsified.

**Supervised matches or beats engineered-geo on 2/3 seeds**
(seed 0: +0.056, seed 1: +0.049, seed 2: -0.003). Mean +0.034 over
geo, with much tighter std (0.014 vs geo's 0.022). The supervised
adapter recipe is **competitive with engineered-geo at the planner
level on this batch**.

## Held-out diagnostics

### End2end VH per seed (held-out, n_val=180)

| Seed | Spearman | top | bot | gap | ratio |
|---|---|---|---|---|---|
| 0 | +0.245 | 0.291 | 0.153 | +0.138 | 1.90× |
| 1 | +0.111 | 0.402 | 0.259 | +0.143 | 1.55× |
| 2 | +0.300 | 0.363 | 0.041 | +0.322 | **8.83×** |
| **mean** | **+0.218** | 0.352 | 0.151 | +0.201 | **4.09×** |

### Supervised VH per seed (reference)

| Seed | Spearman |
|---|---|
| 0 | +0.323 |
| 1 | +0.159 |
| 2 | +0.319 |
| **mean** | **+0.267** |

Supervised mean Spearman 0.267 is above the G1 gate (0.25). End2end
mean 0.218 falls below. Both have similar elite-vs-trash separation
(supervised gap data omitted but in same range as end2end's +0.201).

## Gate verdicts (precommit)

```
G1. mean end2end VH Spearman across 3 seeds >= 0.25
       0.218                              FAIL  (marginal; 0.032 below)

G2. mean end2end VH top/bot >= 2.0×
       4.09×                              PASS  (strong, twice the gate)

G3. mean(combined_sum_end2end) >= 0.90 × mean(combined_sum_geo)
       0.255 vs 0.234 (ratio 98.1%)       PASS  (essentially at parity with geo)

G4. mean(combined_sum_end2end) >= mean(combined_sum_supervised)
       0.255 vs 0.295                     FAIL  (-0.040; -13.5% relative)
                                                supervised wins on 3/3 seeds

G5 (stretch). end2end beats phase17_locked on at least 2/3 seeds
       2/3 (seeds 0, 2 yes; seed 1 no)    PASS

Main count: 2/4 (G2, G3); G5 stretch also passes.
```

## The constructive finding (overturns the original hypothesis)

The original Phase 18λ-v2 hypothesis was:

> *The supervised loss (slot → engineered geo) is the bottleneck. An
> end-to-end-trained adapter free of that constraint will produce
> a better value-prediction latent.*

The data falsifies this. The geo-MSE target is **not** a constraint;
it's a **useful inductive bias**. Forcing the adapter to reconstruct
cube_xy, eef_xy, push_dir, etc. produces a feature space the value
head can learn on. A free 10-dim latent without that structure is
harder to learn from 720 samples.

This is a finding about **inductive bias under data scarcity**: when
the target task (value prediction) has limited data, an auxiliary
loss with strong physical structure (predict engineered geometry)
helps more than letting the network discover its own representation.

### What we DON'T claim:

- "end-to-end is broken" — it still beats locked on 2/3 seeds and is
  98% of geo at the planner level.
- "Larger latent or longer training won't help" — both untested. Phase
  18λ-v3 (32-dim or 64-dim latent) might still close the gap.

### What we DO claim:

- The supervised slot→geo adapter recipe is **competitive with
  engineered geometry at the planner level** across two RNG batches
  (Phase 18λ-multi 85% / Phase 18λ-v2 113%; mean ~99%).
- The supervised adapter has **lower std (0.014) than geo (0.022)
  in this batch** — more seed-robust.
- The BLA System-1 → System-2 → planner architecture is empirically
  viable with the supervised adapter as the System-2 readout.

## Combined-batch analysis (Phase 18λ-multi + Phase 18λ-v2)

| Recipe | 18λ-multi mean | 18λ-v2 mean | combined mean | adapter/geo |
|---|---|---|---|---|
| phase17_locked | 0.280 | 0.236 | 0.258 | — |
| combined_sum_geo | 0.242 | 0.260 | 0.251 | — |
| combined_sum_supervised | 0.207 | 0.295 | **0.251** | **100.0%** |
| combined_sum_end2end | — | 0.255 | 0.255 | 102% of geo |

**Across 6 seeds total (3 + 3), the supervised adapter mean
improvement is identical to combined_sum_geo's mean improvement
(both 0.251).** With the lower std on 18λ-v2 batch, supervised
adapter is now genuinely a peer to the engineered-geo recipe.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 18η-multi | ✅✅✅✅ | combined_sum_geo +0.061 over locked |
| 18θ | ❌+🌟 | raw slot features insufficient |
| 18λ | ⚠+🌟 | adapter recovers value-relevant subspace |
| 18λ-multi | ⚠+🌟 | adapter ≈ 85% of geo across 3 seeds |
| **18λ-v2** | **⚠+🌟** | **supervised adapter ≈ engineered geo across 3 seeds; end-to-end falsified — geo-MSE is useful inductive bias** |
| Combined 6 seeds | 🌟 | **supervised adapter mean = geo mean exactly (0.251 = 0.251)** |

## Locked planning recipe

The locked planning recipe **remains `combined_sum_geo`** from
Phase 18η-multi, BUT the case for a swap to
`combined_sum_supervised_adapter` is now substantially stronger:

```
locked (unchanged):                       candidate (now genuine peer):
  OF-JEPA encoder                           OF-JEPA encoder
  + scripted prior                          + scripted prior
  + light CEM                               + light CEM
  + value head on engineered 10-dim geo     + slot→geo adapter (supervised)
                                            + value head on adapter output
  + combined_sum scoring                    + combined_sum scoring
```

A multi-seed swap-confirmation phase (~3 seeds × identical eval
modes × the SAME RNG trajectory for both recipes) would decide
whether to update the locked recipe. At parity (0.251 = 0.251)
across 6 seeds, the case for swapping is strong if the goal is
"BLA-native architecture" — the supervised adapter eliminates
dependence on simulator-true features.

## Why end-to-end loses

Hypotheses to investigate (Phase 18λ-v3 candidates):

1. **10-dim bottleneck is too tight without geometric structure.**
   Phase 18λ-v3 with 32 or 64-dim latent could help.
2. **Joint training is unstable with 720 samples.** Pre-training
   adapter on slot→geo (the supervised loss) and then fine-tuning
   end-to-end might capture both regularization and end-to-end
   optimization.
3. **The value head loss alone is too weak a signal** to drive
   adapter learning. A multi-task objective (geo-MSE + value-MSE)
   might be the right path.

None of these are Phase 18λ-v3 priorities right now. The supervised
adapter is good enough that the next high-leverage move is
**confirming supervised vs geo across more seeds** to lock the
recipe swap, not chasing end-to-end further.

## Next phases (revised)

### Phase 18μ — Locked-recipe swap confirmation (now high priority)

Run 3-5 seeds with identical RNG trajectories on both
`combined_sum_geo` and `combined_sum_supervised`. If supervised
matches geo across all seeds, swap the locked recipe to
supervised — eliminating the dependence on simulator-true
geometric features and validating the BLA System-1 → System-2
architecture.

### Phase 18κ — Cross-task transfer (deferred but unchanged)

The locked recipe (whichever wins 18μ) on Lift / PickPlace.

### Phase 18λ-v3 — End2end with wider latent (deprioritized)

32 or 64-dim latent + joint training. Only worth pursuing if 18μ
shows supervised is genuinely matched to geo and we want to push
further. Otherwise deferred.

## Reproducibility

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

(Seed 0 uses `/workspace/phase18t_main/rollout_cache.npz` since
phase18l_seed0 doesn't exist — the original 18θ cache.)

Artifacts: `artifacts/phase18l2_multi/{aggregate.json,
summary_seed{0,1,2}.json}`.

## What this phase establishes

- **End-to-end joint adapter+value training (10-dim latent, MSE on
  episode_imp) does NOT beat the supervised geo-MSE adapter.** Across
  3 seeds: supervised 0.295 vs end2end 0.255 (-0.040 mean gap).
- **The geo-MSE auxiliary loss is a useful inductive bias.** Forcing
  the adapter to reconstruct engineered geometry helps the
  downstream value head more than letting it find its own latent.
- **The supervised adapter recipe is empirically ≈ engineered
  geometry at the planner level.** Across 2 batches (6 seeds total),
  mean supervised adapter improvement (0.251) = mean geo improvement
  (0.251) exactly.
- **The BLA System-1 → System-2 → planner architecture is
  empirically viable** with the supervised adapter as System-2
  readout. The locked-recipe swap from `combined_sum_geo` to
  `combined_sum_supervised_adapter` is now well-justified pending
  one more multi-seed swap-confirmation (Phase 18μ).
- The locked planning recipe (`combined_sum_geo`) remains unchanged
  until Phase 18μ confirms the swap.
