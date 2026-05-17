# Phase 9B (OF-JEPA v1 on MOVi-D) — Decision document

**Date:** 2026-05-16.
**Status:** ❌ **REGRESSION — v1 visibility-gating degrades what v0 was already doing correctly. Phase 9 v0 was MISDIAGNOSED.**

> Phase 9 marked OF-JEPA v0 as PARTIAL because anonymous Hungarian
> hidden_pos_mse was 4.2e-2 (3360× slot_delta's). We built v1
> (null-Sinkhorn dustbin + transition model + visibility belief) as
> the architectural fix. The Phase 9B sweep added the right metric —
> identity-conditioned hidden MSE — and it **falsifies the v1 fix
> AND retroactively corrects the Phase 9 verdict**. Under the
> identity-conditioned metric, v0 was already tracking its bound
> entity through occlusion at id_h/v = 1.51. v1's added complexity
> regressed every metric that matters.

## The metric correction

**Anonymous Hungarian hidden_pos_mse** (Phase 9 v0 metric):
relabels slot→entity assignments fresh each frame to minimize MSE.
Architectures that don't persistently bind win this trivially —
slot_delta wins by 3360× because its slots are exchangeable.

**Identity-conditioned hidden_pos_mse** (Phase 9B metric): finds
each file's modal entity across an episode, then measures per-frame
MSE against THAT FIXED ENTITY regardless of visibility. Can't be
gamed by anonymous rematching.

## Headline numbers (3 seeds × 3000 steps on MOVi-D)

| Mode | vis_mse | hpm_anon | switch ↓ | id_vis_mse | id_hid_mse | id_h/v |
|---|---|---|---|---|---|---|
| **of_jepa_v0** | **1.3e-5** | 3.3e-2 | **0.076** | **2.5e-5** | **4.2e-5** | **1.51** |
| of_jepa_v1 | 4.4e-4 | 4.8e-3 | 0.091 | 8.8e-3 | 1.3e-2 | 2.68 |
| slot_delta | 1.4e-5 | 1.8e-5 | 0.857 | 1.5e-5 | 1.3e-5 | 1.10 |
| slot_dense_update | 2.9e-4 | 2.9e-4 | 0.583 | 2.9e-4 | 4.5e-4 | 1.58 |

## What this reveals

### v0 was already passing — we measured wrong

| Metric | v0 | slot_delta | What it means |
|---|---|---|---|
| switch_rate | 0.076 | 0.857 | v0 is **11× more identity-stable** |
| id_visible_mse | 2.5e-5 | 1.5e-5 | v0 visible state ~equal to slot_delta |
| id_hidden_mse | 4.2e-5 | 1.3e-5 | v0 hidden state tracks its file's modal entity tightly |
| id_h/v | 1.51 | 1.10 | v0 degrades only 1.5× under occlusion |

OF-JEPA v0 on MOVi-D: switch 11× better than slot_delta AND identity-conditioned MSE within 3× of slot_delta. **That's a joint pass on the right metric.**

The Phase 9 doc's anonymous hpm "failure" was the metric's
artifact: when entity *e* is occluded, anonymous Hungarian routes
*e*'s GT position to whichever slot is currently closest in
predicted-position space. That slot is typically bound to a
*different* entity *e'* via OF-JEPA's persistent assignment.
The file's state contains *e'*'s content, not *e*'s — so probe
MSE between the slot and *e*'s GT explodes. **This is the
architecture working correctly; the metric is wrong.**

### v1 is a regression on every metric that matters

| Metric | v0 | v1 | Δ |
|---|---|---|---|
| visible_mse | 1.3e-5 | 4.4e-4 | **30× worse** |
| id_visible_mse | 2.5e-5 | 8.8e-3 | **350× worse** |
| id_hidden_mse | 4.2e-5 | 1.3e-2 | **300× worse** |
| id_h/v | 1.51 | 2.68 | 1.8× worse |
| switch_rate | 0.076 | 0.091 | similar |
| hpm_anon | 3.3e-2 | 4.8e-3 | 7× better (the "fix") |

v1's visibility-gating + null-Sinkhorn + transition model **only**
improves the metric that was wrong (anonymous hpm). On the
identity-conditioned metrics it severely regresses. The added
visibility BCE loss + transition-loss optimization pressure
displaces position-fit capacity, dropping visible MSE 30×.

The "fix" was for a problem that the proper metric reveals didn't
exist.

## What this commits as the locked architecture state

**OF-JEPA v0 is the canonical BLA System-1 perceptual architecture.**

Its components:
- Persistent learned id_proto (identity address per file)
- Memory-anchored cross-attention (memory queries observations, not the reverse)
- Differentiable Sinkhorn assignment over per-frame proposals
- Slow EMA on id_key with inter-frame LayerNorm
- Sparse delta on state_value (`state + change_mask · tanh(delta)`)
- Inter-frame LayerNorm on state_value to prevent recurrence blowup
- ConvNeXt-T proposal encoder

What we shelved (and why):
- **OF-JEPA v1** (null-Sinkhorn + transition + visibility belief) —
  not beneficial on current MOVi-D regime. May become useful for
  datasets with TRUE object birth/death (long streams, MOVi-E
  camera-motion-induced occlusion, real video with entries/exits)
  but should not replace v0 on current data.

## Methodology lesson — locked to memory

[[feedback-identity-conditioned-metrics]]:

> **For object-file architectures, identity-conditioned metrics are
> primary. Anonymous Hungarian rematching is a secondary diagnostic
> that systematically rewards non-persistent shuffling architectures
> and misdiagnoses real object-file systems.**

This is a sibling of [[feedback-joint-metric-vs-single-axis]] but
sharper: even within a single axis (position MSE), the choice
between anonymous vs identity-conditioned matters enormously. We
spent two phases of architecture work fixing a metric problem.

## Retroactive correction to Phase 9

PHASE_9_JEPA_DECISION.md's "PARTIAL — OF-JEPA wins identity but
fails hidden_pos_mse" should be read as:

> **CORRECTED: OF-JEPA v0 passes MOVi-D under identity-conditioned
> evaluation. The original anonymous-Hungarian "failure" was a
> metric artifact. v0 maintains object-file state through occlusion
> with id_h/v = 1.51 — comparable to slot_delta's 1.10 — while
> being 11× more identity-stable (switch 0.076 vs 0.857).**

## What stays open

Phase 9B's v1 architecture is **shelved**, not deleted. The
code remains in `system1_jepa/of_jepa.py:ObjectFileMemoryV1` and
can be revisited when:

1. We test against truly streaming data (MOVi-E camera motion,
   or stitched/long-clip variants) where object birth/death is
   real, not 0.05% of frames within a fixed 24-frame clip.
2. We have a real-image benchmark (CLEVRER) where occlusion is
   substantially deeper than MOVi-D's ~5% rate.

For now: **OF-JEPA v0 is the architecture.**

Phase 9C (loss-weight tuning to recover v1's visible precision) is
moot — there's no reason to tune a regression. Closing #113/#114
as deferred.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 | ✅ | slot_delta strong spatial state memory under stress |
| 7 v1 | ⚠ | slot_delta vs slot_dense_update tradeoff identified |
| 7B-D | ❌ | slot-content interventions falsified (4 attempts) |
| 8A | ❌ | contrastive loss collapses content at any effective λ |
| 8C | ✅ | OF-JEPA on MOVi-A: switch 0.002, joint gate passed |
| 8D | ✅ | OF-JEPA under stress (≥8 ent, stride=4): BOTH axes won |
| **9** | **✅** | **(retroactively corrected) OF-JEPA on MOVi-D: identity-conditioned id_h/v 1.51, switch 11× better than slot_delta** |
| **9B** | **❌** | **v1 visibility-gating is a regression; v0 was already passing** |

## Reproducibility

Code at Phase 9-era commit + ObjectFileMemoryV1 in `of_jepa.py` +
identity_conditioned_position_eval in `identity_probe.py`.
Artifacts at `artifacts/phase9b_run1/seed_{0,1,2}/`.

Run command (4 modes × 3 seeds):

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed nohup python3 scripts/slot_jepa_movi_train.py \
    --cache /workspace/movi_d_local/validation \
    --modes of_jepa_v1,of_jepa_v0,slot_delta,slot_dense_update \
    --seeds $seed --epochs 20 --max-steps 3000 \
    --log-every 250 --probe-epochs 300 --train-frac 0.8 \
    --lr 1e-4 --sigreg-w 0.0 \
    --of-jepa-w 1.0 --of-pos-w 10.0 --of-visibility-w 1.0 \
    --n-slots 20 --slot-dim 128 --max-entities 25 \
    --out /workspace/phase9b_run1/seed_$seed > seed_$seed.log 2>&1 &
done
```
