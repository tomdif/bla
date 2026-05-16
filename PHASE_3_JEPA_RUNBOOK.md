# Phase 3 (JEPA Track) — Runbook

**Scope.** Stress-test the Phase-2 slot-delta result on harder envs and
multiple seeds. The Phase-2 ablation already established that sparse
delta is the load-bearing mechanism (`PHASE_2_JEPA_DECISION.md`). Phase 3
asks: does that conclusion survive when the env adds more targets, more
distractors, distractor motion, partial observability, and longer
occlusion windows?

If yes → the slot-delta module ships as the memory layer of the broader
BLA stack. If no → we learn exactly which axis breaks it.

## Run command (pod target)

```bash
python scripts/slot_jepa_phase3.py \
  --seeds 0,1,2,3,4 \
  --targets 3,5,8 \
  --distractors 2,5,10 \
  --K 3,5,10 \
  --J 10,20,40,80 \
  --J-train 10 \
  --modes slot_delta,slot_dense_update,dense_jepa_flatten,copy \
  --moving-distractors \
  --partial-observability \
  --obs-radius 8 \
  --steps 3000 \
  --probe-episodes 32 \
  --out artifacts/phase3_slot_delta_stress/
```

That matrix expands to **5 seeds × 4 modes × 3 targets × 3 distractors ×
3 K = 540 sub-runs**, each training once and evaluating at four J values
(2160 probe-eval rows). Add `--partial-observability` only if you want
that stress dimension turned on globally for the whole sweep; otherwise
omit it. Same for `--moving-distractors`.

For a smaller local smoke:

```bash
python scripts/slot_jepa_phase3.py \
  --seeds 0,1 --targets 3 --distractors 2 --K 5 --J 5,10 \
  --modes slot_delta,dense_jepa_flatten,copy \
  --steps 600 --probe-episodes 16 --probe-epochs 100 \
  --out /tmp/phase3_smoke/
```

## Compute estimate

Per sub-run on CPU (Phase 2 numbers, 3000 steps + four probe evals):
- slot_delta: ~32s
- slot_dense_update: ~32s
- dense_jepa_flatten: ~9s (no slot pipeline)
- copy: ~12s

Wall-clock for the full pod matrix is dominated by the 540 sub-runs.
At ~30s each on a single H200 → **~4.5 hr serial**. Trivial to shard
across the 6 GPUs of the standard pod by partitioning the seed × mode
product → **~45 min wall on 6 GPUs**. The orchestrator is fork-safe;
the simplest sharding is N copies pinned via `CUDA_VISIBLE_DEVICES`
with disjoint `--seeds` lists.

## Output layout

```
artifacts/phase3_slot_delta_stress/
├── manifest.json                # git commit, full config, phase-2 reference
├── raw_results.jsonl            # one row per (sub-run × J) — 2160 rows
├── aggregate.csv                # per (mode, K, n_targets, n_distractors, J)
│                                # with mean ± stderr ± 95% CI
├── gates.json                   # automatic win/loss gate per cell
└── runs/
    ├── seed=0_mode=slot_delta_K=3_nt=3_nd=2/
    │   ├── final.pt
    │   ├── probe_eval.json
    │   └── stdout.log
    └── ... (540 directories total)
```

The manifest is the reproducibility anchor — git commit, command line,
all flag values, Phase-2 reference metrics, slot config knobs. Do not
delete it.

## Gates (Phase 3 pass criteria)

Applied automatically per `(K, n_targets, n_distractors, J)` cell, with
mean over seeds:

1. `hidden_mse[slot_delta] ≤ 0.75 × hidden_mse[dense_jepa_flatten]`
2. `hidden_mse[slot_delta] ≤ 0.75 × hidden_mse[slot_dense_update]`

A cell **passes** if both gates pass. Phase 3 as a whole **passes** if
≥ 75% of cells pass (allows some breakdown at the hardest configurations
without invalidating the overall claim).

Headline metrics to lift into the writeup:
- median margin of slot_delta over the better of the two controls
- gates pass-rate per stress axis (which axis breaks first)
- per-target-count degradation slope J=10→80

## Secondary diagnostics (already logged per sub-run)

These come "for free" out of the existing trainer:
- slot collapse rate (= 1 - mask_mean)
- prediction loss curve (in `stdout.log`)
- slot↔target identity stability (`id_stable`)
- per-hidden-step MSE breakdown (`per_hidden_step_mse` in `probe_eval.json`)

If we want **slot swapping rate** explicitly, that's a small add — record
the argmax slot-to-target assignment at each visible→hidden transition
and check it stays constant across the hidden window. Not in this
runbook's scope; flag as a Phase-3b extension.

## What success looks like

A Phase-3 pass would say:

> The slot-delta memory advantage over fair patch-level dense JEPA
> survives 5× more distractors, distractor motion, partial
> observability, and J=80 occlusion windows. The sparse-delta
> mechanism continues to outperform slot+dense-update by a similar
> margin to Phase 2. Per-stress-axis breakdowns identify [X, Y, Z] as
> the directions that erode the advantage fastest.

If we see slot_delta lose ground on a specific axis (e.g. moving
distractors), that's not a failure — it's a precise constraint for the
next iteration of the design. Either way the experiment is informative.

## After Phase 3

If pass → integrate the slot-delta module into the broader BLA latent
stack (replace/augment the current pooled-state path). Decision doc to
follow.

If fail → identify the failure axis, add the targeted fix (typed slot
heads? recurrence? hierarchical predictor?), and re-test on the
failing cell only.
