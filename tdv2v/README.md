# Two-view TDV — Tier-1 harness

Tests whether **two concurrent camera angles** improve TDV-style self-supervised video representations —
specifically whether a cross-view objective (a) **recovers the semantic gap** TDV has and (b) **boosts
depth**, without any hand-crafted augmentation. Tier-0 (a frozen-encoder probe) already gave the green
light on the semantic-invariance signal and showed the depth benefit needs a *learned patch-level*
cross-view module — which is what this harness pretrains.

## The objective (one script, flags select the condition)
Per step we form the **(camera × time) grid** `(A,t) (A,t+1) (B,t) (B,t+1)` and combine:
- **Temporal (TDV)** on each camera: `ẑ_{t+1} = z_t + motion(Δx; z_t)` vs an **EMA-teacher** `z_{t+1}` (MSE).
- **Cross-view** (`--crossview`): a **CroCo-style head** predicts camera-B tokens from camera-A tokens (MSE).
- **Commutativity** (`--commut`): `T_A + V_{t+1} = V_t + T_B` — motion-then-view = view-then-motion.
- **Anti-collapse**: **SIGReg** (default) — random 1-D projections of the `[CLS]` batch forced to `N(0,1)`
  (isotropic-Gaussian features ⇒ no collapse); `--anticollapse dino` for the DINO alternative.

## The decisive comparison + the control
```
bash run.sh         # trains: mono / xview / commut / shuffle  (ViT-S, identical data & compute)
```
- **Hypothesis confirmed iff** `xview` KNN **≫** `mono` KNN **and** `xview` **≫** `shuffle`.
- `shuffle` (`--shuffle-control`) breaks the A–B pairing (camera B is permuted within the batch), so the
  cross-view loss sees mismatched scenes. If the gain were just "more features / extra regularization,"
  shuffle would match xview. If the gain is **genuine cross-view geometry**, shuffle collapses to `mono`.
  This is the control that makes the result publishable (rules out the dimensionality/regularization
  confound).

## What to log (already wired)
`temp / xview / commut / ac` loss components · **`eff_rank`** (participation-ratio effective rank of the
`[CLS]` batch — collapse ⇒ ~1) · periodic **KNN** (object-identity on synthetic = semantic proxy; swap for
ImageNet on the real run). Metrics dumped to `runs/*/metrics_*.json`.

## Going to the real data (Tier-1 proper)
1. Implement `SceneFlowVideo` / `KITTIStereoVideo` in `data.py` (stubs present): left/right → cameras A/B,
   consecutive frames → t/t+1. SceneFlow is synthetic stereo video already in TDV's depth/flow eval, so
   it's the cleanest controlled testbed; KITTI/nuScenes for realism.
2. Add the downstream **stereo-depth (SceneFlow) and optical-flow (MPI-Sintel)** fine-tunes (the dorsal
   metrics) — reuse the paper's CroCo/Midway decoder + DPT head on a *frozen* backbone.
3. Replace the synthetic KNN probe with **ImageNet KNN** (`eval.py`).
4. Scale: ViT-S → ViT-B, `--steps 20000 → 200000`, `--bs 128 → 256`. Paper used 2×H100×48h for full SSv2.

## Smoke test (CPU/MPS, ~1 min)
```
python train.py --img 64 --patch 16 --dim 64 --depth 2 --heads 2 --motion-depth 1 --xview-depth 1 \
                --bs 8 --steps 40 --warmup 5 --knn-every 20 --log-every 10 --n-classes 8 --crossview \
                --out runs/smoke
```

## Files
`model.py` (ViT + motion + cross-view + EMA) · `losses.py` (temporal/xview/commut/**SIGReg**/dino + eff_rank)
· `data.py` (synthetic OOB + SceneFlow/KITTI stubs) · `eval.py` (KNN) · `train.py` · `run.sh`.

## Honest caveats
- Synthetic data understates the case; the real signal needs SceneFlow/KITTI.
- The additive composition `z+Δz` and the cross-view branch *are* assumptions (just weaker/geometric ones);
  if `commut` helps, the geometry is real; if `xview` needs a *learned* (non-additive) view delta on wide
  baselines, restrict to near-stereo.
- Wide baselines break additivity — start near-stereo (SceneFlow/KITTI are small-baseline).
