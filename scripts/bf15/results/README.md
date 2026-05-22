# BF-0.7 → BF-0.14 SAM Perception Calibration Results

Reproducibility data from the perception characterization run on 2026-05-22.
All measurements on PickPlaceCan PNG/JPEG frames rendered at 480×480 via robosuite
+ MuJoCo (offscreen EGL) on a RunPod 2×H100 instance.

## Phases

| Phase | Directory | Headline metric |
|---|---|---|
| BF-0.7 | `bf07_sam_calibration/` | Mean 2.10 cm / max 2.33 cm 3D pose error, 0 switches, 135/135 valid |
| BF-0.7 backbone sweep | `bf07_sam_calibration/backbone_sweep.json` | Tiny @ 27 FPS, Small @ 25 FPS, Base+ @ 19, Large @ 11.5 — accuracy flat |
| BF-0.7 latency | `bf07_sam_calibration/latency.json` | Hiera-Large 87 ms p95 / 11.5 FPS at 480×480 |
| BF-0.8 | `bf08_sam_occlusion/` | Full occlusion: 0/26 masks during, 0-frame recovery, all 4 gates PASS |
| BF-0.9 | `bf09_sam_partial_occ/` | Partial occlusion: 4.3 px max drift, identity bulletproof |
| BF-0.10 | `bf15_flat_demo/streaming_benchmark.json` | 27 FPS sustained end-to-end; memory linear ~0.75 MB/frame |
| BF-0.11 | `bf15_flat_demo/watchdog_test.json` | Fiducial watchdog: 40 → 3 zero-mask frames (-92.5%), 0-frame recovery |
| BF-0.12 noise | `bf12_noise_sweep/results.json` | Gaussian σ ∈ [0, 40]: mean error 0.40-0.48 cm flat |
| BF-0.12 blur | `bf12_blur_sweep/results.json` | Motion blur k ≤ 25: blur ELIMINATES static-occluder failure |
| BF-0.13 | `bf13_image_quality/results.json` | Lighting 0.3-2.5× + JPEG q≥30 both pass with margin |
| BF-0.14 single | `bf14_distractor_proximity/result.json` | 10cm offset, scene-rando spike: 1.29 cm mean |
| BF-0.14 sweep | `bf14_offset_sweep/results.json` | Offsets {20,10,5}cm: 2.00 / 0.41 / 0.32 cm mean — proximity *helps* |

## Pipeline (per the locked spec §5.0)

```
fiducial detection
  → initial SAM 2.1 click seed
  → SAM 2.1 Hiera-Tiny propagation
  → mask centroid + plane/RGB-D projection
  → if mask silence ≥ silence_threshold AND fiducial visible:
       re-seed via add_new_points_or_box(clear_old_points=True)
  → RollingObjectFileTracker
  → DemoRetriever
  → action replay / EpisodeLogger
  → reset SAM state at episode boundary (cap memory growth)
```

## Implementation

- `bla/forge/sam_perception.py` — SAMPerception (mock_static + sam2.1 backends, watchdog support)
- `bla/forge/deploy_loop_sam.py` — build_sam_deployment_loop (BF-1.5 + BF-1.6)
- `tests/test_bf15_sam_integration.py` — 7 unit tests on the mock backend
- `scripts/bf15/*.py` — pod-side validation scripts (require CUDA)

## Renders + sample frames

Large renders (demo.npz with full RGB+depth, 53MB each) and representative
sample frames are not committed. They live at:

```
~/bla_artifacts/bf_perception_2026_05_22/
  bf07_sam_calibration/demo.npz         (53 MB, full BF-0.7 lift demo)
  bf08_sam_occlusion/demo.npz           (53 MB, Milk-as-moving-occluder)
  bf09_sam_partial_occ/demo.npz         (53 MB, partial occlusion sweep)
  bf14_distractor_proximity/demo.npz    (4 KB, metadata only)
  bf15_flat_demo/demo.npz               (3 KB, metadata only)
  sample_frames/                         (representative JPEGs)
```

Regeneratable by re-running the matching `scripts/bf15/*.py` script on a
CUDA-equipped pod.
