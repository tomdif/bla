# Phase 18κ Regime 3 — Lift fine-tune (Decision)

**Date:** 2026-05-18.
**Status:** ✅ **Run completed (3 seeds × 6 modes × 30 eval eps).**
**Verdict:** Registered gates G1/G4 FAIL. Unregistered control `demo_no_cem` dominates 3/3 seeds.

## Single-line headline

> **For Lift, expert-demo replay beats all CEM-refined variants.**
> The transferable object is the demonstration manifold, not CEM exploration around it.

## Three-seed aggregate

| Mode | imp_mean | imp_std | z_gain (m) | succ_3/3 |
|---|---:|---:|---:|---:|
| **demo_no_cem** | **0.333** | **0.054** | **0.108** | **3/3** |
| phase17_locked | 0.244 | 0.079 | 0.067 | 3/3 |
| combined_sum_supervised | 0.200 | 0.027 | 0.058 | 3/3 |
| combined_sum_geo | 0.144 | 0.068 | 0.035 | 3/3 |
| combined_sum_end2end | 0.122 | 0.057 | 0.031 | 3/3 |
| naive_cem | 0.000 | 0.000 | −0.010 | 3/3 (floor) |

`demo_no_cem` beats every CEM-augmented mode at every seed.

Per-seed `demo_no_cem` vs best CEM:

| Seed | demo_no_cem | best CEM mode | Δ |
|---:|---:|---:|---:|
| 0 | 0.333 | phase17_locked 0.300 | +0.033 |
| 1 | 0.400 | phase17_locked 0.300 | +0.100 |
| 2 | 0.267 | combined_sum_supervised 0.233 | +0.033 |

## Registered gates (from `PHASE_18K_REGIME3_LIFT_PRECOMMIT.md`)

- **G1** (best learned recipe ≥ locked + 0.02): best learned `combined_sum_supervised` = 0.200, locked = 0.244, 0.200 < 0.264 → **FAIL** (and locked itself is below `demo_no_cem`)
- **G2** (pretrain+ft within 0.02 of sup and end2end): pretrain+ft head trained but not eval'd in focused set → **N/A**
- **G3** (some learned-adapter ≥ 0.90 × geo): sup 0.200 ≥ 0.130 (0.90 × geo 0.144) → technical PASS but **moot** (geo is the weakest)
- **G4** (adapter cube_z/eef_z Spearman ≥ 0.30): cube_z = −0.059, eef_z = 0.092 → **FAIL**

Adapter overall is healthy (mean_val_spearman 0.444 ± 0.035 across 10 geo features) — it recovers planar features (x, y, push_dir) cleanly but does **not** recover vertical (z) features at this 160/40-sample fine-tune. The adapter is fine; the held-out z-channel just isn't there to recover at this sample size.

## Updated architectural lesson

The previous lesson from the Lift pilot (n=5) was:
> *"demo prior + value/adapters transfers to Lift."*

The updated lesson, at 3 seeds × n=30, is:
> **"demo prior transfers; CEM around the demo harms Lift."**
> **The correct search budget around expert demonstrations is zero, or near-zero structured search.**

This is the extreme case of [[less-search-when-score-anti-correlated]]: when the prior is already on a fragile manifold, any Gaussian perturbation produces off-manifold samples that no learned scorer can reliably rank back.

## Why this makes sense

Lift is contact- and timing-sensitive. The demo encodes:
- approach pose,
- gripper-close timing,
- contact point,
- lift trajectory,
- small wrist/EEF corrections.

Even after the **Path A fix** (per-dim sigma masking, σ_gripper = 0), Gaussian noise on dims 0–5 corrupts the approach/lift geometry. Candidates land off-manifold; the value head cannot reliably re-rank them back onto the demo manifold. So `demo_no_cem` wins.

Push (Stack) didn't show this because the scripted FSM prior was noisy enough to leave headroom for CEM and a learned scorer. Expert demos don't leave headroom.

## What this phase falsifies / validates

| Claim | Status |
|---|---|
| BLA recipe family (Stack locked → Lift) transfers via fine-tune | **PARTIAL** — transfers in form (no errors, training works), but not in **advantage** over the demo prior alone |
| Action-space CEM with σ=0.12 refines an expert demo prior | **FALSIFIED** for Lift |
| The transferable object is the value head + adapter geometry | **FALSIFIED** here — the transferable object is the demonstration manifold |
| Gripper-bit semantic preservation is necessary for demo-replay + CEM | **VALIDATED** (without it, mean_imp = 0; with it, demo_no_cem reaches 0.333) |

## Engineering unblock (durable)

- `scripts/phase16_policy_prior_mpc.py`: `cem_with_prior` now accepts per-dim `sigma` and `sigma_floor` (scalar or vector, broadcast to action_dim).
- `scripts/phase18k_r3_lift.py`: `rollout_demo_lift_prior` loads robomimic demo actions and replays them as the prior.
- `scripts/phase18k_r3_full.py`: `lift_sigma()` helper produces `[σ, σ, σ, σ, σ, σ, 0]` to preserve the gripper bit; `demo_no_cem` mode added.

This combination (demo-replay prior + per-dim sigma mask) is reusable for any future expert-demo-based phase.

## Next recipe directions (suggested, not yet committed)

Replace action-space Gaussian CEM with one of:

1. **Demo selection only** — rank a small bank of raw demos by value head, pick best, **no perturbation**.
2. **Time-warp search** — adjust demo speed / phase, not action values.
3. **Low-dimensional residual search** — only 2–4 smooth parameters (e.g. approach height, grasp delay).
4. **Tiny residual noise** — σ = 0.01–0.02 only on XY pre-grasp (not Z, not post-grasp).

Common thread: search in a structured subspace that respects the demo manifold, not in raw per-step action space.

## What does NOT change

- BLA recipe family verdict on Stack push (Phase 18η/λ/μ/ν): **unchanged**. The combined_sum recipes win on scripted-FSM-prior tasks.
- The locked planning recipe for Stack push (Recipe A through D): **unchanged**.
- Object-file architecture, value-head architecture, adapter architecture: **all unchanged**.
- The supervised adapter recovers planar geometry on Lift (0.44 Spearman across features). The bottleneck is the z-channel, not the adapter.

## Files

- Decision doc: this file (`docs/phases/PHASE_18K_REGIME3_LIFT_DECISION.md`).
- Pre-commit: `docs/phases/PHASE_18K_REGIME3_LIFT_PRECOMMIT.md`.
- Deferred-and-then-unblocked memo: `docs/phases/PHASE_18K_REGIME3_DEFERRED.md` (Path 1 + Path 4 both attempted, both worked, but produced this falsification).
- Scripts: `scripts/phase18k_r3_lift.py`, `scripts/phase18k_r3_full.py`, `scripts/phase16_policy_prior_mpc.py` (per-dim sigma).
- Pod artifacts: `/workspace/phase18k_r3_seed{0,1,2}/{summary.json,per_episode_*.jsonl,rollout_cache_lift.npz}`.

## Final R3 interpretation

> **The transferable object across tasks is the demonstration manifold, not CEM exploration around it.**

That falsifies one specific extension of the BLA recipe family (action-space CEM refinement of expert demos for Lift) while validating the underlying engineering (per-dim sigma, demo-replay prior). Useful negative result.
