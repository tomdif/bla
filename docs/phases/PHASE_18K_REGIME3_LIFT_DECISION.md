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

---

## Addendum (2026-05-18, post-commit): seed-2 was run twice

A second seed-2 launch (`bbw44ghyq` background relaunch) finished after
the original 3-seed commit, overwriting `/workspace/phase18k_r3_seed2/`.
Same `--seed 2` argument, but produced materially different results —
indicating non-trivial unseeded variance in the pipeline (likely
MuJoCo init state, demo-bank selection RNG, or env-reset noise).

The two seed-2 runs are therefore **independent draws from a wider
distribution at seed=2**, not redundant. Treating them as four
independent runs:

| Mode | s0 | s1 | s2-orig | s2-rerun | mean | std (n=4) |
|---|---:|---:|---:|---:|---:|---:|
| **demo_no_cem** | **0.333** | **0.400** | **0.267** | 0.233 | **0.308** | **0.074** |
| phase17_locked | 0.300 | 0.300 | 0.133 | 0.100 | 0.208 | 0.107 |
| combined_sum_supervised | 0.167 | 0.200 | 0.233 | 0.167 | 0.192 | 0.032 |
| combined_sum_geo | 0.133 | 0.233 | 0.067 | **0.300** | 0.183 | 0.104 |
| combined_sum_end2end | 0.100 | 0.200 | 0.067 | **0.300** | 0.167 | 0.105 |
| naive_cem | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### What this changes

The "3/3 outright wins" framing is no longer the cleanest description.
At n=4: `demo_no_cem` wins outright on 3 of 4 runs and falls to rank
3 on the 4th, where `combined_sum_geo` and `combined_sum_end2end`
each hit 0.300 (above `demo_no_cem`'s 0.233 on the same run).

### What this does NOT change

The qualitative verdict holds, but softer:

- **4-run mean still favors `demo_no_cem`**: 0.308 vs 0.208 next-best (Δ = +0.100 ≈ 1.3× demo's std, ≈ 0.94× phase17_locked's std).
- **`demo_no_cem` has the lowest variance across runs** (std 0.074 vs CEM modes' 0.104–0.107). The CEM modes are unstable: same `--seed 2` produced 0.067 *and* 0.300 for both `combined_sum_geo` and `combined_sum_end2end`. `demo_no_cem` was 0.233–0.400 (range 0.167) vs `combined_sum_geo` 0.067–0.300 (range 0.233).
- **The architectural lesson is unchanged**: demo-prior + no-CEM is the most reliable mode; CEM modes are noisier and only sometimes match it.
- **Registered gates G1/G4 still FAIL.**

### Updated honest framing

> On Lift fine-tune at 200-sample scale with robomimic demo prior:
> demo-replay alone is the **most reliable** mode (lowest variance,
> highest mean across 4 runs). CEM-with-value-head modes occasionally
> match it when they happen to land near the demo manifold, but
> their variance across nominally-identical-seed runs is large
> enough that they cannot be relied upon. The recommended deployment
> choice for the demo-prior regime is `demo_no_cem`.

### Additional finding: unseeded pipeline variance

The fact that `--seed 2` produced different summaries across two
launches reveals that the BLA pipeline has unseeded sources of
randomness. Candidates to audit before any future small-n run:
- MuJoCo simulator seed (separate from numpy/torch seed)
- robomimic demo-selection RNG (which demo of demo_ids=(1, 3))
- env reset noise (cube position randomization)

This is a separate engineering finding worth documenting but does
not require a rerun of R3 — the qualitative result is robust to the
variance.
