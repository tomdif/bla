# Phase D4 — NutAssemblySquare Doctrine Validation (Decision)

**Date:** 2026-05-19.
**Status:** ✅ **STRONG PASS — all 4 gates clear, strong-pass triggered.**
**Precommit:** `docs/phases/PHASE_D4_PRECOMMIT.md` (commit `e2b8ad1`).

## Headline

> **The demo-prior doctrine holds on NutAssemblySquare too.**
> `demo_no_cem` beats `phase17_locked` by +0.400 improvement and
> +38.6pp success across 3 seeds × 30 episodes. naive_cem stays
> at floor. Recipe E now has **three independent cross-task
> validations**: Lift (Phase 18κ R3), PickPlaceCan (Phase D3-main),
> and NutAssemblySquare (Phase D4).

## Three-seed aggregate (n=30 per seed per mode)

| Mode | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| **demo_no_cem** | imp 0.663 / 63% | imp 0.665 / 63% | imp 0.626 / 60% | **0.651 ± 0.022 / 62% ± 2pp** |
| phase17_locked | imp 0.197 / 17% | imp 0.188 / 17% | imp 0.367 / 37% | 0.251 ± 0.099 / 24% ± 12pp |
| naive_cem | 0.000 / 0% | 0.000 / 0% | 0.000 / 0% | 0.000 / 0% (floor) |

## Gate evaluation

| Gate | Threshold | Observed | Result |
|---|---|---:|---|
| **G1** (Δ_imp ≥ +0.10) | +0.10 | **+0.400** | ✅ PASS by 4× |
| **G2** (Δ_succ ≥ +10pp) | +10pp | **+38.6pp** | ✅ PASS by 3.9× |
| **G3** (demo > locked on ≥ 2/3 seeds) | 2/3 | **3/3** | ✅ PASS |
| **G4** (naive_cem ≤ 0.10) | ≤ 0.10 | 0.000 | ✅ PASS |
| **Strong-pass** (Δ≥+0.25 AND Δ_succ≥+25pp) | both | Δ=+0.40, Δ_succ=+38.6pp | ✅ **YES** |

## Variance pattern (P5 again confirmed, even stronger)

```
demo_no_cem    σ_imp = 0.022   (lowest variance yet of any nontrivial mode)
phase17_locked σ_imp = 0.099   (Square is noisier than PickPlaceCan;
                                seed 2's lucky-CEM inflated this)
naive_cem      σ     = 0.000   (deterministic floor)
```

demo_no_cem is **even more reliable on Square than on PickPlaceCan**
(σ 0.022 vs 0.043). The recipe's value is becoming clearer with each
task: not just better mean, but lower variance — the demonstration
manifold is the most STABLE policy object.

## Why the effect is smaller than on PickPlaceCan

| Task | Δ_imp | Δ_succ |
|---|---:|---:|
| Lift (4-run aggregate) | +0.10 | +10pp |
| PickPlaceCan (3 seeds) | +0.564 | +56.7pp |
| **NutAssemblySquare (3 seeds)** | **+0.400** | **+38.6pp** |

Two factors:
1. **Demo bank**: 17 of 20 Square demos work on state-matched reset
   (vs 5 of 20 for PickPlaceCan). With more working demos, the
   "demo_no_cem" baseline gets higher (62% success on Square vs 71%
   on PickPlaceCan — comparable). But also `phase17_locked` benefits
   when CEM noise happens to compress demo timing into the eval
   window (Square at plan_horizon=30 cuts off some demos' natural
   lift point, which CEM occasionally rescues).
2. **Seed 2 outlier**: seed 2's phase17_locked = 0.367, well above
   seeds 0 / 1 at ~0.19. The "lucky CEM" mechanism is present but
   doesn't dominate — even on the lucky seed, demo_no_cem at 0.626
   still beats it by +0.259.

## Falsification scenarios (from precommit)

| Scenario | Result |
|---|---|
| F1 (CEM matches demo within ±0.02) | ❌ Did not fire (Δ=+0.40) |
| F2 (demo_no_cem ≤ 0.10) | ❌ Did not fire (demo_no_cem = 0.651) |
| F3 (phase17_locked beats demo) | ❌ Did not fire (3/3 seeds favor demo) |
| F4 (demo_no_cem < naive_cem) | ❌ Did not fire (650× better than floor) |

**Zero falsification triggers.** The doctrine is robust across three
independent contact-sensitive task families now.

## Doctrine implication

The BLA architecture spec's regime-map prediction is now validated
on:
- **Grasp-and-lift** (Lift)
- **Grasp-and-place** (PickPlaceCan)
- **Grasp-and-insert** (NutAssemblySquare)

These cover three meaningfully different contact-sensitive
constraint structures. The "demo manifold is the transferable
object" rule has predictive power across this family.

The unified statement:

> In demo-prior / contact-sensitive regimes, `demo_no_cem` is the
> default. The demonstration manifold is both the **best-performing**
> AND **most reliable** policy object. Action-space CEM around the
> demo is destructive on average and only occasionally helps when
> the demo's timing doesn't fit the eval horizon.

## What this does NOT yet establish

- **P4 (combined_sum_\* doesn't recover demo_no_cem)**: still deferred.
  After three cross-task wins for Recipe E, the falsification test
  "can value-guided CEM ever recover demo_no_cem?" becomes more
  interesting but also less likely to overturn the doctrine.
- **Articulated dynamics** (Door / lever / valve): still not tested
  directly. Square is precise insertion, which is contact-sensitive
  but not articulated. Future work.
- **FSM-prior regime variants**: Stack push was the original
  validation in Phases 14-18; the FSM side of the regime map hasn't
  been re-stress-tested cross-task. Lower priority since the
  FSM-prior side has 4+ phases of validation already.

## Updated cross-task evidence table

```
Task                           Δ_imp(demo − locked)    Δ_succ   Notes
─────────────────────────────────────────────────────────────────────
Lift (Phase 18κ R3, 4 runs)         +0.10              +10pp   first demo-prior validation
PickPlaceCan (Phase D3, 3 seeds)    +0.564             +56.7pp  largest effect
NutAssemblySquare (Phase D4, 3 sd)  +0.400             +38.6pp  precise-insertion
```

This belongs in `BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §5 (regime
map) and §8 (evidence table).

## Decision

**The demo-prior doctrine is now cross-task validated on three
independent contact-sensitive task families.** Recipe E (`demo_no_cem`)
is established as the default for the demo-prior regime, with the
applicability rule strengthened:

> "When the prior is an expert demonstration manifold, `demo_no_cem`
> is the default. The demonstration manifold is the policy. Do not
> add CEM unless there is a calibrated trust-region reason to do so."

Locked next:
1. Update architecture spec §5 + §8 with D4 row.
2. Memory update: cross-task validation now N=3.
3. Defer further task-arc growth (Door if demos surface, ToolHang,
   Transport) until there's a specific reason to add them.
4. Consider P4 value-head recovery test next session if the doctrine
   needs a deeper falsification probe.

## Files

- Precommit: `docs/phases/PHASE_D4_PRECOMMIT.md` (commit `e2b8ad1`)
- Decision: this file
- Task primitives: `scripts/phase_d4_square.py`
- Pilot/main eval: `scripts/phase_d4_pilot.py`
- Pod artifacts: `/workspace/phase_d4_main_seed{0,1,2}/summary.json`

## Locked
