# Phase 18κ Regime 3 — Lift fine-tune (Deferred)

**Date:** 2026-05-18.
**Status:** ⏸️ **DEFERRED — implementation blocker on scripted_lift prior.**

## Why this phase is deferred (not failed)

The Phase 18κ R3 architecture and pipeline are fully implemented:
- `scripts/phase18k_r3_lift.py` — Lift env + 10-dim engineered geo +
  3-stage FSM scripted prior + closed-loop oracle.
- `scripts/phase18k_r3_full.py` — collection + 4-recipe training +
  decile diagnostics + 5-recipe eval, parallel across seeds.

Both scripts smoke-tested cleanly on pod: Lift env builds, OF-JEPA
encodes Lift frames, value-head architectures train.

**The blocker is the scripted_lift prior.** Multiple iterations
(reach_height sweep, longer grasp, slower lift) all fail to produce
nonzero `cube_z_gain` across 20 pilot episodes. The pod-side trace
shows the gripper closes at a position where fingertips don't make
contact with the cube — likely a robosuite-specific gripper-offset
detail that requires more careful reverse-engineering than this
session has time for.

Without an informative scripted prior:
- Collection produces all-zero labels → value heads learn nothing
- Adapter geo-recovery on z-features is ~zero (height Spearman -0.32 / -0.37)
- All eval modes regress to floor

This is an implementation gap, not an architectural finding. The
BLA recipe family verdict from Phase 18η/18λ/18μ/18κ R2/18ν stands.

## What would unblock R3

Three potential approaches (any one):

1. **Use robomimic Lift demos as the scripted prior** — Phase 16
   established that we have access to robomimic data; demos
   reliably grasp and lift. Substitute these for the FSM scripted
   prior in `rollout_scripted_lift_prior`.

2. **Carefully tune the FSM** with robosuite-specific gripper
   offsets, possibly using a published Panda Lift heuristic from
   robosuite examples.

3. **Reframe Phase 18κ R3** to a different task that's easier to
   script (e.g., NutAssembly, Door, or even a Push variant with
   different cube mass/friction) — though those move away from
   the "vertical geometry" test the original R3 was specifically
   designed for.

## What this phase doesn't change

The Phase 18κ Regime 2 (OOD goal-distance shift) result stands as
the cross-distribution transfer test:
- supervised retained 94% on OOD
- end2end inverted to OOD winner
- adapter geo-recovery essentially unchanged across distributions

The Phase 18ν result stands for in-dist vs OOD recipe family
analysis (annealed fails OOD; pretrain+ft is the best of the
schedule variants but doesn't strictly unify).

The locked planning recipe family (Recipes A/B/C/D, deployment-
conditional) is unchanged.

## Reproducibility

If/when R3 is unblocked:

```bash
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$SEED \
  nohup python3 -u scripts/phase18k_r3_full.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 200 --n-eval-episodes 30 \
    --out /workspace/phase18k_r3_seed${SEED} --seed $SEED &
done
```

with whichever scripted prior fix is chosen (1, 2, or 3 above).

The precommit doc `PHASE_18K_REGIME3_LIFT_PRECOMMIT.md` (commit
`9d88ad4`) remains the locked design + gates.
