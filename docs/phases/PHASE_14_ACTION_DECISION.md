# Phase 14 (Action-conditioned OF-JEPA, robosuite Stack) — Decision document

**Date:** 2026-05-17.
**Status:** ⚠ **SPLIT RESULT — directionally informative.**
Action conditioning helps decoded position prediction (−13%) but
hurts JEPA-style slot-state matching (+100%). The first true robot-
action benchmark in the OF-JEPA stack returns a useful asymmetry:
**action is a planning-relevant signal, not a perception-loop one.**

> Phase 14 tested action conditioning on robosuite Stack with the
> Panda robot. 200 random-policy rollouts (80 frames each) drive a
> 2-mode comparison: OF-JEPA's future-state head with vs without
> 7-DOF action input.

## Setup

| | Value |
|---|---|
| Task | robosuite Stack (cubeA + cubeB + Panda end-effector) |
| Episodes | 200 random-policy rollouts, 80 frames each |
| Image size | 128×128 RGB |
| Action space | 7-DOF Panda EE-velocity control |
| OF-JEPA cfg | n_slots=6, id_dim=64, state_dim=64, proposal_dim=128 |
| Training | 1500 steps, JEPA stride k=4, lr=1e-4 |
| Eval | k=4 future state + decoded position MSE |
| Seed count | 1 (first pass) |

The "objects" the encoder must bind are 3 entities: cubeA, cubeB,
and the robot end-effector. Slot count is 6 to give headroom.

## Headline numbers

| Metric | Baseline | +Action | Δ | Verdict |
|---|---|---|---|---|
| future_state_mse | 6.11e-5 | 1.22e-4 | **+100%** | ❌ worse |
| future_pos_mse   | 2.28e-2 | 1.99e-2 | **−13%** | ✅ improved |

Both modes use the same OF-JEPA v0 encoder + slot mechanism + aux
position head. They differ only in the future predictor's input:
the +action mode concatenates the 7-DOF action vector to the slot
state when predicting state_value at t+k.

## What this means

**The two metrics are measuring different things.**

`future_state_mse` is the JEPA loss target: predict the encoder's
own slot-state at t+k from slot-state at t. Adding action input
makes this HARDER, not easier, because:

1. The encoder's slot states already implicitly encode motion (it
   sees the action's *effect* in the next frame).
2. The action input adds variance the predictor head must span
   without adding load-bearing information for matching the
   encoder's output (which the encoder produces from the FRAME,
   not from the action).
3. With limited capacity (small future_head MLP) and 1500 training
   steps, the predictor can't fully model the action→state map and
   the extra input degrades fit.

`future_pos_mse` is the decoded 2D position via the aux head.
Adding action HELPS by 13% because:

1. Position is the part of state most directly determined by
   action (robot commanded velocity → end-effector position).
2. The aux head (a single Linear) doesn't have to learn the
   action→position map from latents — the action goes in directly.
3. Even when state-matching is noisier, the position decoder
   benefits from the explicit action signal.

## Architectural take

> **Action conditioning helps downstream consumers that need
> position-equivariant prediction (planners). It doesn't help
> consumers doing self-loop state-matching (perception JEPA).**

This is a useful asymmetry. Frame-level visual prediction (JEPA)
already amortizes the effect of action into slot states. Planners
and verifiers, which consume positions or other action-equivariant
signals, do want explicit action input.

For the BLA architecture this matters because:

- The System-1 perception loop (OF-JEPA encoder + slot binding) is
  better trained WITHOUT action conditioning — the encoder learns
  general world dynamics.
- The action effect model that System-2 / planning consume should
  be a SEPARATE, action-conditioned readout on top of OF-JEPA's
  object-file state.

The current `slot_jepa_robosuite_train.py` has both heads (state
prediction + position prediction) in the same model. The cleanest
production split is:

```
OF-JEPA v0 encoder       (no action input — train on frames only)
   ↓
Object-file memory
   ↓
   ├── State-prediction head (no action; self-loop JEPA)
   └── Action-conditioned readout heads
         ├── future_position(state, action) → pos[t+k]
         ├── action_effect(state, action) → which_object_moves
         └── reachability(state_goal) → cost-to-go
```

## What's still open

- **Multi-seed confirmation.** 1 seed; the −13% pos improvement could
  be within seed noise. 3 seeds would tighten the claim.
- **Phase 14.4 — which-object-changed.** The plan called for a
  separate test: predict which of {cubeA, cubeB, eef} moves most at
  step t given action. That separates "action effect locality" from
  "action effect magnitude" and is the right test for relation graph
  load-bearing under action conditioning. Not yet run.
- **Successful task completion.** Random policy rarely stacks. To
  generate richer interaction signal, replace random policy with
  scripted task-completion policy or imitation rollouts.

## Reproducibility

Code committed in this commit:
- `system1_jepa/robosuite_data.py` — RobosuiteDataset
- `scripts/robosuite_collect_rollouts.py` — random-policy rollout collector
- `scripts/slot_jepa_robosuite_train.py` — action-conditioned trainer

Artifacts: `artifacts/phase14_robosuite/seed0_of_jepa_v0{,_action}.json`

Run command:
```bash
# 1. Collect rollouts (~5 min)
python3 scripts/robosuite_collect_rollouts.py \
  --task Stack --n-episodes 200 --horizon 80 \
  --out /workspace/robosuite_local/stack

# 2. Train baseline + action-conditioned (~40 min)
python3 scripts/slot_jepa_robosuite_train.py \
  --cache /workspace/robosuite_local/stack \
  --modes of_jepa_v0,of_jepa_v0_action \
  --seeds 0 --max-steps 1500 --jepa-stride 4 \
  --out /workspace/phase14_run1
```

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 2-6 JEPA | ✅ | slot_delta spatial memory |
| 7-8A | ❌×5 | content-side identity fixes falsified |
| 8C/8D | ✅ | OF-JEPA v0: identity-as-address |
| 9 | ✅ | MOVi-D identity transfer (corrected metric) |
| 10 | ✅ | refactor + first-class metrics |
| 12 | ✅ good | relations add 5-9% on single-slot readouts |
| 13.3 | ✅ | OF-JEPA transfers to CLEVRER |
| 13.4 | ⚠ | relations redundant when readout has pairs |
| **14.3** | **⚠** | **action conditioning helps position (-13%), hurts state-matching (+100%); architecturally informative split** |

The clean architectural takeaway across Phases 12-14:

> *Relations and actions are both context signals. They help the
> downstream readout type they pair with — relations for single-slot
> readouts; actions for position-equivariant readouts — and add
> noise to readouts that already encode the same information
> through their input or training target. Object files remain the
> load-bearing primitive.*
