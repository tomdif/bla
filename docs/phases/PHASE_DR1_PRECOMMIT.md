# Phase DR1 — Demo Retrieval Prior (Precommit)

**Date:** 2026-05-19.
**Status:** 🔒 **Gates locked before run.**
**Parent:** `docs/BLA_SCALING_ROADMAP.md` §4 + `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §4.

## Purpose

Scale Recipe E from "replay one fixed demo" to "**retrieve the closest
useful demo from a bank**". Test on PickPlaceCan with a 50-demo bank
(vs Phase D3's 5-demo cycling bank). The hypothesis: NN demo retrieval
beats fixed cycling because more of the env's reset distribution is
covered by some demo in the bank.

## Falsifiable claim

```
demo_retrieval_top1 (50-demo bank, retrieved by NN on initial
(can_pos, eef_pos)) outperforms demo_no_cem (fixed 5-demo cycling
bank, D3 protocol) on PickPlaceCan with state-matched reset to a
randomly chosen demo from the FULL 50-demo set.
```

The cycling bank only matches the env init when ep_id mod 5 lands
on the same demo as the random reset target. NN retrieval should
match much more often.

## Protocol

```
Task     : robosuite PickPlaceCan
Demo bank: first 50 demos from can_demo_v141.hdf5
Init     : state-matched reset to a RANDOMLY CHOSEN demo from
           the 50-demo set (not the cycling subset)
Modes    : demo_no_cem_cycle, demo_no_cem_oracle, demo_retrieval_top1,
           demo_retrieval_top3_avg, phase17_locked, naive_cem
Seeds    : 0, 1, 2 (parallel on GPUs 0/1/2)
Episodes : 30 per seed per mode (5 for pilot)
Sigma    : per-dim, σ_motion=0.12, σ_gripper=0
```

Mode definitions:

```
demo_no_cem_cycle         — D3 baseline; cycle through fixed 5 working demos
demo_no_cem_oracle        — ceiling; use the demo we reset to (perfect info)
demo_retrieval_top1       — NN by L2 on (cubeA_init_xy, eef_init_xy)
demo_retrieval_top3_avg   — average the top-3 retrieved action sequences elementwise
phase17_locked            — CEM σ=0.12 around demo_retrieval_top1
naive_cem                 — pure CEM (floor)
```

Retrieval features (initial state):

```
key = [
  can_pos_xy   (2),
  eef_pos_xy   (2),
  can_z, eef_z (2),
]
shape = (6,)
distance = L2 in this 6-D space
```

## Pre-committed gates

```
G1: demo_retrieval_top1 ≥ demo_no_cem_cycle − 0.05
G2: demo_retrieval_top3_avg ≥ demo_no_cem_cycle
G3: demo_retrieval_top1 beats phase17_locked by ≥ +0.10 improvement
G4: demo_retrieval_top1 success ≥ phase17_locked + 10pp
```

Strong pass:

```
SP1: demo_retrieval_top3_avg ≥ demo_no_cem_cycle on 3-seed mean
SP2: demo_retrieval_top1 has lower variance than demo_no_cem_cycle
```

Oracle ceiling (not a gate, just a reference):

```
demo_no_cem_oracle: expected ≈ 0.65–0.75 imp (matches Phase D3 numbers
on a per-demo basis; oracle picks the right demo every time)
```

## Pre-committed predictions

```
demo_no_cem_cycle:        expected imp 0.10–0.25 (cycling matches the
                          random reset target only ~10% of the time;
                          most episodes fail)

demo_no_cem_oracle:       expected imp 0.65–0.80 (perfect retrieval ceiling)

demo_retrieval_top1:      expected imp 0.40–0.70 (NN retrieval often
                          finds a close-enough demo; not always
                          perfect)

demo_retrieval_top3_avg:  expected imp 0.30–0.60 (averaging may hurt
                          slightly vs top1 because per-step action
                          averaging blurs grasp timing — but might
                          help on edge cases)

phase17_locked:           expected imp 0.10–0.25 (CEM around top-1
                          retrieval; same destructive-search pattern
                          as Phase D3/D4)

naive_cem:                expected imp 0.00 (floor)
```

## What this phase does NOT do

- **Does not change runtime.** Runtime is rolling-window K=5 per D1b,
  but this phase uses the existing batched evaluation runtime to keep
  variables separate (per user's caveat).
- **Does not bring CEM back as the primary mode.** phase17_locked is
  included only as the comparison baseline. The claim is that
  retrieval scales Recipe E *without* search.
- **Does not introduce a learned proposal policy.** Retrieval is a
  pure-logic nearest-neighbor over the demo bank. A learned policy
  is Phase DR2+.

## Falsification scenarios

```
F1: demo_retrieval_top1 ≤ demo_no_cem_cycle − 0.05 →
    NN retrieval doesn't help; Recipe E may not scale beyond
    cycling, OR the retrieval features (initial state geometry) are
    insufficient. Try richer features in DR2.

F2: demo_retrieval_top1 ≤ phase17_locked → CEM around retrieval helps
    more than retrieval alone. The doctrine "no CEM around demos"
    has a scope limit at fresh-NN-retrieved demos. Counter-finding.

F3: demo_retrieval_top1 ≥ demo_no_cem_oracle → retrieval beats oracle.
    Suspicious; suggests oracle isn't the right ceiling. Investigate.

F4: All modes ≤ 0.10 → environment regressed. Sanity fails.
```

## Locked
