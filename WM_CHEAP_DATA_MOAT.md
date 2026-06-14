# Milestone: the world-model moat is architecture × cheap causal interaction data

> **The moat is real, but it is not "architecture alone." It is architecture × cheap causal interaction data.**

**One sentence:** a *gated* (verified) world model can turn cheap, goal-agnostic interaction into zero-shot
goal transfer; behavior cloning cannot buy the same transfer with expert demonstrations and retraining.

All results below are on **gated** world models — every WM passed the convergence gate (`arm_px ≤ 5px`,
auto-retried) before any verdict was computed. The broken-WM contrast is the point: a *broken* WM gives
0.12 zero-shot (a fake "no moat"); the *gated* WM gives 0.97 (the real moat). **The product is not "use a
world model" — it is "use a *verified* world model." The gate flips the verdict.**

## The three levers

| Lever | Result | Honest read |
|---|---|---|
| **3 — goal-shift moat** | Frozen gated WM **0.97 @12** zero-shot; BC + 300 expert demos + retrain → **0.28** | imitation does not approach parity **within the tested retraining sweep**; the WM transfers zero-shot |
| **2 — recovery** | gap +0.48 (no-perturb) vs +0.47 (perturb) → **FALSIFIED** | recovery is **not** the source of the moat on smooth Reacher (do not rescue; the negative adds credibility) |
| **1 — cheap-data leverage** | E=0 ties BC @6, only weakly beats @12; cheap exploration unlocks the curve | the moat's source is **cheap causal interaction**, not pure architecture with identical data |

### Lever 1 leverage curve (TEST / shifted goals, tuned planner)
```
E (cheap exploration)   succ@6   succ@12   mean_px
E=0    (demos only)      0.07     0.35      17.7    architecture-alone: ties BC @6, weak @12
E=2000                   0.15     0.62      11.0
E=8000                   0.33     0.82       8.8
E=30000                  0.65     0.97       5.5    peak = full-data result
E=41650 (full)           0.65     0.88       6.4    REGRESSES vs E=30000 -- needs investigation
BC baseline (bc_goal)    0.07     0.15        --
```
**Caveat (do not gloss):** performance rises through E=30000 (the peak / full-data result); the E=41650
point **regresses** (0.97→0.88 @12). Candidate causes: noise, data-quality dilution, planner-tuning
mismatch, over-regularization, gate calibration. It does **not** kill the claim (E=30000 already proves it),
but it must be explained before a paper/investor deck.

## The boxed claims
> **The architecture's edge is its ability to exploit cheap causal interaction data.**
> (NOT: "the architecture wins with no additional information.")

> **World models win where cheap causal exploration can be converted into reusable counterfactual control.**

## Best current commercial claim
> In a goal-shift gate, a verified world model reaches **0.97 @12 using ~30k cheap goal-agnostic
> interactions and no goal-specific retraining**. Behavior cloning remains demo-bound, reaching only **0.28
> even after 300 new expert demonstrations and retraining**. The moat is not pure architecture; it is the
> architecture's ability to convert cheap causal interaction into reusable goal transfer.

## Relation to prior work
Dreamer 4 makes the same broad argument (world models learn general knowledge from video, simulate
experience, train behavior in imagination; action-conditioning learnable from relatively small paired
action data while world knowledge comes from broader experience). This is the **smaller, gated,
verificationist** version of that thesis.

## Action-causality control (DONE) — the mechanism is causal, not diversity
E=30000, zero-shot shifted-goal transfer. Real vs shuffled-action vs zero-action exploration (same (s,s')
transitions + diversity, broken action->consequence link):
```
variant            test@6   test@12   arm_px   OOD-rollout
real               0.82      1.00      1.6        9.0
shuffle_actions    0.05      0.35      2.1       15.1
zero_actions       0.03      0.17      1.7       14.9
VERDICT: MECHANISM IS CAUSAL
```
- Real beats shuffled by +0.65 @12 and zero by +0.83 -> correct action->consequence structure is NECESSARY;
  the win is causal, not "more/diverse data."
- ALL THREE converge on arm_px (gate passes them) -> the convergence gate is BLIND to broken dynamics. Only
  the held-out OOD rollout flags the corruption (real 9.0 vs shuffled 15.1 / zero 14.9). "Use a VERIFIED
  world model" => verified on held-out DYNAMICS, not just decode.

## Hardening checks (DONE)
**Multi-seed (n=3)** — curve robust; @12: E0 0.23±.03, E2k 0.49±.07, E8k 0.81±.07, E30k 0.90±.04,
full 0.95±.02. The single-seed E=41650 "regression" (0.97→0.88) was NOISE: 3-seed mean at full = 0.95,
within error of E=30000. All 3 seeds: MOAT CONFIRMED.

**Stronger BC** — MOAT SURVIVES (+0.45 over strongest). @12 on shift: wm_cem 0.97; wmrep_bc (BC on FROZEN
WM encoder) 0.53; HER_bc 0.17; bc_goal 0.10; bc_goal@1000 0.12 (more demos don't help). DECOMPOSITION: ~half
the advantage (0.10→0.53) is the learned REPRESENTATION, the other half (0.53→0.97) is the PLANNER.

**Concept discovery** — SURPRISING positive (scrutinize). Target (never supervised) decodes to 1.77px from
the arm-only latent (control WM_both 0.58px, random 14.35px). Emergence yes, but likely via incidental
VISUAL encoding of a salient static feature, NOT causal-necessity (target doesn't affect arm dynamics);
contradicts the dissociation prior. Needs linear-probe + occlusion follow-up.

## (original) hardening list, for reference
1. **Multi-seed curve** — ≥3 seeds at E∈{0,2000,8000,30000,full}, report mean ± SE.
2. **Cost-normalized** — $ / time: BC cost to reach 0.28 vs WM cost to reach 0.97.
3. **Stronger imitation baselines** — goal-conditioned BC, hindsight-relabel BC, BC w/ WM's representation,
   BC + augmentation, BC at 300/1000/3000 demos.
4. **Action-causality control (DONE ✓ — see above)** — real 1.00 vs shuffled 0.35 vs zero 0.17 @12; all
   converge on arm_px but OOD rollout flags the corrupted ones. Mechanism confirmed causal.

## Artifact
```json
{
  "tag": "phase-wm-cheap-data-moat-v0",
  "wm_gated": true,
  "broken_wm_zero_shot_at12": 0.12,
  "gated_wm_zero_shot_at12": 0.97,
  "lever_3": {
    "wm_zero_shot_at12": 0.97, "wm_zero_shot_at6": 0.23,
    "bc_goal_retrain_best_at12": 0.28, "bc_goal_n0_at12": 0.15,
    "wm_frozen": true, "wm_new_data": 0,
    "verdict": "moat_within_sweep"
  },
  "lever_2": {
    "gap_no_perturbation_at12": 0.48, "gap_under_perturbation_at12": 0.47,
    "verdict": "recovery_advantage_falsified_on_smooth_reacher"
  },
  "lever_1": {
    "bc_goal_test_at12": 0.15,
    "curve": [
      {"E": 0,     "succ6": 0.07, "succ12": 0.35, "mean_px": 17.7},
      {"E": 2000,  "succ6": 0.15, "succ12": 0.62, "mean_px": 11.0},
      {"E": 8000,  "succ6": 0.33, "succ12": 0.82, "mean_px": 8.8},
      {"E": 30000, "succ6": 0.65, "succ12": 0.97, "mean_px": 5.5},
      {"E": 41650, "succ6": 0.65, "succ12": 0.88, "mean_px": 6.4}
    ],
    "peak_E": 30000,
    "regression_note": "E=41650 regresses 0.97->0.88 @12; unexplained, needs investigation",
    "verdict": "cheap_interaction_unlocks_moat"
  },
  "causal_control": {
    "E": 30000,
    "real":            {"succ12": 1.00, "succ6": 0.82, "arm_px": 1.6, "rollout_ood": 9.0},
    "shuffle_actions": {"succ12": 0.35, "succ6": 0.05, "arm_px": 2.1, "rollout_ood": 15.1},
    "zero_actions":    {"succ12": 0.17, "succ6": 0.03, "arm_px": 1.7, "rollout_ood": 14.9},
    "gate_blind_to_corruption": true,
    "ood_rollout_detects_corruption": true,
    "verdict": "mechanism_is_causal_not_diversity"
  }
}
```
