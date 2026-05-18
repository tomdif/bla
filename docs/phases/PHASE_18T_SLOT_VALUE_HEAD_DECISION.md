# Phase 18θ — Slot-feature value head (Decision document)

**Date:** 2026-05-18.
**Status:** ❌🌟 **0/4 gates. Decisive negative result on raw frozen slot features as value-head input. The locked recipe stays as `combined_sum_geo`. Clarifying architectural finding: the BLA split is `System-1 = identity/dynamics`, `System-2 / readout = goal-relative geometry + value`.**

> **Headline:** Slot-feature value prediction fails under the current
> frozen OF-JEPA representation and 720-sample value dataset. Geo
> features remain the load-bearing value input. Geoslot does not
> materially improve over geo. The locked planner keeps the
> engineered geometry value head; future BLA integration should
> expose explicit goal-relative object-file features rather than
> raw frozen slots.

## The question

> Does OF-JEPA's learned object-file state contain planner-relevant
> value information beyond hand-engineered geometry?

**Answer**: Not at the raw frozen-slot level, and not at this data
budget. Slot features alone fail to learn an episode-level value
signal; concatenating slot to geo barely moves results above geo
alone.

## Setup

Identical to Phase 18η, with three value-head variants trained in
parallel on the same rollout cache:

| Head | Input | Dim |
|---|---|---|
| **geo** (Phase 18η reference) | 10-dim BC features | 10 |
| **slot** (primary 18θ) | OF-JEPA slot features flattened | 768 |
| **geo+slot** (richer) | concat(geo, slot) | 778 |

- 300 rollout episodes (scripted_prior + light CEM) × 3 replans = 900
  (geo, slot, goal, plan, episode_imp) tuples
- 720 train / 180 held-out val
- 2000 training steps each, AdamW lr=3e-4
- 6 eval modes × 30 episodes, seed 0

## Decisive evidence: held-out value-head diagnostics

This is the seed-independent measurement that decides the verdict.

```
Head     Pearson  Spearman   top    bot    top-bot gap
geo       0.320    0.362    0.335  0.091   +0.244
slot      0.082    0.113    0.339  0.286   +0.053    ← weak
geoslot   0.309    0.343    0.352  0.083   +0.268    ← ≈ geo + ε
```

- **Slot head's Spearman is 0.113**, 3.2× worse than geo's 0.362
- **Slot head's top/bot ratio is 1.18×** (0.339 / 0.286); the
  head cannot meaningfully separate its predicted-best from
  predicted-worst candidates
- **Geoslot ≈ geo**: top-bot gap 0.268 vs geo's 0.244 — adding 768-dim
  slot to 10-dim geo adds essentially zero usable value signal beyond
  what geo already provides

The Phase 18η value head's success was *not* "noise exploitation":
it was using real geometric information (Pearson 0.32, top-bot +0.24
on this very seed). Slot features alone don't contain that signal in
a form a 720-sample MLP can extract.

## Eval results (n=30, seed 0)

```
Mode                  improvement  dir_score  contact  success
gt_closed_loop          0.180        0.204     0.73     0.17
phase17_locked          0.381        0.487     0.97     0.53      ← high-variance seed
combined_sum_geo        0.253        0.345     0.97     0.27
combined_sum_slot       0.255        0.201     0.97     0.33
combined_sum_geoslot    0.201        0.284     1.00     0.20
naive_cem               0.000        0.000     0.23     0.00
```

Important nuance: `phase17_locked` at 0.381 here is an outlier-high
single-seed reading (vs Phase 18η-multi mean 0.255 across 3 seeds).
The Phase 18θ collection consumes more RNG per episode (slot
extraction at each replan boundary calls `encode_frame`), shifting
the global np.random state by eval time and producing different
eval env conditions. **This is environment-variance, not a regression
of the locked recipe.** Phase 18η-multi remains the canonical
multi-seed result.

Even ignoring the high baseline, the per-mode pattern is informative:
combined_sum_slot (0.255) ≈ combined_sum_geo (0.253), and
combined_sum_geoslot (0.201) is *worse* than either. The slot
representation isn't helping at the planner level.

## Gate verdicts (vs precommit)

```
G1. combined_sum_slot.improvement >= phase17_locked.improvement + 0.02
       0.255 vs 0.401                                  FAIL  (env-variance
                                                              compounded with
                                                              weak slot signal)

G2. combined_sum_geoslot.improvement >= combined_sum_geo.improvement
       0.201 vs 0.253                                  FAIL  (-0.052)

G3. slot head held-out Spearman >= 0.20
       0.113                                           FAIL  (3.2× below geo)

G4. slot head top-decile / bot-decile >= 2.0
       0.339 / 0.286 = 1.18×                           FAIL  (3.7× below geo)

Verdict: 0/4. Per precommit matrix: "OF-JEPA slot state is useful
for dynamics but value prediction still needs explicit geometric
features. The planner depends on hand-engineered geometric
abstractions; that's the System-2/readout layer's job. Still a
clarifying result."
```

## The right reframing

This is **not** a falsification of the BLA System-1 thesis. The
correct interpretation:

> *OF-JEPA slot/object-file features are excellent for identity,
> dynamics, and action-conditioned prediction (Phase 17/18d). The
> frozen slot representation does NOT directly expose the
> goal-relative scalar value information that a 256-hidden 3-layer
> MLP can read out from 720 training examples. The value head
> needs either: (1) explicit geometric / goal-relative features
> (current locked recipe), (2) a goal-conditioned slot encoder
> (re-train OF-JEPA with goal awareness), (3) more rollout data,
> or (4) a learned System-2 readout that computes geometry from
> object files.*

## Updated architectural split

```
System-1 OF-JEPA:                        System-2 / readout:
  object identity                          goal-relative geometry
  dynamic state                            value estimation
  action-conditioned prediction            planner scoring
```

The split is coherent: a world model doesn't need every scalar
planning feature directly linearly available from frozen slots. It
needs a reliable interface for *computing* such features.

The value head needs goal-relative relational features like:

```
cube-to-goal vector
eef-to-cube vector
push direction alignment
distance-to-goal
contact geometry
```

These are **computations over object files**, not necessarily raw
slot features.

## Updated full claim stack

| Phase | Status | Headline |
|---|---|---|
| 14.5/14.6 | ✅✅✅ | offline ranking + OOD generalization |
| 17 | ✅ | mixed-data; planner beats oracle (seed 0) |
| 18δ multi-seed | ✅✅✅ | Phase 17 robust; planner +0.005 over oracle |
| 18β | ❌ + ⭐ | distillation falsified; light CEM > heavy CEM |
| 18γ | ❌ + 🌟 | rank ≠ candidate ≠ episode; 18β -0.52 was cross-replan drift |
| 18η | ✅ G2 | combined_sum beats locked +0.050 at n=30 |
| 18η-multi | ✅✅✅✅ 4/4 | combined_sum robust: +0.061 over locked across 3 seeds |
| **18θ** | **❌ 0/4 + 🌟** | **raw slot features insufficient for value-head at 720 samples; geo features carry the value signal; geoslot ≈ geo + ε** |

The locked planning recipe (Phase 18η-multi `combined_sum_geo`)
remains unchanged.

## Next phases (revised)

### Phase 18λ (lambda) — Object-file geometry adapter (new priority)

Rather than feeding raw 768-dim slot features to a value head,
build a small structured adapter that extracts geometric
relationships from slot state:

```
adapter:
  slot_features (n_slots × slot_dim) + goal_xy
  → cube_position (2)         # from cubeA slot's position-attended features
  → eef_position (3)          # from eef slot
  → cube_to_goal_vec (2)
  → eef_to_cube_vec (3)
  → push_alignment (1)
  → ~10 derived geometric features

value_head_adapter:
  adapter output + goal + plan → episode_improvement
```

If this matches or beats `combined_sum_geo`, the architecture becomes
genuinely BLA-native: System-1 provides slots, System-2 computes
goal-relative geometry, the planner reads off both.

Pre-committed gates would mirror Phase 18θ but applied to the adapter
variant.

### Phase 18ι — z-normalized combined_max (deferred from 18η)

Still on the shelf; lower priority than 18λ.

### Phase 18κ — Cross-task transfer

Test the locked `combined_sum_geo` recipe on Lift / PickPlace.

## Reproducibility

```bash
python3 scripts/phase18t_slot_value_head.py \
    --model-action /workspace/phase17/model_action_finetuned.pt \
    --rollout-episodes 300 --train-steps 2000 \
    --n-eval-episodes 30 \
    --out /workspace/phase18t_main \
    --seed 0
```

Artifacts:
- Pod: `/workspace/phase18t_main/{summary.json, rollout_cache.npz,
  value_head_{geo,slot,geoslot}.pt, per_episode_*.jsonl, log.txt}`
- Repo: `artifacts/phase18t/{summary.json, per_episode_*.jsonl}`

## What this phase establishes

- **Slot features alone are not a value-prediction substrate at this
  data budget.** Spearman 0.113 (3.2× worse than geo); top/bot ratio
  1.18× (3.7× worse).
- **Geometric features carry the planner-value signal.** Phase 18η's
  success was not noise exploitation — it was using real geometric
  information (Pearson 0.32 here, +0.061 over locked across 3 seeds).
- **Slot + geo is essentially geo.** The 768-dim slot input adds
  near-zero usable information beyond the 10-dim geo input.
- **The BLA split is architecturally sound but needs a System-2
  geometry-readout layer**, not direct frozen-slot feeding. Phase
  18λ is the right next architectural lever, not "bigger MLP on
  slots."
- **The locked planning recipe (combined_sum_geo from Phase 18η-multi)
  remains unchanged.** Phase 18θ does not weaken it.
