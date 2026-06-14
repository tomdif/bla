# Next phase — multi-object contact world model (per-slot, control-trained)

> The one build the whole investigation earned the right to run. It is the *only* scene where the moat, OF-JEPA
> structure, the faithfulness lever, and the planner question all matter together.

## Why this, why now (what the prior investigation proved)
- **The moat is real** — a *verified* world model beats imitation on shifted goals (2D, 3D-Cartesian, 3D-torque).
- **The binding constraint is world-model faithfulness** — action→effector direction plateaus at **cosine ≈ 0.42**,
  and that ceiling is the **JEPA reconstruction objective**, not the planner, perception, scale, or representation
  structure (proven by exhaustion: CEM-horizon, MPPI, learned policies, governor, ensemble, resolution, *and* the
  OF-JEPA slot encoder all failed to move it).
- **The two real levers only pay off on multi-object structure:** (A) a **control objective** (TD-MPC2-style latent
  trained for control, not reconstruction), and (B) **per-slot OF-JEPA dynamics** (the action conditions the
  controllable slot; object-object interactions propagate contact). On *one* controllable object both are null
  (confirmed). This phase tests both, where they can finally separate.

## Hypotheses (each a falsifiable gate)
- **H1 (faithfulness):** a per-slot, *control-trained* latent breaks the 0.42 ceiling on **object** dynamics —
  pushing block *i* moves block-slot *i* faithfully. **Gate M1: per-object action-cosine > 0.42.**
- **H2 (moat):** the moat is *sharper* in a multi-object volume — imitation's coverage gap is worst when there are
  decoys + a goal volume. **Gate M2: WM ≫ BC on shifted goals + correct-object selection, margin > the single-object torque moat.**
- **H3 (planner):** in the multimodal/decoy/contact regime, the proposal stack (MPPI + diverse proposers + governor)
  finally beats CEM, and ensemble disagreement *now* flags exploitation (multi-object → genuine epistemic structure).
  **Gate M3: stack > CEM @success; disagreement separation > 1.6×.**

## Environment — minimal multi-object contact scene
**Task: push the *target* object to a goal region, among decoys.** Exercises multi-object, per-object dynamics,
contact, decoys, and multimodal planning (which object, which side to push) — the exact structure missing before.

Two routes (recommend **A**, fall back to **B**):
- **A. `FetchPush`-multi (gymnasium-robotics, already installed).** Cartesian gripper (trivial gripper dynamics) +
  2–3 sliding blocks (rich *contact* dynamics — the hard, interesting part). Isolates **object** dynamics from
  control. Extend the single-block env to N blocks; target = one color, others decoys; goal = a region. Reuses our
  `r3_fetch3d` plumbing (obs has per-object positions → per-slot grounding for free).
- **B. Custom MuJoCo tabletop pusher** (reuse the `r3_torque` arm + a table + free pucks). More control over
  contact/geometry, more build. Use only if FetchPush-multi can't express decoys/goal cleanly.

Validate with a probe (the discipline that's paid off every time): a privileged expert pushes the target to the
goal among decoys; a *direct* greedy push fails on contact-multimodality → confirms it's a genuine task. **(M0)**

## World model — per-slot RELATIONAL dynamics, control-trained latent
The flat OF-JEPA result refuted the *cheap drop-in* (slots → flatten → same dynamics → single object → reconstruction),
not OF-JEPA. The literature (TD-MPC2, Dreamer 4, C-SWM, SlotFormer, SOLD, Interaction Networks / GNS) is unanimous:
object structure pays off only with **object-level relational dynamics + control/value objectives + multi-step
prediction + stable binding**. So the model is:

- **Slots — ORACLE FIRST (decisive methodological fix).** `M1a` uses **simulator object states** as slots
  (pusher + each puck + goal) — *no* slot attention. This isolates the real question (can per-slot control dynamics
  break 0.42?) from binding fragility (the 4-attempt problem). Only `M1b` swaps in **learned visual slots**, and only
  then with the published binding fixes (SAVi temporal slots / DINOSAUR feature-recon / Slot-Contrast / first-frame cue)
  — *never* test learned slots + dynamics together first, or a binding failure masquerades as a dynamics failure.
- **Dynamics — RELATIONAL message passing, NOT flat-MLP** (Interaction Networks / GNS / C-SWM). Slots are graph nodes;
  contact/proximity/action are edges:
  ```
  gripper_delta   = f_g(g_t, a_t, Σ_j m(g_t, o_j, e))
  object_delta_i  = o_i + f_o(o_i, Σ_j m(o_i, s_j, e_ij))   # action enters ONLY the gripper node;
                                                            # contact messages propagate to objects; decoys inert until hit
  ```
  Flattening slots → MLP destroys the structure — that was the flat-swap failure.
- **Objective — CONTROL owns the latent, reconstruction is auxiliary** (TD-MPC2 decoder-free; Dreamer 4 x-prediction):
  | loss | role |
  |---|---|
  | `L_action_cos` | action → controlled-object delta **direction** (the 0.42 metric, now primary) |
  | `L_goal_value` | predicts goal progress / reward / success (control latent) |
  | `L_multistep` | predicted object states match **t+k endpoints** (x-prediction, not 1-step velocity) |
  | `L_contact` | predicts contact event / which object moved |
  | `L_identity` + `L_scof` | slot persistence (no swaps) + residual info retained |
  | `L_recon` | small auxiliary health-check only |
  Use **RMS loss normalization** across objectives (Dreamer 4). Add **policy/reward/value heads** trained in imagination.

## Milestones / build sequence
| # | Milestone | Gate | Status |
|---|---|---|---|
| **M0** | multi-object push env + probe | direct fails (0/5), contact works (4/5), expert solves clean cases (3/5) | **DONE** (`probe_push.py`) |
| **M1a** | **ORACLE-slot** dynamics: monolithic-MLP vs flat-slot vs **relational/contact** on sim states | per-object (contacted-puck) **action-cosine > 0.42**; contact-class works; multi-step stable | **next** |
| **M1b** | learned **visual** slots (SAVi/DINOSAUR/Slot-Contrast/first-frame cue) | recovers most of M1a; binding stable across seeds (no 4× fragility) | after M1a |
| **M2** | moat eval (shifted goals + right-object among decoys) | WM ≫ BC, margin > single-object torque moat | after M1b |
| **M3** | planner: stack / TD-MPC / learned proposals (Diffusion-MPC, Dream-MPC) + ensemble | stack > CEM; disagreement flags exploitation; gain isn't model-error exploitation | after M2 |

**M1a is the true core test** and it's *cheap* (state-based — no rendering): given *perfect* object states, can per-slot
**relational/contact** dynamics + a control objective predict the controlled-object delta faithfully (>0.42)?
- **If yes** → the 0.42 ceiling was **perception** (the reconstruction latent), and the fix is a control-trained latent
  feeding relational dynamics → M1b/M2.
- **If even oracle-state relational dynamics is ~0.42** → the bottleneck is the **contact dynamics / objective itself**,
  not perception — a deeper scientific result.
- **Decompose, don't conflate:** M1a removes perception (oracle slots) so a low cosine can't be blamed on binding;
  M1b adds perception back only once the dynamics question is answered.

**Deferred per literature:** learned-slot binding fixes → M1b only; learned action proposals / gradient planning
(Diffusion-MPC, Dream-MPC) → M3 only (planner upgrades are meaningless until the model is faithful).

## Reuse inventory (don't rebuild)
- **Recipe & discipline:** convergence gate, multi-step decode-consistency, action-coverage exploration,
  `action_authority` cosine diagnostic, held-out/OOD rollout — all in `r3_torque.py`.
- **OF-JEPA:** `slot_encoder.py`, `system1_jepa/{slot_predictor,slot_existence,id_consistency,of_jepa/*}`.
- **Planners/verify:** `proposal_stack.py` (TorchWMAdapter drop-in), `imagination_policy.py`, `ensemble_dynamics.py`.
- **Ops lessons:** long runs in **tmux + logfile**; monitor with `grep -q` (not `-c`); kill orphans **by PID**
  (never `pkill -f` — self-matches); seed planners + ≥40 eval eps (eval variance is high).

## Risks & mitigations
- **Slot binding fragility** → gate + specialization/existence loss + multi-seed; budget retries.
- **Contact dynamics hard** → contact-rich exploration (push-into-objects episodes) + multi-step consistency.
- **Dynamics-limit recurs** (M1 fails) → that *is* the result: object structure + control objective still can't
  break 0.42 → the ceiling is deeper than the objective; pivot to representation/data-scale or declare the limit.
- **Planner still ties (M3)** → only meaningful if M1 passed (faithful model); if it ties on a *faithful* model,
  that finally falsifies the planner-matters thesis cleanly.

## First step
**M0: stand up the multi-object push env + probe** (validate it's a genuine contact/decoy task before any WM) —
exactly the disciplined sequence that worked for torque and the zone task.
