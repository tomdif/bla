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

## World model — per-slot, control-trained latent
Combine OF-JEPA structure (B) with a control objective (A), on the hard-won recipe:
- **Encoder:** `SlotEncoder` (slot_encoder.py) → K slots = {gripper slot, object slots}. Slot binding is fragile
  (4× retries seen) → keep the **convergence gate** + add a slot-specialization / existence objective
  (`system1_jepa/slot_existence.py`, `id_consistency.py`) so slots bind to objects.
- **Dynamics:** **per-slot** (`system1_jepa/slot_predictor.SlotDeltaPredictor`) — action conditions the gripper
  slot; slot-attention propagates contact to object slots. This is the piece the flat drop-in *didn't* use.
- **Objective (the lever):** *not* pure reconstruction. Train for control:
  1. light per-object decode grounding (each slot → its object position),
  2. **multi-step decode-consistency** (the torque lesson — rolled slots decode to the true per-object trajectory),
  3. a **value/reward head** (TD-MPC2-style) so the latent is shaped for the push task, not for pixels.
- **Per-object decode:** each object-slot decodes its object's position (the object-file).

## Milestones / build sequence
| # | Milestone | Gate | Reuses |
|---|---|---|---|
| **M0** | multi-object push env + probe | direct-push fails, expert-push works, decoys present | r3_fetch3d / r3_torque, probe pattern |
| **M1** | per-slot control-latent WM + per-object action-authority | **object-cosine > 0.42** | SlotEncoder, SlotDeltaPredictor, gate, consistency, `action_authority` |
| **M2** | moat eval (shifted goals + right-object) | WM ≫ BC, margin > torque moat | eval_method3d, BC3D |
| **M3** | planner A/B + ensemble re-test | stack > CEM; disagreement sep > 1.6× | proposal_stack, ensemble_dynamics, imagination_policy |

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
