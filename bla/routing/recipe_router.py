"""Recipe Router — choose a recipe from a task descriptor.

The deployment doctrine, from `docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md`
§5 and `docs/BLA_SCALING_ROADMAP.md` §1:

    if FSM-prior / broad action search:
        Recipe A (engineered geo) or B (supervised adapter)
    if demo-prior / contact-sensitive:
        Recipe E1 (narrow init) or E2 (wide init)
    if OOD goal shift:
        Recipe D (pretrain+ft) preferred, fallback C (end2end)
    if simulator-true features available:
        Recipe A preferred over B

The router is pure logic; it makes one decision per call. The
output is a RouterDecision pairing the chosen Recipe with its
locked RecipeConfig and a short rationale string for logging.

Falsifiable claim: across pre-committed task descriptors, this
function should choose the empirically-best recipe ≥ 70% of the
time (Phase Scale-1 pass criterion).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bla.recipes.registry import Recipe, RecipeConfig, RECIPE_REGISTRY


PriorKind = Literal["fsm", "demo", "none"]


@dataclass(frozen=True)
class TaskDescriptor:
    """Structured description of a task for routing.

    Field semantics (all locked to validated regimes):

    - prior_kind: what's available as a prior?
        "fsm"  — a scripted FSM that gets rough motion right
        "demo" — expert demonstrations available (e.g. robomimic)
        "none" — neither (router will fall through to floor)

    - contact_sensitive: does success depend on precise contact
      timing (grasp, insert) vs broad motion (push)?

    - out_of_distribution: does the eval distribution differ from
      training (goal-distance shift, friction shift, init shift)?

    - sim_true_features: are simulator-true engineered geometry
      features available (cube_xy, eef_xyz, etc.) for the value head?

    - init_distribution_wide: for demo-prior tasks, is the env's
      initial-state distribution wider than the demo bank can cover
      on fresh random resets? Lift = False; PickPlaceCan = True.
    """
    prior_kind: PriorKind
    contact_sensitive: bool = False
    out_of_distribution: bool = False
    sim_true_features: bool = False
    init_distribution_wide: bool = False
    task_name: str | None = None


@dataclass(frozen=True)
class RouterDecision:
    recipe: Recipe
    config: RecipeConfig
    rationale: str


def recipe_router(task: TaskDescriptor) -> RouterDecision:
    """Map a TaskDescriptor to the recipe the BLA doctrine predicts wins.

    Decision tree (priority order matches the doctrine's strength):

      1. Demo-prior + contact-sensitive → Recipe E (locked from D3+D4)
         E1 if narrow init, E2 if wide init.
      2. Out-of-distribution shift → Recipe D (pretrain+ft, from 18ν)
      3. Simulator-true features → Recipe A (engineered geo, 18η-multi)
      4. Otherwise → Recipe B (supervised adapter, 18λ)

    Rule 1 dominates because Phase D3+D4 established it with three
    cross-task validations and zero falsification triggers.
    """
    if task.prior_kind == "demo" and task.contact_sensitive:
        if task.init_distribution_wide:
            chosen = Recipe.E2
            why = ("demo-prior + contact-sensitive + wide init distribution "
                    "→ Recipe E2 (state-matched reset). "
                    "Validated: PickPlaceCan (D3), NutAssemblySquare (D4).")
        else:
            chosen = Recipe.E1
            why = ("demo-prior + contact-sensitive + narrow init distribution "
                    "→ Recipe E1 (fresh-reset replay). "
                    "Validated: Lift (Phase 18κ R3).")
        return RouterDecision(chosen, RECIPE_REGISTRY[chosen], why)

    if task.out_of_distribution:
        chosen = Recipe.D
        why = ("OOD distribution shift → Recipe D (pretrain+ft). "
                "Validated: Phase 18ν.")
        return RouterDecision(chosen, RECIPE_REGISTRY[chosen], why)

    if task.sim_true_features:
        chosen = Recipe.A
        why = ("FSM-prior + simulator-true features available "
                "→ Recipe A (engineered geometry value head). "
                "Validated: Phase 18η-multi (Stack push).")
        return RouterDecision(chosen, RECIPE_REGISTRY[chosen], why)

    chosen = Recipe.B
    why = ("FSM-prior + slot-only features → Recipe B (supervised "
            "adapter value head). Validated: Phase 18λ; cross-task "
            "transfer claim depends on adapter fidelity.")
    return RouterDecision(chosen, RECIPE_REGISTRY[chosen], why)
