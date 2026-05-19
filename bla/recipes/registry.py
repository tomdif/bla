"""Recipe registry — the five locked recipes A/B/C/D/E.

Each Recipe has a RecipeConfig capturing its planning-stack
hyperparameters as locked by the validating phase. See
`docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md` §4 for the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Recipe(str, Enum):
    A = "engineered_geo"          # Phase 18η-multi
    B = "supervised_adapter"      # Phase 18λ
    C = "end2end"                 # Phase 18λ-v2; OOD-favored
    D = "pretrain_ft"             # Phase 18ν
    E1 = "demo_no_cem_fresh_reset"      # Phase 18κ R3 (Lift)
    E2 = "demo_no_cem_state_matched"    # Phase D3/D4 (PickPlaceCan, Square)
    E2_FAST = "demo_retrieval_top1"     # Phase DR1; highest mean, higher variance
    E2_STABLE = "demo_retrieval_top3_avg"  # Phase DR1; lower mean, better stability


@dataclass(frozen=True)
class RecipeConfig:
    """Locked planning-stack hyperparameters for a recipe."""
    recipe: Recipe
    prior_kind: str                       # "scripted_fsm" / "demo_replay" / "none"
    use_cem: bool
    cem_K: int = 32
    cem_iters: int = 1
    cem_sigma_motion: float = 0.12
    cem_sigma_gripper: float = 0.0
    cem_sigma_floor: float = 0.05
    cem_elite_frac: float = 0.2
    value_head_input: str = "none"        # "engineered_geo" / "supervised_adapter" /
                                          # "end2end" / "pretrain_ft" / "none"
    combined_lambda: float = 0.5
    plan_horizon: int = 20
    jepa_stride: int = 4
    notes: tuple[str, ...] = field(default_factory=tuple)


RECIPE_REGISTRY: dict[Recipe, RecipeConfig] = {
    Recipe.A: RecipeConfig(
        recipe=Recipe.A,
        prior_kind="scripted_fsm",
        use_cem=True,
        value_head_input="engineered_geo",
        notes=("Stack push family; simulator-true features required",),
    ),
    Recipe.B: RecipeConfig(
        recipe=Recipe.B,
        prior_kind="scripted_fsm",
        use_cem=True,
        value_head_input="supervised_adapter",
        notes=("BLA-native; cross-task transfer when sim-true features unavailable",),
    ),
    Recipe.C: RecipeConfig(
        recipe=Recipe.C,
        prior_kind="scripted_fsm",
        use_cem=True,
        value_head_input="end2end",
        notes=("OOD-favored; higher variance than A/B in-distribution",),
    ),
    Recipe.D: RecipeConfig(
        recipe=Recipe.D,
        prior_kind="scripted_fsm",
        use_cem=True,
        value_head_input="pretrain_ft",
        notes=("Best in-dist of schedule variants; balances in-dist + OOD",),
    ),
    Recipe.E1: RecipeConfig(
        recipe=Recipe.E1,
        prior_kind="demo_replay",
        use_cem=False,
        value_head_input="none",
        notes=("Lift regime: demos work on fresh env reset",
                "narrow initial-state distribution"),
    ),
    Recipe.E2: RecipeConfig(
        recipe=Recipe.E2,
        prior_kind="demo_replay",
        use_cem=False,
        value_head_input="none",
        notes=("PickPlaceCan / NutAssemblySquare regime",
                "wide init-state distribution — env.sim.set_state_from_flattened "
                "to demo.states[0] before replay"),
    ),
    Recipe.E2_FAST: RecipeConfig(
        recipe=Recipe.E2_FAST,
        prior_kind="demo_replay",
        use_cem=False,
        value_head_input="none",
        notes=("Phase DR1: NN retrieval over a demo bank, top-1 chosen.",
                "Highest mean (PickPlaceCan 0.346) but higher variance "
                "(σ=0.135). Use when peak performance matters and seed-to-seed "
                "variation is acceptable."),
    ),
    Recipe.E2_STABLE: RecipeConfig(
        recipe=Recipe.E2_STABLE,
        prior_kind="demo_replay",
        use_cem=False,
        value_head_input="none",
        notes=("Phase DR1: NN retrieval over a demo bank, top-3 actions "
                "averaged elementwise. Lower mean (PickPlaceCan 0.280) but "
                "lower variance (σ=0.052). Use when reliability matters."),
    ),
}


def locked_predictor_path() -> str:
    """Pod path to the locked Phase 17 action-conditioned predictor."""
    return "/workspace/phase17/model_action_finetuned.pt"


def locked_adapter_path() -> str:
    """Pod path to the locked Phase 18λ supervised adapter."""
    return "/workspace/phase18l2_seed0/supervised_adapter.pt"
