"""Unit tests for the Recipe Router.

Each test encodes a task BLA has actually validated, and asserts
the router chooses the empirically-winning recipe. These tests
ARE the doctrine's falsifiable claims.
"""
from __future__ import annotations

from bla.recipes import Recipe, RECIPE_REGISTRY
from bla.routing import TaskDescriptor, recipe_router


# ---------- demo-prior regime (Recipe E) ----------

def test_lift_picks_e1_fresh_reset():
    """Lift (Phase 18κ R3): demos work on fresh env reset → E1."""
    task = TaskDescriptor(
        prior_kind="demo", contact_sensitive=True,
        init_distribution_wide=False, task_name="Lift",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.E1
    assert "narrow init" in decision.rationale.lower()


def test_pickplace_can_picks_e2_state_matched():
    """PickPlaceCan (Phase D3): wide init distribution → E2."""
    task = TaskDescriptor(
        prior_kind="demo", contact_sensitive=True,
        init_distribution_wide=True, task_name="PickPlaceCan",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.E2
    assert "wide init" in decision.rationale.lower()


def test_nut_assembly_square_picks_e2_state_matched():
    """NutAssemblySquare (Phase D4): wide init + precise insertion → E2."""
    task = TaskDescriptor(
        prior_kind="demo", contact_sensitive=True,
        init_distribution_wide=True, task_name="NutAssemblySquare",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.E2


# ---------- FSM-prior regime (Recipes A/B) ----------

def test_stack_push_with_sim_features_picks_a():
    """Stack push (Phases 14-18) with sim-true features → Recipe A."""
    task = TaskDescriptor(
        prior_kind="fsm", contact_sensitive=False,
        sim_true_features=True, task_name="StackPush",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.A


def test_stack_push_without_sim_features_picks_b():
    """Same Stack push, but only slot features → Recipe B (supervised adapter)."""
    task = TaskDescriptor(
        prior_kind="fsm", contact_sensitive=False,
        sim_true_features=False, task_name="StackPush-NoSimFeats",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.B


# ---------- OOD regime (Recipe D) ----------

def test_ood_goal_shift_picks_d():
    """Phase 18κ R2 / 18ν: distribution shift → Recipe D (pretrain+ft)."""
    task = TaskDescriptor(
        prior_kind="fsm", out_of_distribution=True,
        sim_true_features=True, task_name="StackPush-OOD",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.D


# ---------- doctrine priority: demo-prior dominates OOD ----------

def test_demo_prior_beats_ood_when_both_apply():
    """Demo-prior + contact-sensitive should take priority even with OOD shift.

    The doctrine puts demo-prior first because Phase D3+D4 had Δ=+0.4 / +0.5
    margins; Phase 18κ R2 OOD finding had smaller effect size. If both apply
    we route to E, not D.
    """
    task = TaskDescriptor(
        prior_kind="demo", contact_sensitive=True,
        out_of_distribution=True, init_distribution_wide=True,
        task_name="PickPlaceCan-OOD-hypothetical",
    )
    decision = recipe_router(task)
    assert decision.recipe == Recipe.E2


# ---------- recipe config sanity ----------

def test_recipe_e_recipes_disable_cem():
    """Recipe E variants are NO-CEM by construction (the doctrine's claim)."""
    for r in (Recipe.E1, Recipe.E2):
        cfg = RECIPE_REGISTRY[r]
        assert cfg.use_cem is False
        assert cfg.prior_kind == "demo_replay"
        assert cfg.value_head_input == "none"


def test_recipe_abcd_use_cem_and_fsm_prior():
    """Recipes A/B/C/D all use light CEM with the FSM-prior protocol."""
    for r in (Recipe.A, Recipe.B, Recipe.C, Recipe.D):
        cfg = RECIPE_REGISTRY[r]
        assert cfg.use_cem is True
        assert cfg.prior_kind == "scripted_fsm"
        assert cfg.cem_K == 32
        assert cfg.cem_iters == 1
        assert cfg.cem_sigma_motion == 0.12
        assert cfg.cem_sigma_gripper == 0.0


def test_decision_carries_full_config():
    """RouterDecision must include the locked RecipeConfig for downstream use."""
    decision = recipe_router(TaskDescriptor(
        prior_kind="demo", contact_sensitive=True,
        init_distribution_wide=True,
    ))
    assert decision.recipe == Recipe.E2
    assert decision.config.recipe == Recipe.E2
    assert decision.config.use_cem is False
    assert decision.rationale  # non-empty


# ---------- floor / fallthrough ----------

def test_no_prior_no_features_falls_through_to_b():
    """No prior, no features, in-dist → router still picks Recipe B.

    Rationale: B is the BLA-native floor (supervised adapter on slots);
    if the deployment context is genuinely unknown, B is the safest
    default before doing more diagnosis.
    """
    task = TaskDescriptor(prior_kind="none")
    decision = recipe_router(task)
    assert decision.recipe == Recipe.B
