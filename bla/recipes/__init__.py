"""Locked recipes A through E with their hyperparameters."""
from bla.recipes.registry import (
    Recipe,
    RecipeConfig,
    RECIPE_REGISTRY,
    locked_predictor_path,
    locked_adapter_path,
)

__all__ = [
    "Recipe",
    "RecipeConfig",
    "RECIPE_REGISTRY",
    "locked_predictor_path",
    "locked_adapter_path",
]
