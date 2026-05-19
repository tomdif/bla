"""Recipe Router — TaskDescriptor → Recipe.

The headline new component for Scale-0. Takes a structured task
descriptor and returns the recipe the BLA architecture spec
predicts will win, plus the recipe's locked config.

See `docs/BLA_SCALING_ROADMAP.md` §1 for the deployment doctrine.
"""
from bla.routing.recipe_router import (
    TaskDescriptor,
    RouterDecision,
    recipe_router,
)

__all__ = ["TaskDescriptor", "RouterDecision", "recipe_router"]
