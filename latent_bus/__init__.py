from .bridge import VetoLoop, veto_loss
from .bus import TokenlessLatentBus, contrastive_infonce

# Lazy-load symbols whose modules have transitive deps on the
# System-2 / verification stack. The OF-JEPA path only needs the
# bus + bridge, so we don't make those deps load-bearing here.
def __getattr__(name):
    if name == "AsyncPrefetcher":
        from .prefetch import AsyncPrefetcher
        return AsyncPrefetcher
    if name in ("EntropyRouter", "RouterDecision", "RouterPlan"):
        from .router import EntropyRouter, RouterDecision, RouterPlan
        return {"EntropyRouter": EntropyRouter,
                "RouterDecision": RouterDecision,
                "RouterPlan": RouterPlan}[name]
    raise AttributeError(f"module 'latent_bus' has no attribute {name!r}")


__all__ = [
    "AsyncPrefetcher",
    "EntropyRouter",
    "RouterDecision",
    "RouterPlan",
    "TokenlessLatentBus",
    "VetoLoop",
    "contrastive_infonce",
    "veto_loss",
]

