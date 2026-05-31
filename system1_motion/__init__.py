"""System-1 motion substrate (system1_motion_spec.md).

Action-conditioned next-state latent prediction (V-JEPA 2-AC) trained to pass
Gate 0 (held-out agent-position decode < 5px). A1-minimal = primary prediction
+ variance hinge + detached decode diagnostic.
"""
from .models import ViTEncoder, LatentDynamics, DecodeHead, ema_update
from .objective import RunningSigma, normalized_mse, variance_hinge, substrate_loss

__all__ = ["ViTEncoder", "LatentDynamics", "DecodeHead", "ema_update",
           "RunningSigma", "normalized_mse", "variance_hinge", "substrate_loss"]
