from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn

from .bus import TokenlessLatentBus


@dataclass
class VetoResult:
    accepted: torch.Tensor
    failure_score: torch.Tensor
    simulated_future: torch.Tensor


class VetoLoop(nn.Module):
    """Projects a DCA plan back into JEPA action space and simulates failure."""

    def __init__(
        self,
        bus: TokenlessLatentBus,
        jepa_predictor: nn.Module,
        failure_scorer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.bus = bus
        self.jepa_predictor = jepa_predictor
        self.failure_scorer = failure_scorer
        self.threshold = threshold

    def default_failure_scorer(self, z_future: torch.Tensor) -> torch.Tensor:
        return z_future.float().pow(2).mean(dim=(-1, -2)).sigmoid()

    def forward(self, z_current: torch.Tensor, dca_plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_latent = self.bus.down(dca_plan)
        simulated = self.jepa_predictor(z_current, action_latent)
        scorer = self.failure_scorer or self.default_failure_scorer
        score = scorer(simulated)
        accepted = score < self.threshold
        return {
            "accepted": accepted,
            "failure_score": score,
            "simulated_future": simulated,
        }

