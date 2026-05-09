from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class RouterDecision:
    entropy: torch.Tensor
    wake_dca: torch.Tensor
    power_state: List[str]


class EntropyRouter(nn.Module):
    """Hardware interrupt gate based on variance across JEPA predictor heads."""

    def __init__(self, threshold_tau: float = 0.02, predictor_heads: Iterable[nn.Module] | None = None):
        super().__init__()
        self.threshold_tau = nn.Parameter(torch.tensor(float(threshold_tau)))
        self.predictor_heads = nn.ModuleList(list(predictor_heads or []))

    def predictive_variance(self, predictions: Iterable[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(list(predictions), dim=0).float()
        variance = stacked.var(dim=0, unbiased=False)
        return variance.mean(dim=tuple(range(1, variance.ndim)))

    def decide(self, predictions: Iterable[torch.Tensor]) -> RouterDecision:
        entropy = self.predictive_variance(predictions)
        wake = entropy >= self.threshold_tau
        power_state = ["WAKE" if flag else "SLEEP" for flag in wake.detach().cpu().tolist()]
        return RouterDecision(entropy=entropy, wake_dca=wake, power_state=power_state)

    def forward(self, z_context: torch.Tensor, action: torch.Tensor) -> RouterDecision:
        if not self.predictor_heads:
            raise RuntimeError("forward requires predictor_heads; use decide(predictions) otherwise")
        predictions = [head(z_context, action) for head in self.predictor_heads]
        return self.decide(predictions)
