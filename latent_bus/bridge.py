from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn

from .bus import TokenlessLatentBus


@dataclass
class VetoResult:
    accepted: torch.Tensor
    failure_score: torch.Tensor
    simulated_future: torch.Tensor


class VetoLoop(nn.Module):
    """Projects a DCA plan back into JEPA action space and simulates failure.

    The default failure scorer measures plan disruptiveness — how far the
    JEPA-predicted future moves away from the current state under the plan.
    Replace with a learned safety classifier for real applications.
    """

    def __init__(
        self,
        bus: TokenlessLatentBus,
        jepa_predictor: nn.Module,
        failure_scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.bus = bus
        self.jepa_predictor = jepa_predictor
        self.failure_scorer = failure_scorer
        self.threshold = threshold

    @staticmethod
    def default_failure_scorer(
        simulated_future: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        diff = (simulated_future - baseline).float()
        return diff.pow(2).mean(dim=tuple(range(1, diff.ndim)))

    def forward(
        self,
        z_current: torch.Tensor,
        dca_plan: torch.Tensor,
        action: torch.Tensor,
        target_positions: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Dict[str, torch.Tensor]:
        plan_pooled = dca_plan.mean(dim=1) if dca_plan.ndim == 3 else dca_plan
        plan_action = self.bus.forward_down(plan_pooled)
        combined = action + plan_action
        simulated = self.jepa_predictor(
            context_tokens=z_current,
            action=combined,
            target_positions=target_positions,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        scorer = self.failure_scorer or self.default_failure_scorer
        baseline = torch.zeros_like(simulated)
        score = scorer(simulated, baseline)
        accepted = score < self.threshold
        return {
            "accepted": accepted,
            "failure_score": score,
            "simulated_future": simulated,
        }


def veto_loss(
    veto: VetoLoop,
    z_current: torch.Tensor,
    dca_plan: torch.Tensor,
    action: torch.Tensor,
    target_positions: torch.Tensor,
    grid_h: int,
    grid_w: int,
    safety_target: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Differentiable veto loss DCA can backprop through.

    Pushes the JEPA-predicted future under the plan toward `safety_target`
    (default: zeros, i.e. a "no disruption" baseline). Lower loss = the plan
    is predicted to leave the world close to the safety baseline.
    """

    out = veto(z_current, dca_plan, action, target_positions, grid_h, grid_w)
    target = safety_target if safety_target is not None else torch.zeros_like(out["simulated_future"])
    loss = (out["simulated_future"].float() - target.float()).pow(2).mean()
    return {"veto_loss": loss, **out}
