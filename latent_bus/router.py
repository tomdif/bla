from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn

from verification.router_action import RouterAction, RouterActionType


@dataclass
class RouterDecision:
    """Legacy two-action decision (wake / sleep). Kept for compatibility
    with anything that imported it before the action-space expansion."""
    entropy: torch.Tensor
    wake_dca: torch.Tensor
    power_state: List[str]


@dataclass
class RouterPlan:
    """Result of a 7-action routing decision per batch element."""
    actions: List[RouterAction]
    entropy: torch.Tensor
    rationale: List[str] = field(default_factory=list)


class EntropyRouter(nn.Module):
    """Budget allocator over the 7-action B.L.A. space:
        {answer, retrieve, simulate, prove, search, ask, defer}

    The router is non-trainable in Phase 2 (a hand-coded entropy threshold
    + simple dispatch). Phase 5 makes it RL-trained.

    Two surfaces for backward compat:
      * `decide(predictions)` returns the legacy wake/sleep RouterDecision.
      * `route(predictions, ...)` returns a 7-action RouterPlan.
    """

    def __init__(
        self,
        threshold_tau: float = 0.02,
        predictor_heads: Iterable[nn.Module] | None = None,
        action_handlers: Optional[Dict[RouterActionType, Callable]] = None,
    ):
        super().__init__()
        self.register_buffer("threshold_tau", torch.tensor(float(threshold_tau)))
        self.predictor_heads = nn.ModuleList(list(predictor_heads or []))
        self.action_handlers: Dict[RouterActionType, Callable] = dict(action_handlers or {})

    # --- legacy 2-action API -------------------------------------------

    def predictive_variance(self, predictions: Iterable[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(list(predictions), dim=0).float()
        variance = stacked.var(dim=0, unbiased=False)
        return variance.mean(dim=tuple(range(1, variance.ndim)))

    def decide(self, predictions: Iterable[torch.Tensor]) -> RouterDecision:
        entropy = self.predictive_variance(predictions)
        wake = entropy >= self.threshold_tau
        power_state = ["WAKE" if flag else "SLEEP" for flag in wake.detach().cpu().tolist()]
        return RouterDecision(entropy=entropy, wake_dca=wake, power_state=power_state)

    # --- new 7-action API ----------------------------------------------

    def route(
        self,
        predictions: Iterable[torch.Tensor],
        budget_remaining: float = 1.0,
    ) -> RouterPlan:
        """Pick one action per batch element from the 7-action space.

        Phase 2 policy (hand-coded):
            * entropy < tau           → ANSWER (cheap)
            * tau ≤ entropy < 2*tau   → RETRIEVE
            * 2*tau ≤ entropy < 4*tau → SIMULATE
            * 4*tau ≤ entropy < 8*tau → PROVE
            * entropy ≥ 8*tau         → SEARCH
            * budget_remaining < 0.1  → DEFER (overrides above)
            * predictions disagree on action class → ASK

        Real router learns these via RL in Phase 5.
        """
        entropy = self.predictive_variance(predictions)
        actions: List[RouterAction] = []
        rationale: List[str] = []
        tau = float(self.threshold_tau)
        for ent in entropy.detach().cpu().tolist():
            if budget_remaining < 0.1:
                action_type = RouterActionType.DEFER
                rat = f"budget exhausted ({budget_remaining:.2f}<0.1)"
            elif ent < tau:
                action_type = RouterActionType.ANSWER
                rat = f"entropy {ent:.4f} < tau {tau:.4f}"
            elif ent < 2 * tau:
                action_type = RouterActionType.RETRIEVE
                rat = f"entropy {ent:.4f} in [tau, 2·tau)"
            elif ent < 4 * tau:
                action_type = RouterActionType.SIMULATE
                rat = f"entropy {ent:.4f} in [2·tau, 4·tau)"
            elif ent < 8 * tau:
                action_type = RouterActionType.PROVE
                rat = f"entropy {ent:.4f} in [4·tau, 8·tau)"
            else:
                action_type = RouterActionType.SEARCH
                rat = f"entropy {ent:.4f} ≥ 8·tau"
            actions.append(RouterAction(type=action_type, budget_flops=int(1e6 * (ent / tau))))
            rationale.append(rat)
        return RouterPlan(actions=actions, entropy=entropy, rationale=rationale)

    def execute(self, plan: RouterPlan, **handler_kwargs):
        """Run each action through its registered handler. Phase 2 keeps
        handler invocation simple — one call per batch element."""
        outputs = []
        for action in plan.actions:
            handler = self.action_handlers.get(action.type)
            if handler is None:
                outputs.append(None)
                continue
            outputs.append(handler(action, **handler_kwargs))
        return outputs

    def forward(self, *args, **kwargs) -> RouterDecision:
        if not self.predictor_heads:
            raise RuntimeError("forward requires predictor_heads; use decide(predictions) otherwise")
        predictions = [head(*args, **kwargs) for head in self.predictor_heads]
        return self.decide(predictions)
