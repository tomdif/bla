"""SimulatorAgreement — runs the candidate plan forward N times under noise
and certifies it iff the success fraction crosses a threshold.

For multi-target navigate, "the plan" is the policy itself. Each rollout
is a fresh environment instance. Success = task completion. The success
fraction across N rollouts becomes the uncertainty estimate written into
the CommitmentObject.

This is the canonical shape of "verified by simulation": a stochastic
check, calibrated against the actual deployment outcome, with a clear
threshold for pass/fail.
"""

from __future__ import annotations

from typing import Any, Callable

from .certifier import Certifier, CertifierResult


class SimulatorAgreement(Certifier):
    name = "simulator_agreement"
    category = "sim"

    def __init__(
        self,
        rollout_fn: Callable[[int], bool],
        n_rollouts: int = 10,
        threshold: float = 0.8,
    ):
        self.rollout_fn = rollout_fn
        self.n_rollouts = n_rollouts
        self.threshold = threshold

    def check(self, candidate: Any, seed_offset: int = 0, **kwargs) -> CertifierResult:
        successes = 0
        for i in range(self.n_rollouts):
            try:
                if self.rollout_fn(seed_offset + i):
                    successes += 1
            except Exception as exc:  # rollout failure counts as failed sim
                continue
        fraction = successes / max(self.n_rollouts, 1)
        passed = fraction >= self.threshold
        return CertifierResult(
            name=self.name,
            passed=passed,
            confidence=fraction,
            details={
                "n_rollouts": self.n_rollouts,
                "successes": successes,
                "threshold": self.threshold,
                "fraction": fraction,
            },
        )
