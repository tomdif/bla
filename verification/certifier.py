"""Certifier ABC. Every concrete certifier returns a CertifierResult that
gets attached to a CommitmentObject (in proofs_run / tests_run /
simulations_run depending on the certifier's category)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CertifierResult:
    name: str
    passed: bool
    confidence: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "confidence": self.confidence,
            "details": self.details,
        }


class Certifier(ABC):
    """A certifier checks a candidate output and returns a structured
    result. Categories drive where the result is filed on the
    CommitmentObject:
      * proof:  proofs_run (formal verification)
      * test:   tests_run (code / unit / property tests)
      * sim:    simulations_run (world-model rollouts)
    """

    name: str = "certifier"
    category: str = "test"

    @abstractmethod
    def check(self, candidate: Any, **kwargs) -> CertifierResult:
        """Run the certifier on a candidate output. Implementations should
        be deterministic given fixed kwargs (seed, etc.) so the
        reproducibility_packet on the CommitmentObject can replay them.
        """

    def attach(self, commitment, candidate: Any, **kwargs) -> CertifierResult:
        result = self.check(candidate, **kwargs)
        bucket = {
            "proof": commitment.proofs_run,
            "test": commitment.tests_run,
            "sim": commitment.simulations_run,
        }.get(self.category)
        if bucket is None:
            raise ValueError(f"unknown certifier category: {self.category}")
        bucket.append(result.to_dict())
        return result
