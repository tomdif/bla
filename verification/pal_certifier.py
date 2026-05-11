"""PALCertifier — wraps the verifier signals (perturbation responsiveness,
ops count, etc.) as a CertifierResult on a CommitmentObject.

This lets the BLA pipeline treat PAL output as a certified claim: the
commitment carries the Python code + execution output + verifier
features in a structured way, with uncertainty derived from
verifier confidence.
"""
from __future__ import annotations

import sys
import os
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.certifier import Certifier, CertifierResult
from scripts.verifier import score_candidate


class PALCertifier(Certifier):
    """Certifier that scores a PAL candidate by structural + perturbation
    features. Marks the candidate as 'passed' if its verifier score
    exceeds a calibrated threshold.

    Returns CertifierResult with:
      passed: True if score > threshold
      confidence: normalized score in [0, 1]
      details: full feature dict
    """

    name = "pal_certifier"
    category = "test"

    def __init__(self, score_threshold: float = 3.0):
        self.score_threshold = score_threshold

    def check(self, candidate: dict, **kwargs) -> CertifierResult:
        """Candidate is a dict with keys:
          code: str
          output: str (stdout from execution)
          pred: str or None (parsed answer)
          problem_numbers: list[str]
        """
        code = candidate.get("code", "")
        output = candidate.get("output", "")
        pred = candidate.get("pred")
        problem_numbers = candidate.get("problem_numbers", [])

        s = score_candidate(code, output, pred, problem_numbers)
        # Score range observed: -100 (no parse) to ~14 (good); typical
        # bad-but-runs ~0-2, good ~6+. Normalize to [0,1] via sigmoid-ish.
        normalized = max(0.0, min(1.0, s["score"] / 12.0))
        return CertifierResult(
            name=self.name,
            passed=bool(s["runs"] and s["score"] >= self.score_threshold),
            confidence=normalized,
            details={
                "raw_score": s["score"],
                "ops": s["ops"],
                "is_computed": s["is_computed"],
                "chain_depth": s["chain_depth"],
                "responsiveness": s["responsiveness"],
                "echo": s["echo"],
            },
        )
