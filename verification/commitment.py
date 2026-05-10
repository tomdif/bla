"""CommitmentObject — the structured output type for B.L.A.

Truth is not stored. It is certified. A commitment object is what B.L.A.
emits in place of a raw answer: the claim plus the evidence trail plus
the certifier results plus a calibrated uncertainty bound plus a
reproducibility packet.

This is the schema. The certifiers (`certifier.py`,
`simulator_agreement.py`, `test_runner.py`, `proof_checker.py`) are what
populate it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CommitmentObject:
    claim: Any
    evidence: list = field(default_factory=list)
    reasoning_trace: dict = field(default_factory=dict)
    proofs_run: list = field(default_factory=list)
    tests_run: list = field(default_factory=list)
    simulations_run: list = field(default_factory=list)
    counterexamples_searched: list = field(default_factory=list)
    uncertainty: float = 1.0
    consensus: Optional[dict] = None
    reproducibility_packet: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def certified(self) -> bool:
        """True iff every attached certifier passed."""
        all_results = self.proofs_run + self.tests_run + self.simulations_run
        if not all_results:
            return False
        return all(r.get("passed", False) for r in all_results)

    @property
    def confidence(self) -> float:
        """Convenience: 1 - uncertainty."""
        return 1.0 - max(0.0, min(1.0, self.uncertainty))

    def to_json(self) -> str:
        return json.dumps(self._serializable(), default=str, indent=2)

    def _serializable(self) -> dict:
        d = asdict(self)
        # ensure things like numpy / torch tensors don't blow up
        for key in ("claim", "evidence", "reasoning_trace"):
            d[key] = _coerce(d[key])
        return d


def _coerce(x):
    """Best-effort conversion to JSON-friendly types."""
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, dict):
            return {k: _coerce(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_coerce(v) for v in x]
        return repr(x)
