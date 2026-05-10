"""TestRunner — runs candidate code against unit / property tests in a
sandboxed Python subprocess. Phase 2 stub; real exercise comes in Phase 4
(text + code generation).

The contract: candidate is a Python source string + a tests source string.
The certifier runs both via `subprocess.run` with a timeout and resource
limits. The certifier is detail-light by design — production would add
nsjail / Docker / RestrictedPython.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .certifier import Certifier, CertifierResult


class TestRunner(Certifier):
    name = "test_runner"
    category = "test"

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def check(self, candidate: Any, tests: str | None = None, **kwargs) -> CertifierResult:
        if tests is None:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": "no tests supplied"},
            )
        if not isinstance(candidate, str):
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": f"candidate must be a Python source string, got {type(candidate).__name__}"},
            )
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "candidate.py").write_text(candidate)
            (tmp / "test_candidate.py").write_text(tests)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "test_candidate.py"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return CertifierResult(
                    name=self.name, passed=False, confidence=0.0,
                    details={"error": "timeout", "timeout_s": self.timeout_s},
                )
            passed = proc.returncode == 0
            return CertifierResult(
                name=self.name,
                passed=passed,
                confidence=1.0 if passed else 0.0,
                details={
                    "stdout": proc.stdout[-2048:],
                    "stderr": proc.stderr[-2048:],
                    "returncode": proc.returncode,
                },
            )
