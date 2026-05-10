"""ProofChecker — verifies SymPy / Z3 propositions. Phase 2 stub; real
exercise comes in Phase 4 (math derivations) and Phase 6 (procedural CPU).

Two backends:
  * "sympy": prove a relation by structural simplification.
  * "z3":    discharge a proposition via SAT/SMT.

Either may not be installed; the certifier degrades to passed=False with
a clear `details["error"]`.
"""

from __future__ import annotations

from typing import Any

from .certifier import Certifier, CertifierResult


class ProofChecker(Certifier):
    name = "proof_checker"
    category = "proof"

    def __init__(self, backend: str = "sympy"):
        if backend not in ("sympy", "z3"):
            raise ValueError(f"unsupported backend: {backend}")
        self.backend = backend

    def check(self, candidate: Any, **kwargs) -> CertifierResult:
        if self.backend == "sympy":
            return self._sympy_check(candidate, **kwargs)
        return self._z3_check(candidate, **kwargs)

    def _sympy_check(self, candidate: Any, lhs: str | None = None, rhs: str | None = None, **kwargs) -> CertifierResult:
        try:
            import sympy
        except ImportError:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": "sympy not installed"},
            )
        if lhs is None or rhs is None:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": "sympy backend requires lhs and rhs strings"},
            )
        try:
            l = sympy.sympify(lhs)
            r = sympy.sympify(rhs)
            equal = sympy.simplify(l - r) == 0
            return CertifierResult(
                name=self.name, passed=equal, confidence=1.0 if equal else 0.0,
                details={"backend": "sympy", "lhs": lhs, "rhs": rhs, "equal": equal},
            )
        except Exception as exc:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"backend": "sympy", "error": str(exc)},
            )

    def _z3_check(self, candidate: Any, proposition: str | None = None, **kwargs) -> CertifierResult:
        try:
            import z3  # type: ignore
        except ImportError:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": "z3 not installed"},
            )
        if proposition is None:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"error": "z3 backend requires a proposition string"},
            )
        try:
            solver = z3.Solver()
            ctx = {**z3.__dict__}
            negated = z3.Not(eval(proposition, ctx))
            solver.add(negated)
            result = solver.check()
            valid = result == z3.unsat
            return CertifierResult(
                name=self.name, passed=valid, confidence=1.0 if valid else 0.0,
                details={"backend": "z3", "proposition": proposition, "z3_result": str(result)},
            )
        except Exception as exc:
            return CertifierResult(
                name=self.name, passed=False, confidence=0.0,
                details={"backend": "z3", "error": str(exc)},
            )
