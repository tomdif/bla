"""WMOS -- World-Model Operating System.

A verificationist operator console for affordance world models. Proposers (language, learned
Δ-estimator, persistent library) put TYPED hypotheses on a bus; the Δachievable verifier and real
action feedback own truth; an action governor refuses to release unverified actions.

    from wmos import Harness, get_adapter, SessionStore
    h = Harness(get_adapter("grid"), SessionStore())

Central invariant:  NO UNVERIFIED PROPOSAL OWNS TRUTH.
"""
from .adapters import get_adapter, list_adapters, register, Adapter
from .engine import Harness, Hypothesis, Memory
from .persistence import SessionStore
from .config import load_config

__version__ = "0.1.0"
__all__ = ["Harness", "Hypothesis", "Memory", "get_adapter", "list_adapters",
           "register", "Adapter", "SessionStore", "load_config", "__version__"]
