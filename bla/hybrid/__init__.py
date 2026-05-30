"""BLA-Hybrid — JEPA-LLM harness (Phase 1: structured-state + frontier LLM).

What this is:
  A working harness for the "JEPA talks to LLM through structured state"
  thesis. Phase 1 stands up the pipeline:

    user input → LLM extracts events → object-file state store →
    predictor proposes consequences → LLM renders answer

  In Phase 1 the predictor is the LLM itself (the trivial swap point). The
  whole MVP is data-collection scaffolding: every loop step is logged as a
  (state_before, action, state_after, outcome) trajectory so that in Phase
  3 a real predictive model (the actual "JEPA" — a learned latent-state
  predictor trained on those trajectories) can be swapped in for
  `LLMPredictor` without changing the surrounding code.

What this is NOT:
  - Not yet a trained JEPA. There's no learned predictor here; the
    interface for one is in place.
  - Not a full agent harness. No tool use, no RAG, no multi-turn planning
    beyond the structured prediction.
  - Not coupled to vision / OF-JEPA. Conceptual object-JEPA — see
    docs/BLA_SYSTEM1_WORLD_MODEL_ARCHITECTURE.md for the visual cousin.

See `bla/hybrid/loop.py` for the wiring and `bla/hybrid/cli.py` for the
runnable entry point.
"""
from bla.hybrid.state import ObjectFile, StateStore
from bla.hybrid.protocol import (
    Observe,
    Update,
    Predict,
    Critique,
    Plan,
    Render,
)
from bla.hybrid.llm_client import (
    LLMClient,
    AnthropicLLMClient,
    MockLLMClient,
)
from bla.hybrid.predictor import Predictor, LLMPredictor, MockPredictor
from bla.hybrid.loop import HybridLoop

__all__ = [
    "ObjectFile",
    "StateStore",
    "Observe",
    "Update",
    "Predict",
    "Critique",
    "Plan",
    "Render",
    "LLMClient",
    "AnthropicLLMClient",
    "MockLLMClient",
    "Predictor",
    "LLMPredictor",
    "MockPredictor",
    "HybridLoop",
]
