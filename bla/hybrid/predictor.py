"""Predictor: proposes consequences of candidate actions.

Phase 1 (this file): predictor IS the LLM. It reads the current
StateStore as JSON, the user's most recent input, and is asked to
propose candidate actions + predicted consequences, then return a typed
Predict packet.

Phase 3 (future): swap LLMPredictor for a learned model (the actual
"JEPA" — a JEPA-style predictive embedding model trained on the
(state_before, action, state_after) trajectories logged by StateStore).
The Predict protocol and call surface stay identical.

This file is deliberately small. The phase-3 work is to train a model
that produces the same Predict payload from a state embedding and a
candidate-action embedding, without going through the LLM.
"""
from __future__ import annotations

import abc
import json
from typing import Any

from bla.hybrid.llm_client import LLMClient, parse_json_response
from bla.hybrid.protocol import Critique, Predict
from bla.hybrid.state import StateStore


_PREDICTOR_SYSTEM_PROMPT = """\
You are the predictive substrate of an object-JEPA / LLM hybrid system.
Your job is NOT to talk to the user. Your job is to look at the current
object-file state and propose:

  1. Candidate actions that could be taken next.
  2. The predicted consequences (state deltas) of each.
  3. A recommended action with confidence.
  4. Critique issues: contradictions, unsupported claims, missing
     evidence in the current state.

You must return a single JSON object with this exact shape:

{
  "predict": {
    "current_state_summary": "1-2 sentences",
    "candidate_actions": [
      {
        "action": "...",
        "predicted_state_delta": ["..."],
        "expected_information_gain": 0.0,
        "risk": 0.0
      }
    ],
    "recommended_action": "...",
    "confidence": 0.0
  },
  "critique": {
    "issues": [
      {
        "kind": "unsupported_claim | contradiction | missing_evidence | stale",
        "claim_or_object_id": "...",
        "severity": "low | medium | high",
        "recommended_revision": "..."
      }
    ]
  }
}

Rules:
  - Be calibrated: low confidence is fine and useful. Do not invent
    certainty.
  - Reference objects by id when commenting on them.
  - Predict in terms of STATE DELTAS, not prose explanations.
  - If you cannot propose any useful actions, return an empty
    candidate_actions list and one critique issue explaining why.
"""


class Predictor(abc.ABC):
    """Generic predictor surface. Phase 1 = LLM; Phase 3 = learned JEPA."""

    @abc.abstractmethod
    def predict(
        self, state: StateStore, user_input: str,
    ) -> tuple[Predict, Critique]:
        """Given current state + last user input, return (Predict, Critique)."""


class LLMPredictor(Predictor):
    """MVP predictor. Uses the LLM itself as the predictive substrate.

    This is the PHASE-3 SWAP POINT. A real JEPA replaces the body of
    `predict()`: it would encode `state` and `user_input` into latent
    embeddings, predict latent state-deltas via a learned dynamics model,
    and decode those deltas into the Predict / Critique packets.
    """

    def __init__(self, llm: LLMClient, max_tokens: int = 4096):
        self.llm = llm
        self.max_tokens = max_tokens

    def predict(
        self, state: StateStore, user_input: str,
    ) -> tuple[Predict, Critique]:
        user_msg = (
            "CURRENT STATE (JSON):\n"
            + json.dumps(state.to_dict(), indent=2, sort_keys=True)
            + "\n\nUSER INPUT:\n"
            + user_input
            + "\n\nProduce the predict + critique JSON."
        )
        raw = self.llm.complete(
            system=_PREDICTOR_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=self.max_tokens,
            json_only=True,
        )
        payload = parse_json_response(raw)
        predict_d = payload.get("predict", {})
        critique_d = payload.get("critique", {})
        return (
            Predict(
                current_state_summary=predict_d.get("current_state_summary", ""),
                candidate_actions=predict_d.get("candidate_actions", []) or [],
                recommended_action=predict_d.get("recommended_action"),
                confidence=float(predict_d.get("confidence", 0.0) or 0.0),
            ),
            Critique(issues=critique_d.get("issues", []) or []),
        )


class MockPredictor(Predictor):
    """Scripted predictor for tests. Returns canned (Predict, Critique)."""

    def __init__(self, scripted: list[tuple[Predict, Critique]]):
        self._scripted: list[tuple[Predict, Critique]] = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def predict(
        self, state: StateStore, user_input: str,
    ) -> tuple[Predict, Critique]:
        self.calls.append({
            "n_objects": len(state), "user_input": user_input,
        })
        if not self._scripted:
            raise RuntimeError(
                "MockPredictor ran out of scripted responses "
                f"(after {len(self.calls)} calls).")
        return self._scripted.pop(0)
