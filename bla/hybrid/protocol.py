"""Typed protocol between JEPA-side state and the LLM.

Six message kinds. Each is a small JSON-serializable dataclass. Order in
the loop (`bla/hybrid/loop.py`):

    OBSERVE  → LLM extracts events/entities from raw input
    UPDATE   → state store applies the changes implied by OBSERVE
    PREDICT  → predictor proposes consequences of candidate actions
    CRITIQUE → predictor flags contradictions / missing evidence
    PLAN     → planner picks the next action
    RENDER   → LLM converts the final packet into natural language

Phase 1 uses the LLM as both OBSERVE-extractor and PREDICT-engine. Phase 3
replaces PREDICT with a learned model trained on logged trajectories; the
message shapes here stay the same.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Observe:
    """LLM-extracted view of raw input.

    `proposed_object_updates` is the bridge to UPDATE. Each entry is a
    small dict: {"op": "add"|"upsert"|"update", "object": {...}} or
    {"op": "update", "id": "...", "changes": {...}}. The loop applies
    these via StateStore when auto_apply_updates is True.
    """
    raw_input: str
    intent: str = ""
    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    extracted_events: list[dict[str, Any]] = field(default_factory=list)
    proposed_object_updates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Update:
    """Deltas the state store should apply.

    Each delta is a small dict: {"op": "add"|"upsert"|"update"|"remove",
    "object": {...} | "id": "...", "changes": {...}}.
    """
    changed_objects: list[dict[str, Any]] = field(default_factory=list)
    new_relations: list[dict[str, Any]] = field(default_factory=list)
    removed_relations: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_changes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Predict:
    """Candidate actions + predicted consequences.

    Phase 1: produced by LLMPredictor. Phase 3: produced by a learned
    JEPA model that consumes the current StateStore and outputs typed
    state-delta predictions.
    """
    current_state_summary: str = ""
    candidate_actions: list[dict[str, Any]] = field(default_factory=list)
    recommended_action: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Critique:
    """Issues the predictor or state store has flagged."""
    issues: list[dict[str, Any]] = field(default_factory=list)
    """
    Each issue: {kind, claim_or_object_id, severity, recommended_revision}.
      kind ∈ {unsupported_claim, contradiction, missing_evidence, stale}
      severity ∈ {low, medium, high}
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    """Selected next action + alternatives considered."""
    recommended_action: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Render:
    """Final natural-language packet for the user.

    `do_not_claim` is a list of constraints the LLM must NOT violate (e.g.
    "do not say slot persistence is solved"). These are derived from
    Critique.issues and from caller-supplied guardrails.
    """
    audience: str = "user"
    style: str = "direct technical"
    content_packet: dict[str, Any] = field(default_factory=dict)
    do_not_claim: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
