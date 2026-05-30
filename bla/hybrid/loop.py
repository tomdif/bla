"""HybridLoop — the observe → update → predict → render cycle.

This is the main wiring. The loop is intentionally short because the work
lives in `predictor.py` and `llm_client.py`; the loop just sequences them
and keeps the StateStore consistent.

Flow per step (one user input):

  1. OBSERVE  — LLM extracts entities/events from raw_input
  2. UPDATE   — apply state deltas implied by OBSERVE (caller-controlled
                or LLM-proposed; MVP keeps this conservative: no
                automatic state mutation from raw OBSERVE, only from
                explicit "update_objects" deltas the LLM emits)
  3. PREDICT  — Predictor produces (Predict, Critique) from the current
                StateStore + raw_input
  4. RENDER   — LLM converts the bundle into natural-language reply,
                with the critique's recommended_revision list flattened
                into Render.do_not_claim guardrails

Each step appends a trajectory record so Phase-3 training has data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from bla.hybrid.llm_client import LLMClient, parse_json_response
from bla.hybrid.predictor import Predictor
from bla.hybrid.protocol import Observe, Predict, Render, Update
from bla.hybrid.state import ObjectFile, StateStore


_OBSERVER_SYSTEM_PROMPT = """\
You extract structured information from one user message in the context
of an ongoing object-file state store. Return a single JSON object:

{
  "intent": "question | claim | proposal | observation | event_update",
  "extracted_entities": [
    {"name": "...", "type": "..."}
  ],
  "extracted_events": [
    {"kind": "experiment_completed | result_observed | hypothesis_revised | other",
     "summary": "..."}
  ],
  "proposed_object_updates": [
    {"op": "add | upsert | update", ...}
  ]
}

OBJECTFILE SCHEMA — required to get updates right:
  An ObjectFile has these top-level fields ONLY:
    id, type, name, state (a sub-dict), confidence (0..1),
    supported_by (list of ids), contradicted_by (list of ids),
    open_questions (list of strings).
  Domain-specific numbers and metrics (e.g. spearman_score,
  sample_count, threshold, observed_value, status) DO NOT live at the
  top level — they go inside the `state` sub-dict.

UPDATE OPERATION SHAPES:

  Add a new object:
    {"op": "add",
     "object": {"id": "phase_19", "type": "phase",
                "name": "Phase 19 — geometry adapter (3 seeds)",
                "state": {"spearman_score": 0.52, "n_seeds": 3},
                "confidence": 0.9}}

  Upsert (add OR replace by id):
    Same shape as add, with "op": "upsert".

  Update an existing object's TOP-LEVEL fields:
    {"op": "update", "id": "phase_19",
     "changes": {"confidence": 0.5}}

  Update an existing object's DOMAIN fields (inside state):
    {"op": "update", "id": "phase_19",
     "changes": {"state": {"spearman_score": 0.48,
                            "note": "corrected from 0.52"}}}

  WRONG — do not put domain fields at the top of changes:
    {"op": "update", "id": "phase_19",
     "changes": {"spearman_score": 0.48}}        // wrong
  The loop will reject this because ObjectFile has no `spearman_score`
  attribute. Always nest under `state` for domain values.

If nothing in the message implies a state mutation, return an empty
proposed_object_updates list. Be conservative — do not invent
mutations.
"""


_RENDER_SYSTEM_PROMPT_TEMPLATE = """\
{domain_preamble}

You are receiving a structured packet from the predictive substrate:

  - current_state_summary: 1-2 sentence summary of the world state
  - candidate_actions: actions the predictor proposed
  - recommended_action: the predictor's top pick
  - confidence: in [0, 1]
  - critique_issues: contradictions or missing evidence the predictor flagged
  - do_not_claim: assertions you MUST NOT make in your reply

Your job: render this into a concise reply to the user. Rules:
  - Lead with the answer or recommendation. Skip preamble.
  - If confidence < 0.4, say so explicitly.
  - Reference object ids when discussing specific entities.
  - Do not invent objects or facts that aren't in the packet.
  - Respect every entry in do_not_claim.
  - 3-6 sentences. Bullets are fine for multi-option answers.
"""


@dataclass
class StepRecord:
    """One trajectory entry (Phase-3 training input)."""
    user_input: str
    observe: dict[str, Any]
    update: dict[str, Any]
    predict: dict[str, Any]
    critique: dict[str, Any]
    render_text: str
    state_changes: list[dict[str, Any]] = field(default_factory=list)


class HybridLoop:
    """One conversational step at a time, with full trajectory logging.

    Args:
      llm:               LLMClient for OBSERVE + RENDER
      predictor:         Predictor (Phase 1: LLMPredictor; Phase 3:
                         learned JEPA)
      state:             StateStore (already loaded / seeded)
      domain_preamble:   domain-specific system-prompt header (e.g. the
                         BLA tracker preamble in `bla_tracker.py`)
      auto_apply_updates: if True (default False), apply LLM-proposed
                         object updates from OBSERVE automatically. If
                         False, return them in StepRecord.update for
                         caller review. Default is conservative — the
                         user/CLI is expected to confirm.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        predictor: Predictor,
        state: StateStore,
        domain_preamble: str = "",
        auto_apply_updates: bool = False,
    ):
        self.llm = llm
        self.predictor = predictor
        self.state = state
        self.domain_preamble = domain_preamble
        self.auto_apply_updates = auto_apply_updates
        self.history: list[StepRecord] = []

    # ---------- pipeline stages ----------
    def observe(self, user_input: str) -> Observe:
        raw = self.llm.complete(
            system=_OBSERVER_SYSTEM_PROMPT,
            user=(
                "STATE OBJECTS (id, type, name):\n"
                + "\n".join(
                    f"  {o.id} | {o.type} | {o.name}"
                    for o in self.state.all()
                )
                + "\n\nUSER MESSAGE:\n"
                + user_input
            ),
            max_tokens=1024,
            json_only=True,
        )
        payload = parse_json_response(raw)
        return Observe(
            raw_input=user_input,
            intent=payload.get("intent", ""),
            extracted_entities=payload.get("extracted_entities", []) or [],
            extracted_events=payload.get("extracted_events", []) or [],
            proposed_object_updates=payload.get("proposed_object_updates", []) or [],
        )

    # The set of ObjectFile top-level field names we accept on an update.
    # Anything else in `changes` gets auto-nested under `state` (see below).
    _OBJECTFILE_TOPLEVEL_FIELDS = frozenset({
        "name", "type", "state", "confidence",
        "supported_by", "contradicted_by", "open_questions",
    })

    def apply_proposed_updates(
        self, proposed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply OBSERVE-proposed object updates. Returns the change list.

        Updates with keys that don't match ObjectFile top-level fields
        are auto-nested under `state` (e.g. `spearman_score: 0.48`
        becomes `state.spearman_score: 0.48`). This is a forgiving
        interpretation of LLM output: the OBSERVE prompt teaches the
        right shape, but if the model slips, we nest rather than drop.
        """
        applied: list[dict[str, Any]] = []
        for upd in proposed:
            op = upd.get("op")
            try:
                if op in ("add", "upsert"):
                    obj_data = upd.get("object") or {}
                    obj = ObjectFile(**obj_data)
                    if op == "add" and obj.id in self.state:
                        # Be forgiving: convert to upsert
                        self.state.upsert(obj)
                    elif op == "add":
                        self.state.add(obj)
                    else:
                        self.state.upsert(obj)
                    applied.append({"op": op, "id": obj.id, "ok": True})
                elif op == "update":
                    obj_id = upd.get("id")
                    changes = dict(upd.get("changes") or {})
                    # Separate top-level vs domain (state-nested) changes.
                    state_patch: dict[str, Any] = {}
                    for k in list(changes.keys()):
                        if k not in self._OBJECTFILE_TOPLEVEL_FIELDS:
                            state_patch[k] = changes.pop(k)
                    if state_patch:
                        # Merge with existing state rather than replace it
                        cur = self.state.find(obj_id)
                        if cur is None:
                            applied.append({
                                "op": "update", "id": obj_id, "ok": False,
                                "reason": f"no object with id {obj_id!r}",
                                "input": upd,
                            })
                            continue
                        merged = dict(cur.state)
                        merged.update(state_patch)
                        # If caller ALSO passed an explicit `state` dict in
                        # changes, that wins for any overlapping keys.
                        if "state" in changes:
                            merged.update(changes["state"])
                        changes["state"] = merged
                    self.state.update(obj_id, **changes)
                    applied.append({
                        "op": "update", "id": obj_id, "ok": True,
                        "auto_nested_into_state": list(state_patch.keys()),
                    })
                else:
                    applied.append({"op": op, "ok": False, "reason": "unknown op"})
            except Exception as e:
                applied.append({
                    "op": op, "ok": False, "reason": str(e), "input": upd,
                })
        return applied

    def render(
        self, user_input: str, predict_pkt: Predict, critique_issues: list[dict[str, Any]],
    ) -> tuple[Render, str]:
        do_not_claim = [
            iss.get("recommended_revision", "")
            for iss in critique_issues
            if iss.get("severity") in ("medium", "high")
            and iss.get("recommended_revision")
        ]
        render_pkt = Render(
            audience="user",
            style="direct technical",
            content_packet={
                "current_state_summary": predict_pkt.current_state_summary,
                "candidate_actions": predict_pkt.candidate_actions,
                "recommended_action": predict_pkt.recommended_action,
                "confidence": predict_pkt.confidence,
                "critique_issues": critique_issues,
            },
            do_not_claim=do_not_claim,
        )
        system = _RENDER_SYSTEM_PROMPT_TEMPLATE.format(
            domain_preamble=self.domain_preamble or "",
        )
        user_msg = (
            "USER INPUT:\n" + user_input + "\n\nPACKET (JSON):\n"
            + json.dumps(render_pkt.to_dict(), indent=2, sort_keys=True)
        )
        text = self.llm.complete(
            system=system, user=user_msg, max_tokens=1024,
        )
        return render_pkt, text.strip()

    # ---------- one full step ----------
    def step(self, user_input: str) -> StepRecord:
        observe_pkt = self._safe_observe(user_input)

        # UPDATE: apply OBSERVE's proposed updates if enabled. State has to
        # actually evolve turn-to-turn for the hybrid's multi-turn pitch
        # to hold; without this it's a glorified single-turn loop.
        proposed = list(observe_pkt.proposed_object_updates)
        applied: list[dict[str, Any]] = []
        if self.auto_apply_updates and proposed:
            applied = self.apply_proposed_updates(proposed)
        update_record: dict[str, Any] = {"proposed": proposed, "applied": applied}

        # Surface update-application outcomes to the predictor so it can
        # reason over the *actual* state-changes that happened (not just
        # the proposals). Without this, RENDER can fabricate state
        # changes that the apply step silently dropped — the T3→T4 bug.
        predict_input = user_input
        if applied:
            apply_summary = "\n\nRECENT STATE CHANGES THIS TURN:\n" + json.dumps(
                applied, indent=2, sort_keys=True,
            )
            predict_input = user_input + apply_summary

        # Predict + critique from current (now possibly updated) state
        predict_pkt, critique_pkt = self.predictor.predict(self.state, predict_input)

        # Render answer
        _, render_text = self.render(
            user_input, predict_pkt, critique_pkt.issues,
        )

        rec = StepRecord(
            user_input=user_input,
            observe=observe_pkt.to_dict(),
            update=update_record,
            predict=predict_pkt.to_dict(),
            critique=critique_pkt.to_dict(),
            render_text=render_text,
            state_changes=list(self.state.change_log),
        )
        self.state.clear_change_log()
        self.history.append(rec)
        return rec

    def _safe_observe(self, user_input: str) -> Observe:
        """Run OBSERVE, falling back to a minimal Observe on parse error.

        The OBSERVE step is best-effort: if the model returns malformed
        JSON or schema-mismatched payload, the rest of the loop is still
        useful (predictor + render run on the existing state). We log the
        failure into Observe.extracted_events so trajectories see it.
        """
        try:
            return self.observe(user_input)
        except (ValueError, KeyError) as e:
            return Observe(
                raw_input=user_input,
                intent="parse_failure",
                extracted_events=[{"kind": "observe_parse_error", "error": str(e)}],
            )
