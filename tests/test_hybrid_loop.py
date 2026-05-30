"""Integration tests for bla.hybrid.loop — all offline via MockLLMClient."""
from __future__ import annotations

import json

import pytest

from bla.hybrid.bla_tracker import BLA_DOMAIN_SYSTEM_PROMPT, seed_bla_tracker
from bla.hybrid.llm_client import MockLLMClient, parse_json_response
from bla.hybrid.loop import HybridLoop
from bla.hybrid.predictor import LLMPredictor, MockPredictor
from bla.hybrid.protocol import Critique, Predict
from bla.hybrid.state import ObjectFile, StateStore


# ---------- parse_json_response ----------
def test_parse_json_response_plain():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_strips_code_fence():
    assert parse_json_response("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_response_strips_unlabeled_fence():
    assert parse_json_response("```\n[1, 2]\n```") == [1, 2]


def test_parse_json_response_extracts_from_prose():
    s = "Here you go:\n{\"a\": 1}\nDone."
    assert parse_json_response(s) == {"a": 1}


def test_parse_json_response_raises_on_garbage():
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_json_response("not json at all")


# ---------- LLMPredictor with MockLLMClient ----------
def test_llm_predictor_parses_well_formed_json():
    canned = json.dumps({
        "predict": {
            "current_state_summary": "X is uncertain",
            "candidate_actions": [
                {"action": "run experiment Z", "predicted_state_delta": ["X.conf↑"],
                 "expected_information_gain": 0.8, "risk": 0.2},
            ],
            "recommended_action": "run experiment Z",
            "confidence": 0.72,
        },
        "critique": {
            "issues": [
                {"kind": "missing_evidence", "claim_or_object_id": "hyp_1",
                 "severity": "medium", "recommended_revision": "test Z first"},
            ],
        },
    })
    llm = MockLLMClient([canned])
    predictor = LLMPredictor(llm)
    state = StateStore()
    state.add(ObjectFile(id="hyp_1", type="hypothesis", name="H"))
    predict, critique = predictor.predict(state, "should we run Y or Z?")
    assert predict.recommended_action == "run experiment Z"
    assert predict.confidence == pytest.approx(0.72)
    assert len(predict.candidate_actions) == 1
    assert critique.issues[0]["severity"] == "medium"


def test_llm_predictor_handles_missing_fields():
    canned = json.dumps({"predict": {}, "critique": {}})
    predictor = LLMPredictor(MockLLMClient([canned]))
    p, c = predictor.predict(StateStore(), "test")
    assert p.candidate_actions == []
    assert p.confidence == 0.0
    assert c.issues == []


# ---------- HybridLoop.step (full pipeline) ----------
def _scripted_loop():
    """Build a loop with two scripted LLM responses (OBSERVE + RENDER)
    and one scripted Predict + Critique from MockPredictor."""
    observe_response = json.dumps({
        "intent": "question",
        "extracted_entities": [{"name": "Phase 18l", "type": "phase"}],
        "extracted_events": [],
        "proposed_object_updates": [],
    })
    render_response = (
        "Based on 18l's single-seed G1 fail, prioritize the 18mu multi-seed "
        "result; confidence 0.7 the multi-seed verdict (acceptable swap) is "
        "the right read."
    )
    llm = MockLLMClient([observe_response, render_response])
    predictor = MockPredictor([(
        Predict(
            current_state_summary="18l G1 failed single-seed; 18mu acceptable swap.",
            candidate_actions=[
                {"action": "trust 18mu multi-seed", "predicted_state_delta": [],
                 "expected_information_gain": 0.5, "risk": 0.2},
            ],
            recommended_action="trust 18mu multi-seed",
            confidence=0.7,
        ),
        Critique(issues=[
            {"kind": "missing_evidence", "claim_or_object_id": "phase_18l",
             "severity": "low", "recommended_revision": "low-severity, no guardrail"},
        ]),
    )])
    state = StateStore()
    seed_bla_tracker(state)
    loop = HybridLoop(
        llm=llm, predictor=predictor, state=state,
        domain_preamble=BLA_DOMAIN_SYSTEM_PROMPT,
    )
    return loop, llm, predictor, state


def test_step_runs_full_pipeline():
    loop, llm, predictor, state = _scripted_loop()
    rec = loop.step("Should I trust 18l or 18mu?")
    assert rec.user_input == "Should I trust 18l or 18mu?"
    assert rec.observe["intent"] == "question"
    assert rec.predict["recommended_action"] == "trust 18mu multi-seed"
    assert rec.render_text.startswith("Based on 18l")
    # Two LLM calls used (observe + render); predict came from MockPredictor
    assert len(llm.calls) == 2
    assert len(predictor.calls) == 1


def test_step_history_accumulates():
    loop, llm, _, _ = _scripted_loop()
    loop.step("Q1")
    # Reload scripted responses for a 2nd step
    second_observe = json.dumps({
        "intent": "question", "extracted_entities": [], "extracted_events": [],
        "proposed_object_updates": [],
    })
    second_render = "Second answer."
    loop.llm._responses.extend([second_observe, second_render])
    loop.predictor._scripted.append((
        Predict(recommended_action="ack", confidence=0.5),
        Critique(),
    ))
    loop.step("Q2")
    assert len(loop.history) == 2
    assert loop.history[0].user_input == "Q1"
    assert loop.history[1].user_input == "Q2"


def test_render_guardrails_propagate_from_critique():
    """High-severity critique recommendations land in Render.do_not_claim."""
    canned_observe = json.dumps({
        "intent": "question", "extracted_entities": [], "extracted_events": [],
        "proposed_object_updates": [],
    })
    llm = MockLLMClient([canned_observe, "OK"])
    predictor = MockPredictor([(
        Predict(recommended_action="x", confidence=0.4),
        Critique(issues=[
            {"kind": "unsupported_claim", "claim_or_object_id": "hyp_x",
             "severity": "high",
             "recommended_revision": "do not claim hyp_x is validated"},
            {"kind": "stale", "claim_or_object_id": "gate_y",
             "severity": "low",
             "recommended_revision": "low-severity — dropped from guardrails"},
        ]),
    )])
    state = StateStore()
    loop = HybridLoop(llm=llm, predictor=predictor, state=state)
    loop.step("ask")
    # Inspect the second (render) call to llm
    render_call = llm.calls[1]
    payload = json.loads(render_call["user"].split("PACKET (JSON):\n", 1)[1])
    assert payload["do_not_claim"] == ["do not claim hyp_x is validated"]


def test_observe_parse_failure_falls_back_gracefully():
    """If OBSERVE returns garbage, the loop still completes via predict+render."""
    llm = MockLLMClient([
        "not json at all",          # observe fails
        "Final answer despite observe failure.",  # render still runs
    ])
    predictor = MockPredictor([(
        Predict(recommended_action="x", confidence=0.5),
        Critique(),
    )])
    state = StateStore()
    loop = HybridLoop(llm=llm, predictor=predictor, state=state)
    rec = loop.step("anything")
    assert rec.observe["intent"] == "parse_failure"
    assert "observe_parse_error" in str(rec.observe["extracted_events"])
    assert rec.render_text.startswith("Final answer")


# ---------- apply_proposed_updates auto-nesting ----------
def test_apply_updates_auto_nests_unknown_keys_into_state():
    """Unknown top-level keys in `changes` should nest under `state`."""
    state = StateStore()
    state.add(ObjectFile(
        id="phase_19", type="phase", name="Phase 19",
        state={"existing": "keep me", "n_seeds": 3},
    ))
    loop = HybridLoop(
        llm=MockLLMClient([]),
        predictor=MockPredictor([]),
        state=state,
    )
    applied = loop.apply_proposed_updates([
        {"op": "update", "id": "phase_19",
         "changes": {"spearman_score": 0.48, "note": "corrected"}},
    ])
    assert applied[0]["ok"] is True
    assert set(applied[0]["auto_nested_into_state"]) == {"spearman_score", "note"}
    cur = state.get("phase_19")
    assert cur.state["spearman_score"] == 0.48
    assert cur.state["note"] == "corrected"
    # Existing state fields preserved
    assert cur.state["existing"] == "keep me"
    assert cur.state["n_seeds"] == 3


def test_apply_updates_top_level_changes_still_work():
    """Real top-level field updates should still be applied directly."""
    state = StateStore()
    state.add(ObjectFile(id="x", type="hypothesis", name="X", confidence=0.5))
    loop = HybridLoop(llm=MockLLMClient([]), predictor=MockPredictor([]), state=state)
    applied = loop.apply_proposed_updates([
        {"op": "update", "id": "x", "changes": {"confidence": 0.9}},
    ])
    assert applied[0]["ok"] is True
    assert applied[0]["auto_nested_into_state"] == []
    assert state.get("x").confidence == 0.9


def test_apply_updates_mixed_top_level_and_domain_fields():
    """Mixed changes — some top-level, some domain — should split cleanly."""
    state = StateStore()
    state.add(ObjectFile(id="x", type="phase", name="X", state={"foo": 1}))
    loop = HybridLoop(llm=MockLLMClient([]), predictor=MockPredictor([]), state=state)
    applied = loop.apply_proposed_updates([
        {"op": "update", "id": "x",
         "changes": {"confidence": 0.7, "spearman": 0.5, "note": "ok"}},
    ])
    assert applied[0]["ok"] is True
    assert set(applied[0]["auto_nested_into_state"]) == {"spearman", "note"}
    cur = state.get("x")
    assert cur.confidence == 0.7
    assert cur.state == {"foo": 1, "spearman": 0.5, "note": "ok"}


def test_apply_updates_missing_object_reports_failure():
    state = StateStore()
    loop = HybridLoop(llm=MockLLMClient([]), predictor=MockPredictor([]), state=state)
    applied = loop.apply_proposed_updates([
        {"op": "update", "id": "nope",
         "changes": {"spearman": 0.5}},
    ])
    assert applied[0]["ok"] is False
    assert "no object" in applied[0]["reason"]


def test_apply_updates_explicit_state_dict_takes_priority():
    """If caller passes both top-level domain key AND explicit state.k,
    the explicit state.k wins on overlap."""
    state = StateStore()
    state.add(ObjectFile(id="x", type="phase", name="X", state={"a": 1}))
    loop = HybridLoop(llm=MockLLMClient([]), predictor=MockPredictor([]), state=state)
    loop.apply_proposed_updates([
        {"op": "update", "id": "x",
         "changes": {"spearman": 0.3, "state": {"spearman": 0.9, "b": 2}}},
    ])
    cur = state.get("x")
    # Explicit state.spearman=0.9 wins over auto-nested spearman=0.3
    assert cur.state["spearman"] == 0.9
    assert cur.state["b"] == 2
    assert cur.state["a"] == 1


def test_step_surfaces_apply_outcomes_to_predictor_prompt():
    """When updates apply, the predictor's user_input gets a RECENT
    STATE CHANGES section appended so the predictor can reason over
    what actually happened — preventing the T3→T4 silent-drop bug."""
    canned_observe = json.dumps({
        "intent": "event_update",
        "extracted_entities": [],
        "extracted_events": [],
        "proposed_object_updates": [
            {"op": "update", "id": "x",
             "changes": {"spearman_score": 0.48}},
        ],
    })
    state = StateStore()
    state.add(ObjectFile(id="x", type="phase", name="X",
                          state={"spearman_score": 0.52}))
    llm = MockLLMClient([canned_observe, "rendered"])
    pred = MockPredictor([(Predict(confidence=0.7), Critique())])
    loop = HybridLoop(
        llm=llm, predictor=pred, state=state, auto_apply_updates=True,
    )
    loop.step("correction: 0.48")
    # Predictor was given the appended apply summary
    assert "RECENT STATE CHANGES THIS TURN" in pred.calls[0]["user_input"]
    # State was actually updated
    assert state.get("x").state["spearman_score"] == 0.48


# ---------- BLA tracker seed ----------
def test_seed_is_idempotent():
    s = StateStore()
    seed_bla_tracker(s)
    n = len(s)
    seed_bla_tracker(s)
    assert len(s) == n
