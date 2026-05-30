"""Unit tests for bla.hybrid.protocol dataclasses."""
from __future__ import annotations

from bla.hybrid.protocol import (
    Critique,
    Observe,
    Plan,
    Predict,
    Render,
    Update,
)


def test_observe_round_trip():
    o = Observe(
        raw_input="hello",
        intent="question",
        extracted_entities=[{"name": "X", "type": "hypothesis"}],
        extracted_events=[{"kind": "result_observed", "summary": "..."}],
    )
    d = o.to_dict()
    assert d["raw_input"] == "hello"
    assert d["intent"] == "question"
    assert len(d["extracted_entities"]) == 1


def test_update_round_trip():
    u = Update(changed_objects=[{"op": "add", "object": {"id": "x"}}])
    assert u.to_dict()["changed_objects"][0]["op"] == "add"


def test_predict_defaults_safe():
    p = Predict()
    d = p.to_dict()
    assert d["candidate_actions"] == []
    assert d["confidence"] == 0.0
    assert d["recommended_action"] is None


def test_critique_severity_passthrough():
    c = Critique(issues=[
        {"kind": "contradiction", "claim_or_object_id": "x",
         "severity": "high", "recommended_revision": "do Y"},
    ])
    assert c.to_dict()["issues"][0]["severity"] == "high"


def test_plan_round_trip():
    p = Plan(
        recommended_action="run X",
        alternatives=[{"action": "run Y"}],
        reason="highest EIG",
    )
    assert p.to_dict()["recommended_action"] == "run X"


def test_render_do_not_claim_carried():
    r = Render(
        content_packet={"x": 1},
        do_not_claim=["do not say X is solved"],
    )
    d = r.to_dict()
    assert d["do_not_claim"] == ["do not say X is solved"]
    assert d["audience"] == "user"
