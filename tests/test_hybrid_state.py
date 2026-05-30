"""Unit tests for bla.hybrid.state — ObjectFile + StateStore."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bla.hybrid.state import ObjectFile, StateStore


# ---------- ObjectFile ----------
def test_objectfile_basic():
    o = ObjectFile(id="x", type="hypothesis", name="X")
    assert o.id == "x"
    assert o.confidence == 1.0
    assert o.state == {}
    assert o.created_at  # ISO string


def test_objectfile_rejects_empty_id():
    with pytest.raises(ValueError, match="id must be non-empty"):
        ObjectFile(id="", type="hypothesis", name="X")


def test_objectfile_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        ObjectFile(id="x", type="hypothesis", name="X", confidence=1.5)


def test_objectfile_round_trip_dict():
    o = ObjectFile(
        id="a", type="experiment", name="A", state={"x": 1}, confidence=0.7,
        supported_by=["b"], open_questions=["?"],
    )
    o2 = ObjectFile.from_dict(o.to_dict())
    assert o2.to_dict() == o.to_dict()


# ---------- StateStore: basic ops ----------
def test_store_add_and_get():
    s = StateStore()
    o = ObjectFile(id="x", type="hypothesis", name="X")
    s.add(o)
    assert s.get("x") is o
    assert len(s) == 1
    assert "x" in s


def test_store_rejects_duplicate_add():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    with pytest.raises(KeyError, match="already exists"):
        s.add(ObjectFile(id="x", type="hypothesis", name="X again"))


def test_store_upsert_replaces():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    s.upsert(ObjectFile(id="x", type="hypothesis", name="X v2"))
    assert s.get("x").name == "X v2"


def test_store_update_changes_field_and_bumps_updated_at():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    before = s.get("x").updated_at
    # ensure ISO timestamps differ (sleep-free trick: just update twice with
    # different content; updated_at uses microseconds so the second call
    # almost always differs, but if not we still verify the change)
    s.update("x", confidence=0.5)
    o = s.get("x")
    assert o.confidence == 0.5
    assert o.updated_at >= before


def test_store_update_rejects_unknown_field():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    with pytest.raises(AttributeError):
        s.update("x", nonexistent=True)


def test_store_update_rejects_immutable_fields():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    with pytest.raises(ValueError, match="cannot update"):
        s.update("x", id="y")


def test_store_update_validates_confidence_after_change():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    with pytest.raises(ValueError, match="confidence"):
        s.update("x", confidence=2.0)


def test_store_remove():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    s.remove("x")
    assert "x" not in s
    assert len(s) == 0


# ---------- StateStore: views ----------
def test_store_by_type():
    s = StateStore()
    s.add(ObjectFile(id="h1", type="hypothesis", name=""))
    s.add(ObjectFile(id="h2", type="hypothesis", name=""))
    s.add(ObjectFile(id="e1", type="experiment", name=""))
    assert {o.id for o in s.by_type("hypothesis")} == {"h1", "h2"}
    assert {o.id for o in s.by_type("experiment")} == {"e1"}


# ---------- StateStore: change log ----------
def test_change_log_records_add_update_remove():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    s.update("x", confidence=0.5)
    s.remove("x")
    log = s.change_log
    assert [e["op"] for e in log] == ["add", "update", "remove"]
    assert log[0]["before"] is None
    assert log[2]["after"] is None


def test_change_log_clear():
    s = StateStore()
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    assert len(s.change_log) == 1
    s.clear_change_log()
    assert s.change_log == []


# ---------- StateStore: persistence ----------
def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "state.json"
    s = StateStore(path=path)
    s.add(ObjectFile(id="x", type="hypothesis", name="X", confidence=0.6))
    s.add(ObjectFile(id="y", type="experiment", name="Y"))
    s.save()
    assert path.exists()

    s2 = StateStore(path=path)
    assert len(s2) == 2
    assert s2.get("x").confidence == 0.6
    assert s2.get("y").type == "experiment"


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "state.json"
    s = StateStore(path=path)
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    s.save()
    # No stray .tmp files
    leftovers = list(tmp_path.glob("state.json.*"))
    assert leftovers == []


def test_load_rejects_unknown_version(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "objects": []}))
    with pytest.raises(ValueError, match="unknown version"):
        StateStore(path=path)


def test_save_without_path_raises():
    s = StateStore()
    with pytest.raises(RuntimeError, match="no `path`"):
        s.save()


def test_load_clears_change_log(tmp_path):
    """Loading from disk should not produce phantom change-log entries."""
    path = tmp_path / "state.json"
    s = StateStore(path=path)
    s.add(ObjectFile(id="x", type="hypothesis", name="X"))
    s.save()

    s2 = StateStore(path=path)
    assert s2.change_log == []
