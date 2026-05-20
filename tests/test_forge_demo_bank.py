"""Unit tests for bla.forge.demo_bank (BF-1.0).

Covers:
  - DemoCollector lifecycle (start / append_action / finalize)
  - DemoCollector input validation
  - DemoBank directory scan + filters
  - DemoBank.add() round-trip to disk
  - DemoBank.to_demo_states() with caller-supplied key_fn
  - End-to-end: build mock bank → wire into DemoRetriever
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    DemoBank,
    DemoCollector,
    DemoRecord,
    build_mock_demo,
    mock_calibration,
    mock_fiducials,
    save_demo,
)
from bla.recipes import DemoRetriever


# ---------- DemoCollector ----------
def test_collector_full_lifecycle():
    bundle = mock_calibration()
    fids = mock_fiducials(
        ids=(0, 1),
        world_xy_per_id={0: (0.05, 0.0), 1: (-0.05, 0.10)},
        bundle=bundle,
    )
    coll = DemoCollector(demo_id=3, task="pickplace", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle,
                  extra={"gripper_open": 1.0})
    for t in range(15):
        coll.append_action(np.zeros(7, dtype=np.float32))
    assert coll.n_actions == 15
    rec = coll.finalize(achieved_outcome=0.7, collector_notes="ok")
    assert rec.demo_id == 3
    assert rec.task == "pickplace"
    assert rec.actions.shape == (15, 7)
    assert rec.achieved_outcome == 0.7
    assert rec.collector_notes == "ok"
    # Initial state has fiducials + extras
    assert "fiducials" in rec.initial_state
    assert "0" in rec.initial_state["fiducials"]
    assert "1" in rec.initial_state["fiducials"]
    assert rec.initial_state["gripper_open"] == 1.0
    # The world_xy fed into mock_fiducials should round-trip back
    np.testing.assert_allclose(
        rec.initial_state["fiducials"]["0"][:2], [0.05, 0.0], atol=1e-2,
    )


def test_collector_append_before_start_raises():
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    with pytest.raises(RuntimeError, match="before start"):
        coll.append_action(np.zeros(7, dtype=np.float32))


def test_collector_finalize_before_start_raises():
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    with pytest.raises(RuntimeError, match="before start"):
        coll.finalize(achieved_outcome=0.5)


def test_collector_start_twice_raises():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle)
    with pytest.raises(RuntimeError, match="twice"):
        coll.start(initial_fiducials=fids, bundle=bundle)


def test_collector_finalize_twice_raises():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle)
    coll.append_action(np.zeros(7, dtype=np.float32))
    coll.finalize(achieved_outcome=0.5)
    with pytest.raises(RuntimeError, match="already finalized"):
        coll.finalize(achieved_outcome=0.5)


def test_collector_finalize_empty_raises():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle)
    with pytest.raises(RuntimeError, match="no appended actions"):
        coll.finalize(achieved_outcome=0.5)


def test_collector_action_shape_validation():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle)
    with pytest.raises(ValueError, match="action must be"):
        coll.append_action(np.zeros(8, dtype=np.float32))   # wrong dim


def test_collector_outcome_range_validation():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle)
    coll.append_action(np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        coll.finalize(achieved_outcome=1.5)


def test_collector_extra_cannot_override_fiducials():
    bundle = mock_calibration()
    fids = mock_fiducials(ids=(0,), bundle=bundle)
    coll = DemoCollector(demo_id=0, task="t", action_dim=7)
    with pytest.raises(ValueError, match="cannot override 'fiducials'"):
        coll.start(initial_fiducials=fids, bundle=bundle,
                      extra={"fiducials": "bogus"})


# ---------- build_mock_demo ----------
def test_build_mock_demo_smoke():
    rec = build_mock_demo(demo_id=7, n_actions=12, cube_xy=(0.05, -0.03))
    assert rec.demo_id == 7
    assert rec.actions.shape == (12, 7)
    assert rec.achieved_outcome == 0.85
    assert "fiducials" in rec.initial_state
    # cube at (0.05, -0.03) should round-trip
    np.testing.assert_allclose(
        rec.initial_state["fiducials"]["0"][:2], [0.05, -0.03], atol=1e-2)


def test_build_mock_demo_is_deterministic_per_seed():
    a = build_mock_demo(demo_id=0, seed=42)
    b = build_mock_demo(demo_id=0, seed=42)
    np.testing.assert_array_equal(a.actions, b.actions)


# ---------- DemoBank ----------
def test_demo_bank_from_empty_directory(tmp_path: Path):
    bank = DemoBank.from_directory(tmp_path)
    assert len(bank) == 0
    assert bank.next_demo_id() == 0


def test_demo_bank_loads_all_demos(tmp_path: Path):
    for did in range(5):
        save_demo(build_mock_demo(demo_id=did), tmp_path / f"demo_{did:04d}.json")
    bank = DemoBank.from_directory(tmp_path)
    assert len(bank) == 5
    assert sorted(r.demo_id for r in bank.records) == [0, 1, 2, 3, 4]
    assert bank.next_demo_id() == 5


def test_demo_bank_task_filter(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0, task="pickplace"),
                tmp_path / "demo_0000.json")
    save_demo(build_mock_demo(demo_id=1, task="lift"),
                tmp_path / "demo_0001.json")
    save_demo(build_mock_demo(demo_id=2, task="pickplace"),
                tmp_path / "demo_0002.json")
    bank = DemoBank.from_directory(tmp_path, task_filter="pickplace")
    assert len(bank) == 2
    assert sorted(r.demo_id for r in bank.records) == [0, 2]


def test_demo_bank_min_outcome_filter(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0, achieved_outcome=0.4),
                tmp_path / "demo_0000.json")
    save_demo(build_mock_demo(demo_id=1, achieved_outcome=0.8),
                tmp_path / "demo_0001.json")
    bank = DemoBank.from_directory(tmp_path, min_outcome=0.5)
    assert len(bank) == 1
    assert bank.records[0].demo_id == 1


def test_demo_bank_skip_invalid_json(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0), tmp_path / "demo_0000.json")
    (tmp_path / "demo_bogus.json").write_text("not json {")
    with pytest.warns(UserWarning, match="skipping"):
        bank = DemoBank.from_directory(tmp_path, skip_invalid=True)
    assert len(bank) == 1


def test_demo_bank_strict_raises_on_invalid(tmp_path: Path):
    (tmp_path / "demo_bogus.json").write_text("not json {")
    with pytest.raises(Exception):
        DemoBank.from_directory(tmp_path, skip_invalid=False)


def test_demo_bank_by_demo_id(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0), tmp_path / "demo_0000.json")
    save_demo(build_mock_demo(demo_id=7), tmp_path / "demo_0007.json")
    bank = DemoBank.from_directory(tmp_path)
    rec = bank.by_demo_id(7)
    assert rec.demo_id == 7
    with pytest.raises(KeyError):
        bank.by_demo_id(99)


def test_demo_bank_add_writes_to_disk(tmp_path: Path):
    bank = DemoBank.from_directory(tmp_path)
    rec = build_mock_demo(demo_id=bank.next_demo_id())
    path = bank.add(rec)
    assert path.exists()
    assert len(bank) == 1
    # Reload from disk: persisted
    bank2 = DemoBank.from_directory(tmp_path)
    assert len(bank2) == 1
    assert bank2.records[0].demo_id == 0


def test_demo_bank_add_rejects_duplicate_id(tmp_path: Path):
    bank = DemoBank.from_directory(tmp_path)
    bank.add(build_mock_demo(demo_id=0))
    with pytest.raises(ValueError, match="already contains"):
        bank.add(build_mock_demo(demo_id=0))


def test_demo_bank_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DemoBank.from_directory(tmp_path / "nope")


# ---------- DemoBank → DemoRetriever bridge ----------
def _key_fn_pickplace(r: DemoRecord) -> np.ndarray:
    """Test key_fn: cube_xy ⊕ marker_xy (4-D, mock)."""
    cube = r.initial_state["fiducials"]["0"][:2]
    marker = r.initial_state["fiducials"]["1"][:2]
    return np.array([*cube, *marker], dtype=np.float32)


def test_demo_bank_to_demo_states_basic(tmp_path: Path):
    for did, cube in enumerate([(0.0, 0.0), (0.05, 0.0), (-0.05, 0.05)]):
        save_demo(build_mock_demo(demo_id=did, cube_xy=cube),
                    tmp_path / f"demo_{did:04d}.json")
    bank = DemoBank.from_directory(tmp_path)
    states = bank.to_demo_states(key_fn=_key_fn_pickplace)
    assert len(states) == 3
    for s, r in zip(states, bank.records):
        assert s.demo_id == r.demo_id
        assert s.outcome_score == r.achieved_outcome
        np.testing.assert_array_equal(s.action_seq, r.actions)
        assert s.key.shape == (4,)
        assert s.metadata["task"] == r.task


def test_demo_bank_to_demo_states_inconsistent_key_shape_raises(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0), tmp_path / "demo_0000.json")
    save_demo(build_mock_demo(demo_id=1), tmp_path / "demo_0001.json")
    bank = DemoBank.from_directory(tmp_path)
    # First call returns 4-D, second returns 6-D → should raise.
    sizes = iter([4, 6])
    def bad_key_fn(r):
        return np.zeros(next(sizes), dtype=np.float32)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        bank.to_demo_states(key_fn=bad_key_fn)


def test_demo_bank_to_demo_states_with_init_state_fn(tmp_path: Path):
    save_demo(build_mock_demo(demo_id=0), tmp_path / "demo_0000.json")
    bank = DemoBank.from_directory(tmp_path)
    states = bank.to_demo_states(
        key_fn=_key_fn_pickplace,
        init_state_fn=lambda r: np.linspace(0, 1, 45, dtype=np.float64),
    )
    assert states[0].init_state is not None
    assert states[0].init_state.shape == (45,)


def test_demo_bank_full_retrieval_pipeline(tmp_path: Path):
    """End-to-end: write 5 demos at distinct cube positions, load the bank,
    convert to states, build a retriever index, and verify NN retrieval
    finds the demo whose cube position is closest to the query."""
    cubes = [(0.00, 0.00), (0.10, 0.00), (-0.10, 0.00),
                (0.00, 0.10), (0.00, -0.10)]
    for did, c in enumerate(cubes):
        save_demo(build_mock_demo(demo_id=did, cube_xy=c),
                    tmp_path / f"demo_{did:04d}.json")
    bank = DemoBank.from_directory(tmp_path)
    states = bank.to_demo_states(key_fn=_key_fn_pickplace)
    retr = DemoRetriever()
    retr.build_index(states)

    # Query at (0.09, 0.01) — closest to demo_id=1 (cube_xy=(0.10,0))
    query = np.array([0.09, 0.01, 0.10, -0.10], dtype=np.float32)
    [top] = retr.retrieve(query, k=1)
    assert top.demo_id == 1
