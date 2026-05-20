"""Unit tests for bla.forge.episode (BF-0.4).

Validates the EpisodeRecord + DemoRecord + EpisodeLogger contracts.
Also exercises `build_mock_episode_loop` — the first BF-0 deliverable
(a mocked camera/gantry loop producing the EXACT real-hardware JSON
schema).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    DemoRecord,
    DEMO_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    EpisodeLogger,
    EpisodeRecord,
    FiducialDetection,
    ObservationStep,
    RollingObjectFileTracker,
    build_mock_episode_loop,
    load_demo,
    load_episode,
    mock_calibration,
    mock_fiducials,
    save_demo,
    save_episode,
)


# ---------- DemoRecord ----------
def test_demo_record_round_trip(tmp_path: Path):
    d = DemoRecord(
        demo_id=42, task="pickplace",
        initial_state={"object_pose": [0, 0, 0.84, 0.0],
                          "eef_pose": [0.1, 0, 1.0, 0.0],
                          "gripper_open": 1.0},
        actions=np.random.RandomState(0).randn(20, 7).astype(np.float32),
        achieved_outcome=0.85,
        collector_notes="test demo",
    )
    path = tmp_path / "demo.json"
    save_demo(d, path)
    loaded = load_demo(path)
    assert loaded.demo_id == 42
    assert loaded.task == "pickplace"
    assert loaded.initial_state == d.initial_state
    np.testing.assert_allclose(loaded.actions, d.actions, atol=1e-6)
    assert loaded.achieved_outcome == 0.85
    assert loaded.collector_notes == "test demo"
    assert loaded.schema_version == DEMO_SCHEMA_VERSION


def test_demo_record_validates_action_shape():
    with pytest.raises(ValueError, match="actions must be"):
        DemoRecord(
            demo_id=0, task="x", initial_state={},
            actions=np.zeros((7,)),  # 1-D, not 2-D
            achieved_outcome=0.0,
        )


def test_demo_record_to_demo_state_for_retriever():
    """A DemoRecord must convert cleanly to bla.recipes.DemoState."""
    d = DemoRecord(
        demo_id=7, task="pickplace",
        initial_state={"object_pose": [0.1, 0.2, 0.84, 0.0]},
        actions=np.random.RandomState(0).randn(30, 7).astype(np.float32),
        achieved_outcome=0.92,
    )
    key = np.array([0.1, 0.2, 0.0, 0.0, 0.84, 1.0], dtype=np.float32)
    ds = d.to_demo_state(key=key)
    # It IS a DemoState (DemoState is duck-typed; we check the fields)
    assert ds.demo_id == 7
    np.testing.assert_array_equal(ds.key, key)
    np.testing.assert_array_equal(ds.action_seq, d.actions)
    assert ds.outcome_score == 0.92
    assert ds.metadata["task"] == "pickplace"
    assert ds.init_state is None  # not provided


def test_demo_record_to_demo_state_with_init_state():
    d = DemoRecord(
        demo_id=1, task="lift", initial_state={},
        actions=np.zeros((10, 7), dtype=np.float32),
        achieved_outcome=0.5,
    )
    flat_state = np.linspace(0, 1, 45, dtype=np.float64)
    ds = d.to_demo_state(
        key=np.zeros(6, dtype=np.float32),
        init_state_flat=flat_state,
    )
    np.testing.assert_array_equal(ds.init_state, flat_state)


def test_demo_record_schema_version_mismatch_warns(tmp_path: Path):
    d = DemoRecord(
        demo_id=0, task="x", initial_state={},
        actions=np.zeros((1, 7), dtype=np.float32),
        achieved_outcome=0.0,
    )
    path = tmp_path / "old.json"
    save_demo(d, path)
    # Tamper with the file: set a different schema version
    raw = json.loads(path.read_text())
    raw["schema_version"] = "0.0"
    path.write_text(json.dumps(raw))
    with pytest.warns(UserWarning, match="schema version mismatch"):
        load_demo(path)


# ---------- EpisodeRecord ----------
def _make_minimal_episode() -> EpisodeRecord:
    """A small fixture EpisodeRecord for serialization tests."""
    T = 5
    return EpisodeRecord(
        ep_id=0,
        timestamp="2026-05-20T12:00:00Z",
        task="pickplace",
        router_decision={"recipe": "E2_FAST", "rationale": "test",
                            "task_descriptor": {"prior_kind": "demo"}},
        outcome={"success": True, "improvement": 1.0,
                   "metric_name": "cube_z_gain", "notes": ""},
        retrieved_demo={"demo_id": 5, "nn_distance": 0.0, "filter_passed": [5]},
        frames=np.zeros((T, 32, 32, 3), dtype=np.uint8),
        slot_states=np.random.RandomState(0).randn(T, 6, 128).astype(np.float32),
        decoded_positions=np.random.RandomState(1).randn(T, 6, 2).astype(np.float32),
        gantry_actions=np.zeros((T, 7), dtype=np.float32),
        gantry_states=np.zeros((T, 9), dtype=np.float32),
        safety_events=[{"timestep": 2, "reason": "test", "action": "log_only"}],
        perturbations=[],
    )


def test_episode_record_save_load_sidecar(tmp_path: Path):
    """Default save uses .npz sidecar for bulky arrays."""
    rec = _make_minimal_episode()
    json_path = tmp_path / "ep0.json"
    save_episode(rec, json_path, use_sidecar=True)
    assert json_path.exists()
    sidecar = tmp_path / "ep0.npz"
    assert sidecar.exists()
    # JSON should NOT contain bulky arrays
    raw = json.loads(json_path.read_text())
    assert raw["arrays_sidecar"] == "ep0.npz"
    assert "frames" not in raw   # offloaded
    assert "slot_states" not in raw

    loaded = load_episode(json_path)
    assert loaded.ep_id == 0
    assert loaded.task == "pickplace"
    assert loaded.outcome == rec.outcome
    np.testing.assert_array_equal(loaded.frames, rec.frames)
    np.testing.assert_array_equal(loaded.slot_states, rec.slot_states)
    np.testing.assert_array_equal(loaded.decoded_positions, rec.decoded_positions)
    np.testing.assert_array_equal(loaded.gantry_actions, rec.gantry_actions)
    np.testing.assert_array_equal(loaded.gantry_states, rec.gantry_states)
    assert loaded.safety_events == rec.safety_events


def test_episode_record_save_load_inline(tmp_path: Path):
    """Inline mode: bulky arrays inside the JSON, no sidecar."""
    rec = _make_minimal_episode()
    json_path = tmp_path / "ep_inline.json"
    save_episode(rec, json_path, use_sidecar=False)
    assert json_path.exists()
    # No sidecar
    assert not (tmp_path / "ep_inline.npz").exists()
    raw = json.loads(json_path.read_text())
    assert "frames" in raw   # inline
    assert "slot_states" in raw
    assert "arrays_sidecar" not in raw

    loaded = load_episode(json_path)
    np.testing.assert_array_equal(loaded.frames, rec.frames)
    np.testing.assert_allclose(loaded.slot_states, rec.slot_states, atol=1e-6)


def test_episode_record_missing_sidecar_raises(tmp_path: Path):
    """If JSON references a sidecar that doesn't exist, load fails clearly."""
    rec = _make_minimal_episode()
    json_path = tmp_path / "ep_orphan.json"
    save_episode(rec, json_path, use_sidecar=True)
    sidecar = tmp_path / "ep_orphan.npz"
    sidecar.unlink()   # delete the sidecar
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_episode(json_path)


def test_episode_record_schema_version_mismatch_warns(tmp_path: Path):
    rec = _make_minimal_episode()
    json_path = tmp_path / "ep_old.json"
    save_episode(rec, json_path, use_sidecar=False)
    raw = json.loads(json_path.read_text())
    raw["schema_version"] = "0.0"
    json_path.write_text(json.dumps(raw))
    with pytest.warns(UserWarning, match="schema version mismatch"):
        load_episode(json_path)


def test_episode_record_sidecar_size_is_smaller_than_inline(tmp_path: Path):
    """Sanity: with 200-step rollout, the sidecar version's JSON is much
    smaller than the inline version's. This is the whole point of
    sidecar mode."""
    T = 200
    rec = EpisodeRecord(
        ep_id=0, timestamp="t", task="pickplace",
        router_decision={}, outcome={}, retrieved_demo={},
        frames=np.zeros((T, 64, 64, 3), dtype=np.uint8),
        slot_states=np.zeros((T, 6, 128), dtype=np.float32),
        decoded_positions=np.zeros((T, 6, 2), dtype=np.float32),
        gantry_actions=np.zeros((T, 7), dtype=np.float32),
        gantry_states=np.zeros((T, 9), dtype=np.float32),
    )
    inline_path = tmp_path / "inline.json"
    sidecar_path = tmp_path / "sidecar.json"
    save_episode(rec, inline_path, use_sidecar=False)
    save_episode(rec, sidecar_path, use_sidecar=True)
    inline_size = inline_path.stat().st_size
    sidecar_size = sidecar_path.stat().st_size + \
        (tmp_path / "sidecar.npz").stat().st_size
    # JSON-only inline file must be much larger than the sidecar's JSON
    # (the .npz can still be large but is binary-efficient).
    assert sidecar_path.stat().st_size < inline_size / 100, (
        f"Sidecar JSON should be tiny ({sidecar_path.stat().st_size}B) "
        f"vs inline JSON ({inline_size}B)")


# ---------- EpisodeLogger ----------
def test_episode_logger_appends_aligned_steps():
    """N append_step calls produce N-step arrays in finalize."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    logger = EpisodeLogger(ep_id=0, task="pickplace")
    fids = mock_fiducials(bundle=bundle)
    for t in range(7):
        obs = tracker.step(np.zeros((64, 64, 3), dtype=np.uint8), fids)
        logger.append_step(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            obs=obs,
            gantry_action=np.zeros(7, dtype=np.float32),
            gantry_state=np.zeros(9, dtype=np.float32),
        )
    assert logger.n_steps == 7
    logger.set_outcome(success=True, improvement=0.5,
                          metric_name="z_gain")
    rec = logger.finalize()
    assert rec.frames.shape[0] == 7
    assert rec.slot_states.shape == (7, 6, 128)
    assert rec.decoded_positions.shape == (7, 6, 2)
    assert rec.gantry_actions.shape == (7, 7)
    assert rec.gantry_states.shape == (7, 9)


def test_episode_logger_set_outcome_validates_range():
    logger = EpisodeLogger(ep_id=0, task="t")
    with pytest.raises(ValueError, match="improvement must be"):
        logger.set_outcome(success=True, improvement=1.5,
                              metric_name="x")


def test_episode_logger_finalize_without_outcome_warns():
    logger = EpisodeLogger(ep_id=0, task="t")
    with pytest.warns(UserWarning, match="set_outcome"):
        rec = logger.finalize()
    assert rec.outcome == {}


def test_episode_logger_safety_events_recorded():
    logger = EpisodeLogger(ep_id=0, task="t")
    logger.log_safety_event(timestep=3, reason="bounds_warn",
                                action="slow_to_zero")
    logger.log_safety_event(timestep=7, reason="e_stop",
                                action="halt")
    logger.set_outcome(success=False, improvement=0.0,
                          metric_name="x", notes="e-stop")
    rec = logger.finalize()
    assert len(rec.safety_events) == 2
    assert rec.safety_events[0]["reason"] == "bounds_warn"
    assert rec.safety_events[1]["action"] == "halt"


def test_episode_logger_perturbations_recorded():
    logger = EpisodeLogger(ep_id=0, task="t")
    logger.log_perturbation({"timestep": 4, "kind": "manual_push",
                                  "magnitude_m": 0.05})
    logger.set_outcome(success=True, improvement=0.5, metric_name="x")
    rec = logger.finalize()
    assert len(rec.perturbations) == 1
    assert rec.perturbations[0]["kind"] == "manual_push"


# ---------- the mocked camera/gantry loop ----------
def test_build_mock_episode_loop_default_args_runs():
    """Top-level: the mock loop produces a valid EpisodeRecord."""
    rec = build_mock_episode_loop(ep_id=42, n_steps=20)
    assert rec.ep_id == 42
    assert rec.task == "pickplace"
    assert rec.frames.shape[0] == 20
    assert rec.slot_states.shape == (20, 6, 128)
    assert rec.decoded_positions.shape == (20, 6, 2)
    assert rec.gantry_actions.shape == (20, 7)
    assert rec.gantry_states.shape == (20, 9)
    assert rec.outcome["success"] is True
    assert rec.outcome["improvement"] == 1.0


def test_build_mock_episode_loop_round_trip_through_json(tmp_path: Path):
    """End-to-end: mocked loop → save → load → identical record.
    This is the BF-0 first-deliverable acceptance test."""
    rec = build_mock_episode_loop(ep_id=7, n_steps=15)
    json_path = tmp_path / "ep7.json"
    save_episode(rec, json_path, use_sidecar=True)
    loaded = load_episode(json_path)
    assert loaded.ep_id == 7
    assert loaded.task == rec.task
    assert loaded.outcome == rec.outcome
    np.testing.assert_array_equal(loaded.frames, rec.frames)
    np.testing.assert_allclose(loaded.slot_states, rec.slot_states, atol=1e-6)
    np.testing.assert_allclose(loaded.decoded_positions,
                                       rec.decoded_positions, atol=1e-6)


def test_mock_loop_decoded_positions_move_with_object():
    """The synthetic object sweeps across the workspace; decoded positions
    of its slot should move monotonically."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=20)
    # The mock loop places ID 0 at start_xy → end_xy. ID 0 should be in
    # one of the slots; find which.
    # Slot 0 should be bound to fiducial 0 (FCFS).
    slot_0_x = rec.decoded_positions[:, 0, 0]   # x-coord of slot 0 over time
    # Object moves from -0.10 → +0.10 in x; slot 0 should track that.
    assert slot_0_x[0] < slot_0_x[-1]    # x increases
    np.testing.assert_allclose(slot_0_x[0], -0.10, atol=1e-3)
    np.testing.assert_allclose(slot_0_x[-1], +0.10, atol=1e-3)


def test_mock_loop_router_decision_recorded():
    rec = build_mock_episode_loop(ep_id=0, n_steps=5)
    assert rec.router_decision["recipe"] == "E2_FAST"
    assert "rationale" in rec.router_decision
    assert "task_descriptor" in rec.router_decision


def test_mock_loop_with_custom_router_decision():
    custom = {"recipe": "A", "rationale": "stack push",
                "task_descriptor": {"prior_kind": "fsm"}}
    rec = build_mock_episode_loop(ep_id=0, n_steps=3,
                                          router_decision=custom)
    assert rec.router_decision == custom


# ---------- demo bank ↔ DemoRetriever bridge ----------
def test_demo_record_drops_into_demo_retriever(tmp_path: Path):
    """The whole point of DemoRecord: produce demos that the existing
    DemoRetriever can index without any glue."""
    from bla.recipes import DemoRetriever
    demos = []
    for did in range(3):
        d = DemoRecord(
            demo_id=did, task="pickplace",
            initial_state={"object_pose": [0.1 * did, 0.0, 0.84, 0.0]},
            actions=np.random.RandomState(did).randn(20, 7).astype(np.float32),
            achieved_outcome=0.5 + 0.1 * did,
        )
        demos.append(d.to_demo_state(
            key=np.array([0.1 * did, 0.0, 0.0, 0.0, 0.84, 1.0],
                            dtype=np.float32),
        ))
    retr = DemoRetriever()
    retr.build_index(demos)
    # Retrieve the demo closest to demo_id=1's key
    top = retr.retrieve(np.array([0.10, 0.0, 0.0, 0.0, 0.84, 1.0],
                                       dtype=np.float32), k=1)[0]
    assert top.demo_id == 1
    assert top.outcome_score == 0.6
