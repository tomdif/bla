"""Unit tests for bla.forge.deploy_loop (BF-1.1).

Verifies the full mock deployment cycle:
  perception → retrieval → action replay → safety → log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bla.forge import (
    DemoBank,
    build_mock_demo,
    build_mock_deployment_loop,
    save_demo,
    save_episode,
    load_episode,
)


def _bank_with_three_cubes(tmp_path: Path) -> DemoBank:
    """Build a 3-demo bank at three distinct cube positions."""
    cubes = [(-0.08, 0.00), (0.00, 0.00), (+0.08, 0.00)]
    bank = DemoBank.from_directory(tmp_path)
    for did, c in enumerate(cubes):
        bank.add(build_mock_demo(demo_id=did, cube_xy=c, n_actions=12,
                                          achieved_outcome=0.7 + 0.05 * did))
    return bank


def _key_cube_xy(scene: dict) -> np.ndarray:
    return np.asarray(scene[0], dtype=np.float32)


# ---------- happy path ----------
def test_deploy_loop_retrieves_nearest_demo(tmp_path: Path):
    bank = _bank_with_three_cubes(tmp_path)
    # Query near demo 2 (cube at +0.08)
    rec = build_mock_deployment_loop(
        bank=bank,
        key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.075, 0.005),
                                  1: (0.10, -0.10)},
    )
    assert rec.retrieved_demo["demo_id"] == 2
    assert rec.retrieved_demo["nn_distance"] < 0.02  # very close
    assert rec.outcome["success"] is True
    # The episode replays demo 2's action_seq (12 steps)
    assert rec.frames.shape[0] == 12
    assert rec.gantry_actions.shape == (12, 7)


def test_deploy_loop_replays_demo_action_sequence(tmp_path: Path):
    """The episode's gantry_actions array MUST match the retrieved
    demo's action_seq element-wise (Recipe E2_FAST: no perturbation)."""
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.00, 0.00), 1: (0.10, -0.10)},
    )
    retrieved_id = rec.retrieved_demo["demo_id"]
    demo = bank.by_demo_id(retrieved_id)
    np.testing.assert_allclose(rec.gantry_actions,
                                       demo.actions[:rec.gantry_actions.shape[0]],
                                       atol=1e-6)


def test_deploy_loop_uses_e2_fast_recipe(tmp_path: Path):
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.0, 0.0), 1: (0.10, -0.10)},
    )
    assert rec.router_decision["recipe"] == "E2_FAST"
    assert rec.router_decision["task_descriptor"]["prior_kind"] == "demo"


def test_deploy_loop_perception_decodes_query_cube(tmp_path: Path):
    """The rolling tracker should decode slot 0 at the queried cube_xy."""
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.05, -0.03), 1: (0.10, -0.10)},
    )
    # Slot 0 binds to fiducial 0 (FCFS). decoded_positions[t, 0]
    # should hover near (0.05, -0.03) modulo small drift.
    np.testing.assert_allclose(rec.decoded_positions[0, 0],
                                       [0.05, -0.03], atol=5e-3)


def test_deploy_loop_max_steps_truncates(tmp_path: Path):
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.0, 0.0), 1: (0.10, -0.10)},
        max_steps=5,
    )
    assert rec.frames.shape[0] == 5


# ---------- safety integration ----------
def test_deploy_loop_safety_breach_halts(tmp_path: Path):
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.0, 0.0), 1: (0.10, -0.10)},
        include_safety=True, safety_breach_at_step=3,
    )
    assert rec.frames.shape[0] == 4   # 0..3 with breach at 3
    assert len(rec.safety_events) == 1
    assert rec.safety_events[0]["action"] == "halt"
    assert rec.outcome["success"] is False


def test_deploy_loop_safety_disabled(tmp_path: Path):
    bank = _bank_with_three_cubes(tmp_path)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.0, 0.0), 1: (0.10, -0.10)},
        include_safety=False, safety_breach_at_step=3,
    )
    # With safety off, breach is ignored → full demo replays
    assert rec.frames.shape[0] == 12
    assert rec.safety_events == []


# ---------- empty bank ----------
def test_deploy_loop_empty_bank_raises(tmp_path: Path):
    bank = DemoBank.from_directory(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        build_mock_deployment_loop(
            bank=bank, key_fn=_key_cube_xy,
            query_world_xy_per_id={0: (0.0, 0.0)},
        )


# ---------- record round-trip ----------
def test_deploy_loop_record_round_trips_through_disk(tmp_path: Path):
    """The deploy-loop's EpisodeRecord must save+load identically (this
    is the same contract BF-0.4 verified, but exercised through the
    full deployment cycle)."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    for did in range(3):
        save_demo(build_mock_demo(demo_id=did, cube_xy=(0.05*did, 0.0),
                                          n_actions=10),
                    bank_dir / f"demo_{did:04d}.json")
    bank = DemoBank.from_directory(bank_dir)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.06, 0.0), 1: (0.10, -0.10)},
    )
    ep_path = tmp_path / "ep0.json"
    save_episode(rec, ep_path, use_sidecar=True)
    loaded = load_episode(ep_path)
    np.testing.assert_allclose(loaded.gantry_actions, rec.gantry_actions,
                                       atol=1e-6)
    assert loaded.retrieved_demo == rec.retrieved_demo
    assert loaded.router_decision == rec.router_decision
