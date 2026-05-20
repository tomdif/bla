"""Unit tests for bla.forge.outcome (BF-1.2)."""
from __future__ import annotations

import numpy as np
import pytest

from bla.forge import (
    DemoBank,
    EpisodeRecord,
    build_mock_demo,
    build_mock_deployment_loop,
    build_mock_episode_loop,
    combine_scores,
    compute_outcome,
    cube_xy_displacement_score,
    episode_length_fraction_score,
    episode_to_demo,
    no_safety_events_score,
)


# ---------- cube_xy_displacement_score ----------
def test_displacement_score_full_episode_sweep():
    """build_mock_episode_loop sweeps the cube 0.20m in XY — that's
    well above the 0.10m target, so the score should saturate at 1.0."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=False)
    s = cube_xy_displacement_score(rec, slot_id=0, target_m=0.10)
    assert s == 1.0


def test_displacement_score_below_target():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=False)
    s = cube_xy_displacement_score(rec, slot_id=0, target_m=0.50)
    # Total displacement is sqrt(0.20² + 0.10²) ≈ 0.224, normalized
    # by 0.50 ≈ 0.447. (start = (-0.10, -0.05), end = (+0.10, +0.05))
    assert 0.40 < s < 0.50


def test_displacement_score_clamped_to_unit():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=False)
    s = cube_xy_displacement_score(rec, slot_id=0, target_m=0.01)
    assert s == 1.0   # clipped


def test_displacement_score_empty_episode_returns_zero():
    """An empty EpisodeRecord (no steps) scores 0."""
    rec = EpisodeRecord(
        ep_id=0, timestamp="t", task="x",
        router_decision={}, outcome={}, retrieved_demo={},
        frames=np.zeros((0, 32, 32, 3), dtype=np.uint8),
        slot_states=np.zeros((0, 6, 128), dtype=np.float32),
        decoded_positions=np.zeros((0, 6, 2), dtype=np.float32),
        gantry_actions=np.zeros((0, 7), dtype=np.float32),
        gantry_states=np.zeros((0, 9), dtype=np.float32),
    )
    assert cube_xy_displacement_score(rec) == 0.0


def test_displacement_score_invalid_slot_raises():
    rec = build_mock_episode_loop(ep_id=0, n_steps=5)
    with pytest.raises(ValueError, match="out of range"):
        cube_xy_displacement_score(rec, slot_id=99)


def test_displacement_score_invalid_target_raises():
    rec = build_mock_episode_loop(ep_id=0, n_steps=5)
    with pytest.raises(ValueError, match="target_m must be"):
        cube_xy_displacement_score(rec, target_m=-1.0)


def test_displacement_score_all_nan_slot_returns_zero():
    """A slot with NaN decoded positions (unbound) scores 0."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=10)
    # Slot 5 (unbound, since only fiducials 0 and 1 exist) should be NaN
    assert np.all(np.isnan(rec.decoded_positions[:, 5, 0]))
    assert cube_xy_displacement_score(rec, slot_id=5) == 0.0


# ---------- no_safety_events_score ----------
def test_no_safety_events_score_clean_episode():
    rec = build_mock_episode_loop(ep_id=0, n_steps=10, include_safety=True)
    assert no_safety_events_score(rec) == 1.0


def test_no_safety_events_score_with_breach():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True,
                                          safety_breach_at_step=3)
    assert no_safety_events_score(rec) == 0.0


# ---------- episode_length_fraction_score ----------
def test_length_fraction_score_full():
    rec = build_mock_episode_loop(ep_id=0, n_steps=10)
    assert episode_length_fraction_score(rec, expected_steps=10) == 1.0


def test_length_fraction_score_partial():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True,
                                          safety_breach_at_step=4)
    # Halts at step 4 → 5 actual steps; expected 30 → 5/30 ≈ 0.167
    s = episode_length_fraction_score(rec, expected_steps=30)
    assert abs(s - 5/30) < 0.01


def test_length_fraction_score_capped():
    rec = build_mock_episode_loop(ep_id=0, n_steps=10)
    # Expected 5, actual 10 → 2.0, capped to 1.0
    assert episode_length_fraction_score(rec, expected_steps=5) == 1.0


def test_length_fraction_invalid_expected_raises():
    rec = build_mock_episode_loop(ep_id=0, n_steps=5)
    with pytest.raises(ValueError, match="expected_steps"):
        episode_length_fraction_score(rec, expected_steps=0)


# ---------- combine_scores ----------
def test_combine_scores_product_safety_gate():
    """A halted episode with big displacement should still score 0
    under product combination with safety."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True,
                                          safety_breach_at_step=15)
    combo = combine_scores(cube_xy_displacement_score, no_safety_events_score,
                              mode="product")
    s = combo(rec)
    assert s == 0.0   # safety_events = halt → multiplier 0


def test_combine_scores_product_clean_passes_through():
    """A clean episode (no safety events, full sweep) should score 1.0."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True)
    combo = combine_scores(cube_xy_displacement_score, no_safety_events_score,
                              mode="product")
    assert combo(rec) == 1.0   # both terms = 1.0


def test_combine_scores_weighted_mean():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True)
    # displacement=1.0, safety=1.0 → weighted mean = 1.0
    combo = combine_scores(
        cube_xy_displacement_score, no_safety_events_score,
        mode="weighted_mean", weights=[2.0, 1.0],
    )
    assert combo(rec) == 1.0


def test_combine_scores_weighted_mean_requires_weights():
    with pytest.raises(ValueError, match="weights required"):
        combine_scores(cube_xy_displacement_score, mode="weighted_mean")


def test_combine_scores_weighted_mean_weights_must_match():
    with pytest.raises(ValueError, match="must match"):
        combine_scores(
            cube_xy_displacement_score, no_safety_events_score,
            mode="weighted_mean", weights=[1.0],
        )


def test_combine_scores_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        combine_scores(cube_xy_displacement_score, mode="bogus")


def test_combine_scores_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        combine_scores()


# ---------- compute_outcome ----------
def test_compute_outcome_above_threshold():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=False)
    out = compute_outcome(rec, score_fn=cube_xy_displacement_score,
                              metric_name="cube_disp", success_threshold=0.5)
    assert out["success"] is True
    assert out["improvement"] == 1.0
    assert out["metric_name"] == "cube_disp"


def test_compute_outcome_below_threshold():
    rec = build_mock_episode_loop(ep_id=0, n_steps=30, include_safety=True,
                                          safety_breach_at_step=2)
    combo = combine_scores(cube_xy_displacement_score, no_safety_events_score,
                              mode="product")
    out = compute_outcome(rec, score_fn=combo, metric_name="combo")
    assert out["success"] is False
    assert out["improvement"] == 0.0


def test_compute_outcome_invalid_threshold():
    rec = build_mock_episode_loop(ep_id=0, n_steps=5)
    with pytest.raises(ValueError, match="success_threshold"):
        compute_outcome(rec, score_fn=cube_xy_displacement_score,
                            metric_name="x", success_threshold=1.5)


# ---------- episode_to_demo ----------
def test_episode_to_demo_uses_scorer():
    rec = build_mock_episode_loop(ep_id=0, n_steps=20, include_safety=False)
    demo = episode_to_demo(
        rec, demo_id=42,
        score_fn=cube_xy_displacement_score,
        collector_notes="from teleop ep0",
    )
    assert demo.demo_id == 42
    assert demo.task == rec.task
    assert demo.actions.shape == rec.gantry_actions.shape
    np.testing.assert_array_equal(demo.actions, rec.gantry_actions)
    assert demo.achieved_outcome == 1.0   # full sweep
    assert "0" in demo.initial_state["fiducials"]
    assert demo.collector_notes == "from teleop ep0"


def test_episode_to_demo_falls_back_to_ep_outcome():
    rec = build_mock_episode_loop(ep_id=0, n_steps=10, include_safety=False)
    # No score_fn → use ep.outcome["improvement"] (= 1.0 from happy path)
    demo = episode_to_demo(rec, demo_id=0)
    assert demo.achieved_outcome == rec.outcome["improvement"]


def test_episode_to_demo_empty_raises():
    rec = EpisodeRecord(
        ep_id=0, timestamp="t", task="x",
        router_decision={}, outcome={}, retrieved_demo={},
        frames=np.zeros((0, 32, 32, 3), dtype=np.uint8),
        slot_states=np.zeros((0, 6, 128), dtype=np.float32),
        decoded_positions=np.zeros((0, 6, 2), dtype=np.float32),
        gantry_actions=np.zeros((0, 7), dtype=np.float32),
        gantry_states=np.zeros((0, 9), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="empty episode"):
        episode_to_demo(rec, demo_id=0)


def test_episode_to_demo_clips_score_to_unit():
    """Even if a buggy scorer returns >1.0, achieved_outcome is clamped."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=10)
    demo = episode_to_demo(rec, demo_id=0,
                                  score_fn=lambda ep: 5.0)
    assert demo.achieved_outcome == 1.0


def test_episode_to_demo_round_trip_into_bank(tmp_path):
    """End-to-end: teleop ep → score → DemoRecord → DemoBank.add()."""
    rec = build_mock_episode_loop(ep_id=0, n_steps=20, include_safety=False)
    bank = DemoBank.from_directory(tmp_path)
    demo = episode_to_demo(rec, demo_id=bank.next_demo_id(),
                                  score_fn=cube_xy_displacement_score)
    path = bank.add(demo)
    assert path.exists()
    # Reload from disk
    bank2 = DemoBank.from_directory(tmp_path)
    assert len(bank2) == 1
    assert bank2.records[0].achieved_outcome == 1.0
