"""Unit tests for bla.forge.rolling_tracker (BF-0.3).

Validates the production-shape rolling K=5 wrapper using the mock_static
backend. The ofjepa backend just raises NotImplementedError until the
real Phase-14 encoder is wired in (tested below).
"""
from __future__ import annotations

import numpy as np
import pytest

from bla.forge import (
    FiducialDetection,
    ObservationStep,
    RollingObjectFileTracker,
    mock_calibration,
    mock_fiducials,
)


# ---------- helpers ----------
def _dummy_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """A throw-away BGR frame for tests that don't care about pixels."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------- constructor validation ----------
def test_constructor_rejects_invalid_K():
    bundle = mock_calibration()
    with pytest.raises(ValueError, match="K must be"):
        RollingObjectFileTracker(bundle, K=0)


def test_constructor_rejects_invalid_n_slots():
    bundle = mock_calibration()
    with pytest.raises(ValueError, match="n_slots must be"):
        RollingObjectFileTracker(bundle, n_slots=0)


def test_constructor_rejects_invalid_slot_dim():
    bundle = mock_calibration()
    with pytest.raises(ValueError, match="slot_dim must be"):
        RollingObjectFileTracker(bundle, slot_dim=0)


def test_constructor_rejects_unknown_backend():
    bundle = mock_calibration()
    with pytest.raises(ValueError, match="unknown backend"):
        RollingObjectFileTracker(bundle, backend="bogus")


def test_constructor_rejects_unknown_missing_policy():
    bundle = mock_calibration()
    with pytest.raises(ValueError, match="unknown missing_policy"):
        RollingObjectFileTracker(bundle, missing_policy="bogus")


# ---------- frame shape validation ----------
def test_step_rejects_wrong_frame_shape():
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    with pytest.raises(ValueError, match="frame must be"):
        tracker.step(np.zeros((10,), dtype=np.uint8))


def test_step_accepts_grayscale_or_bgr():
    """Both H×W and H×W×3 frame shapes are valid."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    tracker.step(np.zeros((64, 64), dtype=np.uint8))    # grayscale
    tracker.step(np.zeros((64, 64, 3), dtype=np.uint8))  # BGR
    assert tracker.buffer_len == 2


# ---------- output shape ----------
def test_observation_step_shapes():
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, K=5, n_slots=6, slot_dim=128)
    fids = mock_fiducials(bundle=bundle)
    step = tracker.step(_dummy_frame(), fids)
    assert isinstance(step, ObservationStep)
    assert step.slot_states.shape == (6, 128)
    assert step.decoded_positions_world.shape == (6, 2)
    assert step.confidence.shape == (6,)
    assert step.timestep == 0
    assert step.buffer_len == 1


# ---------- rolling buffer semantics ----------
def test_rolling_buffer_caps_at_K():
    """After K+ steps, the buffer should hold exactly K frames."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, K=5)
    for _ in range(10):
        tracker.step(_dummy_frame())
    assert tracker.buffer_len == 5
    assert tracker.step_count == 10


def test_buffer_grows_until_K():
    """Before K steps, buffer grows by 1 each call."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, K=5)
    for i in range(1, 6):
        tracker.step(_dummy_frame())
        assert tracker.buffer_len == i


# ---------- identity binding (the Phase 8C invariant) ----------
def test_identity_binding_is_persistent_across_steps():
    """The same fiducial ID gets the SAME slot across all calls."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    # Step 1: see IDs 0, 1, 2, 3
    fids = mock_fiducials(bundle=bundle)
    s1 = tracker.step(_dummy_frame(), fids)
    bindings_1 = tracker.identity_bindings
    # Step 2: see IDs 0, 1, 2, 3 again (same)
    s2 = tracker.step(_dummy_frame(), fids)
    bindings_2 = tracker.identity_bindings
    # Same id-to-slot map
    assert bindings_1 == bindings_2
    # Each step's slot_to_object_id agrees with the bindings
    for slot_idx, fid_id in s1.slot_to_object_id.items():
        assert bindings_1[fid_id] == slot_idx
        assert s2.slot_to_object_id[slot_idx] == fid_id


def test_identity_persists_through_one_occluded_frame():
    """If a tag is briefly missing then visible again, its slot is preserved."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    fids_full = mock_fiducials(bundle=bundle)
    fids_missing_id_2 = [f for f in fids_full if f.id != 2]
    s1 = tracker.step(_dummy_frame(), fids_full)
    slot_of_id_2_step1 = tracker.identity_bindings[2]
    s2 = tracker.step(_dummy_frame(), fids_missing_id_2)
    # ID 2's slot is still mapped, just not bound this frame
    assert tracker.identity_bindings[2] == slot_of_id_2_step1
    assert slot_of_id_2_step1 not in s2.slot_to_object_id
    # Re-emerge:
    s3 = tracker.step(_dummy_frame(), fids_full)
    assert s3.slot_to_object_id[slot_of_id_2_step1] == 2


def test_new_fiducial_gets_next_free_slot():
    """A new ID at step T should claim the next unused slot index."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, n_slots=6)
    # First step: see IDs 0, 1
    fids1 = mock_fiducials(ids=(0, 1), bundle=bundle)
    tracker.step(_dummy_frame(), fids1)
    # Next step: see a new ID 99
    fids2 = mock_fiducials(ids=(0, 1, 99), bundle=bundle,
                                 world_xy_per_id={0: (-0.1, -0.1),
                                                       1: (+0.1, -0.1),
                                                       99: (0.0, +0.1)})
    tracker.step(_dummy_frame(), fids2)
    # ID 99's slot must be 0-or-higher and NOT collide with 0 or 1's slots
    bindings = tracker.identity_bindings
    assert bindings[0] != bindings[99]
    assert bindings[1] != bindings[99]


def test_excess_fiducials_dropped_silently():
    """When more unique IDs than n_slots, late arrivers are dropped."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, n_slots=2)
    fids = mock_fiducials(ids=(0, 1, 2, 3), bundle=bundle)
    step = tracker.step(_dummy_frame(), fids)
    # Only 2 of the 4 fiducials can be bound
    bindings = tracker.identity_bindings
    assert len(bindings) == 2
    # The 2 lowest IDs (FCFS) should win
    assert 0 in bindings and 1 in bindings
    assert 2 not in bindings and 3 not in bindings
    # Confidence should be 0 for unbound slots (there are none here — n_slots=2)
    # but slot_to_object_id should have exactly 2 entries
    assert len(step.slot_to_object_id) == 2


# ---------- decoded positions ----------
def test_decoded_positions_match_fiducial_world_xy():
    """Decoded positions should round-trip through BF-0.1 projection."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    layout = {0: (-0.10, -0.10), 1: (+0.10, +0.05)}
    fids = mock_fiducials(ids=(0, 1), world_xy_per_id=layout, bundle=bundle)
    step = tracker.step(_dummy_frame(), fids)
    for slot_idx, fid_id in step.slot_to_object_id.items():
        np.testing.assert_allclose(
            step.decoded_positions_world[slot_idx],
            layout[fid_id], atol=1e-5,
        )


def test_unbound_slots_have_nan_positions_under_nan_policy():
    """missing_policy='nan' (default): unbound slots show NaN positions."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, n_slots=6,
                                                missing_policy="nan")
    fids = mock_fiducials(ids=(0, 1), bundle=bundle,
                                 world_xy_per_id={0: (-0.1, -0.1), 1: (+0.1, -0.1)})
    step = tracker.step(_dummy_frame(), fids)
    # 4 slots are unbound (n_slots=6, 2 fids)
    bound_slots = set(step.slot_to_object_id.keys())
    for s in range(6):
        if s not in bound_slots:
            assert np.all(np.isnan(step.decoded_positions_world[s]))


def test_carry_policy_keeps_last_known_position_when_fid_missing():
    """missing_policy='carry': if a slot's fid disappears, retain its last
    known world position (with confidence dropping to 0)."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, missing_policy="carry")
    # Step 1: ID 0 at (-0.1, -0.1)
    fids1 = mock_fiducials(ids=(0,), bundle=bundle,
                                  world_xy_per_id={0: (-0.1, -0.1)})
    s1 = tracker.step(_dummy_frame(), fids1)
    slot_0 = tracker.identity_bindings[0]
    np.testing.assert_allclose(s1.decoded_positions_world[slot_0],
                                      [-0.1, -0.1], atol=1e-5)
    # Step 2: nothing visible
    s2 = tracker.step(_dummy_frame(), [])
    # Slot 0's position is CARRIED (not NaN)
    np.testing.assert_allclose(s2.decoded_positions_world[slot_0],
                                      [-0.1, -0.1], atol=1e-5)
    # But confidence drops to 0 because no current observation
    assert s2.confidence[slot_0] == 0.0
    # And slot_to_object_id this step has no entry for slot_0
    assert slot_0 not in s2.slot_to_object_id


# ---------- slot state determinism ----------
def test_mock_slot_state_is_deterministic_per_id():
    """Same fiducial id + same world position → identical slot state.
    This is the mock realization of identity-as-address."""
    bundle = mock_calibration()
    tracker_a = RollingObjectFileTracker(bundle)
    tracker_b = RollingObjectFileTracker(bundle)
    fids = mock_fiducials(bundle=bundle)
    a = tracker_a.step(_dummy_frame(), fids)
    b = tracker_b.step(_dummy_frame(), fids)
    np.testing.assert_array_equal(a.slot_states, b.slot_states)


def test_slot_state_changes_with_position_same_id():
    """Same ID at different positions should have similar identity components
    but different position components → not identical, not radically different."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    fid_pos_a = mock_fiducials(ids=(7,), bundle=bundle,
                                       world_xy_per_id={7: (-0.10, 0.0)})
    fid_pos_b = mock_fiducials(ids=(7,), bundle=bundle,
                                       world_xy_per_id={7: (+0.10, 0.0)})
    a = tracker.step(_dummy_frame(), fid_pos_a)
    # New tracker because we want to compare at the SAME logical slot
    tracker2 = RollingObjectFileTracker(bundle)
    b = tracker2.step(_dummy_frame(), fid_pos_b)
    slot = tracker.identity_bindings[7]
    assert slot == tracker2.identity_bindings[7]
    # Identity half is identical; pose half differs
    half = a.slot_states.shape[1] // 2
    np.testing.assert_array_equal(
        a.slot_states[slot, :half], b.slot_states[slot, :half])
    # Pose halves should differ (we used different world x)
    assert not np.allclose(
        a.slot_states[slot, half:], b.slot_states[slot, half:])


def test_slot_states_differ_across_distinct_ids():
    """Different fiducial IDs should produce different slot embeddings."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    fids = mock_fiducials(ids=(0, 1, 2, 3), bundle=bundle)
    step = tracker.step(_dummy_frame(), fids)
    # Compare identity halves across slots that are bound
    half = step.slot_states.shape[1] // 2
    slot_ids = sorted(step.slot_to_object_id.items())
    for (s_a, _), (s_b, _) in zip(slot_ids, slot_ids[1:]):
        assert not np.allclose(
            step.slot_states[s_a, :half], step.slot_states[s_b, :half])


# ---------- reset ----------
def test_reset_clears_buffer_and_bindings():
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle)
    tracker.step(_dummy_frame(), mock_fiducials(bundle=bundle))
    assert tracker.buffer_len == 1
    assert tracker.identity_bindings
    tracker.reset()
    assert tracker.buffer_len == 0
    assert tracker.identity_bindings == {}
    assert tracker.step_count == 0


# ---------- ofjepa backend stub ----------
def test_ofjepa_backend_raises_not_implemented():
    """Real OF-JEPA backend is reserved for hardware-arrival wiring."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, backend="ofjepa")
    with pytest.raises(NotImplementedError, match="ofjepa"):
        tracker.step(_dummy_frame(), [])


# ---------- streaming sanity ----------
def test_streaming_30_steps_stable():
    """30 consecutive frames with the same fiducial layout — buffer
    stabilizes, identity bindings stay fixed, decoded positions stay
    correct.

    This is the closest analogue we can get to D1b's rolling-window
    behavior without an actual encoder."""
    bundle = mock_calibration()
    tracker = RollingObjectFileTracker(bundle, K=5)
    layout = {0: (-0.10, -0.10), 1: (+0.10, +0.05)}
    fids = mock_fiducials(ids=(0, 1), world_xy_per_id=layout, bundle=bundle)

    steps = [tracker.step(_dummy_frame(), fids) for _ in range(30)]
    assert tracker.buffer_len == 5
    # Identity bindings unchanged after the first step
    final_bindings = tracker.identity_bindings
    for step in steps[1:]:
        for slot_idx, fid_id in step.slot_to_object_id.items():
            assert final_bindings[fid_id] == slot_idx
    # Last-step positions decode to the layout
    last = steps[-1]
    for slot_idx, fid_id in last.slot_to_object_id.items():
        np.testing.assert_allclose(
            last.decoded_positions_world[slot_idx],
            layout[fid_id], atol=1e-5,
        )
