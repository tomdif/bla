"""Smoke test for the OF-JEPA System-1 substrate API.

Verifies the four-method API + the latent-bus bridge:
  observe(frame) → ObjectFileBatch
  predict(k=0)  → ObjectFileBatch
  read("all")    → dict
  metrics()      → dict
  per_file_project(ofb, bus) → [B, N_files, d_core]

Uses small dimensions + random-init weights so it runs in <5s on CPU.
Tests data flow + shape correctness, NOT model quality (no training).
"""
from __future__ import annotations

import torch
import pytest

from latent_bus.bus import TokenlessLatentBus
from system1_jepa.of_jepa_api import (
    ObjectFileBatch, OFJEPAObjectFiles, per_file_project,
)


@pytest.fixture(scope="module")
def small_substrate():
    torch.manual_seed(0)
    return OFJEPAObjectFiles(
        image_size=64, n_files=8, slot_dim=32, version="v0", device="cpu",
    )


def test_episode_lifecycle_observe_then_read(small_substrate):
    """observe(frame) updates memory; read() reflects the update."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    frame = torch.randn(3, 64, 64)

    # Pre-observe: read returns init state.
    init = s.read("all")
    assert init["frame_idx"] == -1
    assert init["n_files"] == 8

    # First observation.
    ofb = s.observe(frame)
    assert isinstance(ofb, ObjectFileBatch)
    assert ofb.id_keys.shape == (1, 8, 16)
    assert ofb.state_values.shape == (1, 8, 16)
    assert ofb.confidences.shape == (1, 8)
    assert ofb.frame_idx == 0

    # read() should reflect the observation.
    state = s.read("all")
    assert state["frame_idx"] == 0
    assert state["id_keys"].shape == (1, 8, 16)


def test_observe_advances_frame_idx(small_substrate):
    """Multiple observations increment the frame index."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    for t in range(4):
        ofb = s.observe(torch.randn(3, 64, 64))
        assert ofb.frame_idx == t


def test_predict_k0_returns_current_state(small_substrate):
    """predict(k=0) returns most recent observation; confidence is 1."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    s.observe(torch.randn(3, 64, 64))
    pred = s.predict(k_steps=0)
    state = s.read("all")
    assert torch.allclose(pred.id_keys, state["id_keys"].to(pred.id_keys.device))
    assert torch.all(pred.confidences == 1.0)


def test_predict_k_positive_raises_on_v0():
    """v0 has no transition_model; predict(k>0) is unsupported."""
    s = OFJEPAObjectFiles(
        image_size=64, n_files=8, slot_dim=32, version="v0", device="cpu",
    )
    s.reset_episode(batch_size=1)
    s.observe(torch.randn(3, 64, 64))
    with pytest.raises(NotImplementedError, match="v1 transition_model"):
        s.predict(k_steps=3)


def test_predict_k_positive_works_on_v1():
    """v1 has the transition_model; predict(k>0) should work."""
    s = OFJEPAObjectFiles(
        image_size=64, n_files=8, slot_dim=32, version="v1", device="cpu",
    )
    s.reset_episode(batch_size=1)
    s.observe(torch.randn(3, 64, 64))
    pred = s.predict(k_steps=3)
    assert pred.frame_idx == 3  # 0 (after first observe) + 3
    assert torch.all(pred.confidences == 0.0)  # forecasted, not observed


def test_read_queries(small_substrate):
    """read() supports multiple query types and returns clean dicts."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    s.observe(torch.randn(3, 64, 64))

    assert "id_keys" in s.read("id_keys")
    assert "state_values" in s.read("state_values")
    assert "active_mask" in s.read("active")
    full = s.read("all")
    assert {"id_keys", "state_values", "frame_idx", "n_files"} <= set(full.keys())

    with pytest.raises(ValueError):
        s.read("bogus_query")


def test_metrics_cache_roundtrip(small_substrate):
    """External evaluator can deposit metrics via cache_metrics, reader sees them."""
    s = small_substrate
    s.cache_metrics({"id_h/v": 1.51, "switch_rate": 0.076})
    m = s.metrics()
    assert m["id_h/v"] == 1.51
    assert m["switch_rate"] == 0.076


def test_full_slot_view(small_substrate):
    """ObjectFileBatch.full_slot concatenates id + state for legacy consumers."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    ofb = s.observe(torch.randn(3, 64, 64))
    full = ofb.full_slot
    assert full.shape == (1, 8, 32)  # 16 + 16
    assert torch.allclose(full[:, :, :16], ofb.id_keys)
    assert torch.allclose(full[:, :, 16:], ofb.state_values)


def test_per_file_projection_through_latent_bus():
    """The bridge adapter sends EACH file through the bus independently.

    System-2 receives [B, N_files, d_core] — a sequence of tokens, not a
    single pooled vector. This is the architectural commitment: object-file
    structure flows through to System-2 instead of being averaged.
    """
    torch.manual_seed(0)
    s = OFJEPAObjectFiles(
        image_size=64, n_files=8, slot_dim=32, version="v0", device="cpu",
    )
    bus = TokenlessLatentBus(d_jepa=32, d_core=128, dtype=torch.float32)

    s.reset_episode(batch_size=1)
    ofb = s.observe(torch.randn(3, 64, 64))

    projected = per_file_project(ofb, bus)

    assert projected.shape == (1, 8, 128), \
        f"Expected [B=1, N_files=8, d_core=128], got {tuple(projected.shape)}"

    # Each file should produce a different projection (assuming non-degenerate
    # id_keys after the first frame).
    file_vars = projected[0].var(dim=0)
    assert (file_vars > 0).all(), "Per-file projections collapsed"


def test_reset_clears_memory(small_substrate):
    """reset_episode() forgets prior bindings — id_key returns to id_proto."""
    s = small_substrate
    s.reset_episode(batch_size=1)
    s.observe(torch.randn(3, 64, 64))
    state_after_obs = s._memory_state["id_key"].clone()

    s.reset_episode(batch_size=1)
    state_after_reset = s._memory_state["id_key"]

    # After reset, id_key should equal the prototype again — not the post-obs state.
    assert not torch.allclose(state_after_obs, state_after_reset, atol=1e-6), \
        "Reset didn't change id_key"
