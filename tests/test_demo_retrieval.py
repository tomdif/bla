"""Unit tests for the DemoRetriever (Phase DR1).

Verifies NN retrieval semantics + the propose() proposal modes,
independent of robosuite. Pure-logic tests; no env dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from bla.recipes import DemoState, DemoRetriever


def _make_demo(demo_id: int, key: list[float],
                  T: int = 10, action_dim: int = 7) -> DemoState:
    """Build a DemoState whose action_seq is a constant vector marked
    with demo_id in its first action dim, so we can verify which demo
    the retriever picked."""
    actions = np.zeros((T, action_dim), dtype=np.float32)
    actions[:, 0] = float(demo_id)
    return DemoState(
        key=np.asarray(key, dtype=np.float32),
        action_seq=actions, demo_id=demo_id,
    )


def test_build_index_rejects_empty():
    r = DemoRetriever()
    with pytest.raises(ValueError, match="Empty demo bank"):
        r.build_index([])


def test_retrieve_top1_picks_closest_demo():
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0]),
        _make_demo(1, [1.0, 1.0]),
        _make_demo(2, [-2.0, -2.0]),
    ])
    # Query very near demo 0
    top = r.retrieve(np.array([0.05, -0.02]), k=1)
    assert len(top) == 1
    assert top[0].demo_id == 0
    # Query very near demo 1
    top = r.retrieve(np.array([0.9, 0.95]), k=1)
    assert top[0].demo_id == 1
    # Query far on the negative side
    top = r.retrieve(np.array([-1.8, -2.1]), k=1)
    assert top[0].demo_id == 2


def test_retrieve_topk_ordering():
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0]),
        _make_demo(1, [1.0, 0.0]),
        _make_demo(2, [3.0, 0.0]),
        _make_demo(3, [10.0, 0.0]),
    ])
    top = r.retrieve(np.array([0.0, 0.0]), k=3)
    ids = [d.demo_id for d in top]
    assert ids == [0, 1, 2]  # ordered by ascending distance


def test_propose_top1_returns_top_demo_actions():
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0], T=15),
        _make_demo(7, [5.0, 5.0], T=15),
    ])
    actions = r.propose(np.array([0.01, 0.01]), reduce="top1")
    assert actions.shape == (15, 7)
    # Demo 0's actions have first-dim = 0; demo 7's would be 7
    assert np.isclose(actions[0, 0], 0.0)


def test_propose_topk_avg_averages_action_sequences():
    r = DemoRetriever()
    r.build_index([
        _make_demo(2, [0.0, 0.0], T=8),
        _make_demo(4, [0.1, 0.0], T=8),
        _make_demo(6, [0.2, 0.0], T=8),
        _make_demo(99, [10.0, 10.0], T=8),
    ])
    actions = r.propose(np.array([0.0, 0.0]), k=3, reduce="topk_avg")
    # Top-3 are demos 2, 4, 6 → mean first-action-dim = (2+4+6)/3 = 4.0
    assert np.isclose(actions[0, 0], 4.0)
    # The far-away demo 99 should NOT be in the top-3
    assert actions.shape == (8, 7)


def test_propose_horizon_truncation():
    r = DemoRetriever()
    r.build_index([_make_demo(0, [0.0, 0.0], T=20)])
    actions = r.propose(np.array([0.0, 0.0]), H=5)
    assert actions.shape == (5, 7)


def test_propose_horizon_padding_repeats_last_action():
    r = DemoRetriever()
    r.build_index([_make_demo(0, [0.0, 0.0], T=3)])
    actions = r.propose(np.array([0.0, 0.0]), H=8)
    assert actions.shape == (8, 7)
    # All entries past T=3 should be a repeat of action[2]
    last = actions[2]
    for t in range(3, 8):
        assert np.allclose(actions[t], last)


def test_propose_rejects_unknown_reduce():
    r = DemoRetriever()
    r.build_index([_make_demo(0, [0.0, 0.0])])
    with pytest.raises(ValueError, match="unknown reduce"):
        r.propose(np.array([0.0, 0.0]), reduce="not_a_mode")


def test_retrieve_rejects_wrong_query_dim():
    r = DemoRetriever()
    r.build_index([_make_demo(0, [0.0, 0.0])])
    with pytest.raises(ValueError, match="query_key dim"):
        r.retrieve(np.array([0.0, 0.0, 0.0]))


def test_retrieve_before_build_index_raises():
    r = DemoRetriever()
    with pytest.raises(RuntimeError, match="build_index has not been called"):
        r.retrieve(np.array([0.0, 0.0]))


def test_k_larger_than_bank_caps_at_bank_size():
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0]),
        _make_demo(1, [1.0, 0.0]),
    ])
    top = r.retrieve(np.array([0.0, 0.0]), k=10)
    assert len(top) == 2  # capped at bank size
