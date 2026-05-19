"""Unit tests for the DemoRetriever (Phase DR1).

Verifies NN retrieval semantics + the propose() proposal modes,
independent of robosuite. Pure-logic tests; no env dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from bla.recipes import DemoState, DemoRetriever


def _make_demo(demo_id: int, key: list[float],
                  T: int = 10, action_dim: int = 7,
                  outcome_score: float = 0.0) -> DemoState:
    """Build a DemoState whose action_seq is a constant vector marked
    with demo_id in its first action dim, so we can verify which demo
    the retriever picked."""
    actions = np.zeros((T, action_dim), dtype=np.float32)
    actions[:, 0] = float(demo_id)
    return DemoState(
        key=np.asarray(key, dtype=np.float32),
        action_seq=actions, demo_id=demo_id,
        outcome_score=outcome_score,
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


def test_retrieve_rerank_by_outcome_picks_highest_outcome_in_topk():
    """Top-k by distance, then highest outcome_score wins (DR2)."""
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0], outcome_score=0.05),
        _make_demo(1, [0.1, 0.0], outcome_score=0.20),   # better outcome
        _make_demo(2, [0.2, 0.0], outcome_score=0.15),
        _make_demo(7, [10.0, 0.0], outcome_score=0.99),  # far away — excluded
    ])
    chosen = r.retrieve_rerank_by_outcome(np.array([0.0, 0.0]), k=3)
    # Top-3 by distance are demos 0,1,2; among those, demo 1 has highest outcome
    assert chosen.demo_id == 1


def test_retrieve_rerank_by_outcome_ignores_far_high_outcome_demo():
    """Reranking must NOT pull in demos outside the top-k distance set."""
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0], outcome_score=0.10),
        _make_demo(1, [0.05, 0.0], outcome_score=0.15),
        _make_demo(7, [10.0, 0.0], outcome_score=1.0),  # high but far
    ])
    chosen = r.retrieve_rerank_by_outcome(np.array([0.0, 0.0]), k=2)
    # Only demos 0 and 1 are in top-2; demo 7 excluded despite high outcome
    assert chosen.demo_id == 1


def test_outcome_score_defaults_to_zero():
    """Demos without explicit outcome_score should default to 0.0,
    so reranking falls back to top-1 by distance order."""
    r = DemoRetriever()
    d = _make_demo(0, [0.0, 0.0])
    assert d.outcome_score == 0.0


# ---------- DR3: constrained rerank ----------

def test_constrained_rerank_degenerates_to_top1_on_exact_match():
    """When query exactly matches a bank entry (d_min = 0), only that
    demo passes the 1.25× filter; constrained rerank = top-1."""
    r = DemoRetriever()
    r.build_index([
        _make_demo(0, [0.0, 0.0], outcome_score=0.05),
        _make_demo(1, [0.1, 0.0], outcome_score=0.99),   # higher outcome but farther
        _make_demo(2, [0.2, 0.0], outcome_score=0.20),
    ])
    chosen = r.retrieve_constrained_rerank(
        np.array([0.0, 0.0]), k=3, filter_ratio=1.25)
    assert chosen.demo_id == 0  # exact match wins; demo 1's high outcome can't override


def test_constrained_rerank_within_filter_picks_highest_outcome():
    """When NN distance is positive and multiple candidates fall
    within 1.25× of it, outcome breaks the tie."""
    r = DemoRetriever()
    # Query at [0.90, 0]; bank has 3 demos within tight band around 1.00
    # plus a far demo with very high outcome that must NOT win.
    r.build_index([
        _make_demo(3, [1.00, 0.0], outcome_score=0.10),  # d=0.10 (NN)
        _make_demo(5, [1.01, 0.0], outcome_score=0.50),  # d=0.11 (within 1.25× band)
        _make_demo(7, [1.05, 0.0], outcome_score=0.30),  # d=0.15 (just outside 1.25× band)
        _make_demo(99, [10.0, 0.0], outcome_score=0.99), # d=9.10 (far)
    ])
    chosen = r.retrieve_constrained_rerank(
        np.array([0.90, 0.0]), k=4, filter_ratio=1.25)
    # NN distance = 0.10 → threshold = 0.125
    # Filter: demos with d ≤ 0.125 → demos 3 (d=0.10) and 5 (d=0.11)
    # Among those, demo 5 has higher outcome (0.50 > 0.10) → demo 5 wins
    assert chosen.demo_id == 5
    # Demo 99 (very high outcome) must not be picked
    assert chosen.demo_id != 99


def test_constrained_rerank_filter_excludes_demos_outside_band():
    """A demo just outside the 1.25× threshold must not win even
    with the highest outcome."""
    r = DemoRetriever()
    # NN at d=1.00; second demo at d=1.30 (just over 1.25× threshold)
    r.build_index([
        _make_demo(0, [1.00, 0.0], outcome_score=0.10),
        _make_demo(1, [1.30, 0.0], outcome_score=0.99),  # high outcome but excluded
    ])
    chosen = r.retrieve_constrained_rerank(
        np.array([0.0, 0.0]), k=2, filter_ratio=1.25)
    # threshold = 1.0 × 1.25 = 1.25; demo 1 at d=1.30 fails the filter
    # Only demo 0 passes → it wins
    assert chosen.demo_id == 0


def test_constrained_rerank_caps_at_k():
    """Filter operates over top-k, not the whole bank."""
    r = DemoRetriever()
    # 10 demos at distances 0, 0.1, 0.2, ..., 0.9
    # With NN=0, threshold=eps, only demo at 0 passes regardless of k
    demos = [_make_demo(i, [0.1 * i, 0.0],
                            outcome_score=0.05 * (10 - i))
                 for i in range(10)]
    r.build_index(demos)
    chosen = r.retrieve_constrained_rerank(
        np.array([0.0, 0.0]), k=3, filter_ratio=1.25)
    assert chosen.demo_id == 0  # exact match → only demo 0 in filter
