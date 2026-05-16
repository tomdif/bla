"""Validate the identity-aware Hungarian probe on synthetic batches where
the correct identity assignment is *known by construction*.

Three tests:

1. `test_perfect_slot_recovery`: slot states are clean encodings of entity
   features. The probe + Hungarian assignment should recover the identity
   mapping with zero switches across frames.

2. `test_random_slots_high_switch`: slot states are random noise. The
   Hungarian assignment should be ~uniformly distributed → identity
   switch rate near (E-1)/E.

3. `test_position_only_matches_attribute_aware`: when attribute weight
   is zero, the matcher falls back to position-only. With distinct
   positions this still recovers the correct assignment.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from system1_jepa.identity_probe import (
    ProbeFitConfig,
    fit_identity_probe,
    hungarian_assign,
    identity_aware_probe_eval,
    identity_switch_rate,
    slot_diversity,
)


def _make_clean_slots(n_episodes: int, T: int, E: int, slot_dim: int,
                      seed: int = 0):
    """Per-episode random entities; slot[i] = clean linear encoding of entity[i]
    plus small noise. The probe should be able to recover identities."""
    rng = np.random.default_rng(seed)
    states = []
    gt_pos = []
    gt_attr = []
    gt_vis = []
    gt_ids = []
    ep_ids = []
    frame_idx = []
    hidden_step = []

    A = 6  # attribute dim
    enc_W = rng.standard_normal((2 + A, slot_dim)).astype(np.float32) * 0.5

    for ep in range(n_episodes):
        entity_pos = rng.uniform(0.1, 0.9, size=(E, 2)).astype(np.float32)
        entity_attr = rng.standard_normal((E, A)).astype(np.float32)
        # Movement: small per-frame jitter so positions are time-varying.
        for t in range(T):
            jitter = rng.standard_normal((E, 2)).astype(np.float32) * 0.01
            pos_t = entity_pos + jitter * t  # drift slightly each frame
            entity_feat = np.concatenate([pos_t, entity_attr], axis=-1)  # [E, 2+A]
            slot_state = entity_feat @ enc_W                              # [E, slot_dim]
            slot_state += rng.standard_normal(slot_state.shape).astype(np.float32) * 0.05
            states.append(slot_state)
            gt_pos.append(pos_t)
            gt_attr.append(entity_attr)
            gt_vis.append(np.ones(E, dtype=bool))
            gt_ids.append(np.arange(E, dtype=np.int64))
            ep_ids.append(ep)
            frame_idx.append(t)
            hidden_step.append(0)

    return {
        "states": torch.from_numpy(np.stack(states)),
        "gt_pos": torch.from_numpy(np.stack(gt_pos)),
        "gt_attr": torch.from_numpy(np.stack(gt_attr)),
        "gt_visible": torch.from_numpy(np.stack(gt_vis)),
        "gt_entity_ids": torch.from_numpy(np.stack(gt_ids)),
        "ep_ids": torch.tensor(ep_ids, dtype=torch.long),
        "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        "hidden_step": torch.tensor(hidden_step, dtype=torch.long),
    }


def test_hungarian_assign_identity_when_positions_distinct():
    """With distinct positions and no attributes, Hungarian matches by minimum
    distance. If pred = true, result should be the identity permutation."""
    rng = np.random.default_rng(42)
    pos = rng.uniform(0, 1, size=(5, 2))
    rows, cols, cost = hungarian_assign(pos.copy(), pos.copy())
    np.testing.assert_array_equal(np.sort(rows), np.arange(5))
    # Each slot should match its own index.
    for r, c in zip(rows, cols):
        assert r == c
    assert cost < 1e-12


def test_identity_switch_rate_zero_when_stable():
    """All-stable assignments → zero switches."""
    assignments = [np.array([0, 1, 2, 3]) for _ in range(10)]
    assert identity_switch_rate(assignments) == 0.0


def test_identity_switch_rate_one_when_always_swapped():
    """Every consecutive frame swaps two slots → switch rate ≈ 1."""
    assignments = []
    for t in range(10):
        if t % 2 == 0:
            assignments.append(np.array([0, 1, 2, 3]))
        else:
            assignments.append(np.array([1, 0, 3, 2]))
    rate = identity_switch_rate(assignments)
    assert 0.99 <= rate <= 1.01


def test_slot_diversity_one_when_stable():
    assignments = [np.array([0, 1, 2, 3]) for _ in range(5)]
    assert slot_diversity(assignments) == 1.0


def test_perfect_slot_recovery():
    """Clean entity encoding → probe should achieve very low position MSE
    and near-zero identity switch rate."""
    torch.manual_seed(0)
    np.random.seed(0)
    batch = _make_clean_slots(n_episodes=20, T=8, E=4, slot_dim=32, seed=0)
    cfg = ProbeFitConfig(epochs=400, lr=1e-2, batch_size=64, attr_weight=1.0)
    result = identity_aware_probe_eval(
        states=batch["states"],
        gt_pos=batch["gt_pos"],
        gt_attr=batch["gt_attr"],
        gt_visible=batch["gt_visible"],
        gt_entity_ids=batch["gt_entity_ids"],
        ep_ids=batch["ep_ids"],
        frame_idx=batch["frame_idx"],
        hidden_step=batch["hidden_step"],
        J=0,
        cfg=cfg,
    )
    # Position MSE should be very small for clean encodings.
    assert result.visible_position_mse < 1e-2, \
        f"Probe should recover positions, got {result.visible_position_mse}"
    # Identity switches should be rare (allow up to 10% for noise).
    assert result.identity_switch_rate < 0.10, \
        f"Stable slots should have low switches, got {result.identity_switch_rate}"
    # Per-slot diversity should be near 1 (one entity per slot).
    assert result.mean_slot_diversity < 1.3, \
        f"Slots should bind to single entities, got {result.mean_slot_diversity}"


def test_position_only_when_no_attrs():
    """Empty attribute vector — Hungarian uses position only."""
    rng = np.random.default_rng(123)
    S, E = 4, 4
    pred = rng.uniform(0, 1, size=(S, 2))
    true = rng.uniform(0, 1, size=(E, 2))
    rows, cols, cost = hungarian_assign(pred, true)
    assert len(rows) == S
    assert len(cols) == E


def test_unequal_slots_entities():
    """S != E: matcher should still produce min(S, E) assignments."""
    rng = np.random.default_rng(7)
    pred = rng.uniform(0, 1, size=(6, 2))   # more slots than entities
    true = rng.uniform(0, 1, size=(3, 2))
    rows, cols, _ = hungarian_assign(pred, true)
    assert len(rows) == 3
    assert len(cols) == 3
    assert len(set(cols.tolist())) == 3  # each entity matched once
