"""Tests for the Phase 7B id_dyn_split predictor mode.

Verifies:
1. Output shape matches input slot shape (no truncation/padding).
2. id half updates SLOWLY (EMA with small alpha) — large changes in
   transformer output produce small changes in id.
3. dyn half updates VIA SPARSE DELTA — when change_mask is forced to 0,
   dyn doesn't change.
4. id and dyn updates are *decoupled* in the right way: with the same
   transformer output, increasing id_ema_alpha increases id changes;
   the dyn half is unaffected.
"""
from __future__ import annotations

import torch
import pytest

from system1_jepa.slot_predictor import SlotDeltaPredictor, SlotPredictorConfig


def _make_predictor(slot_dim=32, id_dim=16, id_ema_alpha=0.05):
    return SlotDeltaPredictor(SlotPredictorConfig(
        slot_dim=slot_dim, obs_dim=slot_dim, action_dim=4,
        n_layers=1, n_heads=2, mlp_ratio=2,
        delta_scale=0.2, mask_bias_init=0.0,
        update_mode="id_dyn_split",
        id_dim=id_dim, id_ema_alpha=id_ema_alpha,
    ))


def test_id_dyn_split_shape():
    """next_slots has same shape as slots."""
    torch.manual_seed(0)
    pred = _make_predictor(slot_dim=32, id_dim=16)
    slots = torch.randn(2, 8, 32)
    obs = torch.randn(2, 12, 32)
    action = torch.zeros(2, 4)
    out = pred(slots, obs, action)
    assert out["next_slots"].shape == slots.shape


def test_id_half_updates_slowly():
    """With small EMA alpha, the id half barely changes per step."""
    torch.manual_seed(0)
    pred = _make_predictor(slot_dim=32, id_dim=16, id_ema_alpha=0.01)
    slots = torch.randn(2, 8, 32)
    obs = torch.randn(2, 12, 32)
    action = torch.zeros(2, 4)
    out = pred(slots, obs, action)
    id_change = (out["next_slots"][..., :16] - slots[..., :16]).abs().mean()
    # Even with large transformer outputs, id alpha=0.01 → max change ~0.01 * |output|.
    # Sanity: change should be small.
    assert id_change < 1.0, f"id change too large: {id_change}"


def test_id_alpha_scales_id_change():
    """Bigger id_ema_alpha → bigger id changes; dyn half is unaffected."""
    torch.manual_seed(42)
    slots = torch.randn(2, 8, 32)
    obs = torch.randn(2, 12, 32)
    action = torch.zeros(2, 4)

    pred_a = _make_predictor(id_ema_alpha=0.01)
    pred_a.load_state_dict(pred_a.state_dict())  # noop, anchors state

    pred_b = _make_predictor(id_ema_alpha=0.5)
    # Copy params so the two only differ in the EMA alpha hyperparameter.
    pred_b.load_state_dict(pred_a.state_dict())

    out_a = pred_a(slots, obs, action)
    out_b = pred_b(slots, obs, action)

    id_change_a = (out_a["next_slots"][..., :16] - slots[..., :16]).abs().mean().item()
    id_change_b = (out_b["next_slots"][..., :16] - slots[..., :16]).abs().mean().item()
    assert id_change_b > 5 * id_change_a, \
        f"alpha=0.5 should give >>5× the id change of alpha=0.01: a={id_change_a}, b={id_change_b}"

    # dyn half should be IDENTICAL across the two predictors.
    dyn_a = out_a["next_slots"][..., 16:]
    dyn_b = out_b["next_slots"][..., 16:]
    diff = (dyn_a - dyn_b).abs().max().item()
    assert diff < 1e-5, f"dyn half should be unaffected by id_ema_alpha; diff={diff}"


def test_dyn_half_respects_change_mask():
    """Force the change mask near zero — dyn half barely changes."""
    torch.manual_seed(7)
    pred = _make_predictor(slot_dim=32, id_dim=16, id_ema_alpha=0.0)
    # Drive the change-mask bias very negative so sigmoid ≈ 0.
    with torch.no_grad():
        pred.change_head.bias.fill_(-10.0)
    slots = torch.randn(2, 8, 32)
    obs = torch.randn(2, 12, 32)
    action = torch.zeros(2, 4)
    out = pred(slots, obs, action)
    dyn_change = (out["next_slots"][..., 16:] - slots[..., 16:]).abs().mean().item()
    assert dyn_change < 0.01, f"dyn should be near-frozen with mask≈0; got {dyn_change}"


def test_gradients_flow_through_both_halves():
    """Backprop reaches both id_delta_head and delta_head."""
    torch.manual_seed(3)
    pred = _make_predictor()
    slots = torch.randn(2, 8, 32, requires_grad=False)
    obs = torch.randn(2, 12, 32)
    action = torch.zeros(2, 4)
    out = pred(slots, obs, action)
    loss = out["next_slots"].pow(2).mean()
    loss.backward()
    assert pred.id_delta_head.weight.grad is not None
    assert pred.id_delta_head.weight.grad.abs().sum() > 0
    assert pred.delta_head.weight.grad is not None
    assert pred.delta_head.weight.grad.abs().sum() > 0
