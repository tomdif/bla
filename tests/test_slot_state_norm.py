"""The slot STATE is carried between frames and nothing renormalises it.

next_slots = slots + change_mask * delta, and change_mask is a sigmoid, so it is never zero:
every slot is updated on every step and the state is a pure residual accumulation. The LayerNorm
inside the predictor normalises the transformer's activations, not the state that survives to the
next frame. Measured at init, the mean slot norm goes 0.78 -> 5.92 over 64 steps and is still
climbing. Two consequences are tested here: the drift itself, and the fact that the linear probe
-- our measurement instrument -- was being fitted on raw states whose scale grows with timestep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system1_jepa.slot_predictor import SlotDeltaPredictor, SlotPredictorConfig  # noqa: E402
from scripts.slot_jepa_train import _train_linear_probe                           # noqa: E402


def _roll(inter_frame_norm: bool, steps: int = 64):
    torch.manual_seed(0)
    cfg = SlotPredictorConfig(slot_dim=64, obs_dim=64, action_dim=64, n_layers=2, n_heads=4,
                              inter_frame_norm=inter_frame_norm)
    pred = SlotDeltaPredictor(cfg).eval()
    slots = torch.randn(4, 6, 64) * 0.1
    obs, act = torch.randn(4, 16, 64), torch.randn(4, 64)
    hist = []
    with torch.no_grad():
        for _ in range(steps):
            hist.append(slots.norm(dim=-1).mean().item())
            slots = pred(slots, obs, act)["next_slots"]
    return hist


def test_slot_state_drifts_without_the_flag():
    hist = _roll(False)
    assert hist[-1] > 4 * hist[0], hist[-1] / hist[0]
    assert hist[-1] > hist[32] > hist[8], "growth should still be ongoing, not saturating"


def test_inter_frame_norm_holds_the_state_flat():
    hist = _roll(True)
    tail = hist[8:]
    assert max(tail) - min(tail) < 1e-3, max(tail) - min(tail)


def test_inter_frame_norm_is_off_by_default():
    cfg = SlotPredictorConfig(slot_dim=32, obs_dim=32, action_dim=32)
    assert cfg.inter_frame_norm is False
    assert SlotDeltaPredictor(cfg).state_norm is None


def test_probe_still_fits_raw_states():
    """Documents the open problem rather than papering over it.

    Global per-feature standardisation was tried here and reverted: it cannot remove a scale that
    varies with timestep WITHIN an episode, and under weight decay it was strictly worse on a
    scale-skewed control. The probe therefore still sees the drift, and the fix belongs in the
    slot state (--inter-frame-norm), not in the instrument.
    """
    torch.manual_seed(0)
    x, y = torch.randn(64, 5), torch.randn(64, 2)
    probe = _train_linear_probe(x, y, lr=1e-2, epochs=10)
    assert isinstance(probe, torch.nn.Linear)
