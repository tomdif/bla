"""Slot existence head + visibility-disagreement surprise.

Replaces raw L2 prediction error as the surprise signal for CCT and
related "evidence novelty" gates. Raw L2 has the wrong sign at occlusion
boundaries: removing content from an observation makes it *easier* to
predict, not harder, so L2(pred, actual) decreases when targets vanish.

The right signal is "violation of expectation" — the predictor's belief
about which slots will be bound to real entities at t+1 vs which slots
actually got binding mass when SlotAttention re-bound the new obs.

Pipeline:
  binding_mass(attn) -> [B, S]      ground truth: what slots actually bound
  head(slots)        -> [B, S, 1]   predictor's belief: per-slot existence
  visibility_disagreement_surprise(pred_exists, gt_exists)
      -> [B] high when pred says "active" and gt says "absent" (occlusion)
              or vice versa (object appeared)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def binding_mass(attn: torch.Tensor) -> torch.Tensor:
    """Per-slot binding mass from a SlotAttention `attn` tensor.

    attn: [B, N_inputs, S] — last-iteration softmax-over-slots competition.
    Returns [B, S] in [0, 1]: mass of inputs each slot won, normalized
    by N_inputs so the values are interpretable as "fraction of inputs
    explained."
    """
    if attn.dim() != 3:
        raise ValueError(f"attn must be [B, N, S], got {tuple(attn.shape)}")
    n_inputs = attn.shape[1]
    return attn.sum(dim=1) / float(n_inputs)


class SlotExistenceHead(nn.Module):
    """Maps a slot vector to a scalar existence probability."""

    def __init__(self, slot_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """slots: [B, S, slot_dim] -> existence prob [B, S] in [0, 1]."""
        return torch.sigmoid(self.net(slots).squeeze(-1))


def slot_existence_loss(
    pred_exists: torch.Tensor,       # [B, S] in [0, 1]
    gt_exists: torch.Tensor,         # [B, S] in [0, 1], target
) -> dict:
    """BCE between predicted and ground-truth per-slot existence.

    gt_exists is treated as a soft target so binding_mass values (which
    are continuous) flow through cleanly. The .detach() on gt is the
    caller's responsibility; we do not detach here.
    """
    pred = pred_exists.clamp(1e-6, 1.0 - 1e-6)
    loss = F.binary_cross_entropy(pred, gt_exists.clamp(0.0, 1.0))
    return {
        "loss": loss,
        "pred_mean": pred.detach().mean(),
        "gt_mean": gt_exists.detach().mean(),
    }


class SceneContentHead(nn.Module):
    """Predicts a scalar 'scene content level' from pooled slots + action.

    Fallback path for when per-slot binding mass is uninformative (e.g.,
    slot system has not yet learned to specialize). Trained against a
    raw observation statistic — typically obs.std() per batch — and gives
    a usable surprise signal even when binding_mass is uniform.
    """

    def __init__(self, slot_dim: int, action_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim + action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_slots: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """pooled_slots: [B, slot_dim], action: [B, action_dim] -> [B]."""
        return self.net(torch.cat([pooled_slots, action], dim=-1)).squeeze(-1)


def scene_content_signal(obs: torch.Tensor) -> torch.Tensor:
    """Per-batch observation-content scalar [B] in approximately [0, 1].

    Uses pixel-std as the simplest content proxy: a frame with more
    rendered entities has higher std; a blank-background frame has near
    zero std. Robust to the slot-binding bottleneck because it bypasses
    slot binding entirely.
    """
    if obs.dim() < 2:
        raise ValueError(f"obs must be [B, ...] with batch dim, got {tuple(obs.shape)}")
    dims = tuple(range(1, obs.dim()))
    return obs.float().std(dim=dims)


def scene_content_surprise(
    pred_content: torch.Tensor,      # [B]
    gt_content: torch.Tensor,        # [B]
    weighting: str = "absolute",
) -> torch.Tensor:
    """Per-batch surprise from disagreement on scalar scene content.

    'absolute'   -> |pred - gt|         (any change in content level)
    'asymmetric' -> max(pred - gt, 0)   (only when predictor expected
                                          more content than actually arrived
                                          — the occlusion direction)
    """
    diff = pred_content - gt_content
    if weighting == "absolute":
        return diff.abs()
    if weighting == "asymmetric":
        return diff.clamp(min=0.0)
    raise ValueError(f"unknown weighting: {weighting}")


def visibility_disagreement_surprise(
    pred_exists: torch.Tensor,       # [B, S]
    gt_exists: torch.Tensor,         # [B, S]
    weighting: str = "absolute",
) -> torch.Tensor:
    """Per-batch scalar surprise from disagreement on slot existence.

    'absolute'    -> mean(|pred - gt|)   over slots, per batch  ([B])
    'asymmetric'  -> mean(max(pred - gt, 0))  — only spike when the
                     predictor expected something the obs lacks (the
                     occlusion direction). Useful when "new object
                     appeared" should NOT count as the same regime as
                     "expected object missing."
    """
    pred = pred_exists.detach() if not pred_exists.requires_grad else pred_exists
    diff = pred - gt_exists
    if weighting == "absolute":
        return diff.abs().mean(dim=-1)
    if weighting == "asymmetric":
        return diff.clamp(min=0.0).mean(dim=-1)
    raise ValueError(f"unknown weighting: {weighting}")
