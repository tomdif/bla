"""Sparse slot-delta predictor — the core mechanism of the KAM-JEPA prototype.

Standard latent world models predict the *full* next state. This module
predicts (a) which slots will change under the action and (b) what those
changes are, then composes the next state as:

    S_{t+1} = S_t + change_mask * delta_scale * tanh(raw_delta)

That keeps slot identity stable by default (no change → S_{t+1} = S_t),
bounds the magnitude of any single update (`tanh * delta_scale`), and
turns the change_mask into an interpretable diagnostic — we can read off
*which* slot the model thinks the action touched.

Conditioning. The predictor sees:
  - current slots `S_t` (the world ledger)
  - current observation features (cross-attention from slots → obs)
  - action vector (broadcast onto slots and obs as a bias)

Mechanism. A small transformer over (slots ⊕ obs_features) with an action
bias, projected back to slot dim. Two heads on top of each slot's output
token: a scalar `change_logit` (→ sigmoid → soft mask) and a `raw_delta`
in slot space.

The mask defaults to *soft sigmoid* during training. A `hard_mask = mask
> 0.5` diagnostic is exposed but not used for training in this version
— forcing hard masks early kills gradient flow to the predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SlotPredictorConfig:
    slot_dim: int = 64
    obs_dim: int = 64
    action_dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    delta_scale: float = 0.1
    dropout: float = 0.0
    mask_bias_init: float = -1.0
    update_mode: str = "delta"  # "delta" (sparse) or "dense" (replace) or "dynamic" (top-k active gate) or "id_dyn_split" (slow id + fast sparse dyn)
    target_active_slots: int = 0  # Phase-5B: 0 = all slots active (delta mode); >0 = top-k active gate
    # Phase 7B: split each slot into [id, dyn]. id updates slowly via EMA on
    # its own delta head (no mask), dyn updates via sparse mask × delta.
    # Used only when update_mode == "id_dyn_split".
    id_dim: int = 0           # if 0 and update_mode=="id_dyn_split", defaults to slot_dim // 2
    id_ema_alpha: float = 0.05  # EMA step size for the id half (small = slow)


class SlotDeltaPredictor(nn.Module):
    """Predict (change_mask, delta) for each slot, given (slots, obs, action)."""

    def __init__(self, cfg: SlotPredictorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.slot_dim

        self.obs_proj = nn.Linear(cfg.obs_dim, d, bias=False)
        self.action_proj = nn.Linear(cfg.action_dim, d, bias=False)
        # A learned role token added to slots vs obs so the transformer
        # knows which sequence position is which.
        self.role_embed = nn.Parameter(torch.zeros(1, 2, d))
        nn.init.trunc_normal_(self.role_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=int(d * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)

        # Two heads per slot: scalar change logit + raw delta vector.
        self.change_head = nn.Linear(d, 1)
        self.delta_head = nn.Linear(d, d)
        # Phase 5B: optional per-slot "active" gate. When update_mode is
        # "dynamic", only the top-`target_active_slots` highest-scoring
        # slots receive a delta update each step. The remaining slots
        # are frozen (act as long-term memory). Trained via a
        # straight-through estimator so gradients still flow to the
        # active_head.
        if cfg.update_mode == "dynamic":
            assert cfg.target_active_slots > 0, "dynamic mode needs target_active_slots > 0"
            self.active_head = nn.Linear(d, 1)
            nn.init.zeros_(self.active_head.bias)
            nn.init.normal_(self.active_head.weight, std=0.02)
        else:
            self.active_head = None
        # Phase 7B: id/dyn split. The id half updates slowly via EMA on a
        # dedicated id_delta head; the dyn half updates via the existing
        # sparse-delta mechanism on the dyn-dim subset of `raw_delta`. No
        # new transformer parameters — the id_delta_head re-projects the
        # transformer output to the id_dim subspace.
        if cfg.update_mode == "id_dyn_split":
            self._id_dim = cfg.id_dim if cfg.id_dim > 0 else d // 2
            assert 0 < self._id_dim < d, f"id_dim {self._id_dim} must be in (0, {d})"
            self.id_delta_head = nn.Linear(d, self._id_dim)
        else:
            self._id_dim = 0
        # Init the change-mask bias negative so updates default to off —
        # the model has to learn to fire the mask. The exact starting
        # point matters: bias=-1 starts at sigmoid(-1)≈0.27, bias=-2 at
        # sigmoid(-2)≈0.12, which gives the L1 sparsity penalty room to
        # operate without immediately fighting the prediction objective.
        nn.init.constant_(self.change_head.bias, cfg.mask_bias_init)

    def forward(
        self,
        slots: torch.Tensor,
        obs_features: torch.Tensor,
        action: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict:
        """slots:        [B, S, slot_dim]
        obs_features: [B, N, obs_dim]
        action:       [B, action_dim]

        Returns dict with:
          next_slots:    S + change_mask * delta_scale * tanh(raw_delta)
          change_mask:   soft sigmoid in [0, 1], shape [B, S, 1]
          delta:         bounded delta vectors, shape [B, S, slot_dim]
          + (optional) raw_delta, hard_mask, change_logits
        """
        b, s, _ = slots.shape
        obs = self.obs_proj(obs_features)
        # Tag slots vs obs with role embeddings.
        slots_in = slots + self.role_embed[:, 0:1]
        obs_in = obs + self.role_embed[:, 1:2]
        # Action bias broadcast across every position in the sequence.
        action_bias = self.action_proj(action).unsqueeze(1)
        seq = torch.cat([slots_in, obs_in], dim=1) + action_bias

        h = self.blocks(seq)
        h = self.norm(h)
        slot_out = h[:, :s, :]

        change_logits = self.change_head(slot_out)                # [B, S, 1]
        change_mask = torch.sigmoid(change_logits)
        raw_delta = self.delta_head(slot_out)                     # [B, S, slot_dim]
        delta = self.cfg.delta_scale * torch.tanh(raw_delta)
        active_mask = None
        if self.cfg.update_mode == "delta":
            # Sparse, bounded update — the proposal.
            next_slots = slots + change_mask * delta
        elif self.cfg.update_mode == "dense":
            # Ablation: the predictor produces the next slot state directly
            # via the same transformer + delta-head architecture, but with
            # no change_mask gating and no residual / identity prior. Tests
            # whether the *sparse-delta* mechanism is load-bearing beyond
            # just having slot binding.
            next_slots = raw_delta
            # Force-report a "mask" of 1.0 so downstream loss reporting still
            # has a consistent field; nothing in the loss penalizes it.
            change_mask = torch.ones_like(change_mask)
        elif self.cfg.update_mode == "id_dyn_split":
            # Phase 7B: each slot is [id (slow, EMA), dyn (fast, sparse-delta)].
            # id: id_next = id + alpha * id_delta  (no mask, slow EMA-style)
            # dyn: dyn_next = dyn + change_mask * delta_scale * tanh(raw_delta_dyn)
            id_dim = self._id_dim
            id_in = slots[..., :id_dim]
            dyn_in = slots[..., id_dim:]
            id_delta = self.id_delta_head(slot_out)                    # [B, S, id_dim]
            id_next = id_in + self.cfg.id_ema_alpha * id_delta
            dyn_delta = delta[..., id_dim:]                            # bounded tanh in slot dim
            dyn_next = dyn_in + change_mask * dyn_delta
            next_slots = torch.cat([id_next, dyn_next], dim=-1)
            # Force the mask reported in `out` to be the actual dyn-half mask
            # (no change-mask applied to the id half — it's always-on but slow).
        elif self.cfg.update_mode == "dynamic":
            # Phase 5B: pool of n_slots, but only the top-K highest-
            # scoring slots get a delta update each step. Inactive slots
            # are frozen (act as long-term memory). Straight-through
            # estimator: forward uses hard top-K mask, backward passes
            # gradient through the sigmoid score.
            k = self.cfg.target_active_slots
            active_logits = self.active_head(slot_out).squeeze(-1)   # [B, S]
            topk_idx = active_logits.topk(k, dim=-1).indices
            hard_active = torch.zeros_like(active_logits)
            hard_active.scatter_(1, topk_idx, 1.0)
            active_soft = torch.sigmoid(active_logits)
            # ST: forward = hard_active, backward = active_soft.
            active = hard_active + active_soft - active_soft.detach()
            active_mask = active.unsqueeze(-1)                          # [B, S, 1]
            next_slots = slots + active_mask * change_mask * delta
        else:
            raise ValueError(f"unknown update_mode: {self.cfg.update_mode}")

        out = {
            "next_slots": next_slots,
            "change_mask": change_mask,
            "delta": delta,
        }
        if active_mask is not None:
            out["active_mask"] = active_mask
        if return_diagnostics:
            out["raw_delta"] = raw_delta
            out["change_logits"] = change_logits
            out["hard_mask"] = (change_mask > 0.5).float()
        return out


def slot_delta_loss(
    pred_next_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    change_mask: torch.Tensor,
    sparsity_weight: float = 1e-3,
    bimodal_weight: float = 0.0,
) -> dict:
    """Composite loss for slot-delta training.

    Three terms:
      * prediction:   smooth-L1 between predicted and EMA-target slots.
      * sparsity:     L1 on the soft mask (push *count* of updates down).
      * bimodality:   entropy of each mask entry (push *values* toward
                       decisive {0, 1}). Defaults off; tiny weights of
                       1e-3 to 5e-3 can give cleaner 'on/off' gating
                       without further suppressing the mean.
    The mask penalty has to stay small or it collapses every slot to
    'never update' — that hurts learning more than collapse hurts
    evaluation."""
    prediction = F.smooth_l1_loss(pred_next_slots.float(),
                                    target_next_slots.detach().float())
    m = change_mask.float()
    sparsity = m.mean()
    # Per-entry Bernoulli entropy. Bounded in [0, log(2)] ≈ [0, 0.693];
    # zero when m ∈ {0, 1}, max at m = 0.5.
    eps = 1e-7
    entropy = -(m * (m + eps).log() + (1.0 - m) * (1.0 - m + eps).log()).mean()
    total = prediction + sparsity_weight * sparsity + bimodal_weight * entropy
    return {
        "loss": total,
        "prediction": prediction.detach(),
        "sparsity": sparsity.detach(),
        "bimodal_entropy": entropy.detach(),
        "mask_mean": change_mask.detach().mean(),
        "mask_max": change_mask.detach().amax(),
    }


def copy_baseline(slots: torch.Tensor) -> torch.Tensor:
    """Trivial baseline: don't change slots at all. Useful to confirm the
    delta predictor is actually learning useful updates rather than just
    riding the persistence prior."""
    return slots
