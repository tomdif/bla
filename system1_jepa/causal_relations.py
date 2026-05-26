"""Layer C: self-supervised causal relation graph over slots.

Complement to of_jepa/relations.py, which trains labeled spatial relations
(near, touching). This module trains causal_strength edges with no
external labels — supervision comes from action-conditioned co-firing of
the SlotDeltaPredictor's change_mask: if the same action fires slot i at
t and slot j at t+1, edge i->j gets evidence.

Outputs per ordered pair (i, j):
  edge_type_logits  — K-way classification head (clusters, unsupervised)
  confidence        — calibrated [0, 1] gate on whether ANY edge exists
  causal_strength   — [0, 1] estimate of dynamical influence i -> j
Plus refined_slots: relation-aware slots after one GNN message pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CausalRelationConfig:
    slot_dim: int = 64
    n_edge_types: int = 8
    hidden: int = 128


@dataclass
class EdgeAnnotations:
    edge_type_logits: torch.Tensor   # [B, S, S, K]
    confidence: torch.Tensor         # [B, S, S]  in [0, 1]
    causal_strength: torch.Tensor    # [B, S, S]  in [0, 1]
    refined_slots: torch.Tensor      # [B, S, D]


class CausalRelationHead(nn.Module):
    def __init__(self, cfg: CausalRelationConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.slot_dim, cfg.hidden
        # Pair features: [s_i, s_j, s_i*s_j, |s_i - s_j|]
        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
        )
        self.edge_type_head = nn.Linear(h, cfg.n_edge_types)
        self.confidence_head = nn.Linear(h, 1)
        self.strength_head = nn.Linear(h, 1)
        # Single message-passing round to produce relation-aware slots.
        self.msg_proj = nn.Linear(h, d)
        self.slot_update = nn.GRUCell(d, d)

    def _pair_features(self, slots: torch.Tensor) -> torch.Tensor:
        b, s, d = slots.shape
        si = slots.unsqueeze(2).expand(b, s, s, d)
        sj = slots.unsqueeze(1).expand(b, s, s, d)
        return torch.cat([si, sj, si * sj, (si - sj).abs()], dim=-1)

    def forward(self, slots: torch.Tensor) -> EdgeAnnotations:
        b, s, d = slots.shape
        h = self.pair_mlp(self._pair_features(slots))     # [B, S, S, H]
        edge_type_logits = self.edge_type_head(h)
        confidence = torch.sigmoid(self.confidence_head(h).squeeze(-1))
        strength = torch.sigmoid(self.strength_head(h).squeeze(-1))
        # Strength-weighted, confidence-gated messages aggregated as incoming-to-i.
        gate = (confidence * strength).unsqueeze(-1)
        msg = self.msg_proj(h) * gate                      # [B, S, S, D]
        agg = msg.sum(dim=2)
        refined = self.slot_update(
            agg.reshape(b * s, d), slots.reshape(b * s, d)
        ).reshape(b, s, d)
        return EdgeAnnotations(edge_type_logits, confidence, strength, refined)


def causal_edge_loss(
    edges: EdgeAnnotations,
    change_mask_t: torch.Tensor,     # [B, S, 1] or [B, S]
    change_mask_t1: torch.Tensor,    # [B, S, 1] or [B, S]
    diag_penalty: float = 1e-2,
) -> dict:
    """Self-supervised causal edges from action-conditioned co-firing.

    Hypothesis: if action a fires slot i at t and the same action fires
    slot j at t+1, that is evidence the model's own dynamics propagate
    i -> j. We supervise causal_strength on this signal.
    """
    m_t = change_mask_t.squeeze(-1) if change_mask_t.dim() == 3 else change_mask_t
    m_t1 = change_mask_t1.squeeze(-1) if change_mask_t1.dim() == 3 else change_mask_t1
    target = (m_t.unsqueeze(2) * m_t1.unsqueeze(1)).detach().clamp(0.0, 1.0)
    edge_pred = (edges.causal_strength * edges.confidence).clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(edge_pred, target)
    eye = torch.eye(target.shape[-1], device=target.device)
    diag = (edges.causal_strength * eye).mean()
    return {
        "loss": bce + diag_penalty * diag,
        "bce": bce.detach(),
        "diag": diag.detach(),
        "target_mean": target.mean().detach(),
        "edge_pred_mean": edge_pred.mean().detach(),
    }
