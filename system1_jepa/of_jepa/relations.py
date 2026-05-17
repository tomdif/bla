"""Phase 12 prototype: pairwise relation graph over object files.

OF-JEPA gives us per-file (id_key, state_value). A world model also
needs to know how objects RELATE to each other. This module adds the
minimal scaffold:

  RelationHead          — bilinear/MLP scorer for pairwise relations
  RelationGraphPredictor — predicts pairwise relations from a batch
                           of ObjectFileBatches (frame t)

Start with one relation that has a clean GT signal from MOVi:

  "near"  — pairwise distance in image_positions below a threshold
            (e.g. < 0.10 in normalized [0,1] coords)

Future relations (per the Phase 12 plan): touching, colliding,
supporting, inside, occluding, blocking, moving_together,
causally_affects. Each is a separate head trained on labeled or
simulated GT.

The relation predictor takes object-file batches and outputs an
[N_files, N_files] logit matrix per relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import ObjectFileBatch


@dataclass
class RelationConfig:
    file_dim: int = 128         # full slot dim (id_key + state_value)
    hidden_dim: int = 128
    n_relations: int = 1        # number of relation types to predict
    asymmetric: bool = False    # False → symmetric pairs (i,j) == (j,i)


class RelationHead(nn.Module):
    """Pairwise relation scorer.

    score[r, i, j] = MLP(concat[file_i, file_j])

    For symmetric relations (default), the scorer is symmetrized by
    averaging score(i,j) and score(j,i) — equivalent to symmetric MLP
    input via concat(file_i+file_j, |file_i-file_j|).
    """

    def __init__(self, cfg: RelationConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.file_dim * 2 if cfg.asymmetric else cfg.file_dim * 2
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.n_relations),
        )

    def forward(self, files: torch.Tensor) -> torch.Tensor:
        """files: [B, N, file_dim] → relation logits [B, N, N, n_relations]"""
        B, N, D = files.shape
        # Build all pairs.
        a = files.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]  (file_i)
        b = files.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]  (file_j)
        if self.cfg.asymmetric:
            pair = torch.cat([a, b], dim=-1)
        else:
            pair = torch.cat([a + b, (a - b).abs()], dim=-1)
        logits = self.scorer(pair)  # [B, N, N, n_relations]
        if not self.cfg.asymmetric:
            # Enforce exact symmetry by averaging with the transpose.
            logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits


class RelationGraphPredictor(nn.Module):
    """Wraps a RelationHead to consume ObjectFileBatch directly.

    Returns a [B, N, N, n_relations] logit tensor and (optional) a
    masked version where self-relations (i, i) and inactive-file rows
    are zeroed out.
    """

    def __init__(self, cfg: RelationConfig):
        super().__init__()
        self.head = RelationHead(cfg)
        self.cfg = cfg

    def forward(self, ofb: ObjectFileBatch,
                 mask_self: bool = True) -> torch.Tensor:
        """Returns [B, N, N, n_relations] logits.

        If mask_self, zeros out the diagonal so (i, i) relations are
        not predicted.
        """
        logits = self.head(ofb.full_slot)
        if mask_self:
            N = logits.shape[1]
            eye = torch.eye(N, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(eye[None, :, :, None], 0.0)
        return logits


# --- Ground-truth label generators for MOVi ---

def near_relation_labels(positions: torch.Tensor,
                          visibility: torch.Tensor,
                          threshold: float = 0.10) -> torch.Tensor:
    """GT label for the "near" relation on MOVi-style data.

    positions:  [N, 2] normalized [0,1] image positions for one frame
    visibility: [N] bool — whether each entity is visible this frame
    threshold:  distance threshold for "near"

    Returns [N, N] bool: True iff both entities are visible AND their
    image distance is below threshold. Self-pairs (i, i) are False.
    """
    N, _ = positions.shape
    dist = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)  # [N, N]
    near = dist < threshold
    eye = torch.eye(N, dtype=torch.bool, device=positions.device)
    near = near & ~eye
    # Mask out pairs where either entity is invisible.
    vis_pair = visibility.unsqueeze(0) & visibility.unsqueeze(1)
    return near & vis_pair


def relation_loss(pred_logits: torch.Tensor,
                   gt_labels: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE loss over pairwise relation logits.

    pred_logits: [B, N, N, n_relations]
    gt_labels:   [B, N, N, n_relations] bool
    mask:        [B, N, N] bool (optional) — restrict to valid pairs

    Returns scalar mean over valid pairs.
    """
    target = gt_labels.float()
    if mask is None:
        return F.binary_cross_entropy_with_logits(pred_logits, target)
    # Per-pair BCE, mask, then mean over valid pairs.
    per_pair = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    )  # [B, N, N, n_relations]
    mask_full = mask.unsqueeze(-1).expand_as(per_pair)
    return (per_pair * mask_full).sum() / mask_full.float().sum().clamp(min=1.0)
