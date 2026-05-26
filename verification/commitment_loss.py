"""Causal Commitment Training — penalize silent belief revision.

A commitment is a latent c = encode(claim, world_state). At a later time
the system re-encodes the same claim against a fresh world state. The
loss penalizes drift in c, except in proportion to how much new evidence
has arrived since the commitment was made. Predictor surprise is the
evidence proxy.

Two terms by design:
  consistency    — (1 - r) * drift   : punishes silent revision
  contradiction  — r * (1 - drift)   : punishes stubbornness under surprise

Without the contradiction term, the encoder trivially minimizes drift by
collapsing every claim to the same vector. The contradiction term makes
collapse lossy whenever surprise is high.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CommitmentEncoderConfig:
    claim_dim: int
    world_dim: int
    hidden: int = 256
    out_dim: int = 64


class CommitmentEncoder(nn.Module):
    def __init__(self, cfg: CommitmentEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.claim_dim + cfg.world_dim, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.out_dim),
        )

    def forward(self, claim: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([claim, world], dim=-1)), dim=-1)


def commitment_consistency_loss(
    c_past: torch.Tensor,        # [B, D] commitment at time t
    c_now: torch.Tensor,         # [B, D] re-encoded at time t+k
    surprise: torch.Tensor,      # [B]    nonneg evidence-novelty signal
    revision_temp: float = 1.0,
    margin: float = 0.1,
) -> dict:
    """r = sigmoid(surprise / temp - margin) authorizes revision."""
    drift = 1.0 - (c_past.detach() * c_now).sum(dim=-1)       # [B], cosine drift
    r = torch.sigmoid(surprise / revision_temp - margin)
    consistency = ((1.0 - r) * drift).mean()
    contradiction = (r * (1.0 - drift)).mean()
    total = consistency + contradiction
    return {
        "loss": total,
        "consistency": consistency.detach(),
        "contradiction": contradiction.detach(),
        "revision_rate": r.detach().mean(),
        "drift": drift.detach().mean(),
    }
