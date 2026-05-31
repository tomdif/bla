"""System-1 motion substrate modules (system1_motion_spec.md §4).

Deliberately simple — the contribution is the OBJECTIVE, not the architecture.
Encoder = compact ViT (no pretrained weights), latent dynamics = small
transformer over (z_t, a_t), EMA target encoder, detached position-decode head.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    """Patchify -> + learned pos -> transformer -> mean-pool token = z [d_z].
    Standard ViT; defaults near ViT-S (d=384, depth 6 for cheap Reacher runs)."""

    def __init__(self, image_size=64, patch=8, in_ch=3, d_z=384, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        assert image_size % patch == 0
        self.n_patches = (image_size // patch) ** 2
        self.d_z = d_z
        self.patch_embed = nn.Conv2d(in_ch, d_z, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_z))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_z, heads, int(d_z * mlp_ratio),
                                           dropout=0.0, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_z)

    def forward(self, x):                                   # x: [B,3,H,W]
        h = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, P, d_z]
        h = self.blocks(h + self.pos)
        return self.norm(h).mean(dim=1)                     # [B, d_z] pooled latent


class LatentDynamics(nn.Module):
    """(z_t, a_t) -> z_hat_{t+1}. Action embedded, then a 2-token transformer;
    read out token 0 as the predicted next latent."""

    def __init__(self, d_z=384, d_a=2, depth=4, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.a_embed = nn.Sequential(nn.Linear(d_a, d_z), nn.GELU(), nn.Linear(d_z, d_z))
        self.type_emb = nn.Parameter(torch.zeros(1, 2, d_z))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_z, heads, int(d_z * mlp_ratio),
                                           dropout=0.0, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(d_z, d_z)

    def forward(self, z, a):                                # z:[B,d_z] a:[B,d_a]
        tok = torch.stack([z, self.a_embed(a)], dim=1) + self.type_emb   # [B,2,d_z]
        return self.head(self.blocks(tok)[:, 0])            # [B,d_z]


class DecodeHead(nn.Module):
    """Diagnostic position decoder (detached input). 2-layer MLP -> (x,y) px."""

    def __init__(self, d_z=384, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1 - momentum)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)
