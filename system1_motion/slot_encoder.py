"""Object-centric (slot) encoder for the grounding probe — the OF-JEPA front-end
ported from arc_local/jepa_wm/system1_jepa (PatchViTEncoder + SlotAttention),
self-contained here so the pod run needs no cross-repo paths.

Why this exists: the monolithic mean-pool ViT failed to ground absolute position
(Gate-0 ~18px) under self-supervised objectives. A slot encoder is spatially
structured — patch tokens carry ABSOLUTE sincos position, and slot attention binds
locally — so it is the natural test of "does object-centric perception ground
position for free?" We swap ONLY the encoder; all probe/grounding/condition code
in train_dissoc.py is identical.

SlotEncoder.forward(x) -> [B, n_slots*slot_dim] flattened-slot latent, matching the
ViTEncoder forward(x)->[B,d_z] interface so it drops straight into the rig. Slots
are seeded DETERMINISTICALLY from the learned per-slot means (no sampling noise) so
slot<->role correspondence is stable across the z_t / z_future encodes (needed for
the prediction loss; the probe itself is correspondence-agnostic).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn


# ---------- ViT patch encoder (sincos 2D pos) ----------
def _sincos_1d(positions, dim):
    half = max(dim // 2, 1)
    omega = torch.arange(half, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half - 1, 1)))
    out = positions.float()[:, None] * omega[None, :]
    emb = torch.cat([out.sin(), out.cos()], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def sincos_2d_position(grid_h, grid_w, dim, device, dtype):
    y = torch.arange(grid_h, device=device); x = torch.arange(grid_w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1); xx = xx.reshape(-1)
    y_dim = dim // 2; x_dim = dim - y_dim
    pos = torch.cat([_sincos_1d(yy, y_dim), _sincos_1d(xx, x_dim)], dim=1)
    return pos.to(dtype=dtype)


class PatchViTEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=192, patch_size=8, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        assert latent_dim % heads == 0
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, latent_dim, patch_size, patch_size)
        layer = nn.TransformerEncoderLayer(latent_dim, heads, int(latent_dim * mlp_ratio),
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x):
        p = self.patch_embed(x)
        _, _, gh, gw = p.shape
        tok = p.flatten(2).transpose(1, 2)                      # [B,N,D]
        pos = sincos_2d_position(gh, gw, tok.shape[-1], tok.device, tok.dtype)
        return self.norm(self.blocks(tok + pos.unsqueeze(0)))   # [B,N,D]


# ---------- Slot Attention (Locatello et al. 2020) ----------
@dataclass
class SlotAttentionConfig:
    n_slots: int = 6
    slot_dim: int = 64
    n_iters: int = 3
    mlp_ratio: int = 4
    eps: float = 1e-8


class SlotAttention(nn.Module):
    def __init__(self, input_dim, cfg=None):
        super().__init__()
        self.cfg = cfg or SlotAttentionConfig()
        d = self.cfg.slot_dim
        self.slots_mu = nn.Parameter(torch.zeros(1, self.cfg.n_slots, d))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, self.cfg.n_slots, d))
        nn.init.trunc_normal_(self.slots_mu, std=0.02)
        nn.init.trunc_normal_(self.slots_log_sigma, std=0.02)
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(d)
        self.norm_pre_mlp = nn.LayerNorm(d)
        self.proj_k = nn.Linear(input_dim, d, bias=False)
        self.proj_v = nn.Linear(input_dim, d, bias=False)
        self.proj_q = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        hidden = d * self.cfg.mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, inputs, init_slots=None, return_attention=False):
        b = inputs.shape[0]; d = self.cfg.slot_dim
        x = self.norm_inputs(inputs)
        k = self.proj_k(x); v = self.proj_v(x)
        slots = init_slots if init_slots is not None else self.slots_mu.expand(b, -1, -1)
        scale = d ** -0.5
        attn = None
        for _ in range(self.cfg.n_iters):
            prev = slots
            q = self.proj_q(self.norm_slots(slots))
            attn = (torch.einsum("bsd,bnd->bns", q, k) * scale).softmax(dim=-1)  # [B,N,S]
            attn_n = attn / (attn.sum(dim=1, keepdim=True) + self.cfg.eps)
            updates = torch.einsum("bns,bnd->bsd", attn_n, v)
            slots = self.gru(updates.reshape(b * self.cfg.n_slots, d),
                             prev.reshape(b * self.cfg.n_slots, d)).reshape(b, self.cfg.n_slots, d)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
        if return_attention:
            return slots, attn                                  # attn: [B, N_patches, n_slots]
        return slots                                            # [B, n_slots, slot_dim]


# ---------- wrapper: forward(x) -> flattened-slot latent ----------
class SlotEncoder(nn.Module):
    """OF-JEPA front-end as a drop-in for ViTEncoder. forward(x:[B,C,H,W]) ->
    [B, n_slots*slot_dim]. Deterministic slot init for stable correspondence."""
    def __init__(self, in_channels=3, vit_dim=192, patch=8, depth=6, heads=6,
                 n_slots=6, slot_dim=64, n_iters=3):
        super().__init__()
        self.vit = PatchViTEncoder(in_channels, vit_dim, patch, depth, heads)
        self.slot = SlotAttention(vit_dim, SlotAttentionConfig(n_slots=n_slots, slot_dim=slot_dim, n_iters=n_iters))
        self.d_z = n_slots * slot_dim

    def forward(self, x):
        tok = self.vit(x)                                       # [B,N,vit_dim]
        b = x.shape[0]
        init = self.slot.slots_mu.expand(b, -1, -1)             # deterministic (no noise)
        slots = self.slot(tok, init_slots=init)                 # [B,K,slot_dim]
        return slots.reshape(b, -1)                             # [B, K*slot_dim]

    def encode_with_attn(self, x):
        """Returns (slot_content [B,K,slot_dim], attn [B,N_patches,K]) for the
        content-vs-routing discrimination. Also returns the patch grid (gh,gw)."""
        tok = self.vit(x)
        b = x.shape[0]
        gh = gw = int(round(tok.shape[1] ** 0.5))
        init = self.slot.slots_mu.expand(b, -1, -1)
        slots, attn = self.slot(tok, init_slots=init, return_attention=True)
        return slots, attn, gh, gw
