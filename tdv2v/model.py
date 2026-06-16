"""Compact, self-contained ViT + TDV motion encoder + CroCo-style cross-view head (torch-only)."""
import copy, math, torch, torch.nn as nn


class Block(nn.Module):
    """Pre-LN self-attention transformer block."""
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp)), nn.GELU(), nn.Linear(int(dim*mlp), dim))

    def forward(self, x):
        h = self.ln1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CrossBlock(nn.Module):
    """Pre-LN cross-attention block: queries `q` attend to context `kv`."""
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.lnq = nn.LayerNorm(dim); self.lnk = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp)), nn.GELU(), nn.Linear(int(dim*mlp), dim))

    def forward(self, q, kv):
        k = self.lnk(kv)
        q = q + self.attn(self.lnq(q), k, k, need_weights=False)[0]
        return q + self.mlp(self.ln2(q))


class PatchEmbed(nn.Module):
    def __init__(self, img, patch, dim, in_ch=3):
        super().__init__()
        self.n = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class ViT(nn.Module):
    """Frame encoder. Returns all tokens (B, N+1, D); token 0 is [CLS]."""
    def __init__(self, img=224, patch=16, dim=384, depth=12, heads=6):
        super().__init__()
        self.embed = PatchEmbed(img, patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.embed.n + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        t = self.embed(x)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1) + self.pos
        for b in self.blocks: t = b(t)
        return self.norm(t)


class MotionEncoder(nn.Module):
    """TDV motion encoder: embeds the RGB difference Δx and cross-attends to z_t,
    producing Δz tokens (same shape as z) so that ẑ_{t+1} = z_t + Δz."""
    def __init__(self, img=224, patch=16, dim=384, heads=6, depth=4):
        super().__init__()
        self.embed = PatchEmbed(img, patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.embed.n + 1, dim))
        self.mcls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, dx, z_t):
        m = self.embed(dx)
        m = torch.cat([self.mcls.expand(m.size(0), -1, -1), m], 1) + self.pos
        for b in self.blocks: m = b(m, z_t)        # motion tokens attend to the current frame rep
        return self.norm(m)


class CrossViewHead(nn.Module):
    """CroCo-style: predict the OTHER camera's tokens from this camera's tokens (patch-level
    cross-attention completion). This is the learned correspondence module Tier-0 showed you need."""
    def __init__(self, n_tokens, dim=384, heads=6, depth=4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, z_src):
        q = self.query.expand(z_src.size(0), -1, -1)
        for b in self.blocks: q = b(q, z_src)
        return self.norm(q)


class ProjHead(nn.Module):
    """MLP projection head (for the optional DINO anti-collapse path)."""
    def __init__(self, dim, hidden=2048, out=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.copy_(bs)


def make_teacher(student: nn.Module):
    t = copy.deepcopy(student)
    for p in t.parameters(): p.requires_grad_(False)
    return t.eval()
