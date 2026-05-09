from __future__ import annotations

import torch
from torch import nn


class DiagonalSSMLayer(nn.Module):
    """Bidirectional diagonal-state SSM block (reference, unfused implementation).

    The forward/backward scan is a Python for-loop over the sequence axis. That
    is asymptotically O(N) but the constant is large; for blueprint-scale runs
    swap this for a fused kernel (e.g. mamba-ssm) or a parallel-scan
    implementation. This module is intentionally dependency-free so the scaffold
    runs on a fresh CPU without compiled extensions.
    """

    def __init__(self, d_model: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.a_logit = nn.Parameter(torch.zeros(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def _scan(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        batch, length, dim = x.shape
        state = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        decay = torch.sigmoid(self.a_logit).to(dtype=x.dtype)[None, :]
        outputs = []
        indices = range(length - 1, -1, -1) if reverse else range(length)
        for idx in indices:
            token = x[:, idx, :]
            update = torch.tanh(self.in_proj(token))
            gate = torch.sigmoid(self.gate_proj(token))
            state = decay * state + (1.0 - decay) * update
            outputs.append(gate * state)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self._scan(y, reverse=False) + self._scan(y, reverse=True)
        x = x + self.out_proj(y)
        return x + self.mlp(self.norm(x))


class BidirectionalSSMScratchpad(nn.Module):
    def __init__(self, d_model: int = 4096, layers: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([DiagonalSSMLayer(d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class WorkingMemory(nn.Module):
    """Folds query vectors and retrieved RAM facts into one fixed hidden state."""

    def __init__(self, d_core: int = 4096, d_ram: int = 4096, layers: int = 64):
        super().__init__()
        self.query_proj = nn.Linear(d_core, d_core)
        self.fact_proj = nn.Linear(d_ram, d_core)
        self.scratchpad = BidirectionalSSMScratchpad(d_core, layers=layers)
        self.state_head = nn.Linear(d_core, d_core)

    def forward(self, query: torch.Tensor, facts: torch.Tensor) -> torch.Tensor:
        if facts.ndim != 3:
            raise ValueError(f"facts must be [B, k, d_ram], got {tuple(facts.shape)}")
        dtype = next(self.parameters()).dtype
        query = query.to(dtype=dtype)
        facts = facts.to(dtype=dtype)
        query_token = self.query_proj(query)[:, None, :]
        fact_tokens = self.fact_proj(facts)
        seq = torch.cat([query_token, fact_tokens], dim=1)
        folded = self.scratchpad(seq)
        pooled = folded.mean(dim=1)
        return self.state_head(pooled)
