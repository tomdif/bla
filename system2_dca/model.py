from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from system1_jepa.model import dtype_from_name

from .decoder import DeterministicLexicalDecoder
from .diffusion import LatentDiffusionEngine, diffusion_score_matching_loss
from .ssm import WorkingMemory


@dataclass
class DCAConfig:
    d_core: int = 4096
    d_ram: int = 4096
    ssm_layers: int = 64
    dit_layers: int = 24
    heads: int = 16
    vocab_size: int = 128_000
    dtype: str = "bfloat16"

    @classmethod
    def tiny(cls) -> "DCAConfig":
        return cls(d_core=64, d_ram=64, ssm_layers=1, dit_layers=1, heads=4, vocab_size=256)

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype)


class DCAEngine(nn.Module):
    """System 2: state-space memory, latent diffusion CPU, deterministic decoder."""

    def __init__(self, config: DCAConfig):
        super().__init__()
        self.config = config
        self.working_memory = WorkingMemory(
            d_core=config.d_core,
            d_ram=config.d_ram,
            layers=config.ssm_layers,
        )
        self.diffusion = LatentDiffusionEngine(
            d_core=config.d_core,
            depth=config.dit_layers,
            heads=config.heads,
        )
        self.decoder = DeterministicLexicalDecoder(
            d_core=config.d_core,
            vocab_size=config.vocab_size,
            dtype=config.torch_dtype,
        )
        self.to(dtype=config.torch_dtype)
        self.decoder.codebook.requires_grad_(False)

    def forward(
        self,
        query: torch.Tensor,
        facts: torch.Tensor,
        canvas: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        memory = self.working_memory(query, facts)
        eps_pred = self.diffusion(canvas, timesteps, memory)
        return {"memory": memory, "eps_pred": eps_pred}

    def training_loss(self, query: torch.Tensor, facts: torch.Tensor, x0: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.working_memory(query, facts)
        loss = diffusion_score_matching_loss(self.diffusion, x0, memory)
        return {"loss": loss, "memory": memory.detach()}

