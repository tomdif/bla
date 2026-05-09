from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from system1_jepa.model import dtype_from_name
from tensor_ram.torch_ram import DifferentiableTensorRAM

from .decoder import DeterministicLexicalDecoder
from .diffusion import LatentDiffusionEngine, diffusion_score_matching_loss
from .ram_reader import RAMReader
from .ssm import WorkingMemory


@dataclass
class DCAConfig:
    d_core: int = 4096
    d_ram: int = 4096
    ssm_layers: int = 64
    dit_layers: int = 24
    heads: int = 16
    vocab_size: int = 128_000
    memory_tokens: int = 4
    ram_query_heads: int = 4
    ram_top_k: int = 8
    ram_sparsity_weight: float = 0.0
    ram_hard: bool = False
    dtype: str = "bfloat16"

    @classmethod
    def tiny(cls) -> "DCAConfig":
        return cls(
            d_core=64,
            d_ram=64,
            ssm_layers=1,
            dit_layers=1,
            heads=4,
            vocab_size=256,
            memory_tokens=2,
            ram_query_heads=2,
            ram_top_k=4,
            dtype="float32",
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype)


class DCAEngine(nn.Module):
    """System 2: state-space memory, latent diffusion CPU, deterministic decoder.

    When a `DifferentiableTensorRAM` is attached, the engine fetches its own
    facts via a learned RAMReader instead of accepting them as input. That
    realizes the blueprint's "differentiable READ from RAM" — the CPU decides
    what to retrieve, the RAM contents stay frozen and non-parametric.
    """

    def __init__(self, config: DCAConfig, ram: Optional[DifferentiableTensorRAM] = None):
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
            memory_tokens=config.memory_tokens,
        )
        self.decoder = DeterministicLexicalDecoder(
            d_core=config.d_core,
            vocab_size=config.vocab_size,
        )
        self.ram_reader: Optional[RAMReader] = None
        if ram is not None:
            if ram.d_ram != config.d_ram:
                raise ValueError(f"RAM d_ram {ram.d_ram} != config.d_ram {config.d_ram}")
            self.ram_reader = RAMReader(
                d_state=config.d_core,
                ram=ram,
                num_query_heads=config.ram_query_heads,
                top_k=config.ram_top_k,
                sparsity_weight=config.ram_sparsity_weight,
                hard=config.ram_hard,
            )

        self.working_memory.to(dtype=config.torch_dtype)
        self.diffusion.to(dtype=config.torch_dtype)
        if self.ram_reader is not None:
            self.ram_reader.to(dtype=config.torch_dtype)

    def fetch_facts(self, query: torch.Tensor, return_aux: bool = False):
        if self.ram_reader is None:
            raise RuntimeError("DCAEngine was constructed without a RAM; pass facts explicitly")
        if return_aux:
            facts, aux = self.ram_reader(query, return_aux=True)
            return facts, aux
        facts, _ = self.ram_reader(query)
        return facts

    def forward(
        self,
        query: torch.Tensor,
        canvas: torch.Tensor,
        timesteps: torch.Tensor,
        facts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if facts is None:
            facts = self.fetch_facts(query)
        memory = self.working_memory(query, facts)
        velocity = self.diffusion(canvas, timesteps, memory)
        return {"memory": memory, "velocity": velocity, "facts": facts}

    def training_loss(
        self,
        query: torch.Tensor,
        x0: torch.Tensor,
        facts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sparsity_loss = None
        if facts is None:
            if self.ram_reader is not None and self.ram_reader.sparsity_weight > 0:
                facts, aux = self.fetch_facts(query, return_aux=True)
                sparsity_loss = aux.get("sparsity_loss")
            else:
                facts = self.fetch_facts(query)
        memory = self.working_memory(query, facts)
        diffusion_loss = diffusion_score_matching_loss(self.diffusion, x0, memory)
        total = diffusion_loss + sparsity_loss if sparsity_loss is not None else diffusion_loss
        out = {"loss": total, "diffusion_loss": diffusion_loss.detach(), "memory": memory.detach(), "facts": facts.detach()}
        if sparsity_loss is not None:
            out["sparsity_loss"] = sparsity_loss.detach()
        return out
