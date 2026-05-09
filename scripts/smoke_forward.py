from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from latent_bus import EntropyRouter, TokenlessLatentBus
from system1_jepa import BLAJEPAModel, JEPAConfig
from system2_dca import DCAConfig, DCAEngine
from tensor_ram import FaissTensorRAM


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    jepa_cfg = JEPAConfig.tiny()
    dca_cfg = DCAConfig.tiny()
    jepa = BLAJEPAModel(jepa_cfg)
    dca = DCAEngine(dca_cfg)
    bus = TokenlessLatentBus(d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core)

    masked = torch.randn(2, 3, 16, 16)
    unmasked = torch.randn_like(masked)
    action = torch.randn(2, jepa_cfg.action_dim)
    z_hat, z_target, z_context = jepa(masked, unmasked, action)
    jepa_metrics = jepa.training_loss(masked, unmasked, action)

    core_prior = bus.forward_up(z_context)
    query = core_prior.mean(dim=1)
    ram = FaissTensorRAM(d_ram=dca_cfg.d_ram, use_faiss=False)
    ram_vectors = np.random.randn(128, dca_cfg.d_ram).astype(np.float32)
    ram.add(ram_vectors)
    retrieved = ram.weighted_retrieve(query.detach().float().cpu().numpy(), k=4)
    facts = torch.from_numpy(retrieved.vectors)

    canvas = torch.randn(2, 8, dca_cfg.d_core)
    t = torch.rand(2)
    dca_out = dca(query, facts, canvas, t)
    tokens = dca.decoder.decode(canvas.to(dtype=dca_cfg.torch_dtype))

    router = EntropyRouter(threshold_tau=0.0)
    decision = router.decide([z_hat, z_hat + 0.01, z_hat - 0.01])

    print(
        json.dumps(
            {
                "z_context": list(z_context.shape),
                "core_prior": list(core_prior.shape),
                "ram_indices": retrieved.indices.tolist(),
                "memory": list(dca_out["memory"].shape),
                "eps_pred": list(dca_out["eps_pred"].shape),
                "decoded_tokens": list(tokens.shape),
                "router_power": decision.power_state,
                "jepa_loss": float(jepa_metrics["loss"].detach().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
