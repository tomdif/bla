from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from latent_bus import EntropyRouter, TokenlessLatentBus, contrastive_infonce
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
    bus = TokenlessLatentBus(d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core, dtype=torch.float32)

    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, jepa_cfg.action_dim)
    z_hat, z_target, mask = jepa(image, action)
    jepa_metrics = jepa.training_loss(image, action, mask=mask)

    core_prior = bus.forward_up(z_hat)
    query = core_prior.mean(dim=1)
    ram = FaissTensorRAM(d_ram=dca_cfg.d_ram, use_faiss=False)
    ram_vectors = np.random.randn(128, dca_cfg.d_ram).astype(np.float32)
    ram.add(ram_vectors)
    retrieved = ram.weighted_retrieve(query.detach().float().cpu().numpy(), k=4)
    facts = torch.from_numpy(retrieved.vectors)

    canvas = torch.randn(2, 8, dca_cfg.d_core)
    t = torch.rand(2)
    dca_out = dca(query, canvas, t, facts=facts)
    jepa_prior = bus.forward_up(z_hat).detach()[:, :8] if z_hat.shape[1] >= 8 else None
    sampled = dca.diffusion.sample(
        dca_out["memory"], seq_len=8, steps=4,
        prior=jepa_prior, t_start=0.5 if jepa_prior is not None else 1.0,
    )
    tokens = dca.decoder.decode(sampled)

    router = EntropyRouter(threshold_tau=0.0)
    decision = router.decide([z_hat, z_hat + 0.01, z_hat - 0.01])

    align_loss = contrastive_infonce(core_prior, dca_out["velocity"]).item()

    print(
        json.dumps(
            {
                "z_hat": list(z_hat.shape),
                "z_target": list(z_target.shape),
                "core_prior": list(core_prior.shape),
                "ram_indices": retrieved.indices.tolist(),
                "memory": list(dca_out["memory"].shape),
                "velocity": list(dca_out["velocity"].shape),
                "sampled_tokens": list(tokens.shape),
                "router_power": decision.power_state,
                "jepa_loss": float(jepa_metrics["loss"].detach().cpu()),
                "alignment_loss": align_loss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
