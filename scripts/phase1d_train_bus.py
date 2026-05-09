from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from latent_bus import TokenlessLatentBus, contrastive_infonce
from system1_jepa import BLAJEPAModel, JEPAConfig
from system2_dca import DCAConfig, DCAEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1D latent-bus alignment smoke training.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--facts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    jepa_cfg = JEPAConfig.tiny()
    dca_cfg = DCAConfig.tiny()
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    dca = DCAEngine(dca_cfg).to(device)
    bus = TokenlessLatentBus(
        d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core, dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(bus.parameters(), lr=args.lr)

    for step in range(args.steps):
        image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        action = torch.randn(args.batch_size, jepa_cfg.action_dim, device=device)
        with torch.no_grad():
            z_hat, _, _ = jepa(image, action)
            query = bus.forward_up(z_hat).mean(dim=1)
            facts = torch.randn(args.batch_size, args.facts, dca_cfg.d_ram, device=device)
            memory = dca.working_memory(query, facts)

        core_jepa = bus.forward_up(z_hat).mean(dim=1)
        loss = contrastive_infonce(core_jepa, memory)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(bus.parameters(), args.grad_clip)
        optimizer.step()
        print(json.dumps({"step": step, "alignment_loss": float(loss.detach().cpu())}))


if __name__ == "__main__":
    main()
