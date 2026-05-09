from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system2_dca import DCAConfig, DCAEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1B DCA diffusion smoke training loop.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--facts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DCAConfig.tiny()
    model = DCAEngine(config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    for step in range(args.steps):
        query = torch.randn(args.batch_size, config.d_core)
        facts = torch.randn(args.batch_size, args.facts, config.d_ram)
        x0 = torch.randn(args.batch_size, args.seq_len, config.d_core)
        metrics = model.training_loss(query, facts, x0)
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        optimizer.step()
        print(json.dumps({"step": step, "loss": float(metrics["loss"].detach().cpu())}))


if __name__ == "__main__":
    main()
