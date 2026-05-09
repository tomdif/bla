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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--overfit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = DCAConfig.tiny()
    model = DCAEngine(config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    fixed_query = torch.randn(args.batch_size, config.d_core, device=device)
    fixed_facts = torch.randn(args.batch_size, args.facts, config.d_ram, device=device)
    fixed_x0 = torch.randn(args.batch_size, args.seq_len, config.d_core, device=device)

    for step in range(args.steps):
        if args.overfit:
            query, facts, x0 = fixed_query, fixed_facts, fixed_x0
        else:
            query = torch.randn(args.batch_size, config.d_core, device=device)
            facts = torch.randn(args.batch_size, args.facts, config.d_ram, device=device)
            x0 = torch.randn(args.batch_size, args.seq_len, config.d_core, device=device)
        metrics = model.training_loss(query, x0, facts=facts)
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
        optimizer.step()
        print(json.dumps({"step": step, "loss": float(metrics["loss"].detach().cpu())}))


if __name__ == "__main__":
    main()
