from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import BLAJEPAModel, JEPAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1A JEPA smoke pre-training loop.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = JEPAConfig.tiny()
    model = BLAJEPAModel(config)
    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    history = []
    for step in range(args.steps):
        masked = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
        unmasked = torch.randn_like(masked)
        action = torch.randn(args.batch_size, config.action_dim)
        metrics = model.training_loss(masked, unmasked, action)
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        optimizer.step()
        model.update_target_ema()
        history.append({k: float(v.detach().cpu()) for k, v in metrics.items()})
        print(json.dumps({"step": step, **history[-1]}))


if __name__ == "__main__":
    main()
