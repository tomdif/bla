from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import BLAJEPAModel, ImageBatchSpec, JEPAConfig, make_image_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1A JEPA smoke pre-training loop.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--data", choices=["randn", "synthetic", "cifar10", "auto"], default="randn")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Reuse a single batch every step to verify the loop can fit it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = JEPAConfig.tiny()
    model = BLAJEPAModel(config).to(device)
    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    if args.data == "randn":
        loader = None
    else:
        spec = ImageBatchSpec(image_size=args.image_size, batch_size=args.batch_size)
        loader = make_image_loader(spec, source=args.data, seed=args.seed)

    def next_batch() -> torch.Tensor:
        if loader is None:
            return torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        return next(loader).to(device)

    fixed_image = next_batch()
    fixed_action = torch.randn(args.batch_size, config.action_dim, device=device)

    for step in range(args.steps):
        if args.overfit:
            image, action = fixed_image, fixed_action
        else:
            image = next_batch()
            action = torch.randn(args.batch_size, config.action_dim, device=device)
        metrics = model.training_loss(image, action)
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        model.update_target_ema()
        line = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        print(json.dumps({"step": step, **line}))


if __name__ == "__main__":
    main()
