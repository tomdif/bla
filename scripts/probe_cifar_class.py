"""Train a linear classifier on frozen JEPA features → CIFAR-10 class label.

This is the honest test of whether the JEPA pretrain learned useful features.
If accuracy is >> 10% (chance), the encoder is doing real work. If at chance,
the pretrain config needs to be revisited.

Usage:
    python3 scripts/probe_cifar_class.py \\
        --checkpoint runs/jepa_cifar10_6gpu/final.pt \\
        --epochs 5 --output runs/jepa_cifar10_6gpu/probe_class.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import BLAJEPAModel, JEPAConfig, pool_patch_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="runs/data/cifar10")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    config = JEPAConfig(
        d_jepa=cfg_dict["d"],
        in_channels=3,
        patch_size=cfg_dict["patch_size"],
        encoder_depth=cfg_dict["encoder_depth"],
        encoder_heads=cfg_dict["encoder_heads"],
        encoder_mlp_ratio=4.0,
        predictor_depth=cfg_dict["predictor_depth"],
        predictor_heads=cfg_dict["predictor_heads"],
        action_dim=cfg_dict["d"],
        ema_tau=cfg_dict["ema_tau"],
        sigreg_weight=cfg_dict["sigreg_weight"],
        target_mask_ratio=cfg_dict["target_ratio"],
        dtype="float32",
    )
    model = BLAJEPAModel(config).to(device)
    model.target_encoder.load_state_dict(ckpt["target"])
    encoder = model.target_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    import torchvision

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg_dict["image_size"]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    probe = nn.Linear(config.d_jepa, 10).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    amp_dtype = torch.bfloat16

    history: list[dict] = []
    for epoch in range(args.epochs):
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                z, _, _ = encoder(images)
            features = pool_patch_tokens(z).float().detach()
            logits = probe(features)
            loss = F.cross_entropy(logits, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += float(loss) * images.shape[0]
            train_correct += int((logits.argmax(dim=-1) == targets).sum())
            train_seen += int(images.shape[0])

        probe.eval()
        test_correct = 0
        test_seen = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    z, _, _ = encoder(images)
                features = pool_patch_tokens(z).float()
                logits = probe(features)
                test_correct += int((logits.argmax(dim=-1) == targets).sum())
                test_seen += int(images.shape[0])

        report = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_seen, 1),
            "train_accuracy": train_correct / max(train_seen, 1),
            "test_accuracy": test_correct / max(test_seen, 1),
        }
        print(json.dumps(report), flush=True)
        history.append(report)

    summary = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "epochs": args.epochs,
        "best_test_accuracy": max(r["test_accuracy"] for r in history),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({"event": "summary", **summary}), flush=True)


if __name__ == "__main__":
    main()
