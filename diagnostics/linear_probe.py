"""Linear probe: train a single linear layer on frozen JEPA features.

If a linear classifier on the encoder's output can predict a property
(patch position, color channel, motion), the encoder has learned to
represent that property. This is the cheapest signal that JEPA features
are doing real work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LinearProbeReport:
    train_loss: float
    eval_loss: float
    eval_accuracy: float | None
    epochs: int


def train_linear_probe(
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    image_loader,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    n_classes: int | None = None,
    epochs: int = 5,
    steps_per_epoch: int = 32,
    eval_steps: int = 16,
    lr: float = 3e-3,
    device: torch.device | None = None,
) -> LinearProbeReport:
    """Train a linear probe on frozen features.

    feature_fn:    image_batch [B, C, H, W] -> features [B, D]    (frozen)
    target_fn:     image_batch [B, C, H, W] -> targets  [B] or [B, T]
    n_classes:     if set, classification (CE loss + accuracy); else regression (MSE).
    """

    device = device or torch.device("cpu")
    sample = next(image_loader).to(device)
    feat = feature_fn(sample).detach()
    d = feat.shape[-1]
    output_dim = n_classes if n_classes is not None else None
    if output_dim is None:
        target_sample = target_fn(sample)
        output_dim = target_sample.shape[-1] if target_sample.ndim == 2 else 1
    probe = nn.Linear(d, output_dim).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        last_loss = 0.0
        for _ in range(steps_per_epoch):
            images = next(image_loader).to(device)
            with torch.no_grad():
                features = feature_fn(images).detach()
            targets = target_fn(images).to(device)
            logits = probe(features)
            if n_classes is not None:
                loss = F.cross_entropy(logits, targets.long())
            else:
                if targets.ndim == 1:
                    targets = targets.unsqueeze(-1)
                loss = F.mse_loss(logits, targets.float())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            last_loss = float(loss.detach())

    eval_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for _ in range(eval_steps):
            images = next(image_loader).to(device)
            features = feature_fn(images)
            targets = target_fn(images).to(device)
            logits = probe(features)
            if n_classes is not None:
                eval_loss += float(F.cross_entropy(logits, targets.long()))
                correct += int((logits.argmax(dim=-1) == targets.long()).sum())
                seen += int(targets.shape[0])
            else:
                if targets.ndim == 1:
                    targets = targets.unsqueeze(-1)
                eval_loss += float(F.mse_loss(logits, targets.float()))
    eval_loss /= max(eval_steps, 1)
    accuracy = correct / seen if seen else None

    return LinearProbeReport(
        train_loss=last_loss,
        eval_loss=eval_loss,
        eval_accuracy=accuracy,
        epochs=epochs,
    )
