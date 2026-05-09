"""Synthetic temporal data: a bright square moves on a black canvas.

The action is the (dx, dy) displacement applied per step (plus optional
zero-padding to action_dim). This gives the temporal predictor a real
structure to learn — given the current frame and a known displacement,
the next frame is determined.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MovingPatchSpec:
    image_size: int = 16
    patch_size: int = 2
    horizon: int = 4
    history: int = 2
    action_dim: int = 32

    @property
    def total_frames(self) -> int:
        return self.history + self.horizon


def _draw(canvas: torch.Tensor, x: torch.Tensor, y: torch.Tensor, p: int) -> None:
    """Draw a p×p bright patch at integer (x, y) on a [B, C, H, W] canvas."""

    batch, _, h, w = canvas.shape
    x = x.clamp(0, w - p).long()
    y = y.clamp(0, h - p).long()
    for b in range(batch):
        canvas[b, :, y[b] : y[b] + p, x[b] : x[b] + p] = 1.0


def make_moving_patch_episodes(
    spec: MovingPatchSpec,
    batch_size: int,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate moving-patch sequences.

    Returns:
      image_history: [B, history, 3, H, W]
      action_chunks: [B, horizon, 1, action_dim]   (chunk_size=1; first 2 dims are (dx,dy))
      image_future:  [B, horizon, 3, H, W]
    """

    device = device or torch.device("cpu")
    h = w = spec.image_size
    p = spec.patch_size
    g = generator

    x = torch.randint(0, w - p + 1, (batch_size,), device=device, generator=g).float()
    y = torch.randint(0, h - p + 1, (batch_size,), device=device, generator=g).float()

    history = torch.zeros(batch_size, spec.history, 3, h, w, device=device)
    future = torch.zeros(batch_size, spec.horizon, 3, h, w, device=device)
    actions = torch.zeros(batch_size, spec.horizon, 1, spec.action_dim, device=device)

    history_actions = torch.randint(-2, 3, (spec.history, batch_size, 2), device=device, generator=g).float()
    for t in range(spec.history):
        canvas = torch.zeros(batch_size, 3, h, w, device=device)
        _draw(canvas, x, y, p)
        history[:, t] = canvas
        x = (x + history_actions[t, :, 0]).clamp(0, w - p)
        y = (y + history_actions[t, :, 1]).clamp(0, h - p)

    for t in range(spec.horizon):
        dxy = torch.randint(-2, 3, (batch_size, 2), device=device, generator=g).float()
        actions[:, t, 0, 0:2] = dxy
        x = (x + dxy[:, 0]).clamp(0, w - p)
        y = (y + dxy[:, 1]).clamp(0, h - p)
        canvas = torch.zeros(batch_size, 3, h, w, device=device)
        _draw(canvas, x, y, p)
        future[:, t] = canvas

    return history, actions, future
