"""Sample visualization helpers.

Saves grids of images / diffusion-canvas trajectories / RAM attention as
PNGs without requiring matplotlib (writes raw RGB via PIL when available;
otherwise falls back to a numpy .npy dump).
"""

from __future__ import annotations

from typing import Optional

import torch


def _as_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images.float().clamp(-1, 1)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    return images


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4) -> None:
    """images: [N, 3, H, W] in [-1, 1] (or [0, 1]). Writes a tiled PNG."""

    if images.ndim != 4:
        raise ValueError(f"expected [N, 3, H, W], got {tuple(images.shape)}")
    n, c, h, w = images.shape
    ncol = (n + nrow - 1) // nrow
    grid = torch.zeros(c, ncol * h, nrow * w, dtype=torch.uint8)
    arr = _as_uint8(images)
    for i in range(n):
        r, q = divmod(i, nrow)
        grid[:, r * h : (r + 1) * h, q * w : (q + 1) * w] = arr[i]
    try:
        from PIL import Image  # type: ignore

        rgb = grid.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(rgb).save(path)
    except Exception:
        import numpy as np

        np.save(path.rsplit(".", 1)[0] + ".npy", grid.cpu().numpy())


def canvas_diffusion_trajectory(
    diffusion,
    memory_state: torch.Tensor,
    decoder,
    seq_len: int,
    steps: int = 8,
    prior: Optional[torch.Tensor] = None,
    t_start: float = 1.0,
) -> dict:
    """Run a denoising trajectory and return per-step canvas / token snapshots.

    Returns {steps: list of dicts with 't', 'canvas_norm', 'token_argmax'}.
    """

    diffusion.eval()
    decoder.eval()
    snapshots = []
    dtype = next(diffusion.parameters()).dtype
    device = next(diffusion.parameters()).device
    batch = memory_state.shape[0]
    if prior is None:
        x = torch.randn(batch, seq_len, diffusion.d_core, device=device, dtype=dtype)
    else:
        x = (1.0 - t_start) * prior.to(device=device, dtype=dtype) + t_start * torch.randn_like(prior).to(device=device, dtype=dtype)
    ts = torch.linspace(t_start, 0.0, steps + 1, device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(steps):
            t_now = ts[i].expand(batch)
            v = diffusion(x, t_now, memory_state)
            tokens = decoder.decode(x)
            snapshots.append({
                "t": float(ts[i]),
                "canvas_norm": float(x.float().norm(dim=-1).mean()),
                "token_argmax": tokens.cpu(),
            })
            dt = (ts[i + 1] - ts[i]).to(dtype=dtype)
            x = x + dt * v
    snapshots.append({
        "t": 0.0,
        "canvas_norm": float(x.float().norm(dim=-1).mean()),
        "token_argmax": decoder.decode(x).cpu(),
    })
    return {"steps": snapshots}
