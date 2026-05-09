"""Real-data wrappers for B.L.A.

`CIFAR10Loader` returns batches of normalized 32x32 RGB tensors. If
torchvision is missing or the download fails, falls back to
`SyntheticImageLoader` which generates 32x32 RGB images of randomly
positioned, randomly colored shapes — non-trivial structure that JEPA
can actually learn features from, unlike pure Gaussian noise.

Both loaders implement the same iterator contract:
    next(loader) -> torch.Tensor [batch, 3, H, W] in roughly [-1, 1]

For temporal data, see `system1_jepa.synthetic.make_moving_patch_episodes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class ImageBatchSpec:
    image_size: int = 32
    channels: int = 3
    batch_size: int = 16


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / 255.0 - 0.5) * 2.0


class SyntheticImageLoader:
    """Random colored shape on a colored background. Works offline, no deps."""

    def __init__(self, spec: ImageBatchSpec, seed: int = 0):
        self.spec = spec
        self.gen = torch.Generator().manual_seed(seed)

    def _draw_one(self) -> torch.Tensor:
        s = self.spec.image_size
        bg_color = torch.rand(3, generator=self.gen) * 255
        img = bg_color[:, None, None].expand(3, s, s).clone()

        shape_size = torch.randint(s // 4, s // 2 + 1, (1,), generator=self.gen).item()
        x0 = torch.randint(0, s - shape_size + 1, (1,), generator=self.gen).item()
        y0 = torch.randint(0, s - shape_size + 1, (1,), generator=self.gen).item()
        fg_color = torch.rand(3, generator=self.gen) * 255
        shape_kind = torch.randint(0, 2, (1,), generator=self.gen).item()
        if shape_kind == 0:
            img[:, y0:y0 + shape_size, x0:x0 + shape_size] = fg_color[:, None, None]
        else:
            cy, cx = y0 + shape_size / 2, x0 + shape_size / 2
            r = shape_size / 2
            yy, xx = torch.meshgrid(torch.arange(s).float(), torch.arange(s).float(), indexing="ij")
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            for c in range(3):
                img[c][mask] = fg_color[c]
        return img

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        batch = torch.stack([self._draw_one() for _ in range(self.spec.batch_size)], dim=0)
        return _normalize(batch)


class CIFAR10Loader:
    """torchvision CIFAR-10 with on-disk caching. Falls back to synthetic on failure."""

    def __init__(
        self,
        spec: ImageBatchSpec,
        root: str = "runs/data/cifar10",
        train: bool = True,
        seed: int = 0,
        augment: bool = False,
    ):
        self.spec = spec
        self.fallback: Optional[SyntheticImageLoader] = None
        try:
            import os as _os

            import torchvision  # type: ignore

            tfs: list = [torchvision.transforms.Resize(spec.image_size)]
            if augment and train:
                tfs.append(torchvision.transforms.RandomCrop(spec.image_size, padding=4, padding_mode="reflect"))
                tfs.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
                tfs.append(torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            tfs.extend([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform = torchvision.transforms.Compose(tfs)
            cached = _os.path.exists(_os.path.join(root, "cifar-10-batches-py", "data_batch_1"))
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=not cached, transform=transform
            )
            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=spec.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                generator=torch.Generator().manual_seed(seed),
            )
            self._iter = iter(self.loader)
        except Exception:
            self.fallback = SyntheticImageLoader(spec, seed=seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self.fallback is not None:
            return next(self.fallback)
        try:
            images, _ = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            images, _ = next(self._iter)
        return images


def make_image_loader(
    spec: ImageBatchSpec,
    source: str = "auto",
    seed: int = 0,
    augment: bool = False,
) -> Iterator[torch.Tensor]:
    """source ∈ {auto, cifar10, synthetic}. auto = cifar10 with synthetic fallback."""

    if source == "synthetic":
        return iter(SyntheticImageLoader(spec, seed=seed))
    if source in ("cifar10", "auto"):
        return iter(CIFAR10Loader(spec, seed=seed, augment=augment))
    raise ValueError(f"unknown source: {source}")
