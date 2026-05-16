"""MOVi-A/B/C/.../F dataset loader for the Phase 7 (Kubric) decisive eval.

Reads .npz cache files produced by `scripts/movi_extract_local.py`. Each
episode .npz contains:
    video             [T, H, W, 3] uint8
    image_positions   [E, T, 2]    float32  -- projected 2D pixel coords
    visibility        [E, T]       uint8    -- 1 if visible, else 0
    color_label       [E]          int32
    shape_label       [E]          int32
    material_label    [E]          int32
    size_label        [E]          int32
    num_instances     scalar       int32
    video_name        scalar       str

The loader pads to a fixed max-entities (default 10) so we can batch
heterogeneous episodes. Padded slots get visibility=0 and entity_id=-1.

Identity is the instance index within an episode. The dataset has no
cross-episode identity (object 0 in episode A != object 0 in episode B).
For Phase 7 evaluation, identity stability is measured per-episode.

Returned PyTorch tensors per batch:
    video          [B, T, 3, H, W] float32 in [0,1]
    positions      [B, T, E_max, 2]  float32
    visibility     [B, T, E_max]     bool
    entity_ids     [B, E_max]        long  (-1 for padded slots)
    attrs          [B, E_max, A]     float32  (one-hot color+shape+material+size)
    n_instances    [B]               long
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Attribute label counts in MOVi-A (verified via tfds.builder().info).
# Other MOVi variants may differ; we use the union and one-hot encode.
COLOR_CLASSES = 8
SHAPE_CLASSES = 3
MATERIAL_CLASSES = 2
SIZE_CLASSES = 2
ATTR_DIM = COLOR_CLASSES + SHAPE_CLASSES + MATERIAL_CLASSES + SIZE_CLASSES


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float32)
    valid = (idx >= 0) & (idx < n)
    out[valid, idx[valid]] = 1.0
    return out


def encode_attrs(color: np.ndarray, shape: np.ndarray,
                  material: np.ndarray, size: np.ndarray) -> np.ndarray:
    """[E] int labels → [E, ATTR_DIM] one-hot concatenation."""
    return np.concatenate([
        _onehot(color, COLOR_CLASSES),
        _onehot(shape, SHAPE_CLASSES),
        _onehot(material, MATERIAL_CLASSES),
        _onehot(size, SIZE_CLASSES),
    ], axis=-1)


@dataclass
class MoviSpec:
    cache_dir: str
    max_entities: int = 10
    frames: int = 24
    image_size: int = 128
    normalize_positions: bool = True  # divide pixel coords by image_size


class MoviDataset(Dataset):
    """Loads MOVi episodes from an extracted .npz cache."""

    def __init__(self, spec: MoviSpec):
        self.spec = spec
        manifest_path = Path(spec.cache_dir) / "manifest.json"
        with open(manifest_path) as f:
            mf = json.load(f)
        self.episodes: List[Dict] = mf["episodes"]
        self.frame_size = mf.get("frame_size", spec.image_size)
        # Filter out episodes with more instances than max_entities (rare).
        self.episodes = [e for e in self.episodes
                         if e["num_instances"] <= spec.max_entities]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.episodes[idx]
        path = Path(self.spec.cache_dir) / meta["file"]
        data = np.load(path)

        video = data["video"].astype(np.float32) / 255.0           # [T, H, W, 3]
        T, H, W, _ = video.shape
        E = int(data["num_instances"])
        E_max = self.spec.max_entities

        image_positions = data["image_positions"].astype(np.float32)  # [E, T, 2]
        visibility = data["visibility"].astype(bool)                  # [E, T]

        # Pad to E_max along the entity axis.
        pad_pos = np.zeros((E_max, T, 2), dtype=np.float32)
        pad_vis = np.zeros((E_max, T), dtype=bool)
        pad_attr = np.zeros((E_max, ATTR_DIM), dtype=np.float32)
        entity_ids = -np.ones((E_max,), dtype=np.int64)

        if E > 0:
            pad_pos[:E] = image_positions
            pad_vis[:E] = visibility
            attrs = encode_attrs(
                data["color_label"], data["shape_label"],
                data["material_label"], data["size_label"],
            )
            pad_attr[:E] = attrs
            entity_ids[:E] = np.arange(E)

        # Normalize positions to [0, 1].
        if self.spec.normalize_positions:
            pad_pos[..., 0] /= max(W, 1)
            pad_pos[..., 1] /= max(H, 1)

        # Reorder to [T, E_max, *]
        pos_te = np.transpose(pad_pos, (1, 0, 2))     # [T, E_max, 2]
        vis_te = np.transpose(pad_vis, (1, 0))         # [T, E_max]

        # Channel-first video
        v = np.transpose(video, (0, 3, 1, 2))  # [T, 3, H, W]

        return {
            "video": torch.from_numpy(v),
            "positions": torch.from_numpy(pos_te),
            "visibility": torch.from_numpy(vis_te),
            "entity_ids": torch.from_numpy(entity_ids),
            "attrs": torch.from_numpy(pad_attr),
            "n_instances": torch.tensor(E, dtype=torch.long),
        }


def make_movi_loader(spec: MoviSpec, batch_size: int = 8,
                      shuffle: bool = True, num_workers: int = 2) -> "torch.utils.data.DataLoader":
    """Convenience factory."""
    from torch.utils.data import DataLoader
    ds = MoviDataset(spec)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
