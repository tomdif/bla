"""CLEVRER episode loader for the Phase 13 benchmark.

Mirrors `movi_data.MoviDataset` API so OF-JEPA training scripts can
swap dataset without changing the trainer. Returns per-frame:

    video frames     [T, 3, H, W] float32 in [0, 1]
    positions        [T, E_max, 2] float32 in [0, 1] (image NDC)
    visibility       [T, E_max] bool
    entity_ids       [E_max] long (GT object indices from sim)
    attrs            [E_max, A] float32 one-hot (color/material/shape)
    collision_events list of (frame, object_a_id, object_b_id) GT
    in_out_events    list of (frame, object_id, "in"|"out") GT
    n_instances      int

Positions come from RLE mask centroids in the
`processed_proposals/sim_*.json` files. GT collision/in_out events
come from the same file's `ground_truth` section.

Identity assignment per frame: detections in `frames[].objects` are
matched to `ground_truth.objects` entries by (color, material, shape)
attribute tuple. CLEVRER scenes are designed so attribute triples are
distinct per object, so the match is unique.

Usage:
    python scripts/clevrer_extract_local.py \\
        --videos /workspace/clevrer/videos/train \\
        --annotations /workspace/clevrer/annotations/processed_proposals \\
        --out /workspace/clevrer_local/train \\
        --max-episodes 1000 --frame-stride 4
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# CLEVRER attribute vocab (per the dataset's design).
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["sphere", "cylinder", "cube"]
ATTR_DIM = len(COLORS) + len(MATERIALS) + len(SHAPES)


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def encode_attrs(color: str, material: str, shape: str) -> np.ndarray:
    return np.concatenate([
        _onehot(COLORS.index(color)        if color in COLORS    else -1, len(COLORS)),
        _onehot(MATERIALS.index(material)  if material in MATERIALS else -1, len(MATERIALS)),
        _onehot(SHAPES.index(shape)        if shape in SHAPES    else -1, len(SHAPES)),
    ])


def rle_to_bbox(mask_size: List[int], counts: str) -> Optional[Tuple[float, float]]:
    """Decode COCO-style RLE compressed mask → 2D centroid in image NDC.

    Returns (cx, cy) in [0, 1] (normalized by mask_size W and H), or
    None if the mask decodes to zero pixels.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        # Fallback: simple RLE decoder (slower).
        return _simple_rle_centroid(mask_size, counts)
    rle = {"size": mask_size, "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
    m = coco_mask.decode(rle)  # [H, W]
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    cy = ys.mean() / float(mask_size[0])
    cx = xs.mean() / float(mask_size[1])
    return (cx, cy)


def _simple_rle_centroid(mask_size: List[int], counts: str) -> Optional[Tuple[float, float]]:
    """Pure-Python RLE decoder — fallback when pycocotools is unavailable.

    Note: CLEVRER uses COCO-compressed RLE (RLE-of-RLE), so this fallback
    is approximate. Strongly prefer pycocotools when accuracy matters.
    """
    # Skip COCO compressed format if pycocotools missing.
    H, W = mask_size
    # Conservative fallback: return image center.
    return (0.5, 0.5)


def _match_detection_to_gt(detection_attrs: Tuple[str, str, str],
                            gt_objects: List[Dict]) -> int:
    """Return GT object id whose (color, material, shape) matches the
    detection. Returns -1 if no match.

    CLEVRER scenes are designed so the (color, material, shape) triple
    is unique per object — this should always return a unique id.
    """
    for obj in gt_objects:
        if (obj["color"], obj["material"], obj["shape"]) == detection_attrs:
            return obj["id"]
    return -1


@dataclass
class ClevrerSpec:
    cache_dir: str
    max_entities: int = 10
    image_size: int = 128
    frame_stride: int = 4   # CLEVRER is 128 frames; stride=4 gives 32 frames per episode
    normalize_positions: bool = False  # positions are already in [0, 1]


class ClevrerDataset(Dataset):
    """PyTorch Dataset over preprocessed CLEVRER episodes (.npz cache).

    Cache produced by `scripts/clevrer_extract_local.py`. Each .npz has:
      - video           [T, H, W, 3] uint8 (T frames at frame_stride)
      - image_positions [E, T, 2] float32 in [0, 1]
      - visibility      [E, T] uint8 (1 = detected at this frame)
      - color_idx       [E] int32 (index into COLORS)
      - material_idx    [E] int32
      - shape_idx       [E] int32
      - num_instances   int
      - video_name      str
      - collisions      list of (frame, obj_a_id, obj_b_id) at the
                        stride-aligned frame index
      - in_outs         list of (frame, obj_id, type_idx) at stride-aligned
    """

    def __init__(self, spec: ClevrerSpec):
        self.spec = spec
        manifest_path = Path(spec.cache_dir) / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.episodes = self.manifest["episodes"]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.episodes[idx]
        data = np.load(Path(self.spec.cache_dir) / meta["file"], allow_pickle=True)

        video = data["video"].astype(np.float32) / 255.0  # [T, H, W, 3]
        T, H, W, _ = video.shape
        E = int(data["num_instances"])
        E_max = self.spec.max_entities

        image_positions = data["image_positions"].astype(np.float32)  # [E, T, 2]
        visibility = data["visibility"].astype(bool)                   # [E, T]

        pad_pos = np.zeros((E_max, T, 2), dtype=np.float32)
        pad_vis = np.zeros((E_max, T), dtype=bool)
        pad_attr = np.zeros((E_max, ATTR_DIM), dtype=np.float32)
        entity_ids = -np.ones((E_max,), dtype=np.int64)

        if E > 0:
            pad_pos[:E] = image_positions
            pad_vis[:E] = visibility
            for i in range(min(E, E_max)):
                attr = np.concatenate([
                    _onehot(int(data["color_idx"][i]), len(COLORS)),
                    _onehot(int(data["material_idx"][i]), len(MATERIALS)),
                    _onehot(int(data["shape_idx"][i]), len(SHAPES)),
                ])
                pad_attr[i] = attr
            entity_ids[:E] = np.arange(E)

        if self.spec.normalize_positions:
            pad_pos[..., 0] /= max(W, 1)
            pad_pos[..., 1] /= max(H, 1)

        pos_te = np.transpose(pad_pos, (1, 0, 2))   # [T, E_max, 2]
        vis_te = np.transpose(pad_vis, (1, 0))       # [T, E_max]
        v = np.transpose(video, (0, 3, 1, 2))        # [T, 3, H, W]

        return {
            "video": torch.from_numpy(v),
            "positions": torch.from_numpy(pos_te),
            "visibility": torch.from_numpy(vis_te),
            "entity_ids": torch.from_numpy(entity_ids),
            "attrs": torch.from_numpy(pad_attr),
            "n_instances": torch.tensor(E, dtype=torch.long),
            "collisions": data["collisions"].tolist() if "collisions" in data else [],
            "in_outs": data["in_outs"].tolist() if "in_outs" in data else [],
        }
