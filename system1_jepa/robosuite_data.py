"""Robosuite rollout loader for Phase 14 action-conditioned OF-JEPA.

Mirrors `movi_data.MoviDataset` API. Returns per-frame video + actions
+ GT object positions (cubeA, cubeB, end-effector). 3 "objects" total
(2 cubes + 1 robot end-effector) — slot-attention learns to bind to
all three.

Per-episode .npz comes from `scripts/robosuite_collect_rollouts.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RobosuiteSpec:
    cache_dir: str
    image_size: int = 128
    max_entities: int = 3   # cubeA, cubeB, end-effector


class RobosuiteDataset(Dataset):
    """PyTorch Dataset over preprocessed robosuite Stack rollouts.

    Each episode .npz has: video, actions, cube_a_pos, cube_b_pos,
    eef_pos, rewards.

    Returns batch dict with:
      video        [T, 3, H, W] float32 in [0,1]
      positions    [T, E_max, 3] float32 — 3D world positions
                   (3 entities: cubeA, cubeB, eef)
      visibility   [T, E_max] bool — always True (no occlusion modeling)
      entity_ids   [E_max] long (0=cubeA, 1=cubeB, 2=eef)
      attrs        [E_max, 1] float32 — entity-type one-hot
      actions      [T, action_dim] float32
      n_instances  int (=3)
    """
    def __init__(self, spec: RobosuiteSpec):
        self.spec = spec
        manifest_path = Path(spec.cache_dir) / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.episodes = self.manifest["episodes"]
        self.action_dim = self.manifest["action_dim"]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.episodes[idx]
        data = np.load(Path(self.spec.cache_dir) / meta["file"], allow_pickle=True)
        video = data["video"].astype(np.float32) / 255.0          # [T, H, W, 3]
        T = video.shape[0]
        E_max = self.spec.max_entities

        # Stack the 3 entity positions: cubeA, cubeB, eef → [T, 3, 3]
        positions_3d = np.stack([
            data["cube_a_pos"], data["cube_b_pos"], data["eef_pos"],
        ], axis=1).astype(np.float32)   # [T, 3, 3]

        # For 2D probing, project to (x, y) ignoring z. Center-normalize.
        positions_2d = positions_3d[..., :2]   # [T, 3, 2]
        # Robosuite's workspace x,y ∈ approx [-0.3, 0.3]. Normalize to [0, 1].
        positions_2d = (positions_2d + 0.3) / 0.6
        positions_2d = np.clip(positions_2d, 0.0, 1.0)

        # Pad to E_max (=3, no padding needed for Stack).
        pad_pos = np.zeros((T, E_max, 2), dtype=np.float32)
        pad_vis = np.ones((T, E_max), dtype=bool)
        pad_attr = np.zeros((E_max, 3), dtype=np.float32)
        for i in range(min(3, E_max)):
            pad_attr[i, i] = 1.0
        if 3 <= E_max:
            pad_pos[:, :3] = positions_2d

        v = np.transpose(video, (0, 3, 1, 2))    # [T, 3, H, W]
        return {
            "video": torch.from_numpy(v),
            "positions": torch.from_numpy(pad_pos),
            "visibility": torch.from_numpy(pad_vis),
            "entity_ids": torch.from_numpy(np.arange(E_max, dtype=np.int64)),
            "attrs": torch.from_numpy(pad_attr),
            "actions": torch.from_numpy(data["actions"].astype(np.float32)),
            "n_instances": torch.tensor(3, dtype=torch.long),
        }
