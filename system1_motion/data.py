"""Cached transition dataset for substrate training.

Decouples training from live rendering: render_dataset.py produces a .npz once
(frames + actions + GT positions + episode ids), and training reads it. Yields
(x_t, a_t, x_{t+1}, pos_t) pairs that don't cross episode boundaries.

npz schema: frames[N,3,H,W] uint8, actions[N,da] f32, pos[N,2] f32 (pixels),
            ep_id[N] int, img_px int.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.frames = d["frames"]            # uint8 [N,3,H,W]
        self.actions = d["actions"].astype(np.float32)
        self.pos = d["pos"].astype(np.float32)
        self.ep = d["ep_id"].astype(np.int64)
        self.img_px = int(d["img_px"]) if "img_px" in d else self.frames.shape[-1]
        self.pairs = np.where(self.ep[:-1] == self.ep[1:])[0]   # i valid iff same episode as i+1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        j = int(self.pairs[i])
        xt = torch.from_numpy(self.frames[j]).float() / 255.0
        xtp1 = torch.from_numpy(self.frames[j + 1]).float() / 255.0
        return xt, torch.from_numpy(self.actions[j]), xtp1, torch.from_numpy(self.pos[j])

    def eval_arrays(self, frac=0.2):
        """Held-out (frames, positions) for the Gate-0 probe — last `frac` of
        frames (deterministic, distinct from the pairs used in training batches)."""
        n = len(self.frames); k = int((1 - frac) * n)
        return self.frames[k:], self.pos[k:]
