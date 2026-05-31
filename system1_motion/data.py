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
    """frame_stack=S: x_t is the S most-recent frames stacked on channels (3*S),
    so the encoder can see velocity (needed for a 2nd-order env — A1 with S=1
    failed Gate 0 because single-frame action-conditioned prediction can't reward
    position-encoding without velocity). x_{t+1} is the stack ending at t+1."""

    def __init__(self, npz_path, frame_stack=1):
        d = np.load(npz_path)
        self.frames = d["frames"]            # uint8 [N,3,H,W]
        self.actions = d["actions"].astype(np.float32)
        self.pos = d["pos"].astype(np.float32)
        self.ep = d["ep_id"].astype(np.int64)
        self.img_px = int(d["img_px"]) if "img_px" in d else self.frames.shape[-1]
        self.S = int(frame_stack)
        # valid i: i and i+1 same episode, AND the S-frame history ending at i+1
        # stays within the same episode (so x_t and x_{t+1} are well-defined).
        ok = (self.ep[:-1] == self.ep[1:])
        for k in range(1, self.S):
            ok[k:] &= (self.ep[:-1 - k] == self.ep[k:-1])      # j-k same episode as j
        self.pairs = np.where(ok)[0]
        self.pairs = self.pairs[self.pairs >= self.S - 1]

    def __len__(self):
        return len(self.pairs)

    def _stack(self, end):                                     # frames[end-S+1 .. end] -> [3S,H,W]
        idx = list(range(end - self.S + 1, end + 1))
        return torch.from_numpy(self.frames[idx].reshape(-1, *self.frames.shape[2:])).float() / 255.0

    def __getitem__(self, i):
        j = int(self.pairs[i])
        return (self._stack(j), torch.from_numpy(self.actions[j]),
                self._stack(j + 1), torch.from_numpy(self.pos[j]))

    def eval_arrays(self, frac=0.2):
        """Held-out (stacked frames [M,3S,H,W], positions) for the Gate-0 probe —
        last `frac` of valid stacks, distinct from training batches."""
        n = len(self.pairs); k = int((1 - frac) * n)
        ends = self.pairs[k:]
        X = np.stack([self.frames[e - self.S + 1:e + 1].reshape(-1, *self.frames.shape[2:])
                      for e in ends]).astype(np.uint8)
        return X, self.pos[ends]
