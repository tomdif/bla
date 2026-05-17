"""OFJEPA wrapper — assembles the proposal encoder + object-file memory
into a video-level model usable by the trainer + substrate API.

Despite the file name, this is the top-level "model assembly" — the
predictor logic lives inside `ObjectFileMemory.step()` (sparse delta
+ Sinkhorn-matched proposal). This module just wires per-frame
encoding to memory updates across an episode and exposes the full
slot state (id_key + state_value) for downstream consumers.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .encoder import ProposalEncoder
from .object_file_memory import (
    OFJEPAConfig, ObjectFileMemory, ObjectFileMemoryV1,
)


class OFJEPA(nn.Module):
    """Object-File JEPA wrapper for video training and inference.

    Pulls per-frame proposals from the encoder, runs the object-file
    memory step-by-step across all T frames of an episode, and exposes:

        slot_states_per_frame: [T, B, N_files, id_dim + state_dim]
        assignments_per_frame: [T, B, N_files, n_proposals]

    The "full slot state" returned for compatibility with the existing
    identity probe is concat(id_key, state_value).
    """
    def __init__(self, image_size: int, cfg: OFJEPAConfig = OFJEPAConfig(),
                 version: str = "v0"):
        super().__init__()
        self.cfg = cfg
        self.version = version
        self.proposal_encoder = ProposalEncoder(image_size, cfg.proposal_dim)
        if version == "v1":
            self.memory = ObjectFileMemoryV1(cfg)
        else:
            self.memory = ObjectFileMemory(cfg)
        # Compatibility shim with existing trainer:
        self.id_dim = cfg.id_dim
        self.use_id_cons = False
        self.use_id_contrast = False
        self.mode = f"of_jepa_{version}"
        self.slot_dim = cfg.id_dim + cfg.state_dim
        self.n_slots = cfg.n_files
        self.slot_to_pos_aux = self.memory.slot_to_pos_aux

    def encode_video(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """video: [T, 3, H, W] → (slot_states [T, n_files, full_dim], proposals [T, n_props, prop_dim])

        For v1, also tracks per-frame visibility logits in
        self._last_vis_logits for the trainer to read out.
        """
        T = video.shape[0]
        proposals = self.proposal_encoder(video)
        memory = self.memory.init_episode(batch_size=1, device=video.device)
        slot_states = []
        vis_logits = []
        for t in range(T):
            memory, diag = self.memory.step(memory, proposals[t:t+1])
            full = torch.cat([memory["id_key"], memory["state_value"]], dim=-1)
            slot_states.append(full[0])
            if "visibility_logit" in diag:
                vis_logits.append(diag["visibility_logit"][0])
        self._last_vis_logits = torch.stack(vis_logits, dim=0) if vis_logits else None
        return torch.stack(slot_states, dim=0), proposals

    def encode_video_grad(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video)
