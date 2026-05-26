"""SAM 2.1 → ObjectFileBatch bridge for the Layer-C / CCT pipeline.

The real-world equivalent of `OFJEPAObjectFiles.observe(frame)`. Lets
the Layer-C scripts (CausalRelationHead + SlotExistenceHead + CCT)
consume SAM 2.1 perception output through the same interface they
already use for OF-JEPA, so the BLA-Forge hardware swap is pure sensor
replacement (matching the BF-0 doctrine).

Two pieces:

  SAMObjectFiles      — the adapter: takes any object that exposes
                         detect(frame_idx) -> list[FiducialDetection]
                         (real SAMPerception or a mock), maintains
                         per-obj_id learned id_keys + state_value encoder,
                         emits ObjectFileBatch matching OF-JEPA's contract.

  SyntheticOcclusionPerception
                      — drop-in mock for offline validation: reads env
                         visibility + entity positions and synthesizes
                         FiducialDetections per frame. Confidence = 1.0
                         when an entity is visible, 0.0 when hidden.
                         Lets us run the SAM path end-to-end on the
                         occluded navigate env without real hardware.

Lifetime: build once with n_files=len(seeds), then call observe(frame_idx)
per step. Batch size is fixed at 1 — SAM 2.1 is single-video by design,
and BLA-Forge hardware deployment is single-camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np
import torch
from torch import nn

from bla.forge.fiducials import FiducialDetection
from system1_jepa.of_jepa.interfaces import ObjectFileBatch


class DetectionSource(Protocol):
    def detect(self, frame_idx: int) -> list[FiducialDetection]: ...


class SAMObjectFiles(nn.Module):
    """Adapter from SAM-2.1-shaped perception to ObjectFileBatch.

    Each known object has a stable obj_id (set at construction by the
    list of seed obj_ids). The adapter holds a persistent learned id_key
    embedding per obj_id (identity-as-address) plus a small encoder
    from `(center_px_normalized, bbox_area_normalized)` to `state_value`.
    Confidence is passed through directly from the perception backend.
    """

    def __init__(
        self,
        perception: DetectionSource,
        obj_ids: List[int],
        image_size: int,
        slot_dim: int = 64,
    ):
        super().__init__()
        self.perception = perception
        self.obj_ids = list(obj_ids)
        self.image_size = float(image_size)
        self.slot_dim = slot_dim
        self.id_dim = slot_dim // 2
        self.state_dim = slot_dim // 2
        self.n_files = len(self.obj_ids)
        # obj_id -> position in the file bank.
        self._obj_idx = {oid: i for i, oid in enumerate(self.obj_ids)}
        # Persistent learned identity addresses per file. These are NOT
        # observation-dependent — that's the OF-JEPA identity-as-address
        # invariant. (Initialized to small random; will train with the
        # Layer-C heads via the standard pretrain loop.)
        self.id_keys = nn.Parameter(
            torch.randn(self.n_files, self.id_dim) * 0.02,
        )
        # State-value encoder: 4 features per detection (cx, cy, w, h
        # normalized to [0, 1]) projected to state_dim.
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.Linear(64, self.state_dim),
        )
        # Per-file last-known state, in case a frame has missing detection
        # for an object (we keep its last state but stamp confidence=0).
        self._last_state: Optional[torch.Tensor] = None
        self._frame_idx: int = -1

    def reset_episode(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError(
                f"SAMObjectFiles is single-video by design (BLA-Forge "
                f"hardware = single camera); got batch_size={batch_size}"
            )
        self._last_state = torch.zeros(
            1, self.n_files, self.state_dim, device=self.id_keys.device,
        )
        self._frame_idx = -1

    @torch.no_grad()
    def _features_from_detection(self, det: FiducialDetection) -> torch.Tensor:
        """Convert a FiducialDetection to a 4-D normalized feature vector."""
        cx, cy = det.center_px
        xs = det.pixel_corners[:, 0]
        ys = det.pixel_corners[:, 1]
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        f = np.array(
            [cx / self.image_size, cy / self.image_size,
              w / self.image_size, h / self.image_size],
            dtype=np.float32,
        )
        return torch.from_numpy(f).to(self.id_keys.device)

    def observe(self, frame_idx: int) -> ObjectFileBatch:
        """Per-frame update.

        Reads detections at `frame_idx` from the perception backend,
        encodes them into state_values, and returns an ObjectFileBatch
        with the persistent id_keys + per-file confidences.
        """
        if self._last_state is None:
            raise RuntimeError("Call reset_episode() before observe().")
        detections = self.perception.detect(frame_idx)
        feats = torch.zeros(self.n_files, 4, device=self.id_keys.device)
        confidences = torch.zeros(self.n_files, device=self.id_keys.device)
        visibility = torch.zeros(self.n_files, device=self.id_keys.device)
        seen = set()
        for det in detections:
            slot = self._obj_idx.get(det.id)
            if slot is None:
                continue
            feats[slot] = self._features_from_detection(det)
            confidences[slot] = float(det.confidence)
            visibility[slot] = float(det.confidence > 0.0)
            seen.add(det.id)
        # Files with no detection this frame: keep last state, confidence 0.
        new_state = self.state_encoder(feats).unsqueeze(0)        # [1, N, state_dim]
        for oid in self.obj_ids:
            if oid not in seen:
                slot = self._obj_idx[oid]
                new_state[0, slot] = self._last_state[0, slot]
        self._last_state = new_state.detach()
        self._frame_idx = frame_idx

        return ObjectFileBatch(
            id_keys=self.id_keys.unsqueeze(0).expand(1, -1, -1).contiguous(),
            state_values=new_state,
            confidences=confidences.unsqueeze(0),
            frame_idx=self._frame_idx,
            visibility=visibility.unsqueeze(0),
        )


class SyntheticOcclusionPerception:
    """Mock perception that synthesizes FiducialDetections from env state.

    For each step the caller passes the env, we read positions of agent +
    targets and emit detections with confidence = 1.0 for visible
    entities and 0.0 for occluded ones. Lets the Layer-C pipeline test
    its SAM-path plumbing without real video / real SAM.

    Usage:
      perception = SyntheticOcclusionPerception(env, mock_box_half=15.0)
      perception.snap(frame_idx)        # call once per env step
      detections = perception.detect(frame_idx)
    """
    def __init__(self, env, mock_box_half: float = 8.0):
        self.env = env
        self.mock_box_half = float(mock_box_half)
        self._snapshots: dict[int, list[FiducialDetection]] = {}

    def snap(self, frame_idx: int) -> None:
        """Capture detections derived from the env's CURRENT state."""
        spec = self.env.spec
        # Agent: always visible. obj_id = 0.
        ax = float(self.env.x[0]); ay = float(self.env.y[0])
        dets = [self._make_det(0, ax + spec.patch_size / 2, ay + spec.patch_size / 2,
                                 confidence=1.0)]
        # Targets: visible only when not in the hidden window. obj_id = 1..n_targets.
        is_hidden = self.env._is_hidden_step()
        for i in range(spec.n_targets):
            tx = float(self.env.tx[0, i]); ty = float(self.env.ty[0, i])
            conf = 0.0 if is_hidden else 1.0
            dets.append(self._make_det(
                i + 1, tx + spec.patch_size / 2, ty + spec.patch_size / 2,
                confidence=conf,
            ))
        self._snapshots[frame_idx] = dets

    def detect(self, frame_idx: int) -> list[FiducialDetection]:
        return self._snapshots.get(frame_idx, [])

    def _make_det(self, obj_id: int, cx: float, cy: float,
                   confidence: float) -> FiducialDetection:
        h = self.mock_box_half
        corners = np.array(
            [[cx - h, cy - h], [cx + h, cy - h],
              [cx + h, cy + h], [cx - h, cy + h]],
            dtype=np.float64,
        )
        return FiducialDetection(
            id=int(obj_id), family="synthetic_occlusion",
            pixel_corners=corners,
            center_px=np.array([cx, cy], dtype=np.float64),
            confidence=float(confidence),
        )
