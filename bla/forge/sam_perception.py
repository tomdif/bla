"""BF-1.5 — SAM 2.1 perception module: drop-in replacement for mock_fiducials.

Per BF-0.7-0.9 calibration findings:
  - SAM 2.1 Hiera-Tiny click-prompted video predictor tracks objects to
    2 cm 3D pose accuracy when seeded with a click at the known initial
    pixel.
  - Full occlusion: SAM correctly emits a 0-area mask (abstains).
  - Partial occlusion: SAM tracks the visible portion with <5 px drift.
  - Latency: 27 FPS on H100 with Hiera-Tiny.

This module wraps SAM 2.1 so that each step produces FiducialDetection-
shaped output (center_px = mask centroid, pixel_corners = mask bbox).
That keeps the BF-0.2/0.3/1.x downstream pipeline identical — perception
is the only layer that changes.

Mock-first pattern (matches BF-0.1/0.2/0.3): a "mock_static" backend
returns deterministic detections without GPU; the "sam2.1" backend
requires CUDA + the meta sam2 package.

Empty masks (full occlusion) are represented as ZERO confidence
FiducialDetection objects rather than dropped entirely — that gives
downstream code a clean "object known to exist, currently not seen"
signal. Callers may filter them out before feeding into the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bla.forge.fiducials import FiducialDetection


@dataclass
class SAMSeed:
    """One object to track: an id (becomes the FiducialDetection.id) and
    the initial-frame pixel where it lives.

    Fields:
      obj_id    integer identifier (becomes FiducialDetection.id)
      pixel_uv  (u, v) tuple in pixels — the click prompt on frame 0
    """
    obj_id: int
    pixel_uv: tuple[float, float]


class SAMPerception:
    """SAM 2.1 video tracker that emits FiducialDetection-shaped output.

    Two backends:
      "mock_static" : deterministic per-frame detections with the seed
                      pixel as center_px and a fixed 30 px square as
                      pixel_corners; no GPU needed. Suitable for unit
                      tests + BF-1.x dev work before SAM is available.
      "sam2.1"      : real SAM 2.1 click-prompted video predictor.
                      Requires the meta sam2 package installed and CUDA
                      available. Calls predictor.init_state on the
                      provided video_path (JPEG folder) and propagates
                      offline for all frames, then serves cached masks.

    Lifetime: build once per video, then call detect(frame_idx) per step.
    """

    def __init__(
        self,
        *,
        video_path: Path | str,
        seeds: list[SAMSeed],
        backend: str = "mock_static",
        sam_model: str = "facebook/sam2.1-hiera-tiny",
        family: str = "sam2_track",
        device: str = "cuda",
    ):
        if not seeds:
            raise ValueError("SAMPerception requires at least one seed")
        if backend not in ("mock_static", "sam2.1"):
            raise ValueError(
                f"backend must be 'mock_static' or 'sam2.1', got {backend!r}")

        self.video_path = Path(video_path)
        self.seeds = list(seeds)
        self.backend = backend
        self.sam_model = sam_model
        self.family = family
        self.device = device

        self._mock_box_half = 15.0  # px — half-side of synthetic box

        # Pre-computed: {frame_idx: {obj_id: mask uint8 H x W}}
        self._masks: dict[int, dict[int, np.ndarray]] = {}
        # For mock backend: we don't have a real video; emit seeds as-is
        # for all frame_idx requested. Confidence is always 1.0 there.
        self._mock_mode = backend == "mock_static"

        if not self._mock_mode:
            self._build_sam_session()

    def _build_sam_session(self):
        """Load SAM 2.1, init the video session, seed clicks, batch-propagate."""
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        if not torch.cuda.is_available() and self.device == "cuda":
            raise RuntimeError(
                "SAMPerception sam2.1 backend requires CUDA. "
                "Set backend='mock_static' for CPU-only dev.")
        predictor = SAM2VideoPredictor.from_pretrained(
            self.sam_model, device=self.device)
        state = predictor.init_state(video_path=str(self.video_path))
        for seed in self.seeds:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=seed.obj_id,
                points=np.array([[seed.pixel_uv[0], seed.pixel_uv[1]]],
                                  dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
        with torch.inference_mode():
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                frame_masks = {}
                for i, oid in enumerate(obj_ids):
                    m = mask_logits[i]
                    if m.ndim == 3: m = m[0]
                    m_np = (m.float() > 0).cpu().numpy().astype(np.uint8)
                    frame_masks[int(oid)] = m_np
                self._masks[int(frame_idx)] = frame_masks

    def detect(self, frame_idx: int) -> list[FiducialDetection]:
        """Emit per-step detections (drop-in shape for mock_fiducials)."""
        if self._mock_mode:
            return [self._mock_detection(seed) for seed in self.seeds]
        return self._sam_detections(frame_idx)

    # ----- mock -----
    def _mock_detection(self, seed: SAMSeed) -> FiducialDetection:
        u, v = seed.pixel_uv
        h = self._mock_box_half
        corners = np.array([
            [u - h, v - h], [u + h, v - h],
            [u + h, v + h], [u - h, v + h],
        ], dtype=np.float64)
        return FiducialDetection(
            id=seed.obj_id, family=self.family,
            pixel_corners=corners,
            center_px=np.array([u, v], dtype=np.float64),
            confidence=1.0,
        )

    # ----- sam -----
    def _sam_detections(self, frame_idx: int) -> list[FiducialDetection]:
        frame_masks = self._masks.get(frame_idx, {})
        out = []
        for seed in self.seeds:
            mask = frame_masks.get(seed.obj_id)
            if mask is None or mask.sum() == 0:
                # Object known but currently not visible (full occlusion / OOF).
                # Emit a zero-confidence FiducialDetection at the last-known
                # seed pixel so downstream tracker can apply its missing-policy.
                out.append(self._mock_detection(seed))
                out[-1].confidence = 0.0
                continue
            ys, xs = np.where(mask)
            u_c, v_c = float(xs.mean()), float(ys.mean())
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            corners = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1],
            ], dtype=np.float64)
            # Confidence ∈ [0, 1] based on mask area; saturates at 500 px.
            confidence = float(min(1.0, mask.sum() / 500.0))
            out.append(FiducialDetection(
                id=seed.obj_id, family=self.family,
                pixel_corners=corners,
                center_px=np.array([u_c, v_c], dtype=np.float64),
                confidence=confidence,
            ))
        return out

    def __len__(self):
        """Number of frames in the propagated video (0 for mock)."""
        return len(self._masks)
