"""OF-JEPA v0 as a reusable System-1 substrate for the BLA stack.

The user-locked architecture from the Phase 7→8 falsification arc has
the following invariants:

  - Identity lives in a persistent memory ADDRESS (id_proto), not in
    slot content. See [[feedback-identity-as-address]].
  - Prediction and assignment are SEPARATE training signals. See
    [[feedback-prediction-vs-assignment]].
  - Identity-conditioned metrics are PRIMARY for evaluation;
    anonymous Hungarian is a secondary diagnostic. See
    [[feedback-identity-conditioned-metrics]].

This module exposes that architecture as a stable substrate API for
the broader BLA stack (System-2, planning, retrieval, verification).

System-2 should NOT see raw slots. It should see **object files**:
structured records with persistent id_keys, dynamic state_values, and
visibility/confidence beliefs. The four-method API:

  observe(frame_t)        → ObjectFileBatch         (per-frame update)
  predict(k_steps=0)      → ObjectFileBatch         (forward prediction)
  read(query="all")       → structured dict         (System-2 readable)
  metrics()               → identity-conditioned diagnostics

The locked v0 architecture has no transition model, so predict(k > 0)
is currently a no-op that returns the most recent observation. v1's
transition_model is shelved and can be revived if a streaming/missing-
detection regime makes it useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import OFJEPA
from .object_file_memory import OFJEPAConfig


@dataclass
class ObjectFileBatch:
    """One observation's worth of object-file state.

    Fields are batch-shaped so the substrate composes with multi-env
    rollouts. For single-env use, batch_size = 1.
    """
    id_keys:       torch.Tensor   # [B, N_files, id_dim] — persistent identity addresses
    state_values:  torch.Tensor   # [B, N_files, state_dim] — dynamic content
    confidences:   torch.Tensor   # [B, N_files] — Sinkhorn match confidence in [0, 1]
    frame_idx:     int            # which frame in the current episode this state reflects
    visibility:    Optional[torch.Tensor] = None  # [B, N_files] — v1's belief, None for v0

    @property
    def full_slot(self) -> torch.Tensor:
        """Concatenated [id_key, state_value] view — for compatibility with
        existing pipelines that consumed raw slots."""
        return torch.cat([self.id_keys, self.state_values], dim=-1)

    @property
    def n_files(self) -> int:
        return self.id_keys.shape[1]


class OFJEPAObjectFiles(nn.Module):
    """The locked OF-JEPA v0 primitive, exposed as a narrow substrate API.

    Usage by System-2:

        substrate = OFJEPAObjectFiles(checkpoint_path=..., device=...)
        substrate.reset_episode(batch_size=1)
        for frame in video:
            ofb = substrate.observe(frame)
            tokens_for_system2 = bridge.project_per_file(ofb)
            ...

    The substrate is intentionally stateful (carries memory across
    observe() calls). Call reset_episode() to clear it.
    """

    def __init__(
        self,
        image_size: int = 128,
        n_files: int = 12,
        slot_dim: int = 128,
        version: str = "v0",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        cfg = OFJEPAConfig(
            n_files=n_files,
            id_dim=slot_dim // 2,
            state_dim=slot_dim // 2,
            proposal_dim=slot_dim,
        )
        self.of_jepa = OFJEPA(image_size=image_size, cfg=cfg, version=version)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            self.of_jepa.load_state_dict(state, strict=False)
        self.of_jepa.to(device)
        self.device = device
        self._memory_state: Optional[Dict] = None
        self._frame_idx: int = -1
        self._batch_size: int = 0
        # Cache of recent metrics, populated by external eval and read back.
        self._metric_cache: Dict = {}

    @property
    def id_dim(self) -> int:
        return self.of_jepa.cfg.id_dim

    @property
    def state_dim(self) -> int:
        return self.of_jepa.cfg.state_dim

    @property
    def n_files(self) -> int:
        return self.of_jepa.cfg.n_files

    # ---------- Lifecycle ----------------------------------------------------

    def reset_episode(self, batch_size: int = 1) -> None:
        """Start a new episode. Clears memory so identity bindings restart."""
        self._memory_state = self.of_jepa.memory.init_episode(
            batch_size=batch_size, device=self.device,
        )
        self._frame_idx = -1
        self._batch_size = batch_size

    # ---------- The four-method substrate API --------------------------------

    @torch.no_grad()
    def observe(self, frame: torch.Tensor) -> ObjectFileBatch:
        """Per-frame update.

        frame: [B, 3, H, W] or [3, H, W] (will be unsqueezed).
        Returns the post-observation object-file state.

        Internally: encode → proposals → memory.step → returns new state.
        """
        if self._memory_state is None:
            raise RuntimeError("Call reset_episode() before observe().")
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        proposals = self.of_jepa.proposal_encoder(frame)
        new_memory, diag = self.of_jepa.memory.step(self._memory_state, proposals)
        self._memory_state = new_memory
        self._frame_idx += 1

        confidence = self._derive_confidence(diag)
        visibility = diag.get("visibility_logit", None)
        if visibility is not None:
            visibility = torch.sigmoid(visibility)

        return ObjectFileBatch(
            id_keys=new_memory["id_key"],
            state_values=new_memory["state_value"],
            confidences=confidence,
            frame_idx=self._frame_idx,
            visibility=visibility,
        )

    @torch.no_grad()
    def predict(self, k_steps: int = 0) -> ObjectFileBatch:
        """Forward-predict the object-file state k steps ahead WITHOUT observation.

        For OF-JEPA v0 (no transition model), only k_steps == 0 is meaningful;
        it returns the most recently observed state. k_steps > 0 raises
        NotImplementedError on v0 — revive v1's transition_model first.
        """
        if self._memory_state is None:
            raise RuntimeError("Call reset_episode() before predict().")
        if k_steps == 0:
            return ObjectFileBatch(
                id_keys=self._memory_state["id_key"],
                state_values=self._memory_state["state_value"],
                confidences=torch.ones(
                    self._batch_size, self.n_files, device=self.device,
                ),
                frame_idx=self._frame_idx,
            )
        if self.of_jepa.version != "v1":
            raise NotImplementedError(
                f"predict(k_steps={k_steps}) requires the v1 transition_model "
                "(currently shelved per Phase 9B). Reactivate v1 or supply "
                "an action-conditioned predictor before calling with k > 0."
            )
        # v1 path: roll the transition_model forward k_steps without observation.
        state = self._memory_state["state_value"]
        for _ in range(k_steps):
            state = state + 0.05 * self.of_jepa.memory.transition(state)
            state = F.layer_norm(state, (self.state_dim,))
        return ObjectFileBatch(
            id_keys=self._memory_state["id_key"],
            state_values=state,
            confidences=torch.zeros(
                self._batch_size, self.n_files, device=self.device,
            ),  # zero confidence: this is an unobserved forecast
            frame_idx=self._frame_idx + k_steps,
        )

    @torch.no_grad()
    def read(self, query: str = "all") -> Dict:
        """System-2 readable structured-state dump. Serializable.

        Queries:
          "all"           — full object-file batch
          "id_keys"       — just the persistent addresses (per-file identity)
          "state_values"  — just the dynamic content
          "active"        — files with confidence > 0.5
        """
        if self._memory_state is None:
            raise RuntimeError("Call reset_episode() before read().")
        id_keys = self._memory_state["id_key"]
        state_values = self._memory_state["state_value"]
        if query == "id_keys":
            return {"id_keys": id_keys.detach().cpu()}
        if query == "state_values":
            return {"state_values": state_values.detach().cpu()}
        if query == "all":
            return {
                "id_keys": id_keys.detach().cpu(),
                "state_values": state_values.detach().cpu(),
                "frame_idx": self._frame_idx,
                "n_files": self.n_files,
            }
        if query == "active":
            return {
                "id_keys": id_keys.detach().cpu(),
                "state_values": state_values.detach().cpu(),
                "active_mask": torch.ones(
                    self._batch_size, self.n_files, dtype=torch.bool,
                ),  # v0 has no per-file active gate; all files active
            }
        raise ValueError(f"Unknown query: {query!r}")

    def metrics(self) -> Dict:
        """Identity-conditioned diagnostics from the most recent eval pass.

        Populated externally by the evaluator after running
        identity_conditioned_position_eval. Returns empty dict if no eval
        has been run since the substrate was instantiated.
        """
        return dict(self._metric_cache)

    def cache_metrics(self, metrics: Dict) -> None:
        """External hook: evaluator calls this to deposit fresh diagnostics."""
        self._metric_cache = dict(metrics)

    # ---------- Helpers ------------------------------------------------------

    @staticmethod
    def _derive_confidence(diag: Dict) -> torch.Tensor:
        """Map memory.step diagnostics to per-file confidence in [0, 1].

        For v0: confidence = max-over-proposals of assignment row. High
        confidence means a clear best-match proposal exists.
        For v1: confidence = 1 - null_match (already computed and exposed
        as match_confidence in the diagnostics).
        """
        if "match_confidence" in diag:
            return diag["match_confidence"].squeeze(-1)
        if "assignment" in diag:
            # v0: max-row from Sinkhorn matrix.
            return diag["assignment"].max(dim=-1).values
        # Fallback: ones.
        bsz, n_files = (
            diag["change_mask"].shape[0], diag["change_mask"].shape[1]
        ) if "change_mask" in diag else (1, 1)
        return torch.ones(bsz, n_files)


def per_file_project(ofb: ObjectFileBatch, bus: "TokenlessLatentBus") -> torch.Tensor:
    """Bridge adapter: project EACH object file through the latent bus
    independently. System-2 receives [B, N_files, d_core] — a sequence of
    tokens, one per object file, NOT a single pooled vector.

    This is the architectural commitment: object-file structure flows
    through to System-2 instead of being averaged away.
    """
    full = ofb.full_slot                                  # [B, N_files, slot_dim]
    B, N, D = full.shape
    flat = full.reshape(B * N, D)
    projected = bus.forward_up(flat)                       # [B*N, d_core]
    return projected.reshape(B, N, -1)
