"""Occluded multi-target navigate environment.

Wraps `MultiTargetNavigateEnv` with a visible/hidden cycle. During the
"visible" window the canvas renders all targets normally; during the
"hidden" window the targets are *removed from the observation* (still
present in the env state for reward computation, but the agent does not
see them). Optionally adds a fixed pool of visual distractors that
appear in every frame as colour-coded patches so the agent can't trivially
solve the task by background subtraction.

Why this env: it's the smallest task where a representation must carry
target identity *through observations that no longer contain the target*.
Dense JEPA encodes the current frame only; under occlusion it loses the
target. Slot memory with sparse delta updates should keep the target
slot stable across the hidden window.

State during occlusion is ground-truth only used for evaluation and the
identity diagnostic — it is *not* fed to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from .navigate_env import MultiTargetNavigateEnv, MultiTargetNavigateSpec
from .synthetic import _draw


@dataclass
class OccludedNavigateSpec(MultiTargetNavigateSpec):
    visible_steps: int = 5      # K — frames where targets are rendered
    hidden_steps: int = 10      # J — frames where targets are removed from obs
    n_distractors: int = 0       # additional visual clutter patches that never matter for reward


class OccludedMultiTargetNavigateEnv(MultiTargetNavigateEnv):
    def __init__(self, spec: OccludedNavigateSpec, batch_size: int = 4,
                 device: torch.device | None = None, seed: int = 0):
        # Initialise distractor slots *before* super().__init__() because
        # the parent's __init__ calls reset()→observe() which expects
        # them to exist on self.
        self.dx_pos: torch.Tensor | None = None
        self.dy_pos: torch.Tensor | None = None
        super().__init__(spec=spec, batch_size=batch_size, device=device, seed=seed)
        self.spec: OccludedNavigateSpec = spec  # type: ignore[assignment]
        self._reset_distractors()

    def _reset_distractors(self) -> None:
        if self.spec.n_distractors <= 0:
            self.dx_pos = self.dy_pos = None
            return
        max_xy = self.spec.image_size - self.spec.patch_size
        n = self.spec.n_distractors
        self.dx_pos = torch.randint(
            0, max_xy + 1, (self.batch_size, n), device=self.device, generator=self.gen
        ).float()
        self.dy_pos = torch.randint(
            0, max_xy + 1, (self.batch_size, n), device=self.device, generator=self.gen
        ).float()

    def reset(self) -> torch.Tensor:
        out = super().reset()
        self._reset_distractors()
        return out

    def _is_hidden_step(self) -> bool:
        """True when the current step falls inside a hidden window."""
        cycle = self.spec.visible_steps + self.spec.hidden_steps
        if cycle == 0:
            return False
        return (self.t % cycle) >= self.spec.visible_steps

    def _draw_distractors(self, canvas: torch.Tensor) -> None:
        if self.dx_pos is None:
            return
        p = self.spec.patch_size
        for b in range(self.batch_size):
            for i in range(self.spec.n_distractors):
                x = int(self.dx_pos[b, i].item())
                y = int(self.dy_pos[b, i].item())
                # Distractors paint in the blue channel so the encoder
                # can in principle distinguish them from targets (red).
                canvas[b, 2, y : y + p, x : x + p] = 1.0

    def observe(self) -> torch.Tensor:
        canvas = torch.zeros(
            self.batch_size, 3, self.spec.image_size, self.spec.image_size,
            device=self.device,
        )
        _draw(canvas, self.x, self.y, self.spec.patch_size)
        canvas[:, 0, :, :] = 0.0
        if not self._is_hidden_step():
            canvas[:, 0, :, :] += self._all_targets_overlay() * 1.0
        self._draw_distractors(canvas)
        return canvas.clamp(0, 1)

    def visibility_mask(self) -> torch.Tensor:
        """[B] bool — True if the current step shows targets, False if hidden.
        The agent never sees this; use it for evaluation/diagnostics."""
        is_hidden = self._is_hidden_step()
        return torch.full((self.batch_size,), not is_hidden,
                           dtype=torch.bool, device=self.device)
