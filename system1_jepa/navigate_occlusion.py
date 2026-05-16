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
    moving_distractors: bool = False     # distractors random-walk each step
    distractor_move_max: float = 1.0     # per-step random displacement bound
    partial_observability: bool = False  # mask obs outside a circle around the agent
    obs_radius: float = 8.0              # observation circle radius (in pixels)
    rendered_patches: bool = True        # informational flag for the manifest; observations are always pixel-rendered in this env
    perceptual_noise: float = 0.0        # Gaussian pixel noise σ; >0 turns "rendered image" into a noisy perception channel
    randomize_colors: bool = False       # if True, sample fresh entity-colour channels each reset


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

    def _move_distractors(self) -> None:
        """Each distractor takes a random step bounded by `distractor_move_max`,
        clamped to the canvas. Called once per env step when
        `moving_distractors` is set."""
        if self.dx_pos is None or not self.spec.moving_distractors:
            return
        n = self.spec.n_distractors
        max_xy = self.spec.image_size - self.spec.patch_size
        m = self.spec.distractor_move_max
        # Uniform random walk in [-m, m] per axis. Using the env's RNG so
        # reproducibility holds.
        dx = (torch.rand(self.batch_size, n, device=self.device, generator=self.gen) * 2 - 1) * m
        dy = (torch.rand(self.batch_size, n, device=self.device, generator=self.gen) * 2 - 1) * m
        self.dx_pos = (self.dx_pos + dx).clamp(0, max_xy)
        self.dy_pos = (self.dy_pos + dy).clamp(0, max_xy)

    def _apply_partial_observability(self, canvas: torch.Tensor) -> torch.Tensor:
        """Zero out pixels outside a circle of radius `obs_radius` around the
        agent. Implemented as a per-pixel mask; not differentiable, which is
        fine because env outputs are not on the autograd path."""
        if not self.spec.partial_observability:
            return canvas
        size = self.spec.image_size
        radius = self.spec.obs_radius
        y = torch.arange(size, device=self.device).float()
        x = torch.arange(size, device=self.device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H, W]
        # Agent center (top-left of patch + half patch).
        cx = (self.x + self.spec.patch_size / 2).view(-1, 1, 1)
        cy = (self.y + self.spec.patch_size / 2).view(-1, 1, 1)
        dist_sq = (xx.unsqueeze(0) - cx) ** 2 + (yy.unsqueeze(0) - cy) ** 2
        mask = (dist_sq <= radius * radius).float()                 # [B, H, W]
        return canvas * mask.unsqueeze(1)                            # broadcast over channels

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
        canvas = canvas.clamp(0, 1)
        canvas = self._apply_partial_observability(canvas)
        # Phase 4A: simulate image-like perceptual noise. Sampled via the
        # env's RNG so reproducibility holds. Applied *after* the partial-
        # observability mask so masked-out pixels stay exactly zero (which
        # is what an honest occluder would do — no noise in unobserved
        # regions). Final clamp keeps values in valid pixel range.
        if self.spec.perceptual_noise > 0.0:
            noise = torch.empty_like(canvas).normal_(
                mean=0.0, std=self.spec.perceptual_noise, generator=self.gen,
            )
            if self.spec.partial_observability:
                # Re-apply the visibility mask to the noise so we don't leak
                # signal into the masked region.
                noise = self._apply_partial_observability(noise)
            canvas = (canvas + noise).clamp(0, 1)
        return canvas

    def step(self, dxy: torch.Tensor, success_bonus: float = 5.0):
        # Override to let distractors move when `moving_distractors` is on.
        # We move them *before* the parent's step so the agent's
        # post-step observation reflects the new distractor positions.
        if self.spec.moving_distractors:
            self._move_distractors()
        return super().step(dxy, success_bonus=success_bonus)

    def visibility_mask(self) -> torch.Tensor:
        """[B] bool — True if the current step shows targets, False if hidden.
        The agent never sees this; use it for evaluation/diagnostics."""
        is_hidden = self._is_hidden_step()
        return torch.full((self.batch_size,), not is_hidden,
                           dtype=torch.bool, device=self.device)
