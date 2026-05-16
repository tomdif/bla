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
    randomize_colors: bool = False       # legacy alias for color_randomization (kept for back-compat)
    color_randomization: bool = False    # Phase-4B: sample random RGB per entity at each reset; breaks the channel-as-label shortcut
    background_randomization: bool = False  # Phase-4B: per-pixel random background canvas (low magnitude) sampled at each reset
    background_magnitude: float = 0.15   # max background pixel value when background_randomization is on


class OccludedMultiTargetNavigateEnv(MultiTargetNavigateEnv):
    def __init__(self, spec: OccludedNavigateSpec, batch_size: int = 4,
                 device: torch.device | None = None, seed: int = 0):
        # Initialise distractor slots + color buffers *before* super().__init__()
        # because the parent's __init__ calls reset()→observe() which expects
        # them to exist on self.
        self.dx_pos: torch.Tensor | None = None
        self.dy_pos: torch.Tensor | None = None
        self.agent_color: torch.Tensor | None = None
        self.target_colors: torch.Tensor | None = None
        self.distractor_colors: torch.Tensor | None = None
        self.background: torch.Tensor | None = None
        super().__init__(spec=spec, batch_size=batch_size, device=device, seed=seed)
        self.spec: OccludedNavigateSpec = spec  # type: ignore[assignment]
        self._reset_distractors()
        self._sample_colors_and_background()

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
        self._sample_colors_and_background()
        return out

    def _sample_colors_and_background(self) -> None:
        """At each reset, set the per-entity colors and (optionally) a
        random per-pixel background canvas. When `color_randomization`
        is off the colors default to the Phase-3/4A scheme: agent in
        green+blue (0, 1, 1), targets in red (1, 0, 0), distractors in
        blue (0, 0, 1). Phase 4B turns randomization on to break the
        channel-as-label shortcut.

        `randomize_colors` is kept as an alias for `color_randomization`
        for back-compatibility — either flag enables randomization."""
        do_random = self.spec.color_randomization or self.spec.randomize_colors
        B = self.batch_size
        nt = self.spec.n_targets
        nd = self.spec.n_distractors
        if do_random:
            self.agent_color = torch.rand(
                B, 3, device=self.device, generator=self.gen,
            )
            self.target_colors = torch.rand(
                B, nt, 3, device=self.device, generator=self.gen,
            )
            self.distractor_colors = torch.rand(
                B, nd, 3, device=self.device, generator=self.gen,
            ) if nd > 0 else None
        else:
            cyan = torch.tensor([0.0, 1.0, 1.0], device=self.device)
            red = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            blue = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            self.agent_color = cyan.unsqueeze(0).expand(B, 3).contiguous()
            self.target_colors = red.view(1, 1, 3).expand(B, nt, 3).contiguous()
            self.distractor_colors = (
                blue.view(1, 1, 3).expand(B, nd, 3).contiguous()
                if nd > 0 else None
            )
        if self.spec.background_randomization:
            self.background = (
                torch.rand(
                    B, 3, self.spec.image_size, self.spec.image_size,
                    device=self.device, generator=self.gen,
                ) * self.spec.background_magnitude
            )
        else:
            self.background = None

    def _draw_colored(self, canvas: torch.Tensor, x: torch.Tensor,
                       y: torch.Tensor, color: torch.Tensor) -> None:
        """Overwrite a patch on `canvas` ([B, 3, H, W]) at (x, y) with
        a per-batch color ([B, 3])."""
        p = self.spec.patch_size
        size = self.spec.image_size
        for b in range(canvas.shape[0]):
            xi = max(0, min(size - p, int(x[b].item())))
            yi = max(0, min(size - p, int(y[b].item())))
            for ch in range(3):
                canvas[b, ch, yi:yi + p, xi:xi + p] = color[b, ch]

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
        # Lazy color init: the parent's __init__ calls reset()→observe()
        # before our own __init__ has populated the color buffers; on
        # that very first call we sample them on demand.
        if self.agent_color is None:
            self._sample_colors_and_background()
        # Build canvas: random background (if enabled) else zeros, then
        # overwrite per-entity colored patches in the order agent →
        # targets (if visible) → distractors. The order matters when
        # entities overlap; later draws win.
        if self.background is not None:
            canvas = self.background.clone()
        else:
            canvas = torch.zeros(
                self.batch_size, 3,
                self.spec.image_size, self.spec.image_size,
                device=self.device,
            )
        self._draw_colored(canvas, self.x, self.y, self.agent_color)
        if not self._is_hidden_step():
            for i in range(self.spec.n_targets):
                self._draw_colored(
                    canvas, self.tx[:, i], self.ty[:, i],
                    self.target_colors[:, i],
                )
        if self.dx_pos is not None and self.distractor_colors is not None:
            for i in range(self.spec.n_distractors):
                self._draw_colored(
                    canvas, self.dx_pos[:, i], self.dy_pos[:, i],
                    self.distractor_colors[:, i],
                )
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
