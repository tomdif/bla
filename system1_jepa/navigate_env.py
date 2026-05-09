"""Navigate-to-target environment over the moving-patch world.

A bright patch sits at (x, y) on a black canvas. Each step the agent
applies a 2D displacement (dx, dy) and is rewarded by the negative
Euclidean distance to a target (x*, y*). Success = within distance
threshold within `max_steps`.

The environment is fully torch and stateless across batch elements, so
it can drive parallel CEM rollouts on CPU at a reasonable speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .synthetic import _draw


@dataclass
class NavigateSpec:
    image_size: int = 16
    patch_size: int = 2
    max_steps: int = 8
    success_radius: float = 1.5
    action_dim: int = 32
    move_max: float = 2.0


class NavigateEnv:
    def __init__(self, spec: NavigateSpec, batch_size: int = 4, device: torch.device | None = None, seed: int = 0):
        self.spec = spec
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.tx: torch.Tensor
        self.ty: torch.Tensor
        self.t = 0
        self.reset()

    def _new_pos(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_xy = self.spec.image_size - self.spec.patch_size
        x = torch.randint(0, max_xy + 1, (self.batch_size,), device=self.device, generator=self.gen).float()
        y = torch.randint(0, max_xy + 1, (self.batch_size,), device=self.device, generator=self.gen).float()
        return x, y

    def reset(self) -> torch.Tensor:
        self.x, self.y = self._new_pos()
        self.tx, self.ty = self._new_pos()
        self.t = 0
        return self.observe()

    def observe(self) -> torch.Tensor:
        canvas = torch.zeros(self.batch_size, 3, self.spec.image_size, self.spec.image_size, device=self.device)
        _draw(canvas, self.x, self.y, self.spec.patch_size)
        canvas[:, 0, :, :] += self._target_overlay() * 0.5
        return canvas.clamp(0, 1)

    def _target_overlay(self) -> torch.Tensor:
        ov = torch.zeros(self.batch_size, self.spec.image_size, self.spec.image_size, device=self.device)
        for b in range(self.batch_size):
            tx = int(self.tx[b].item())
            ty = int(self.ty[b].item())
            ov[b, ty : ty + self.spec.patch_size, tx : tx + self.spec.patch_size] = 1.0
        return ov

    def step(
        self,
        dxy: torch.Tensor,
        shaped_reward: bool = True,
        success_bonus: float = 5.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """dxy: [B, 2] displacement. Returns (obs, reward, done).

        Reward modes:
          shaped_reward=True (default): r_t = (d_{t-1} - d_t) + success_bonus * I[reached].
            Positive every step the agent moves toward the carrot, negative if away.
            Continuous gradient at every step instead of one sparse signal at the end.
          shaped_reward=False: r_t = -d_t (raw distance, sparse-ish).
        """

        prev_dist = ((self.x - self.tx) ** 2 + (self.y - self.ty) ** 2).sqrt()
        dx = dxy[:, 0].clamp(-self.spec.move_max, self.spec.move_max)
        dy = dxy[:, 1].clamp(-self.spec.move_max, self.spec.move_max)
        max_xy = self.spec.image_size - self.spec.patch_size
        self.x = (self.x + dx).clamp(0, max_xy)
        self.y = (self.y + dy).clamp(0, max_xy)
        dist = ((self.x - self.tx) ** 2 + (self.y - self.ty) ** 2).sqrt()
        reached = dist < self.spec.success_radius
        if shaped_reward:
            reward = (prev_dist - dist) + success_bonus * reached.float()
        else:
            reward = -dist
        done = reached | (self.t >= self.spec.max_steps - 1)
        self.t += 1
        return self.observe(), reward, done

    def expert_action(self) -> torch.Tensor:
        """Optimal greedy action: clip(target - position, ±move_max). Shape [B, 2]."""
        dx = (self.tx - self.x).clamp(-self.spec.move_max, self.spec.move_max)
        dy = (self.ty - self.y).clamp(-self.spec.move_max, self.spec.move_max)
        return torch.stack([dx, dy], dim=1)

    def encode_action(self, dxy: torch.Tensor) -> torch.Tensor:
        """Pack (dx, dy) into the first two slots of a zero-padded action vector."""

        action = torch.zeros(dxy.shape[0], self.spec.action_dim, device=self.device)
        action[:, 0] = dxy[:, 0]
        action[:, 1] = dxy[:, 1]
        return action

    def decode_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., :2]
