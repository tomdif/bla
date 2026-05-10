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


@dataclass
class MultiTargetNavigateSpec:
    image_size: int = 32
    patch_size: int = 2
    n_targets: int = 3
    max_steps: int = 24
    success_radius: float = 1.5
    action_dim: int = 32
    move_max: float = 2.0


class MultiTargetNavigateEnv:
    """Visit N targets within max_steps. Targets are visually identical and
    persistent on the canvas — visited-state is hidden. BC should fail because
    the observation alone can't tell which targets remain to be visited.

    Action: (dx, dy) displacement.
    Observation: rendered canvas with agent + N persistent target markers.
    Reward: shaped (-distance to nearest unvisited) + bonus on visiting an
            unvisited target. Repeated visits to a visited target give zero.
    Done: visited all N, or t >= max_steps.

    State (hidden from observation): which of the N targets has been visited.
    """

    def __init__(
        self,
        spec: MultiTargetNavigateSpec,
        batch_size: int = 4,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        self.spec = spec
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.tx: torch.Tensor  # [B, n_targets]
        self.ty: torch.Tensor
        self.visited: torch.Tensor  # [B, n_targets] bool
        self.t = 0
        self.reset()

    def _new_pos_single(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_xy = self.spec.image_size - self.spec.patch_size
        x = torch.randint(0, max_xy + 1, (self.batch_size,), device=self.device, generator=self.gen).float()
        y = torch.randint(0, max_xy + 1, (self.batch_size,), device=self.device, generator=self.gen).float()
        return x, y

    def _new_target_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_xy = self.spec.image_size - self.spec.patch_size
        n = self.spec.n_targets
        tx = torch.randint(0, max_xy + 1, (self.batch_size, n), device=self.device, generator=self.gen).float()
        ty = torch.randint(0, max_xy + 1, (self.batch_size, n), device=self.device, generator=self.gen).float()
        return tx, ty

    def reset(self) -> torch.Tensor:
        self.x, self.y = self._new_pos_single()
        self.tx, self.ty = self._new_target_set()
        self.visited = torch.zeros(self.batch_size, self.spec.n_targets, dtype=torch.bool, device=self.device)
        self.t = 0
        return self.observe()

    def observe(self) -> torch.Tensor:
        """Canvas: agent in white-ish (channel 1+2), all targets in red — visually identical regardless of visited state."""
        canvas = torch.zeros(self.batch_size, 3, self.spec.image_size, self.spec.image_size, device=self.device)
        _draw(canvas, self.x, self.y, self.spec.patch_size)
        canvas[:, 0, :, :] = 0.0  # agent shows in green/blue only
        canvas[:, 0, :, :] += self._all_targets_overlay() * 1.0
        return canvas.clamp(0, 1)

    def _all_targets_overlay(self) -> torch.Tensor:
        ov = torch.zeros(self.batch_size, self.spec.image_size, self.spec.image_size, device=self.device)
        p = self.spec.patch_size
        for b in range(self.batch_size):
            for i in range(self.spec.n_targets):
                tx = int(self.tx[b, i].item())
                ty = int(self.ty[b, i].item())
                ov[b, ty : ty + p, tx : tx + p] = 1.0
        return ov

    def _distances_to_targets(self) -> torch.Tensor:
        """Distance from agent to each target, [B, n_targets]."""
        dx = self.tx - self.x.unsqueeze(1)
        dy = self.ty - self.y.unsqueeze(1)
        return (dx ** 2 + dy ** 2).sqrt()

    def step(
        self,
        dxy: torch.Tensor,
        success_bonus: float = 5.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """dxy: [B, 2] displacement. Reward = (prev nearest-unvisited dist - new) + bonus on hitting unvisited target."""

        prev_dists = self._distances_to_targets()
        # for unvisited targets, take min distance; visited contribute nothing
        prev_unvisited = prev_dists.masked_fill(self.visited, float("inf"))
        prev_min, _ = prev_unvisited.min(dim=1)

        dx = dxy[:, 0].clamp(-self.spec.move_max, self.spec.move_max)
        dy = dxy[:, 1].clamp(-self.spec.move_max, self.spec.move_max)
        max_xy = self.spec.image_size - self.spec.patch_size
        self.x = (self.x + dx).clamp(0, max_xy)
        self.y = (self.y + dy).clamp(0, max_xy)

        new_dists = self._distances_to_targets()
        # newly-reached unvisited targets
        within = new_dists < self.spec.success_radius
        newly_visited = within & ~self.visited
        bonus = success_bonus * newly_visited.any(dim=1).float()
        self.visited = self.visited | within

        new_unvisited = new_dists.masked_fill(self.visited, float("inf"))
        # if all visited, set new_min = 0 to avoid inf in shaped reward
        all_visited = self.visited.all(dim=1)
        new_min, _ = new_unvisited.min(dim=1)
        new_min = torch.where(all_visited, torch.zeros_like(new_min), new_min)
        prev_min = torch.where(prev_min == float("inf"), torch.zeros_like(prev_min), prev_min)

        shaped = prev_min - new_min
        reward = shaped + bonus

        done = all_visited | (self.t >= self.spec.max_steps - 1)
        self.t += 1
        return self.observe(), reward, done

    def expert_action(self) -> torch.Tensor:
        """Optimal greedy: head to nearest unvisited target. [B, 2]."""
        dists = self._distances_to_targets()
        masked = dists.masked_fill(self.visited, float("inf"))
        target_idx = masked.argmin(dim=1)
        b_idx = torch.arange(self.batch_size, device=self.device)
        target_x = self.tx[b_idx, target_idx]
        target_y = self.ty[b_idx, target_idx]
        # if all visited, no action; just return zeros
        all_visited = self.visited.all(dim=1)
        dx = (target_x - self.x).clamp(-self.spec.move_max, self.spec.move_max)
        dy = (target_y - self.y).clamp(-self.spec.move_max, self.spec.move_max)
        dx = torch.where(all_visited, torch.zeros_like(dx), dx)
        dy = torch.where(all_visited, torch.zeros_like(dy), dy)
        return torch.stack([dx, dy], dim=1)

    def encode_action(self, dxy: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(dxy.shape[0], self.spec.action_dim, device=self.device)
        action[:, 0] = dxy[:, 0]
        action[:, 1] = dxy[:, 1]
        return action

    def decode_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., :2]

    def success_mask(self) -> torch.Tensor:
        """Episode succeeds when all targets visited."""
        return self.visited.all(dim=1)

    def encode_action(self, dxy: torch.Tensor) -> torch.Tensor:
        """Pack (dx, dy) into the first two slots of a zero-padded action vector."""

        action = torch.zeros(dxy.shape[0], self.spec.action_dim, device=self.device)
        action[:, 0] = dxy[:, 0]
        action[:, 1] = dxy[:, 1]
        return action

    def decode_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., :2]
