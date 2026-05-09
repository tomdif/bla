"""Cross-Entropy Method planning over the TemporalPredictor.

CEM samples action chunks from a Gaussian, scores each by rolling out the
temporal predictor + reward head, and refits the sampling Gaussian to the
top-k elites. Standard MPC formulation.

This is the evaluation harness the blueprint promises: once the temporal
predictor has a reward head trained on a real task, CEM gives an honest
success-rate measurement of whether the world model is good enough to plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .temporal import TemporalPredictor


@dataclass
class CEMConfig:
    horizon: int = 4
    iterations: int = 4
    population: int = 64
    elite_frac: float = 0.125
    init_std: float = 1.0
    min_std: float = 0.1
    discount: float = 0.95


def cem_plan(
    predictor: TemporalPredictor,
    frame_embeds: torch.Tensor,
    action_dim: int,
    cfg: CEMConfig = CEMConfig(),
    reward_fn=None,
) -> torch.Tensor:
    """Return the best action chunk sequence for each batch item.

    frame_embeds: [B, T, D] — context embeddings (encoder output).
    action_dim:   d_a — per-step action dimensionality.
    reward_fn:    optional override; signature reward_fn(z_traj) -> [B, H].
                  Default uses the predictor's own reward head, summed over
                  the action chunk.

    Returns action sequences of shape [B, horizon, action_dim].
    """

    batch = frame_embeds.shape[0]
    device = frame_embeds.device
    dtype = frame_embeds.dtype
    n_elite = max(1, int(cfg.population * cfg.elite_frac))

    mu = torch.zeros(batch, cfg.horizon, action_dim, device=device, dtype=dtype)
    sigma = torch.full_like(mu, cfg.init_std)
    discounts = (cfg.discount ** torch.arange(cfg.horizon, device=device, dtype=dtype))

    for _ in range(cfg.iterations):
        noise = torch.randn(batch, cfg.population, cfg.horizon, action_dim, device=device, dtype=dtype)
        actions = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise

        flat_actions = actions.reshape(batch * cfg.population, cfg.horizon, 1, action_dim)
        flat_ctx = frame_embeds.unsqueeze(1).expand(-1, cfg.population, -1, -1)
        flat_ctx = flat_ctx.reshape(batch * cfg.population, *frame_embeds.shape[1:])

        traj = predictor.rollout(flat_ctx, flat_actions)
        if reward_fn is None:
            rewards = traj["r"].sum(dim=-1)
        else:
            rewards = reward_fn(traj["z"])
        returns = (rewards * discounts).sum(dim=-1)
        returns = returns.view(batch, cfg.population)

        elite_idx = returns.topk(n_elite, dim=1).indices
        elite_actions = torch.gather(
            actions,
            1,
            elite_idx[:, :, None, None].expand(-1, -1, cfg.horizon, action_dim),
        )
        mu = elite_actions.mean(dim=1)
        sigma = elite_actions.std(dim=1).clamp(min=cfg.min_std)

    return mu
