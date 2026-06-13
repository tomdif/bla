from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import BLAJEPAModel, JEPAConfig, TemporalConfig, TemporalPredictor, pool_patch_tokens
from system1_jepa.spectral_temporal import (
    SpectralAugmentedTemporalPredictor,
    SpectralBlendTemporalPredictor,
    SpectralFeatureTemporalPredictor,
    SpectralResidualTemporalPredictor,
    SpectralTemporalConfig,
    carrier_prior,
)


@dataclass
class BreakerSpec:
    image_size: int = 16
    patch_size: int = 2
    history: int = 4
    action_dim: int = 32
    accel: int = 2
    max_vel: float = 4.0


class LastActionMLPPredictor(nn.Module):
    """Low-cost action baseline: next latent from last latent plus action."""

    def __init__(self, d: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, frame_embeds: torch.Tensor, a_chunk: torch.Tensor) -> torch.Tensor:
        last = frame_embeds[:, -1]
        action = a_chunk.reshape(a_chunk.shape[0], -1)
        return last + self.net(torch.cat([last, action], dim=-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Breaker benchmark for spectral/carrier temporal prediction on action-dependent JEPA latents."
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--train-horizon", type=int, default=4)
    parser.add_argument("--eval-horizon", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--latent-view", type=str, default="flat", choices=["flat", "pooled"])
    parser.add_argument("--prior-kind", type=str, default="last", choices=["last", "affine", "quadratic"])
    parser.add_argument("--spectral-mode", type=str, default="augmented", choices=["augmented", "blend", "feature", "residual"])
    parser.add_argument("--direct-weight", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="artifacts/phase1g_spectral_breaker/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    spec = BreakerSpec(image_size=args.image_size, history=args.history)
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.action_dim = spec.action_dim
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    latent_dim = _latent_dim(args, jepa_cfg)

    temporal_cfg = TemporalConfig(
        d=latent_dim,
        action_dim=spec.action_dim,
        chunk_size=1,
        n_layers=1,
        num_heads=_num_heads(latent_dim),
        mlp_ratio=2.0,
        max_context=args.history + args.eval_horizon + 2,
        dropout=0.0,
    )
    direct = TemporalPredictor(temporal_cfg).to(device)
    spectral_cfg = SpectralTemporalConfig(
        temporal=copy.deepcopy(temporal_cfg),
        prior_kind=args.prior_kind,
        residual_scale=args.residual_scale,
        direct_weight=args.direct_weight,
    )
    if args.spectral_mode == "augmented":
        spectral = SpectralAugmentedTemporalPredictor(spectral_cfg).to(device)
        spectral.predictor.load_state_dict(copy.deepcopy(direct.state_dict()))
    elif args.spectral_mode == "feature":
        spectral = SpectralFeatureTemporalPredictor(spectral_cfg).to(device)
        spectral.residual.load_state_dict(copy.deepcopy(direct.state_dict()))
    elif args.spectral_mode == "residual":
        spectral = SpectralResidualTemporalPredictor(spectral_cfg).to(device)
        spectral.residual.load_state_dict(copy.deepcopy(direct.state_dict()))
    else:
        spectral = SpectralBlendTemporalPredictor(spectral_cfg).to(device)
        spectral.direct.load_state_dict(copy.deepcopy(direct.state_dict()))
    action_mlp = LastActionMLPPredictor(latent_dim, spec.action_dim).to(device)

    direct_optim = torch.optim.AdamW(direct.parameters(), lr=args.lr)
    spectral_optim = torch.optim.AdamW(spectral.parameters(), lr=args.lr)
    action_optim = torch.optim.AdamW(action_mlp.parameters(), lr=args.lr)

    short_eval = _make_eval_batches(args, spec, device, args.train_horizon, args.seed + 10_000)
    long_eval = _make_eval_batches(args, spec, device, args.eval_horizon, args.seed + 20_000)
    initial = {
        "short": _evaluate(args, spec, jepa, direct, spectral, action_mlp, short_eval),
        "long": _evaluate(args, spec, jepa, direct, spectral, action_mlp, long_eval),
    }
    print(json.dumps({"event": "initial_eval", **initial}))

    logs = []
    for step in range(args.steps):
        history, actions, future = make_breaker_episodes(
            spec, args.batch_size, args.train_horizon, device=device
        )
        with torch.no_grad():
            history_emb = encode_latents(args, jepa, history)
            future_emb = encode_latents(args, jepa, future)

        direct_loss, direct_single = latent_rollout_loss(
            direct, history_emb, actions, future_emb, temporal_cfg.max_context
        )
        _step(direct_optim, direct_loss, direct.parameters(), args.grad_clip)

        spectral_loss, spectral_single = latent_rollout_loss(
            spectral, history_emb, actions, future_emb, temporal_cfg.max_context
        )
        _step(spectral_optim, spectral_loss, spectral.parameters(), args.grad_clip)

        action_loss, action_single = latent_rollout_loss(
            action_mlp, history_emb, actions, future_emb, temporal_cfg.max_context
        )
        _step(action_optim, action_loss, action_mlp.parameters(), args.grad_clip)

        row = {
            "event": "train",
            "step": step,
            "direct_rollout": float(direct_loss.detach()),
            "spectral_rollout": float(spectral_loss.detach()),
            "action_mlp_rollout": float(action_loss.detach()),
            "direct_single": float(direct_single),
            "spectral_single": float(spectral_single),
            "action_mlp_single": float(action_single),
        }
        if hasattr(spectral, "direct_logit"):
            row["spectral_direct_weight"] = float(spectral.direct_logit.sigmoid().detach())
        if hasattr(spectral, "log_residual_scale"):
            row["spectral_residual_scale"] = float(spectral.log_residual_scale.exp().detach())
        if hasattr(spectral, "log_correction_scale"):
            row["spectral_correction_scale"] = float(spectral.log_correction_scale.exp().detach())
        logs.append(row)
        if step % max(args.log_every, 1) == 0 or step == args.steps - 1:
            print(json.dumps(row))

    final = {
        "short": _evaluate(args, spec, jepa, direct, spectral, action_mlp, short_eval),
        "long": _evaluate(args, spec, jepa, direct, spectral, action_mlp, long_eval),
    }
    summary = {
        "config": vars(args),
        "latent_dim": latent_dim,
        "initial_eval": initial,
        "final_eval": final,
        "train_tail": logs[-5:],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "final_eval", "out": str(out_path), **final}))


def make_breaker_episodes(
    spec: BreakerSpec,
    batch_size: int,
    horizon: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = w = spec.image_size
    p = spec.patch_size
    limit = float(w - p)
    x = torch.randint(0, w - p + 1, (batch_size,), device=device, generator=generator).float()
    y = torch.randint(0, h - p + 1, (batch_size,), device=device, generator=generator).float()
    vel = torch.randint(-2, 3, (batch_size, 2), device=device, generator=generator).float()

    history = torch.zeros(batch_size, spec.history, 3, h, w, device=device)
    future = torch.zeros(batch_size, horizon, 3, h, w, device=device)
    actions = torch.zeros(batch_size, horizon, 1, spec.action_dim, device=device)

    for t in range(spec.history):
        _draw_patch(history[:, t], x, y, p)
        accel = _sample_accel(spec, batch_size, device, generator)
        x, y, vel = _advance(x, y, vel, accel, limit, spec.max_vel)

    for t in range(horizon):
        accel = _sample_accel(spec, batch_size, device, generator)
        actions[:, t, 0, 0:2] = accel
        x, y, vel = _advance(x, y, vel, accel, limit, spec.max_vel)
        _draw_patch(future[:, t], x, y, p)

    return history, actions, future


def _draw_patch(canvas: torch.Tensor, x: torch.Tensor, y: torch.Tensor, patch_size: int) -> None:
    x_i = x.clamp(0, canvas.shape[-1] - patch_size).long()
    y_i = y.clamp(0, canvas.shape[-2] - patch_size).long()
    for b in range(canvas.shape[0]):
        canvas[b, :, y_i[b] : y_i[b] + patch_size, x_i[b] : x_i[b] + patch_size] = 1.0


def _sample_accel(
    spec: BreakerSpec,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randint(
        -spec.accel,
        spec.accel + 1,
        (batch_size, 2),
        device=device,
        generator=generator,
    ).float()


def _advance(
    x: torch.Tensor,
    y: torch.Tensor,
    vel: torch.Tensor,
    accel: torch.Tensor,
    limit: float,
    max_vel: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vel = (vel + accel).clamp(-max_vel, max_vel)
    x = x + vel[:, 0]
    y = y + vel[:, 1]
    x, vel_x = _bounce_1d(x, vel[:, 0], limit)
    y, vel_y = _bounce_1d(y, vel[:, 1], limit)
    return x, y, torch.stack([vel_x, vel_y], dim=1)


def _bounce_1d(pos: torch.Tensor, vel: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    below = pos < 0
    pos = torch.where(below, -pos, pos)
    vel = torch.where(below, -vel, vel)
    above = pos > limit
    pos = torch.where(above, 2.0 * limit - pos, pos)
    vel = torch.where(above, -vel, vel)
    return pos.clamp(0, limit), vel


@torch.no_grad()
def encode_latents(args: argparse.Namespace, jepa: BLAJEPAModel, images_btchw: torch.Tensor) -> torch.Tensor:
    batch, steps = images_btchw.shape[:2]
    flat = images_btchw.reshape(batch * steps, *images_btchw.shape[2:])
    z, _, _ = jepa.target_encoder(flat)
    if args.latent_view == "pooled":
        encoded = pool_patch_tokens(z)
    else:
        encoded = z.reshape(z.shape[0], -1)
    return encoded.reshape(batch, steps, -1)


def latent_rollout_loss(
    predictor: nn.Module,
    history_emb: torch.Tensor,
    actions: torch.Tensor,
    future_emb: torch.Tensor,
    max_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = history_emb
    losses = []
    for h in range(actions.shape[1]):
        z_h = predictor(ctx, actions[:, h])
        losses.append(F.mse_loss(z_h.float(), future_emb[:, h].float()))
        ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
        if ctx.shape[1] > max_context:
            ctx = ctx[:, -max_context:]
    return torch.stack(losses).mean(), losses[0].detach()


@torch.no_grad()
def latent_prior_rollout_loss(
    history_emb: torch.Tensor,
    future_emb: torch.Tensor,
    prior_kind: str,
    max_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = history_emb
    losses = []
    for h in range(future_emb.shape[1]):
        z_h = carrier_prior(ctx, prior_kind)
        losses.append(F.mse_loss(z_h.float(), future_emb[:, h].float()))
        ctx = torch.cat([ctx, z_h.unsqueeze(1)], dim=1)
        if ctx.shape[1] > max_context:
            ctx = ctx[:, -max_context:]
    return torch.stack(losses).mean(), losses[0].detach()


@torch.no_grad()
def _evaluate(args, spec, jepa, direct, spectral, action_mlp, batches) -> dict:
    metrics = {
        "direct_rollout": [],
        "spectral_rollout": [],
        "action_mlp_rollout": [],
        "prior_rollout": [],
        "copy_last_rollout": [],
        "direct_single": [],
        "spectral_single": [],
        "action_mlp_single": [],
        "prior_single": [],
        "copy_last_single": [],
    }
    max_context = args.history + args.eval_horizon + 2
    for history, actions, future in batches:
        history_emb = encode_latents(args, jepa, history)
        future_emb = encode_latents(args, jepa, future)
        _append(metrics, "direct", latent_rollout_loss(direct, history_emb, actions, future_emb, max_context))
        _append(metrics, "spectral", latent_rollout_loss(spectral, history_emb, actions, future_emb, max_context))
        _append(metrics, "action_mlp", latent_rollout_loss(action_mlp, history_emb, actions, future_emb, max_context))
        _append(metrics, "prior", latent_prior_rollout_loss(history_emb, future_emb, args.prior_kind, max_context))
        _append(metrics, "copy_last", latent_prior_rollout_loss(history_emb, future_emb, "last", max_context))
    return {key: _mean(values) for key, values in metrics.items()}


def _append(metrics: dict[str, list[float]], prefix: str, result: tuple[torch.Tensor, torch.Tensor]) -> None:
    loss, single = result
    metrics[f"{prefix}_rollout"].append(float(loss))
    metrics[f"{prefix}_single"].append(float(single))


def _step(optim, loss: torch.Tensor, params, grad_clip: float) -> None:
    optim.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(list(params), grad_clip)
    optim.step()


def _make_eval_batches(
    args: argparse.Namespace,
    spec: BreakerSpec,
    device: torch.device,
    horizon: int,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return [
        make_breaker_episodes(spec, args.batch_size, horizon, device=device, generator=generator)
        for _ in range(args.eval_batches)
    ]


def _latent_dim(args: argparse.Namespace, jepa_cfg: JEPAConfig) -> int:
    if args.latent_view == "pooled":
        return jepa_cfg.d_jepa
    grid = args.image_size // jepa_cfg.patch_size
    return grid * grid * jepa_cfg.d_jepa


def _num_heads(d: int) -> int:
    for heads in (8, 4, 2, 1):
        if d % heads == 0:
            return heads
    return 1


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
