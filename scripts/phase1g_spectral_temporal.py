from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import (
    BLAJEPAModel,
    JEPAConfig,
    MovingPatchSpec,
    TemporalConfig,
    TemporalPredictor,
    make_moving_patch_episodes,
    pool_patch_tokens,
)
from system1_jepa.spectral_temporal import (
    SpectralAugmentedTemporalPredictor,
    SpectralBlendTemporalPredictor,
    SpectralFeatureTemporalPredictor,
    SpectralResidualTemporalPredictor,
    SpectralTemporalConfig,
    generic_multistep_rollout_loss,
    prior_multistep_rollout_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1E direct rollout against a spectral/carrier-residual variant."
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prior-kind", type=str, default="last", choices=["last", "affine", "quadratic"])
    parser.add_argument("--spectral-mode", type=str, default="blend", choices=["augmented", "blend", "feature", "residual"])
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--direct-weight", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="artifacts/phase1g_spectral_temporal/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    spec = MovingPatchSpec(
        image_size=args.image_size,
        patch_size=2,
        horizon=args.horizon,
        history=args.history,
    )
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.action_dim = spec.action_dim
    jepa = BLAJEPAModel(jepa_cfg).to(device)

    temporal_cfg = TemporalConfig.tiny()
    temporal_cfg.action_dim = spec.action_dim
    temporal_cfg.max_context = spec.history + spec.horizon + 2

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

    direct_optim = torch.optim.AdamW(direct.parameters(), lr=args.lr)
    spectral_optim = torch.optim.AdamW(spectral.parameters(), lr=args.lr)

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = jepa.target_encoder(flat)
        pooled = pool_patch_tokens(z)
        return pooled.reshape(b, t, -1)

    eval_batches = _make_eval_batches(args, spec, device)
    initial = _evaluate_all(args, direct, spectral, encode_pool, eval_batches, temporal_cfg.max_context)
    print(json.dumps({"event": "initial_eval", **initial}))

    logs = []
    for step in range(args.steps):
        history, actions, future = make_moving_patch_episodes(spec, args.batch_size, device=device)

        direct_loss, direct_single = generic_multistep_rollout_loss(
            direct, encode_pool, history, actions, future
        )
        direct_optim.zero_grad(set_to_none=True)
        direct_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(direct.parameters(), args.grad_clip)
        direct_optim.step()

        spectral_loss, spectral_single = generic_multistep_rollout_loss(
            spectral, encode_pool, history, actions, future
        )
        spectral_optim.zero_grad(set_to_none=True)
        spectral_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(spectral.parameters(), args.grad_clip)
        spectral_optim.step()

        row = {
            "event": "train",
            "step": step,
            "direct_rollout_loss": float(direct_loss.detach()),
            "direct_single_step_loss": float(direct_single),
            "spectral_rollout_loss": float(spectral_loss.detach()),
            "spectral_single_step_loss": float(spectral_single),
        }
        if hasattr(spectral, "log_residual_scale"):
            row["spectral_residual_scale"] = float(spectral.log_residual_scale.exp().detach())
        if hasattr(spectral, "log_correction_scale"):
            row["spectral_correction_scale"] = float(spectral.log_correction_scale.exp().detach())
        if hasattr(spectral, "direct_logit"):
            row["spectral_direct_weight"] = float(spectral.direct_logit.sigmoid().detach())
        logs.append(row)
        print(json.dumps(row))

    final = _evaluate_all(args, direct, spectral, encode_pool, eval_batches, temporal_cfg.max_context)
    summary = {
        "config": vars(args),
        "initial_eval": initial,
        "final_eval": final,
        "train_tail": logs[-5:],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "final_eval", "out": str(out_path), **final}))


def _make_eval_batches(args, spec: MovingPatchSpec, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 10_000)
    batches = []
    for _ in range(args.eval_batches):
        batches.append(
            make_moving_patch_episodes(
                spec,
                args.batch_size,
                device=device,
                generator=generator,
            )
        )
    return batches


@torch.no_grad()
def _evaluate_all(args, direct, spectral, encode_pool, batches, max_context: int) -> dict:
    direct_losses = []
    spectral_losses = []
    prior_losses = []
    copy_last_losses = []
    direct_singles = []
    spectral_singles = []
    prior_singles = []
    copy_last_singles = []
    for history, actions, future in batches:
        loss, single = generic_multistep_rollout_loss(direct, encode_pool, history, actions, future)
        direct_losses.append(float(loss))
        direct_singles.append(float(single))
        loss, single = generic_multistep_rollout_loss(spectral, encode_pool, history, actions, future)
        spectral_losses.append(float(loss))
        spectral_singles.append(float(single))
        loss, single = prior_multistep_rollout_loss(
            encode_pool, history, actions, future, args.prior_kind, max_context
        )
        prior_losses.append(float(loss))
        prior_singles.append(float(single))
        loss, single = prior_multistep_rollout_loss(
            encode_pool, history, actions, future, "last", max_context
        )
        copy_last_losses.append(float(loss))
        copy_last_singles.append(float(single))
    return {
        "direct_eval_rollout": _mean(direct_losses),
        "direct_eval_single": _mean(direct_singles),
        "spectral_eval_rollout": _mean(spectral_losses),
        "spectral_eval_single": _mean(spectral_singles),
        "prior_eval_rollout": _mean(prior_losses),
        "prior_eval_single": _mean(prior_singles),
        "copy_last_eval_rollout": _mean(copy_last_losses),
        "copy_last_eval_single": _mean(copy_last_singles),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
