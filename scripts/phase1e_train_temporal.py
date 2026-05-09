from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import (
    BLAJEPAModel,
    JEPAConfig,
    MovingPatchSpec,
    TemporalConfig,
    TemporalPredictor,
    make_moving_patch_episodes,
    multistep_rollout_loss,
    pool_patch_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1E temporal predictor smoke training (multi-step rollout on moving-patch sequences)."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    predictor = TemporalPredictor(temporal_cfg).to(device)

    optim = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = jepa.target_encoder(flat)
        pooled = pool_patch_tokens(z)
        return pooled.reshape(b, t, -1)

    for step in range(args.steps):
        history, actions, future = make_moving_patch_episodes(spec, args.batch_size, device=device)
        loss, single = multistep_rollout_loss(predictor, encode_pool, history, actions, future)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
        optim.step()
        print(json.dumps({
            "step": step,
            "rollout_loss": float(loss.detach()),
            "single_step_loss": float(single),
        }))


if __name__ == "__main__":
    main()
