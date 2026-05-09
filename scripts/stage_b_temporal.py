"""Stage B: train TemporalPredictor + reward head with frozen JEPA encoder.

Loads the JEPA pretrain checkpoint, freezes the target encoder, generates
moving-patch sequences, trains the temporal predictor with multi-step
rollout supervision. Saves the predictor checkpoint for stages C/D.
"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--temporal-d", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ckpt = torch.load(args.jepa_checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    image_size = cfg_dict["image_size"]
    d = cfg_dict["d"]

    spec = MovingPatchSpec(
        image_size=image_size, patch_size=cfg_dict["patch_size"],
        horizon=args.horizon, history=args.history, action_dim=d,
    )
    jepa_cfg = JEPAConfig(
        d_jepa=d, in_channels=3, patch_size=cfg_dict["patch_size"],
        encoder_depth=cfg_dict["encoder_depth"], encoder_heads=cfg_dict["encoder_heads"],
        predictor_depth=cfg_dict["predictor_depth"], predictor_heads=cfg_dict["predictor_heads"],
        action_dim=d, ema_tau=cfg_dict["ema_tau"], sigreg_weight=cfg_dict["sigreg_weight"],
        target_mask_ratio=cfg_dict["target_ratio"], dtype="float32",
    )
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    jepa.target_encoder.load_state_dict(ckpt["target"])
    encoder = jepa.target_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    temporal_cfg = TemporalConfig(
        d=args.temporal_d, action_dim=d, chunk_size=1,
        n_layers=args.n_layers, num_heads=8,
        max_context=args.history + args.horizon + 4,
    )
    if args.temporal_d != d:
        # ensure dims line up — match temporal d to JEPA d
        temporal_cfg.d = d
    predictor = TemporalPredictor(temporal_cfg).to(device)
    optim = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = encoder(flat)
        pooled = pool_patch_tokens(z)
        return pooled.reshape(b, t, -1)

    history_log = []
    for step in range(args.steps):
        history_imgs, actions, future_imgs = make_moving_patch_episodes(spec, args.batch_size, device=device)
        loss, single = multistep_rollout_loss(predictor, encode_pool, history_imgs, actions, future_imgs)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
        optim.step()
        if (step + 1) % args.log_every == 0:
            entry = {"step": step + 1, "rollout_loss": float(loss.detach()), "single_step_loss": float(single)}
            history_log.append(entry)
            print(json.dumps(entry), flush=True)

    torch.save({
        "predictor": predictor.state_dict(),
        "temporal_config": temporal_cfg.__dict__,
        "moving_patch_spec": spec.__dict__,
        "history": history_log,
        "step": args.steps,
    }, args.output)
    print(json.dumps({"event": "saved", "path": args.output}), flush=True)


if __name__ == "__main__":
    main()
