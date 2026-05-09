"""Stage D: CEM eval on NavigateEnv using trained JEPA encoder + temporal predictor."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import (
    BLAJEPAModel,
    CEMConfig,
    JEPAConfig,
    NavigateEnv,
    NavigateSpec,
    TemporalConfig,
    TemporalPredictor,
    cem_plan,
    pool_patch_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-population", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    jepa_ckpt = torch.load(args.jepa_checkpoint, map_location=device)
    cfg_dict = jepa_ckpt["config"]
    d = cfg_dict["d"]
    image_size = cfg_dict["image_size"]

    jepa_cfg = JEPAConfig(
        d_jepa=d, in_channels=3, patch_size=cfg_dict["patch_size"],
        encoder_depth=cfg_dict["encoder_depth"], encoder_heads=cfg_dict["encoder_heads"],
        predictor_depth=cfg_dict["predictor_depth"], predictor_heads=cfg_dict["predictor_heads"],
        action_dim=d, ema_tau=cfg_dict["ema_tau"], sigreg_weight=cfg_dict["sigreg_weight"],
        target_mask_ratio=cfg_dict["target_ratio"], dtype="float32",
    )
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    jepa.target_encoder.load_state_dict(jepa_ckpt["target"])
    encoder = jepa.target_encoder
    encoder.eval()

    temp_ckpt = torch.load(args.temporal_checkpoint, map_location=device)
    temp_cfg_d = temp_ckpt["temporal_config"]
    temp_cfg = TemporalConfig(**temp_cfg_d)
    predictor = TemporalPredictor(temp_cfg).to(device)
    predictor.load_state_dict(temp_ckpt["predictor"])
    predictor.eval()

    nav_spec = NavigateSpec(
        image_size=image_size,
        patch_size=cfg_dict["patch_size"],
        max_steps=8,
        action_dim=d,
    )

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = encoder(flat)
        return pool_patch_tokens(z).reshape(b, t, -1)

    successes = 0
    total = 0
    cem_cfg = CEMConfig(horizon=args.horizon, iterations=args.cem_iterations, population=args.cem_population)

    n_batches = (args.episodes + args.batch_size - 1) // args.batch_size
    for batch in range(n_batches):
        env = NavigateEnv(nav_spec, batch_size=args.batch_size, device=device, seed=42 + batch)
        obs = env.reset()
        history_emb = encode_pool(obs.unsqueeze(1))
        done_mask = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        success_mask = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        for _ in range(nav_spec.max_steps):
            plan = cem_plan(predictor, history_emb, action_dim=d, cfg=cem_cfg)
            first_action = plan[:, 0]
            dxy = env.decode_action(first_action)
            obs, reward, done = env.step(dxy)
            history_emb = torch.cat([history_emb, encode_pool(obs.unsqueeze(1))], dim=1)
            if history_emb.size(1) > predictor.frame_pos_embed.size(1):
                history_emb = history_emb[:, -predictor.frame_pos_embed.size(1):]
            new_success = done & (reward > -nav_spec.success_radius) & (~done_mask)
            success_mask = success_mask | new_success
            done_mask = done_mask | done
            if done_mask.all():
                break
        successes += int(success_mask.sum())
        total += args.batch_size

    rate = successes / max(total, 1)
    summary = {
        "success_rate": rate,
        "successes": successes,
        "total": total,
        "cem_population": args.cem_population,
        "cem_iterations": args.cem_iterations,
        "horizon": args.horizon,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
