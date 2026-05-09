"""Phase 1F: Train temporal predictor + reward head on the navigate-to-target
task, then evaluate with CEM planning.

This is the first downstream-task evaluation in the project: success rate is
measured by closing the loop (observe → encode → CEM → execute → repeat) and
counting how often the agent reaches the target within the step budget.
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
    parser = argparse.ArgumentParser(description="Train + plan on navigate-to-target.")
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-population", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    nav_spec = NavigateSpec(image_size=16, patch_size=2, max_steps=8, action_dim=32)
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.action_dim = nav_spec.action_dim
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    encoder = jepa.target_encoder

    temporal_cfg = TemporalConfig.tiny()
    temporal_cfg.action_dim = nav_spec.action_dim
    temporal_cfg.max_context = args.horizon + 8
    predictor = TemporalPredictor(temporal_cfg).to(device)
    optim = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = encoder(flat)
        return z.mean(dim=1).reshape(b, t, -1)

    env = NavigateEnv(nav_spec, batch_size=4, seed=args.seed)

    for step in range(args.train_steps):
        obs0 = env.reset()
        obs_seq = [obs0]
        action_seq = []
        reward_seq = []
        for _ in range(args.horizon):
            dxy = (torch.rand(env.batch_size, 2) * 4 - 2)
            action_seq.append(env.encode_action(dxy))
            obs, reward, _ = env.step(dxy)
            obs_seq.append(obs)
            reward_seq.append(reward)

        obs_stack = torch.stack(obs_seq, dim=1)
        actions = torch.stack(action_seq, dim=1).unsqueeze(2)
        rewards = torch.stack(reward_seq, dim=1)

        history = obs_stack[:, :1]
        future = obs_stack[:, 1:]
        history_emb = encode_pool(history)
        future_emb = encode_pool(future)

        ctx = history_emb
        max_ctx = predictor.frame_pos_embed.size(1)
        z_loss = 0.0
        r_loss = 0.0
        for h in range(args.horizon):
            out = predictor(ctx, actions[:, h], return_aux=True)
            target = future_emb[:, h].detach()
            z_loss = z_loss + (out["z"].float() - target.float()).pow(2).mean()
            r_loss = r_loss + (out["r"][:, 0].float() - rewards[:, h].float()).pow(2).mean()
            ctx = torch.cat([ctx, out["z"].unsqueeze(1)], dim=1)
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]

        loss = z_loss / args.horizon + r_loss / args.horizon
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optim.step()
        if step % 10 == 0:
            print(json.dumps({"step": step, "z_loss": float(z_loss / args.horizon), "r_loss": float(r_loss / args.horizon)}))

    eval_env = NavigateEnv(nav_spec, batch_size=args.eval_episodes, seed=args.seed + 1000)
    obs = eval_env.reset()
    history_emb = encode_pool(obs.unsqueeze(1))
    successes = torch.zeros(args.eval_episodes, dtype=torch.bool)

    for t in range(nav_spec.max_steps):
        plan = cem_plan(
            predictor,
            history_emb,
            action_dim=nav_spec.action_dim,
            cfg=CEMConfig(
                horizon=min(args.horizon, nav_spec.max_steps - t),
                iterations=args.cem_iterations,
                population=args.cem_population,
            ),
        )
        first_action = plan[:, 0]
        dxy = eval_env.decode_action(first_action)
        obs, reward, done = eval_env.step(dxy)
        history_emb = torch.cat([history_emb, encode_pool(obs.unsqueeze(1))], dim=1)
        if history_emb.size(1) > predictor.frame_pos_embed.size(1):
            history_emb = history_emb[:, -predictor.frame_pos_embed.size(1):]
        successes |= done & (reward > -nav_spec.success_radius)
        if done.all():
            break

    print(json.dumps({
        "eval_success_rate": float(successes.float().mean()),
        "eval_episodes": args.eval_episodes,
    }))


if __name__ == "__main__":
    main()
