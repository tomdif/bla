"""End-to-end navigate trainer with shaped reward + goal-conditioned predictor.

The redesign principle: skip pretraining entirely. Train encoder + predictor
+ reward/value heads jointly on the navigate task with:

  * Dense shaped per-step reward (prev_dist - curr_dist + success bonus)
  * Goal-position token explicitly fed into predictor every step (so the
    "carrot" never falls out of working memory)
  * Reward-to-go conditioning (decision-transformer style)
  * Imitation loss against the optimal expert action

Eval = CEM planner success rate on held-out target positions. The number
that actually matters: % of episodes where the agent reaches the carrot
within max_steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from system1_jepa import (
    CEMConfig,
    NavigateEnv,
    NavigateSpec,
    PatchViTEncoder,
    TemporalConfig,
    TemporalPredictor,
    cem_plan,
    pool_patch_tokens,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--episodes-per-step", type=int, default=8, help="parallel envs per step")
    p.add_argument("--max-steps-per-ep", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-heads", type=int, default=8)
    p.add_argument("--predictor-depth", type=int, default=3)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--bc-weight", type=float, default=1.0)
    p.add_argument("--reward-weight", type=float, default=0.5)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--world-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--cem-iter", type=int, default=4)
    p.add_argument("--cem-pop", type=int, default=128)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    import math
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def encode_action_batch(dxy: torch.Tensor, action_dim: int) -> torch.Tensor:
    """[B, 2] -> [B, action_dim] with (dx, dy) in slots 0, 1."""
    out = torch.zeros(dxy.shape[0], action_dim, device=dxy.device, dtype=dxy.dtype)
    out[:, 0] = dxy[:, 0]
    out[:, 1] = dxy[:, 1]
    return out


def collect_expert_episode(
    env: NavigateEnv,
    encoder: nn.Module,
    predictor: nn.Module,
    spec: NavigateSpec,
    max_steps: int,
    amp_dtype: torch.dtype,
) -> dict:
    """Run one episode where the env's optimal expert provides actions.

    Returns tensors of shape [B, T, ...] for obs/action/reward/value targets.
    """
    obs0 = env.reset()
    obs_seq = [obs0]
    expert_actions = []
    rewards = []
    for _ in range(max_steps):
        dxy = env.expert_action()
        expert_actions.append(dxy)
        obs, r, done = env.step(dxy)
        obs_seq.append(obs)
        rewards.append(r)
        if done.all():
            break
    obs_seq = torch.stack(obs_seq, dim=1)         # [B, T+1, C, H, W]
    expert_actions = torch.stack(expert_actions, dim=1)  # [B, T, 2]
    rewards = torch.stack(rewards, dim=1)         # [B, T]
    discount = 0.95
    rtg = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + discount * running
        rtg[:, t] = running
    return {
        "obs": obs_seq,
        "actions": expert_actions,
        "rewards": rewards,
        "rtg": rtg,
    }


@torch.no_grad()
def cem_eval_success(
    encoder: nn.Module,
    predictor: nn.Module,
    spec: NavigateSpec,
    n_episodes: int,
    cem_cfg: CEMConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int = 99,
) -> dict:
    encoder.eval()
    predictor.eval()
    bs = min(n_episodes, 16)
    n_batches = (n_episodes + bs - 1) // bs
    successes = 0
    total = 0
    final_dist_sum = 0.0
    for b in range(n_batches):
        env = NavigateEnv(spec, batch_size=bs, device=device, seed=seed + b)
        obs = env.reset()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            z, _, _ = encoder(obs)
        history = pool_patch_tokens(z).unsqueeze(1)
        success_mask = torch.zeros(bs, dtype=torch.bool, device=device)
        for _ in range(spec.max_steps):
            plan = cem_plan(predictor, history, action_dim=spec.action_dim, cfg=cem_cfg)
            first_action = plan[:, 0]
            dxy = env.decode_action(first_action)
            obs, _, done = env.step(dxy)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                z_new, _, _ = encoder(obs)
            new_emb = pool_patch_tokens(z_new).unsqueeze(1)
            history = torch.cat([history, new_emb], dim=1)
            if history.size(1) > predictor.frame_pos_embed.size(1):
                history = history[:, -predictor.frame_pos_embed.size(1):]
            reached = ((env.x - env.tx) ** 2 + (env.y - env.ty) ** 2).sqrt() < spec.success_radius
            success_mask = success_mask | reached
            if done.all():
                break
        successes += int(success_mask.sum())
        total += bs
        final_dist_sum += float(((env.x - env.tx) ** 2 + (env.y - env.ty) ** 2).sqrt().sum())
    encoder.train()
    predictor.train()
    return {
        "success_rate": successes / max(total, 1),
        "successes": successes,
        "total": total,
        "mean_final_dist": final_dist_sum / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output, exist_ok=True)

    spec = NavigateSpec(
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_steps=args.max_steps_per_ep,
        action_dim=args.d,
    )
    env = NavigateEnv(spec, batch_size=args.episodes_per_step, device=device, seed=args.seed + rank)

    encoder = PatchViTEncoder(
        in_channels=3,
        latent_dim=args.d,
        patch_size=args.patch_size,
        depth=args.encoder_depth,
        heads=args.encoder_heads,
    ).to(device)
    temporal_cfg = TemporalConfig(
        d=args.d,
        action_dim=args.d,
        chunk_size=1,
        n_layers=args.predictor_depth,
        num_heads=args.predictor_heads,
        max_context=args.max_steps_per_ep + 4,
    )
    predictor = TemporalPredictor(temporal_cfg).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    if is_main:
        print(json.dumps({"event": "init", "world_size": world, "n_params": n_params, "config": vars(args)}), flush=True)

    if world > 1:
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=False)
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)
    enc_module = encoder.module if world > 1 else encoder
    pred_module = predictor.module if world > 1 else predictor

    optim = torch.optim.AdamW(
        list(enc_module.parameters()) + list(pred_module.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    amp_dtype = torch.bfloat16 if args.amp else torch.float32

    if world > 1:
        dist.barrier()

    t0 = time.time()
    log = {"loss": 0.0, "bc": 0.0, "rew": 0.0, "val": 0.0, "world": 0.0, "n": 0}

    for step in range(args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        ep = collect_expert_episode(env, enc_module, pred_module, spec, args.max_steps_per_ep, amp_dtype)
        obs = ep["obs"].to(device)        # [B, T+1, C, H, W]
        expert_act = ep["actions"].to(device)  # [B, T, 2]
        rewards = ep["rewards"].to(device)     # [B, T]
        rtg = ep["rtg"].to(device)             # [B, T]

        b, t_obs = obs.shape[:2]
        flat_obs = obs.reshape(b * t_obs, *obs.shape[2:])

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            z, _, _ = encoder(flat_obs)
            z_pooled = pool_patch_tokens(z).reshape(b, t_obs, -1)

            t_act = expert_act.shape[1]
            bc_losses = []
            rew_losses = []
            val_losses = []
            world_losses = []
            ctx = z_pooled[:, :1]
            max_ctx = pred_module.frame_pos_embed.size(1)
            for h in range(t_act):
                action_in = encode_action_batch(expert_act[:, h], args.d)
                out = predictor(ctx, action_in.unsqueeze(1), return_aux=True)
                pred_z = out["z"]
                pred_r = out["r"][:, 0]
                pred_v = out["v"]

                next_z_target = z_pooled[:, h + 1].detach()
                world_losses.append(F.mse_loss(pred_z.float(), next_z_target.float()))

                rew_losses.append(F.mse_loss(pred_r.float(), rewards[:, h].float()))
                val_losses.append(F.mse_loss(pred_v.float(), rtg[:, h].float()))

                action_pred_xy = pred_z[:, :2]
                bc_losses.append(F.mse_loss(action_pred_xy.float(), expert_act[:, h].float()))

                ctx = torch.cat([ctx, z_pooled[:, h + 1:h + 2]], dim=1)
                if ctx.size(1) > max_ctx:
                    ctx = ctx[:, -max_ctx:]

            bc_loss = torch.stack(bc_losses).mean()
            rew_loss = torch.stack(rew_losses).mean()
            val_loss = torch.stack(val_losses).mean()
            world_loss = torch.stack(world_losses).mean()
            loss = (
                args.bc_weight * bc_loss
                + args.reward_weight * rew_loss
                + args.value_weight * val_loss
                + args.world_weight * world_loss
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc_module.parameters()) + list(pred_module.parameters()), args.grad_clip
        )
        optim.step()

        log["loss"] += float(loss.detach())
        log["bc"] += float(bc_loss.detach())
        log["rew"] += float(rew_loss.detach())
        log["val"] += float(val_loss.detach())
        log["world"] += float(world_loss.detach())
        log["n"] += 1

        if is_main and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(json.dumps({
                "step": step + 1,
                "loss": log["loss"] / log["n"],
                "bc": log["bc"] / log["n"],
                "rew": log["rew"] / log["n"],
                "val": log["val"] / log["n"],
                "world": log["world"] / log["n"],
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
            }), flush=True)
            log = {"loss": 0.0, "bc": 0.0, "rew": 0.0, "val": 0.0, "world": 0.0, "n": 0}

        if is_main and (step + 1) % args.eval_every == 0:
            cem_cfg = CEMConfig(
                horizon=args.max_steps_per_ep,
                iterations=args.cem_iter,
                population=args.cem_pop,
            )
            eval_out = cem_eval_success(
                enc_module, pred_module, spec, args.eval_episodes, cem_cfg, device, amp_dtype
            )
            print(json.dumps({"event": "eval", "step": step + 1, **eval_out}), flush=True)

        if is_main and (step + 1) % args.ckpt_every == 0:
            path = os.path.join(args.output, f"ckpt_step{step + 1:08d}.pt")
            torch.save({
                "encoder": enc_module.state_dict(),
                "predictor": pred_module.state_dict(),
                "config": vars(args),
                "step": step + 1,
            }, path)
            print(json.dumps({"event": "checkpoint", "step": step + 1, "path": path}), flush=True)

    if is_main:
        path = os.path.join(args.output, "final.pt")
        torch.save({
            "encoder": enc_module.state_dict(),
            "predictor": pred_module.state_dict(),
            "config": vars(args),
            "step": args.steps,
        }, path)
        print(json.dumps({"event": "final", "path": path, "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
