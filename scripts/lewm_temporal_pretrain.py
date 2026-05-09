"""LeWM-style temporal pretrain on moving-patch sequences.

The recipe:
  * Single encoder for context AND target paths (no EMA target encoder)
  * No stop-gradient — gradient flows through both paths via shared encoder
  * SIGReg-only anti-collapse (Epps-Pulley) with linear lambda warmup
  * Normalized MSE prediction loss (no Smooth L1)
  * Multi-step rollout: predict h steps, slide predicted features back as context

Compared to phase1e_train_temporal.py:
  * Encoder is trained jointly (vs frozen target_encoder there)
  * SIGReg on all features per step (vs none there)
  * Lambda warmup ramps regularizer in over warmup_steps (vs constant)

Run with torchrun:
    torchrun --nproc_per_node=6 --master_port=29505 \\
        scripts/lewm_temporal_pretrain.py \\
        --steps 8000 --batch-size 64 --horizon 4 --history 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from system1_jepa import (
    MovingPatchSpec,
    PatchViTEncoder,
    TemporalConfig,
    TemporalPredictor,
    make_moving_patch_episodes,
    pool_patch_tokens,
    sigreg_epps_pulley,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64, help="per-GPU batch")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--encoder-heads", type=int, default=6)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-heads", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--sigreg-lambda", type=float, default=1.0)
    p.add_argument("--sigreg-directions", type=int, default=16)
    p.add_argument("--sigreg-warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--probe-every", type=int, default=500)
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


def linear_warmup(step: int, warmup: int, lambda_max: float) -> float:
    if warmup <= 0 or step >= warmup:
        return lambda_max
    return lambda_max * (step + 1) / warmup


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """Normalize the MSE by the running variance of target features. Keeps
    the prediction loss scale-invariant, which is what cser-jepa-v2 / LeWM use."""
    return F.mse_loss(pred.float(), target.float()) / sigma2.clamp(min=1e-3)


@torch.no_grad()
def navigate_lookahead_probe(
    encoder: nn.Module,
    predictor: nn.Module,
    spec: MovingPatchSpec,
    device: torch.device,
    n_episodes: int = 64,
) -> dict:
    """Cheap probe: given a history, can the predictor predict where the
    moving patch lands after `horizon` known actions? Measure feature MSE
    against ground-truth-encoded future frames. Lower = better world model.
    """
    encoder.eval()
    predictor.eval()
    history, actions, future = make_moving_patch_episodes(spec, n_episodes, device=device)
    b, t = history.shape[:2]
    flat = history.reshape(b * t, *history.shape[2:])
    z, _, _ = encoder(flat)
    history_emb = pool_patch_tokens(z).reshape(b, t, -1)
    bf, tf = future.shape[:2]
    future_flat = future.reshape(bf * tf, *future.shape[2:])
    z_fut, _, _ = encoder(future_flat)
    future_emb = pool_patch_tokens(z_fut).reshape(bf, tf, -1).detach()

    ctx = history_emb
    max_ctx = predictor.frame_pos_embed.size(1)
    losses = []
    for h in range(actions.shape[1]):
        z_pred = predictor(ctx, actions[:, h])
        target = future_emb[:, h]
        losses.append(F.mse_loss(z_pred.float(), target.float()).item())
        ctx = torch.cat([ctx, z_pred.unsqueeze(1)], dim=1)
        if ctx.size(1) > max_ctx:
            ctx = ctx[:, -max_ctx:]
    encoder.train()
    predictor.train()
    return {
        "lookahead_mse_h1": losses[0],
        "lookahead_mse_mean": float(sum(losses) / len(losses)),
    }


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output, exist_ok=True)

    spec = MovingPatchSpec(
        image_size=args.image_size,
        patch_size=args.patch_size,
        horizon=args.horizon,
        history=args.history,
        action_dim=args.action_dim,
    )
    encoder = PatchViTEncoder(
        in_channels=3,
        latent_dim=args.d,
        patch_size=args.patch_size,
        depth=args.encoder_depth,
        heads=args.encoder_heads,
    ).to(device)
    temporal_cfg = TemporalConfig(
        d=args.d,
        action_dim=args.action_dim,
        chunk_size=1,
        n_layers=args.predictor_depth,
        num_heads=args.predictor_heads,
        max_context=args.history + args.horizon + 4,
    )
    predictor = TemporalPredictor(temporal_cfg).to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    if is_main:
        print(json.dumps({"event": "init", "world_size": world, "n_params": n_params, "config": vars(args)}), flush=True)

    if world > 1:
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=False)
        # Predictor's reward_head + value_head don't receive gradient in this
        # loss (LeWM-recipe: only z_pred is used). find_unused_parameters=True
        # tells DDP not to wait for those buckets.
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)

    enc_module = encoder.module if world > 1 else encoder
    pred_module = predictor.module if world > 1 else predictor

    optim = torch.optim.AdamW(
        list(enc_module.parameters()) + list(pred_module.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    sigma2 = torch.tensor(1.0, device=device)
    sigma2_momentum = 0.99

    if world > 1:
        dist.barrier()

    t0 = time.time()
    log = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "n": 0}

    for step in range(args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        sigreg_lambda = linear_warmup(step, args.sigreg_warmup, args.sigreg_lambda)
        for pg in optim.param_groups:
            pg["lr"] = lr

        history, actions, future = make_moving_patch_episodes(spec, args.batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            b = history.shape[0]
            history_flat = history.reshape(b * args.history, *history.shape[2:])
            future_flat = future.reshape(b * args.horizon, *future.shape[2:])
            z_hist, _, _ = encoder(history_flat)
            z_fut, _, _ = encoder(future_flat)

            history_emb = pool_patch_tokens(z_hist).reshape(b, args.history, -1)
            future_emb = pool_patch_tokens(z_fut).reshape(b, args.horizon, -1)

            ctx = history_emb
            max_ctx = pred_module.frame_pos_embed.size(1)
            pred_losses = []
            for h in range(args.horizon):
                z_pred = predictor(ctx, actions[:, h])
                target = future_emb[:, h]
                pred_losses.append(F.mse_loss(z_pred.float(), target.float()) / sigma2.clamp(min=1e-3))
                ctx = torch.cat([ctx, z_pred.unsqueeze(1)], dim=1)
                if ctx.size(1) > max_ctx:
                    ctx = ctx[:, -max_ctx:]
            pred_loss = torch.stack(pred_losses).mean()

            with torch.no_grad():
                target_var = future_emb.detach().float().var().clamp(min=1e-3)
                sigma2 = sigma2_momentum * sigma2 + (1.0 - sigma2_momentum) * target_var

            all_features = torch.cat([
                pool_patch_tokens(z_hist).reshape(-1, args.d),
                pool_patch_tokens(z_fut).reshape(-1, args.d),
            ], dim=0)
            sig_loss = sigreg_epps_pulley(all_features.float(), n_directions=args.sigreg_directions)

            loss = pred_loss + sigreg_lambda * sig_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc_module.parameters()) + list(pred_module.parameters()), args.grad_clip
        )
        optim.step()

        log["loss"] += float(loss.detach())
        log["pred"] += float(pred_loss.detach())
        log["sig"] += float(sig_loss.detach())
        log["n"] += 1

        if is_main and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            ips = (step + 1) * args.batch_size * world / max(elapsed, 1e-6)
            print(json.dumps({
                "step": step + 1,
                "loss": log["loss"] / log["n"],
                "pred_loss": log["pred"] / log["n"],
                "sigreg_loss": log["sig"] / log["n"],
                "sigreg_lambda": sigreg_lambda,
                "sigma2": float(sigma2),
                "lr": lr,
                "imgs_per_sec": round(ips, 1),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)
            log = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "n": 0}

        if is_main and (step + 1) % args.probe_every == 0:
            probe = navigate_lookahead_probe(enc_module, pred_module, spec, device, n_episodes=64)
            print(json.dumps({"event": "probe", "step": step + 1, **probe}), flush=True)

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
