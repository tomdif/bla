"""DDP-aware JEPA pretraining on CIFAR-10 (or synthetic fallback).

Run with torchrun:
    torchrun --nproc_per_node=5 scripts/jepa_pretrain.py \\
        --steps 20000 --batch-size 256 --d 512 --depth 8 \\
        --output runs/jepa_pretrain --probe-every 500

Each rank holds its own model + optimizer; gradients are all-reduced through
DDP. AMP autocast is bf16 (free on B200). EMA target encoder is held in
fp32 to avoid the standard bf16-EMA underflow.

Logs JSON per step, saves checkpoints every --ckpt-every steps, runs a
linear probe every --probe-every steps to give an honest signal that
features are learning useful structure (linear-classify image color
quadrant, accuracy >> 25% chance is the smoke test).
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
    BLAJEPAModel,
    ImageBatchSpec,
    JEPAConfig,
    make_image_loader,
    pool_patch_tokens,
    sigreg_epps_pulley,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256, help="per-GPU batch")
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=8)
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-tau", type=float, default=0.996)
    parser.add_argument("--data", choices=["cifar10", "synthetic", "auto"], default="auto")
    parser.add_argument("--augment", action="store_true",
                        help="Random crop + flip + color jitter on training images.")
    parser.add_argument("--action-mode", choices=["random", "zero"], default="random",
                        help="random=feed torch.randn as action, zero=feed zeros (effectively action-free).")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="runs/jepa_pretrain")
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--probe-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def is_main(rank: int) -> bool:
    return rank == 0


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


def linear_probe(
    encoder: nn.Module,
    loader_iter,
    device: torch.device,
    dtype: torch.dtype,
    n_quadrant_classes: int = 4,
    train_steps: int = 100,
    eval_steps: int = 50,
    probe_d: int = 512,
) -> dict:
    """Probe whether features can predict 'where is the brightest 8x8 region of the image?'

    Cheap, label-free: brightness quadrant is well-defined for any image, makes
    a 4-way classification problem with 25% chance accuracy.
    """

    probe = nn.Linear(probe_d, n_quadrant_classes).to(device=device)
    optim = torch.optim.AdamW(probe.parameters(), lr=3e-3)

    def labels_from(images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        h2, w2 = h // 2, w // 2
        brightness = images.mean(dim=1)
        q = torch.stack([
            brightness[:, :h2, :w2].mean(dim=(1, 2)),
            brightness[:, :h2, w2:].mean(dim=(1, 2)),
            brightness[:, h2:, :w2].mean(dim=(1, 2)),
            brightness[:, h2:, w2:].mean(dim=(1, 2)),
        ], dim=1)
        return q.argmax(dim=1)

    encoder.eval()
    for _ in range(train_steps):
        images = next(loader_iter).to(device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            z, _, _ = encoder(images)
        feat = pool_patch_tokens(z).float().detach()
        targets = labels_from(images)
        logits = probe(feat)
        loss = F.cross_entropy(logits, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    correct = 0
    seen = 0
    with torch.no_grad():
        for _ in range(eval_steps):
            images = next(loader_iter).to(device=device)
            with torch.amp.autocast("cuda", dtype=dtype):
                z, _, _ = encoder(images)
            feat = pool_patch_tokens(z).float()
            targets = labels_from(images)
            logits = probe(feat)
            correct += int((logits.argmax(dim=-1) == targets).sum())
            seen += int(targets.numel())
    return {"probe_accuracy": correct / max(seen, 1), "probe_eval_seen": seen}


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main(rank):
        os.makedirs(args.output, exist_ok=True)

    config = JEPAConfig(
        d_jepa=args.d,
        in_channels=3,
        patch_size=args.patch_size,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        encoder_mlp_ratio=4.0,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        action_dim=args.d,
        ema_tau=args.ema_tau,
        sigreg_weight=args.sigreg_weight,
        target_mask_ratio=args.target_ratio,
        dtype="float32",
    )
    model = BLAJEPAModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main(rank):
        print(json.dumps({"event": "init", "world_size": world, "n_params": n_params, "config": vars(args)}), flush=True)

    if world > 1:
        model.context_encoder = DDP(model.context_encoder, device_ids=[local_rank], find_unused_parameters=False)
        model.predictor = DDP(model.predictor, device_ids=[local_rank], find_unused_parameters=False)

    trainable = list((model.context_encoder.module if world > 1 else model.context_encoder).parameters()) + \
                list((model.predictor.module if world > 1 else model.predictor).parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    spec = ImageBatchSpec(image_size=args.image_size, batch_size=args.batch_size)
    train_loader = make_image_loader(spec, source=args.data, seed=args.seed + rank * 1000, augment=args.augment)
    probe_loader = make_image_loader(
        ImageBatchSpec(image_size=args.image_size, batch_size=128),
        source=args.data,
        seed=args.seed + 7777,
    )
    if world > 1:
        dist.barrier()
        if is_main(rank):
            print(json.dumps({"event": "loaders_ready", "world": world}), flush=True)

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    start_step = 0
    if args.resume and is_main(rank) and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        (model.context_encoder.module if world > 1 else model.context_encoder).load_state_dict(ckpt["context"])
        model.target_encoder.load_state_dict(ckpt["target"])
        (model.predictor.module if world > 1 else model.predictor).load_state_dict(ckpt["predictor"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt.get("step", 0)
        print(json.dumps({"event": "resumed", "step": start_step}), flush=True)

    t0 = time.time()
    log_loss = 0.0
    log_pred = 0.0
    log_sig = 0.0
    log_n = 0

    for step in range(start_step, args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        images = next(train_loader).to(device=device, non_blocking=True)
        if args.action_mode == "zero":
            action = torch.zeros(images.shape[0], config.action_dim, device=device)
        else:
            action = torch.randn(images.shape[0], config.action_dim, device=device)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            metrics = model.training_loss(images, action)
        loss = metrics["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optim.step()
        model.update_target_ema()

        log_loss += float(loss.detach())
        log_pred += float(metrics["prediction"])
        log_sig += float(metrics["sigreg"])
        log_n += 1

        if is_main(rank) and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            ips = (step + 1 - start_step) * args.batch_size * world / max(elapsed, 1e-6)
            print(json.dumps({
                "step": step + 1,
                "loss": log_loss / log_n,
                "prediction": log_pred / log_n,
                "sigreg": log_sig / log_n,
                "lr": lr,
                "imgs_per_sec": round(ips, 1),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)
            log_loss = log_pred = log_sig = 0.0
            log_n = 0

        if is_main(rank) and (step + 1) % args.probe_every == 0:
            target_for_probe = model.target_encoder
            probe = linear_probe(target_for_probe, probe_loader, device, amp_dtype, probe_d=args.d)
            print(json.dumps({"event": "probe", "step": step + 1, **probe}), flush=True)

        if is_main(rank) and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.output, f"ckpt_step{step + 1:08d}.pt")
            torch.save({
                "context": (model.context_encoder.module if world > 1 else model.context_encoder).state_dict(),
                "target": model.target_encoder.state_dict(),
                "predictor": (model.predictor.module if world > 1 else model.predictor).state_dict(),
                "optim": optim.state_dict(),
                "config": vars(args),
                "step": step + 1,
            }, ckpt_path)
            print(json.dumps({"event": "checkpoint", "step": step + 1, "path": ckpt_path}), flush=True)

    if is_main(rank):
        final_path = os.path.join(args.output, "final.pt")
        torch.save({
            "context": (model.context_encoder.module if world > 1 else model.context_encoder).state_dict(),
            "target": model.target_encoder.state_dict(),
            "predictor": (model.predictor.module if world > 1 else model.predictor).state_dict(),
            "config": vars(args),
            "step": args.steps,
        }, final_path)
        print(json.dumps({"event": "final", "path": final_path, "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
