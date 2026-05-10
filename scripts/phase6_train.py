"""Phase 6 — train the procedural CPU on the synthetic logic curriculum.

FSDP across N GPUs. bf16 mixed precision. Cosine LR with warmup. Tied
input/output embeddings. Standard causal LM loss with prompt/target
masking — only the target tokens contribute to the loss; the prompt is
context.

Run:
    torchrun --nproc_per_node=3 scripts/phase6_train.py \\
        --curriculum runs/phase6/curriculum.jsonl \\
        --steps 30000 --batch-size 16 --d 1280 --n-layers 24 --n-heads 20 \\
        --output runs/phase6/run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
import torch.nn as nn

from system2_dca.procedural_core import ProceduralCore, ProceduralCoreConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum", required=True, help="JSONL file from build_curriculum.py")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=16, help="per-GPU")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--d", type=int, default=1280)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--output", required=True)
    p.add_argument("--no-fsdp", action="store_true", help="single-GPU smoke")
    p.add_argument("--amp", action="store_true", default=True)
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


def cosine_lr(step: int, warmup: int, total: int, peak: float, min_lr: float = 3e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


class CurriculumDataset:
    """Streams JSONL examples and tokenizes them on the fly. Each yielded
    record has prompt_ids, target_ids, full_ids, label_ids (with prompt
    tokens replaced by -100 so they don't contribute to loss)."""

    def __init__(self, path: str, tokenizer, seq_len: int, seed: int = 0):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self._lines = self._load()

    def _load(self) -> list[str]:
        with open(self.path) as f:
            lines = [l for l in f if l.strip()]
        return lines

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        for _ in range(10):  # bounded retry
            line = self.rng.choice(self._lines)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = rec["prompt"]
            target = rec["target"] + self.tokenizer.eos_token
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            full = prompt_ids + target_ids
            if len(full) >= self.seq_len:
                full = full[: self.seq_len]
            # labels: -100 on prompt positions; target tokens kept
            labels = [-100] * len(prompt_ids) + target_ids
            labels = labels[: self.seq_len]
            return {
                "input_ids": torch.tensor(full, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "source": rec.get("source", "unknown"),
            }
        raise StopIteration


def collate(batch: list[dict], pad_id: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each sequence to seq_len. Labels padded with -100."""
    L = seq_len
    B = len(batch)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for i, rec in enumerate(batch):
        ids = rec["input_ids"]
        lbs = rec["labels"]
        n = min(len(ids), L)
        input_ids[i, :n] = ids[:n]
        labels[i, :n] = lbs[:n]
    return input_ids, labels


def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output, exist_ok=True)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    cfg = ProceduralCoreConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    )
    model = ProceduralCore(cfg).to(device, dtype=torch.bfloat16)
    n_params = model.n_parameters()
    if is_main:
        print(json.dumps({"event": "init", "world": world, "n_params": n_params,
                          "config": {"d": args.d, "layers": args.n_layers,
                                     "heads": args.n_heads, "seq_len": args.seq_len,
                                     "vocab": cfg.vocab_size}}), flush=True)

    if world > 1 and not args.no_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from system2_dca.procedural_core import TransformerBlock

        mp = MixedPrecision(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.bfloat16,
                            buffer_dtype=torch.bfloat16)
        wrap_policy = lambda module, recurse, nonwrapped_numel: isinstance(module, TransformerBlock)
        # Embedding, final_norm, LM head stay replicated rather than
        # sharded — small in param count and don't tolerate 1-D shards
        # (F.embedding wants 2-D weight, F.rms_norm wants weight matching
        # normalized_shape).
        ignored = [model.tok_embed, model.final_norm, model.head]
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
            ignored_modules=ignored,
        )
    elif world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = CurriculumDataset(args.curriculum, tokenizer, args.seq_len, seed=args.seed + rank)
    pad_id = tokenizer.pad_token_id

    if world > 1:
        dist.barrier()

    t0 = time.time()
    log = {"loss": 0.0, "n": 0}
    for step in range(args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        batch = [next(dataset) for _ in range(args.batch_size)]
        input_ids, labels = collate(batch, pad_id, args.seq_len)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = model.module.loss(input_ids, labels) if hasattr(model, "module") else model.loss(input_ids, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        log["loss"] += float(loss.detach())
        log["n"] += 1

        if is_main and (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.seq_len * world / max(elapsed, 1e-6)
            print(json.dumps({
                "step": step + 1,
                "loss": log["loss"] / log["n"],
                "lr": lr,
                "tokens_per_sec": round(tps, 0),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)
            log = {"loss": 0.0, "n": 0}

        if is_main and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.output, f"ckpt_step{step + 1:08d}.pt")
            state = (model.module if hasattr(model, "module") else model).state_dict()
            torch.save({
                "state_dict": state,
                "config": {"d": args.d, "n_layers": args.n_layers,
                           "n_heads": args.n_heads, "seq_len": args.seq_len,
                           "vocab": cfg.vocab_size},
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            print(json.dumps({"event": "checkpoint", "step": step + 1, "path": ckpt_path}), flush=True)

    if is_main:
        final_path = os.path.join(args.output, "final.pt")
        state = (model.module if hasattr(model, "module") else model).state_dict()
        torch.save({
            "state_dict": state,
            "config": {"d": args.d, "n_layers": args.n_layers,
                       "n_heads": args.n_heads, "seq_len": args.seq_len,
                       "vocab": cfg.vocab_size},
            "step": args.steps,
            "args": vars(args),
        }, final_path)
        print(json.dumps({"event": "final", "path": final_path,
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
