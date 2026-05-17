"""Phase 6 trainer, MT-SSM variant.

Identical to scripts/phase6_train.py but instantiates MTSSMCore
(`system2_dca/mt_ssm_core.py`) instead of the standard transformer
ProceduralCore. Same CLI flags + dataset pipeline; the substrate is
the only change.

The model's `.loss(input_ids, labels)` method returns (lm_loss +
pred_loss_weight * pred_loss); the trainer treats it as a single
scalar so no other code changes are required.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch.distributed as dist
import torch

# B300 (sm_103) compatibility — same as phase6_train.py
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

import torch.nn as nn
from system2_dca.mt_ssm_core import MTSSMCore, MTSSMConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum", required=True)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--state-fast", type=int, default=256)
    p.add_argument("--state-med", type=int, default=512)
    p.add_argument("--state-slow", type=int, default=1024)
    p.add_argument("--pred-loss-weight", type=float, default=0.05)
    p.add_argument("--use-memory", action="store_true",
                   help="Enable Phase 12 working memory slots")
    p.add_argument("--n-slots", type=int, default=16)
    p.add_argument("--slot-chunk", type=int, default=64)
    p.add_argument("--use-attractor", action="store_true",
                   help="Enable Phase 13 attractor refinement on tied output embeddings")
    p.add_argument("--attractor-layers", type=int, default=1)
    p.add_argument("--attractor-train-iters", type=int, default=3)
    p.add_argument("--attractor-infer-iters", type=int, default=8)
    p.add_argument("--attractor-n-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--output", required=True)
    p.add_argument("--no-fsdp", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2500)
    return p.parse_args()


# Reuse the dataset class verbatim
import re, hashlib

class CurriculumDataset:
    def __init__(self, path, tokenizer, seq_len, seed=0, split="train", val_frac=0.05):
        self.path = path; self.tokenizer = tokenizer; self.seq_len = seq_len
        self.rng = random.Random(seed); self.split = split; self.val_frac = val_frac
        self._lines = self._load()

    def _load(self):
        with open(self.path) as f:
            lines = [l for l in f if l.strip()]
        train, val = [], []
        for l in lines:
            h = int(hashlib.md5(l[:200].encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)
            (val if h < self.val_frac else train).append(l)
        return val if self.split == "val" else train

    def sample(self, n):
        batch = []
        for _ in range(n):
            line = self.rng.choice(self._lines)
            rec = json.loads(line)
            prompt_ids = self.tokenizer.encode(rec["prompt"])
            target_ids = self.tokenizer.encode(rec["target"])
            full_ids = prompt_ids + target_ids
            full_ids = full_ids[: self.seq_len]
            labels = [-100] * min(len(prompt_ids), len(full_ids))
            labels += full_ids[len(prompt_ids): len(full_ids)]
            pad_id = self.tokenizer.eos_token_id
            pad_n = self.seq_len - len(full_ids)
            full_ids = full_ids + [pad_id] * pad_n
            labels = labels + [-100] * pad_n
            batch.append((full_ids, labels))
        ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
        lbl = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return ids, lbl


def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        local_rank, rank, world = 0, 0, 1

    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        os.makedirs(args.output, exist_ok=True)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    cfg = MTSSMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d, n_layers=args.n_layers,
        state_fast=args.state_fast, state_med=args.state_med, state_slow=args.state_slow,
        max_seq_len=args.seq_len, dropout=args.dropout,
        pred_loss_weight=args.pred_loss_weight,
        use_memory=args.use_memory, n_slots=args.n_slots, slot_chunk=args.slot_chunk,
        use_attractor=args.use_attractor,
        attractor_layers=args.attractor_layers,
        attractor_train_iters=args.attractor_train_iters,
        attractor_infer_iters=args.attractor_infer_iters,
        attractor_n_heads=args.attractor_n_heads,
    )
    model = MTSSMCore(cfg).to(device, dtype=torch.bfloat16)
    n_params = model.n_parameters()
    if is_main:
        print(json.dumps({"event": "init", "world": world, "n_params": n_params,
                          "config": {"d": args.d, "layers": args.n_layers,
                                     "state_fast": args.state_fast,
                                     "state_med": args.state_med,
                                     "state_slow": args.state_slow,
                                     "pred_loss_weight": args.pred_loss_weight,
                                     "use_memory": args.use_memory,
                                     "n_slots": args.n_slots,
                                     "slot_chunk": args.slot_chunk,
                                     "use_attractor": args.use_attractor,
                                     "attractor_layers": args.attractor_layers,
                                     "attractor_train_iters": args.attractor_train_iters,
                                     "attractor_infer_iters": args.attractor_infer_iters,
                                     "seq_len": args.seq_len,
                                     "vocab": cfg.vocab_size,
                                     "backbone": "mt_ssm"}}), flush=True)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = CurriculumDataset(args.curriculum, tokenizer, args.seq_len,
                                 seed=args.seed + rank, split="train", val_frac=args.val_frac)
    val_dataset = CurriculumDataset(args.curriculum, tokenizer, args.seq_len,
                                     seed=args.seed + 7777, split="val", val_frac=args.val_frac)
    if is_main:
        print(json.dumps({"event": "data", "n_train": len(dataset._lines),
                          "n_val": len(val_dataset._lines)}), flush=True)

    # Cosine LR with linear warmup
    import math
    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        prog = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    @torch.no_grad()
    def compute_val_loss():
        model.eval()
        total, n = 0.0, 0
        for _ in range(args.val_batches):
            ids, lbl = val_dataset.sample(args.batch_size)
            ids = ids.to(device); lbl = lbl.to(device)
            unwrapped = model.module if hasattr(model, "module") else model
            loss = unwrapped.loss(ids, lbl)
            total += loss.item(); n += 1
        model.train()
        return total / max(n, 1)

    model.train()
    t0 = time.time()
    last_log_t = t0
    n_tokens = 0
    for step in range(1, args.steps + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        ids, lbl = dataset.sample(args.batch_size)
        ids = ids.to(device); lbl = lbl.to(device)
        unwrapped = model.module if hasattr(model, "module") else model
        loss = unwrapped.loss(ids, lbl)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        n_tokens += ids.numel() * world

        if step % args.log_every == 0 and is_main:
            now = time.time()
            tps = n_tokens / (now - t0)
            print(json.dumps({"step": step, "loss": loss.item(),
                              "lr": lr_at(step), "tokens_per_sec": round(tps, 1),
                              "elapsed_s": round(now - t0, 1)}), flush=True)
        if step % args.val_every == 0 and is_main:
            v = compute_val_loss()
            print(json.dumps({"event": "val", "step": step, "val_loss": round(v, 4)}), flush=True)
        if step % args.ckpt_every == 0 and is_main:
            unwrapped = model.module if hasattr(model, "module") else model
            path = os.path.join(args.output, f"ckpt_step{step:08d}.pt")
            torch.save({
                "state_dict": unwrapped.state_dict(),
                "config": {"d": args.d, "n_layers": args.n_layers,
                           "state_fast": args.state_fast,
                           "state_med": args.state_med,
                           "state_slow": args.state_slow,
                           "use_memory": args.use_memory,
                           "n_slots": args.n_slots,
                           "slot_chunk": args.slot_chunk,
                           "use_attractor": args.use_attractor,
                           "attractor_layers": args.attractor_layers,
                           "attractor_train_iters": args.attractor_train_iters,
                           "attractor_infer_iters": args.attractor_infer_iters,
                           "attractor_n_heads": args.attractor_n_heads,
                           "seq_len": args.seq_len, "vocab": cfg.vocab_size,
                           "backbone": "mt_ssm"},
                "step": step, "args": vars(args),
            }, path)
            print(json.dumps({"event": "checkpoint", "step": step, "path": path}),
                  flush=True)

    if is_main:
        unwrapped = model.module if hasattr(model, "module") else model
        final_path = os.path.join(args.output, "final.pt")
        torch.save({
            "state_dict": unwrapped.state_dict(),
            "config": {"d": args.d, "n_layers": args.n_layers,
                       "state_fast": args.state_fast, "state_med": args.state_med,
                       "state_slow": args.state_slow,
                       "use_memory": args.use_memory,
                       "n_slots": args.n_slots,
                       "slot_chunk": args.slot_chunk,
                       "seq_len": args.seq_len, "vocab": cfg.vocab_size,
                       "backbone": "mt_ssm"},
            "args": vars(args),
        }, final_path)
        print(json.dumps({"event": "final", "path": final_path,
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
