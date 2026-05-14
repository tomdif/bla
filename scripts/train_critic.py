"""Train a small critic to predict P(correct) for (problem, code) pairs.

Data: labeled candidates from prior runs (problem, code, correct?).
Model: small transformer encoder (DistilBERT or similar).
Use case: at inference, rerank N PAL candidates and pick the one with
highest predicted P(correct).

Decision metric: does critic-pick beat mode-vote (currently 3.5-4%)
on the held-out run17 candidates?
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def load_records_labeled(paths):
    """Read multiple candidate JSONL files; each record is one problem
    with N candidates. Yield (problem, code, correct?) tuples."""
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            gold = r["gold"].rstrip(".")
            for c in r["candidates"]:
                if c.get("pred") is None:
                    continue
                pred = str(c["pred"]).rstrip(".")
                code = c.get("code", "")
                if not code:
                    continue
                yield (r["question"], code, int(pred == gold))


def train_critic(args):
    from transformers import (AutoTokenizer, AutoModel,
                              get_linear_schedule_with_warmup)

    train_paths = args.train.split(",")
    held_path = args.held

    train_data = list(load_records_labeled(train_paths))
    held_data = list(load_records_labeled([held_path])) if held_path else []
    print(f"train: {len(train_data)} pairs, positives: {sum(y for _,_,y in train_data)}")
    print(f"held:  {len(held_data)} pairs, positives: {sum(y for _,_,y in held_data)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModel.from_pretrained(args.model)
    backbone.train()

    class Critic(nn.Module):
        def __init__(self, backbone, hidden):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(hidden, 2)
        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Pool with first token
            pooled = h[:, 0]
            return self.head(pooled)

    hidden = backbone.config.hidden_size
    model = Critic(backbone, hidden).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    rng = random.Random(0)
    def batches(data, bs):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [data[j] for j in idx[i:i+bs]]

    def encode(batch):
        texts = [f"Problem: {q}\nCode:\n{c}" for q, c, _ in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt").to("cuda")
        y = torch.tensor([y for _, _, y in batch]).long().to("cuda")
        return enc, y

    n_steps = (len(train_data) // args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=100,
                                                num_training_steps=n_steps)
    pos_weight = (len(train_data) - sum(y for _,_,y in train_data)) / max(sum(y for _,_,y in train_data), 1)
    print(f"pos_weight: {pos_weight:.2f}")
    weights = torch.tensor([1.0, pos_weight]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in batches(train_data, args.batch_size):
            enc, y = encode(batch)
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                with torch.no_grad():
                    pred = logits.argmax(-1)
                    acc = (pred == y).float().mean().item()
                    print(json.dumps({"event": "train", "step": step, "epoch": epoch,
                                      "loss": loss.item(), "batch_acc": acc,
                                      "elapsed_s": round(time.time() - t0, 1)}),
                          flush=True)

    # Evaluate critic on held-out: per-problem, pick the candidate
    # with highest P(correct), check if it matches gold.
    model.eval()
    print(json.dumps({"event": "evaluating"}), flush=True)

    # Re-organize held data by problem
    held_by_problem = {}
    held_records = [json.loads(l) for l in open(args.held)]
    for r in held_records:
        gold = r["gold"].rstrip(".")
        candidates_with_label = []
        for c in r["candidates"]:
            pred = c.get("pred")
            code = c.get("code", "")
            if pred is None or not code:
                continue
            label = int(str(pred).rstrip(".") == gold)
            candidates_with_label.append((code, str(pred).rstrip("."), label))
        if candidates_with_label:
            held_by_problem[r["question"]] = (gold, candidates_with_label)

    correct_critic = 0
    correct_mode = 0
    correct_oracle = 0
    n_problems = 0
    with torch.no_grad():
        for q, (gold, cands) in held_by_problem.items():
            n_problems += 1
            # Score each candidate
            texts = [f"Problem: {q}\nCode:\n{code}" for code, _, _ in cands]
            scores = []
            for chunk_start in range(0, len(texts), args.batch_size):
                chunk = texts[chunk_start:chunk_start + args.batch_size]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=args.max_len, return_tensors="pt").to("cuda")
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=-1)[:, 1]  # P(correct)
                scores.extend(probs.cpu().tolist())
            # Critic pick
            best_idx = int(np.argmax(scores))
            critic_pred = cands[best_idx][1]
            if critic_pred == gold:
                correct_critic += 1
            # Mode vote (baseline)
            import collections
            votes = collections.Counter(pred for _, pred, _ in cands)
            mode_pred = votes.most_common(1)[0][0]
            if mode_pred == gold:
                correct_mode += 1
            # Oracle
            if any(label for _, _, label in cands):
                correct_oracle += 1

    print(json.dumps({
        "event": "held_eval",
        "n_problems": n_problems,
        "critic_correct": correct_critic,
        "critic_acc": correct_critic / n_problems,
        "mode_correct": correct_mode,
        "mode_acc": correct_mode / n_problems,
        "oracle_correct": correct_oracle,
        "oracle_acc": correct_oracle / n_problems,
    }, indent=2))

    # Save model
    torch.save({"model_state": model.state_dict(),
                "config": {"model_name": args.model}}, args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True,
                   help="comma-separated paths to candidate JSONL files for training")
    p.add_argument("--held", required=True,
                   help="held-out candidate JSONL for evaluation")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    train_critic(args)


if __name__ == "__main__":
    main()
