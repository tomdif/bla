"""Phase 17 — Fine-tune (mixed-data retrain) action-conditioned OF-JEPA on
focused-contact action distributions.

Goal: fix the calibration failure Phase 16 found where the v3-trained
predictor's ranking goes NEGATIVE on goal-directed-push action
distributions (corr=-0.39).

Approach: train fresh model on a 50/50 batch mix of:
  - v3 broad-scripted distribution (existing cache)
  - goal-directed-push focused-contact distribution (new cache)

Uses the same training loop as slot_jepa_robosuite_train.py
(reuses train_one_run) so the architecture and hyperparameters
match Phase 14/15/16's action-conditioned model.

Usage:
    python scripts/phase17_finetune.py \\
        --train-caches /workspace/robosuite_local/stack_scripted,\\
/workspace/robosuite_local/stack_goal_directed \\
        --train-mix 0.5,0.5 \\
        --max-steps 1500 --jepa-stride 4 --seed 0 \\
        --model-out /workspace/phase17/model_action_finetuned.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.robosuite_data import RobosuiteDataset, RobosuiteSpec
from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.identity_probe import hungarian_assign
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA


class MixedDataset(torch.utils.data.Dataset):
    """Sample-weighted concatenation: each draw picks a sub-dataset by weight."""
    def __init__(self, datasets, weights, length_per_epoch):
        self.datasets = datasets
        self.weights = np.array(weights, dtype=np.float64) / sum(weights)
        self.length = length_per_epoch
        self.n_each = [len(d) for d in datasets]

    def __len__(self): return self.length

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        which = int(rng.choice(len(self.datasets), p=self.weights))
        sub_idx = int(rng.randint(0, self.n_each[which]))
        return self.datasets[which][sub_idx]


def train_one_run_mixed(model, mixed_dataset, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(mixed_dataset, batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    step = 0; t0 = time.time()
    for epoch in range(200):
        for batch in loader:
            if step >= args.max_steps: break
            video = batch["video"][0].to(device)
            actions = batch["actions"][0].to(device)
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()
            T = video.shape[0]

            opt.zero_grad(set_to_none=True)
            slot_states, _ = model.encode_video_grad(video)
            id_dim = model.cfg.id_dim
            stride = args.jepa_stride

            future_loss = 0.0
            for t in range(T - stride):
                state_pred = model.predict_state_delta(slot_states[t:t+1], actions[t:t+1])
                state_target = slot_states[t + stride, :, id_dim:].detach().unsqueeze(0)
                future_loss = future_loss + F.mse_loss(state_pred, state_target)
            future_loss = future_loss / max(T - stride, 1)

            pred_pos = model.slot_to_pos_aux(slot_states)
            pos_loss = 0.0; pos_count = 0
            for t in range(T):
                vm = gt_vis[t]
                if not vm.any(): continue
                pp_t = pred_pos[t].unsqueeze(0)
                gt_t = gt_pos[t][vm].unsqueeze(0)
                if gt_t.shape[1] == 0: continue
                rows, cols, _ = hungarian_assign(pp_t[0].detach().cpu().numpy(),
                                                  gt_t[0].detach().cpu().numpy())
                if len(rows) > 0:
                    rs = torch.from_numpy(rows).to(device)
                    cs = torch.from_numpy(cols).to(device)
                    pos_loss = pos_loss + F.mse_loss(pp_t[0, rs], gt_t[0, cs])
                    pos_count += 1
            pos_loss = pos_loss / max(pos_count, 1)
            loss = future_loss + 10.0 * pos_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                print(f"  step {step}/{args.max_steps} "
                       f"future={float(future_loss):.4f} pos={float(pos_loss):.4f} "
                       f"t={time.time()-t0:.0f}s", flush=True)
        if step >= args.max_steps: break
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-caches", required=True,
                    help="Comma-separated paths to v3 and goal-directed caches")
    p.add_argument("--train-mix", default="0.5,0.5",
                    help="Comma-separated mixture weights matching train-caches")
    p.add_argument("--model-out", required=True)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--length-per-epoch", type=int, default=160,
                    help="Virtual epoch length for mixed sampling.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    caches = [c.strip() for c in args.train_caches.split(",")]
    weights = [float(w) for w in args.train_mix.split(",")]
    datasets = []
    for c in caches:
        ds = RobosuiteDataset(RobosuiteSpec(cache_dir=c, image_size=args.image_size))
        datasets.append(ds)
        print(f"Loaded cache: {c}  episodes={len(ds)}", flush=True)
    mixed = MixedDataset(datasets, weights, length_per_epoch=args.length_per_epoch)
    print(f"Mixed dataset: weights={weights}, virtual length={args.length_per_epoch}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
    model = ActionConditionedOFJEPA(image_size=args.image_size, cfg=cfg,
                                     action_dim=7, use_action=True).to(args.device)
    train_one_run_mixed(model, mixed, args, args.device)
    torch.save(model.state_dict(), args.model_out)
    print(f"Saved {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
