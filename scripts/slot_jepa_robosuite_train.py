"""Phase 14.3 — action-conditioned OF-JEPA training on robosuite Stack.

Adds explicit action conditioning to the OF-JEPA predictor: each step,
the 7-DOF action vector is broadcast across files and concatenated to
the predictor's input. The JEPA target is the encoder's slot state at
t+stride (same self-supervised target as MOVi/CLEVRER).

Two modes:
  of_jepa_v0          — no action conditioning (baseline; tests whether
                        an unconditioned model can still match)
  of_jepa_v0_action   — action vector injected into predictor

The Phase 14 question: does action conditioning measurably improve
future-state prediction on robosuite, where actions deterministically
drive object motion (robot pushes a cube)?

Usage:
    python scripts/slot_jepa_robosuite_train.py \\
        --cache /workspace/robosuite_local/stack \\
        --modes of_jepa_v0,of_jepa_v0_action \\
        --max-steps 1500 --out /workspace/phase14_run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.robosuite_data import RobosuiteDataset, RobosuiteSpec
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.identity_probe import hungarian_assign


class ActionConditionedOFJEPA(nn.Module):
    """Wraps an OFJEPA model with an action-conditioned future predictor head.

    The base OFJEPA model encodes video → object-file slot states. We
    add a small predictor that takes (slot_state[t], action[t]) and
    predicts slot_state[t+stride].
    """
    def __init__(self, image_size: int, cfg: OFJEPAConfig, action_dim: int,
                 use_action: bool = True):
        super().__init__()
        self.of_jepa = OFJEPA(image_size=image_size, cfg=cfg, version="v0")
        self.use_action = use_action
        self.action_dim = action_dim
        slot_dim = cfg.id_dim + cfg.state_dim
        in_dim = slot_dim + (action_dim if use_action else 0)
        self.future_head = nn.Sequential(
            nn.Linear(in_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, cfg.state_dim),  # predict only state_value delta
        )
        self.cfg = cfg

    def encode_video(self, video):
        return self.of_jepa.encode_video(video)

    def encode_video_grad(self, video):
        return self.of_jepa.encode_video_grad(video)

    def predict_state_delta(self, slot_t, action_t):
        """slot_t [B, S, slot_dim]; action_t [B, action_dim]. Return [B, S, state_dim]."""
        if self.use_action:
            a = action_t.unsqueeze(1).expand(-1, slot_t.size(1), -1)
            x = torch.cat([slot_t, a], dim=-1)
        else:
            x = slot_t
        return self.future_head(x)

    @property
    def slot_to_pos_aux(self):
        return self.of_jepa.slot_to_pos_aux


def train_one_run(model, dataset, train_idx, args, device, use_action: bool):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    step = 0; t0 = time.time()
    for epoch in range(200):
        for batch in loader:
            if step >= args.max_steps: break
            video = batch["video"][0].to(device)              # [T, 3, H, W]
            actions = batch["actions"][0].to(device)           # [T, action_dim]
            gt_pos = batch["positions"][0].to(device)          # [T, E, 2]
            gt_vis = batch["visibility"][0].to(device).bool()
            T = video.shape[0]

            opt.zero_grad(set_to_none=True)
            slot_states, _ = model.encode_video_grad(video)
            id_dim = model.cfg.id_dim
            state_dim = model.cfg.state_dim
            stride = args.jepa_stride

            # Action-conditioned state-delta prediction.
            future_loss = 0.0
            for t in range(T - stride):
                state_pred = model.predict_state_delta(
                    slot_states[t:t+1], actions[t:t+1],
                )  # [1, S, state_dim]
                state_target = slot_states[t + stride, :, id_dim:].detach().unsqueeze(0)
                future_loss = future_loss + F.mse_loss(state_pred, state_target)
            future_loss = future_loss / max(T - stride, 1)

            # Aux position supervision (same as MOVi/CLEVRER).
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
                print(f"  ({'action' if use_action else 'base'}) step {step}/{args.max_steps} "
                       f"future={float(future_loss):.4f} pos={float(pos_loss):.4f} t={time.time()-t0:.0f}s", flush=True)
        if step >= args.max_steps: break
    return step


def eval_future_mse(model, dataset, eval_idx, args, device, use_action: bool):
    """Evaluate k=stride-step future state prediction MSE."""
    model.eval()
    state_errors = []  # MSE between predicted state and true encoder state
    pos_errors = []     # Probe-decoded position error
    # Future state prediction with action vs without
    with torch.no_grad():
        for ep_i in eval_idx:
            s = dataset[ep_i]
            video = s["video"].to(device)
            actions = s["actions"].to(device)
            gt_pos = s["positions"].to(device)
            slot_states, _ = model.encode_video(video)
            T = video.shape[0]
            id_dim = model.cfg.id_dim
            stride = args.jepa_stride
            for t in range(T - stride):
                state_pred = model.predict_state_delta(slot_states[t:t+1], actions[t:t+1])
                state_target = slot_states[t + stride, :, id_dim:].unsqueeze(0)
                state_errors.append(float(F.mse_loss(state_pred, state_target).item()))
                # Position via Hungarian-match (n_files slots vs E_max entities).
                full_pred = torch.cat([slot_states[t, :, :id_dim].unsqueeze(0), state_pred], dim=-1)
                pred_pos = model.slot_to_pos_aux(full_pred)[0].cpu().numpy()
                true_pos = gt_pos[t + stride].cpu().numpy()
                rows, cols, _ = hungarian_assign(pred_pos, true_pos)
                if len(rows) > 0:
                    err = float(((pred_pos[rows] - true_pos[cols]) ** 2).sum(-1).mean())
                    pos_errors.append(err)
    return {
        "future_state_mse": float(np.mean(state_errors)),
        "future_pos_mse":   float(np.mean(pos_errors)),
        "n_samples":        len(state_errors),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--modes", default="of_jepa_v0,of_jepa_v0_action")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset = RobosuiteDataset(RobosuiteSpec(cache_dir=args.cache, image_size=args.image_size))
    n = len(dataset); print(f"Episodes: {n}, action_dim={dataset.action_dim}", flush=True)
    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    train_idx, eval_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    results = {}

    for seed in seeds:
        for mode in modes:
            use_action = mode.endswith("_action")
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"\n=== seed={seed} mode={mode} use_action={use_action} ===", flush=True)
            cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                                state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
            model = ActionConditionedOFJEPA(
                image_size=args.image_size, cfg=cfg,
                action_dim=dataset.action_dim, use_action=use_action,
            ).to(args.device)
            steps = train_one_run(model, dataset, train_idx, args, args.device, use_action)
            metrics = eval_future_mse(model, dataset, eval_idx, args, args.device, use_action)
            metrics["steps"] = steps; metrics["mode"] = mode; metrics["seed"] = seed
            key = f"seed{seed}_{mode}"
            results[key] = metrics
            with open(out / f"{key}.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  result: {metrics}", flush=True)

    # Aggregate + comparison
    print("\n=== Action-conditioning comparison ===")
    for seed in seeds:
        b = results.get(f"seed{seed}_of_jepa_v0")
        a = results.get(f"seed{seed}_of_jepa_v0_action")
        if b and a:
            for k in ["future_state_mse", "future_pos_mse"]:
                delta = (a[k] - b[k]) / max(b[k], 1e-9) * 100
                verdict = "✅ improved" if a[k] < b[k] * 0.95 else ("⚠ flat" if 0.95 <= a[k] / max(b[k], 1e-9) <= 1.05 else "❌ worse")
                print(f"  seed{seed} {k}: base={b[k]:.4e}  +action={a[k]:.4e}  ({delta:+.1f}%)  {verdict}")


if __name__ == "__main__":
    main()
