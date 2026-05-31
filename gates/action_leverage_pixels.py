#!/usr/bin/env python3
"""Pixel-space action-leverage check on the RENDERED dataset.

The state-space preflight gave r_action=0.996 on Reacher's joint state. This
confirms the leverage survives the rendering pipeline (e.g. low-res aliasing
could erase small joint motions). Measured on FIXED RANDOM CNN features of the
rendered frames (representation-agnostic proxy for "is the action signal visible
in pixels"): r_action = (MSE_history_only - MSE_action_conditioned)/MSE_history_only
on one-step feature prediction. Milestone bar: r_action > 0.5.

    python -m gates.action_leverage_pixels --data runs/reacher_transitions.npz
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn


class RandomCNN(nn.Module):
    """Fixed (untrained) conv stack -> d-dim feature. Deterministic given seed."""
    def __init__(self, d=256, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),     # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),   # 16->8
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(128 * 4, d))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


def train_predictor(X, Y, steps, dev, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X, device=dev); Y = torch.tensor(Y, device=dev)
    n = len(X); idx = np.random.RandomState(seed).permutation(n); ntr = int(0.8 * n)
    tr = torch.tensor(idx[:ntr], device=dev); te = torch.tensor(idx[ntr:], device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        sel = tr[torch.randint(0, len(tr), (256,), generator=g, device=dev)]
        loss = ((net(X[sel]) - Y[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((net(X[te]) - Y[te]) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="gates/action_leverage_pixels.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    frames, actions, ep = d["frames"], d["actions"].astype(np.float32), d["ep_id"]
    phi = RandomCNN(seed=0).to(dev)
    # embed all frames in batches
    Z = []
    for i in range(0, len(frames), 1024):
        x = torch.tensor(frames[i:i + 1024], dtype=torch.float32, device=dev) / 255.0
        Z.append(phi(x).cpu().numpy())
    Z = np.concatenate(Z)
    pairs = np.where(ep[:-1] == ep[1:])[0]
    Zt, A, Ztp1 = Z[pairs], actions[pairs], Z[pairs + 1]
    # z-score target per-dim (ratio is scale-invariant; keeps MSE dimensionless)
    mu, sd = Ztp1.mean(0), Ztp1.std(0) + 1e-6
    Yn, Zn = (Ztp1 - mu) / sd, (Zt - mu) / sd
    copy_mse = float(((Yn - Zn) ** 2).mean())
    mse_hist = train_predictor(Zn, Yn, args.steps, dev)
    mse_act = train_predictor(np.concatenate([Zn, A], 1), Yn, args.steps, dev)
    r = (mse_hist - mse_act) / max(mse_hist, 1e-9)
    res = {"n_pairs": int(len(pairs)), "feature": "random_cnn_d256",
           "copy_forward_mse": round(copy_mse, 4),
           "mse_history_only": round(mse_hist, 4),
           "mse_action_conditioned": round(mse_act, 4),
           "r_action_pixels": round(float(r), 4), "threshold": args.threshold,
           "pass": bool(r > args.threshold)}
    print(json.dumps(res, indent=2))
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nPIXEL LEVERAGE: {'PASS' if res['pass'] else 'FAIL'} "
          f"(r_action={res['r_action_pixels']} vs {args.threshold}; state-space was 0.996)")


if __name__ == "__main__":
    main()
