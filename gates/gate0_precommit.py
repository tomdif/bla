#!/usr/bin/env python3
"""Gate 0 precommit harness (system1_motion_spec.md §6).

Substrate-agnostic pass/fail gate: given frozen-substrate embeddings Z and
ground-truth agent positions P (in PIXELS), train a FRESH small decoder on a
held-out split and report mean L2 position error. Pass iff < threshold_px.

Design invariants (from the spec):
  * decoder is fresh per evaluation (no carryover) and small (2-layer MLP)
  * deterministic given seed (fixed steps/batch order/init)
  * trained on sg(Z) only — the substrate never sees decoder gradients here
  * records render resolution (img_px) + hashes so pass/fail is comparable
  * cheap (<10 min) so it can run as a training-time early-stopping diagnostic

decode_gate(...) is importable for the training-time diagnostic; the CLI runs it
on a .npz of {Z:[N,D] float32, P:[N,2] pixels, img_px:int}.

    python -m gates.gate0_precommit --npz emb.npz --out gate0.json
"""
from __future__ import annotations

import argparse, json, hashlib, time, os
import numpy as np
import torch
import torch.nn as nn


def _hash(arr) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()[:12]


def decode_gate(Z, P, img_px, *, steps=5000, hidden=256, seed=0, split=0.8,
                threshold_px=5.0, device=None):
    """Z:[N,D], P:[N,2] in pixels. Fresh deterministic decoder. -> result dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Z = np.asarray(Z, dtype=np.float32); P = np.asarray(P, dtype=np.float32)
    n = len(Z)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    ntr = int(split * n)
    tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(seed)
    Zt = torch.tensor(Z, device=device); Pt = torch.tensor(P, device=device)
    dec = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    trt = torch.tensor(tr, device=device)
    bs = min(256, len(trt))
    for _ in range(steps):
        sel = trt[torch.randint(0, len(trt), (bs,), generator=g, device=device)]
        loss = ((dec(Zt[sel]) - Pt[sel]) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    tet = torch.tensor(te, device=device)
    with torch.no_grad():
        pred = dec(Zt[tet])
        l2 = ((pred - Pt[tet]) ** 2).sum(-1).sqrt()
        per_axis = torch.sqrt(((pred - Pt[tet]) ** 2).mean(0))
    mean_px = float(l2.mean())
    return {
        "mean_l2_px": round(mean_px, 3),
        "rmse_x_px": round(float(per_axis[0]), 3),
        "rmse_y_px": round(float(per_axis[1]), 3),
        "img_px": int(img_px),
        "threshold_px": float(threshold_px),
        "pass": bool(mean_px < threshold_px),
        "n_train": int(len(tr)), "n_test": int(len(te)), "z_dim": int(Z.shape[1]),
        "decoder_steps": int(steps), "seed": int(seed),
        "embeddings_hash": _hash(Z), "positions_hash": _hash(P),
        "pred_vs_true_sample": [[round(float(a), 1), round(float(b), 1)]
                                for a, b in zip(pred[:8].flatten(0).tolist()[:8],
                                                Pt[tet][:8].flatten(0).tolist()[:8])],
    }


def _scatter(npz_Z, npz_P, res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # re-decode test preds for plot omitted for brevity; plot true distribution
        plt.figure(figsize=(4, 4))
        plt.scatter(npz_P[:, 0], npz_P[:, 1], s=3, alpha=0.3)
        plt.title(f"GT agent positions (mean_l2={res['mean_l2_px']}px, {'PASS' if res['pass'] else 'FAIL'})")
        plt.savefig(path, dpi=80); plt.close()
        return path
    except Exception as e:
        return f"(scatter skipped: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="npz with Z[N,D], P[N,2] pixels, img_px")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--threshold-px", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gate0_precommit.json")
    args = ap.parse_args()
    d = np.load(args.npz)
    Z, P = d["Z"], d["P"]
    img_px = int(d["img_px"]) if "img_px" in d else 0
    res = decode_gate(Z, P, img_px, steps=args.steps, threshold_px=args.threshold_px, seed=args.seed)
    res["timestamp"] = time.time()
    res["npz"] = os.path.abspath(args.npz)
    res["scatter"] = _scatter(Z, P, res, args.out.replace(".json", "_scatter.png"))
    print(json.dumps(res, indent=2))
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nGATE 0: {'PASS' if res['pass'] else 'FAIL'} "
          f"(mean_l2={res['mean_l2_px']}px vs {res['threshold_px']}px threshold @ {img_px}px render)")


if __name__ == "__main__":
    main()
