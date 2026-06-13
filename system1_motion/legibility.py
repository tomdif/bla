#!/usr/bin/env python3
"""Frozen probe-battery scorer — the shared evaluation harness for the encoder
head-to-head. Training is decoupled from evaluation: every (encoder, objective)
cell saves an encoder checkpoint via train_dissoc.py, and THIS scores them all
identically so the comparison is fair.

It rebuilds the encoder from the checkpoint's saved `args` (pool ViT or OF-JEPA
slot), embeds the SAME held-out eval frames, and runs a fresh-linear-readout probe
for each ground-truth state variable:

  ABSOLUTE (px):   arm/finger position, target position
  RELATIVE (R^2):  joint proprioception cos/sin(qpos), angular velocity qvel

Each probe is a fresh MLP trained on 80% / evaluated on 20% (held-out), reported as
R^2 (=1-mse/var) and, for 2D positions, mean L2 in pixels. The composite
"legibility" = mean held-out R^2 across the battery. NOTE: legibility is a
DIAGNOSTIC, not the ranker — select the ideal setup by the planning end-effect
(plan_eval.py); use this to explain *why* a cell wins.

    python -m system1_motion.legibility --ckpt runs/dissoc/substrate_C1_s0.pt \
        --data runs/reacher_transitions.npz --out runs/legibility/C1_s0.json
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch, torch.nn as nn

from system1_motion.models import ViTEncoder
from system1_motion.slot_encoder import SlotEncoder


def eval_ends(ep, N, S, frac):
    """Disjoint-pair ends (matches data.TransitionDataset) without loading frames twice."""
    pairs = np.array([j for j in range(S - 1, N - S) if ep[j - S + 1] == ep[j + S]])
    k = int((1 - frac) * len(pairs))
    return pairs[k:]


def build_encoder(a, in_ch):
    """Reconstruct the encoder from the checkpoint's saved args dict."""
    if a.get("encoder", "pool") == "slot":
        enc = SlotEncoder(in_channels=in_ch, vit_dim=a.get("vit_dim", 192), patch=a["patch"],
                          depth=a["enc_depth"], n_slots=a.get("n_slots", 6), slot_dim=a.get("slot_dim", 64))
    else:
        enc = ViTEncoder(a["image_size"], a["patch"], in_ch, a["d_z"], a["enc_depth"])
    return enc


@torch.no_grad()
def embed(enc, frames, device, bs=512):
    enc.eval(); Z = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i+bs]).float().to(device) / 255.0
        Z.append(enc(x).cpu().numpy())
    return np.concatenate(Z)


def regress_probe(Z, Y, device, steps=5000, hidden=256, seed=0):
    """Fresh-MLP linear-readout probe. Returns held-out R^2 and per-sample preds
    (in original Y units) so callers can compute px for position targets."""
    n = len(Z); rng = np.random.RandomState(seed)
    zmu, zsd = Z.mean(0), Z.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    Zt = torch.tensor((Z - zmu) / zsd, dtype=torch.float32, device=device)
    Yt = torch.tensor((Y - ymu) / ysd, dtype=torch.float32, device=device)
    idx = rng.permutation(n); tr = torch.tensor(idx[:int(.8*n)], device=device); te = torch.tensor(idx[int(.8*n):], device=device)
    net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        s = tr[torch.randint(0, len(tr), (256,), generator=g, device=device)]
        loss = ((net(Zt[s]) - Yt[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_n = net(Zt[te]).cpu().numpy()
    pred = pred_n * ysd + ymu                              # back to original units
    Yte = Y[idx[int(.8*n):]]
    mse = ((pred - Yte) ** 2).mean(); var = Yte.var() + 1e-9
    return {"r2": round(float(1 - mse / var), 4)}, pred, Yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or args.device in ("cpu", "mps")) else "cpu"

    ck = torch.load(args.ckpt, map_location=dev); a = ck["args"]; S = a["frame_stack"]; in_ch = 3 * S
    enc = build_encoder(a, in_ch).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()

    d = np.load(args.data); frames = d["frames"]; H = frames.shape[2]
    ends = eval_ends(d["ep_id"].astype(np.int64), len(frames), S, args.eval_frac)
    past = np.stack([frames[e-S+1:e+1].reshape(-1, H, H) for e in ends]).astype(np.uint8)
    Z = embed(enc, past, dev)
    # COLLAPSE GUARD: a standardized probe can dig signal out of a near-zero latent
    # and make a collapsed encoder look great. Measure the raw embedding scale; if it
    # collapsed, the probe R^2 is an artifact and the cell is INVALID.
    z_std_med = float(np.median(Z.std(0)))
    collapsed = z_std_med < 0.05

    # battery: variable -> (GT array at ends, is_position_px)
    gt = {"arm": (d["pos"][ends], True), "target": (d["target"][ends], True)}
    if "qpos" in d.files:
        q = d["qpos"][ends]; gt["joint_cossin"] = (np.concatenate([np.cos(q), np.sin(q)], -1), False)
    if "qvel" in d.files:
        gt["ang_vel"] = (d["qvel"][ends], False)

    battery = {}
    for name, (Y, is_px) in gt.items():
        Y = np.asarray(Y, np.float32)
        res, pred, Yte = regress_probe(Z, Y, dev)
        if is_px:
            res["px"] = round(float(np.sqrt(((pred - Yte) ** 2).sum(-1)).mean()), 3)
        battery[name] = res

    r2s = [v["r2"] for v in battery.values()]
    out = {"ckpt": os.path.basename(args.ckpt), "encoder": a.get("encoder", "pool"),
           "condition": a.get("condition", "?"), "seed": a.get("seed", 0),
           "reg": a.get("reg", "?"), "z_std_median": round(z_std_med, 4), "collapsed": collapsed,
           "weights": {kk: a.get(kk, 0) for kk in ("inv_weight", "prior_weight", "tgt_weight")},
           "battery": battery,
           "legibility_meanR2": (None if collapsed else round(float(np.mean(r2s)), 4))}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    bat = " | ".join(f"{k}:R2={v['r2']}" + (f"/{v['px']}px" if 'px' in v else "") for k, v in battery.items())
    flag = f"COLLAPSED(z_std={z_std_med:.3f})" if collapsed else f"meanR2={out['legibility_meanR2']} z_std={z_std_med:.2f}"
    print(f"[legibility] {out['encoder']}/{out.get('reg', out['condition'])} {flag} | {bat}", flush=True)


if __name__ == "__main__":
    main()
