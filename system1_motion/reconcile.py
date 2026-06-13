#!/usr/bin/env python3
"""Reconcile the live-probe vs legibility discrepancy on sigreg/qlif checkpoints.

The two probes disagree by ~7px on sigreg/qlif but agree on varhinge. Hypothesis:
decode_gate uses RAW embeddings while legibility's regress_probe STANDARDIZES them,
and sigreg/qlif push the embedding scale so an unstandardized probe fails. Test:
decode arm position from the SAME embeddings three ways and compare px.

    python -m system1_motion.reconcile --ckpt runs/regcmp/substrate_pool_sigreg_s0.pt --data runs/reacher_v2.npz
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from system1_motion.models import ViTEncoder
from system1_motion.slot_encoder import SlotEncoder
from system1_motion.legibility import eval_ends, embed, regress_probe
from gates.gate0_precommit import decode_gate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda"); args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev); a = ck["args"]; S = a["frame_stack"]; in_ch = 3 * S
    if a.get("encoder") == "slot":
        enc = SlotEncoder(in_channels=in_ch, vit_dim=a.get("vit_dim",192), patch=a["patch"], depth=a["enc_depth"],
                          n_slots=a.get("n_slots",6), slot_dim=a.get("slot_dim",64)).to(dev)
    else:
        enc = ViTEncoder(a["image_size"], a["patch"], in_ch, a["d_z"], a["enc_depth"]).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()
    d = np.load(args.data); frames = d["frames"]; H = frames.shape[2]; img_px = int(d["img_px"])
    ends = eval_ends(d["ep_id"].astype(np.int64), len(frames), S, 0.2)
    past = np.stack([frames[e-S+1:e+1].reshape(-1,H,H) for e in ends]).astype(np.uint8)
    Z = embed(enc, past, dev); arm = d["pos"][ends].astype(np.float32)

    zmu, zsd = Z.mean(0), Z.std(0) + 1e-6
    Zstd = (Z - zmu) / zsd
    # effective rank (participation ratio of covariance eigenvalues): catches oblique
    # low-rank collapse that per-dim std misses. eff_rank = (sum lam)^2 / sum(lam^2).
    cov = np.cov((Z - zmu).T)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    eff_rank = float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)) if lam.sum() > 0 else 0.0
    print(f"ckpt={os.path.basename(args.ckpt)}  enc={a.get('encoder')}  reg={a.get('reg')}", flush=True)
    print(f"  embedding scale: |mean|={np.abs(Z.mean()):.3f}  std(per-dim) min/med/max="
          f"{np.percentile(Z.std(0),[0,50,100]).round(3).tolist()}  eff_rank={eff_rank:.1f}/{Z.shape[1]}", flush=True)
    g_raw = decode_gate(Z, arm, img_px, steps=5000, threshold_px=5.0, device=dev)
    g_std = decode_gate(Zstd, arm, img_px, steps=5000, threshold_px=5.0, device=dev)
    rp, pred, Yte = regress_probe(Z, arm, dev)
    px_rp = float(np.sqrt(((pred - Yte) ** 2).sum(-1)).mean())
    print(f"  ARM px:  decode_gate(raw)={g_raw['mean_l2_px']:.2f}   decode_gate(standardized)={g_std['mean_l2_px']:.2f}"
          f"   regress_probe(standardized)={px_rp:.2f}", flush=True)
    print("  => if raw>>std and std~regress_probe: the live probe failed on SCALE; regress_probe/legibility is trustworthy.", flush=True)


if __name__ == "__main__":
    main()
