#!/usr/bin/env python3
"""Content-vs-routing discriminator for the SLOT_C0 target PASS (4.84px).

The confound: slot content = attn . value(tokens), and tokens carry SINCOS absolute
position, so a slot's content passively inherits the position of whatever patches it
attends to — a center-of-mass readout baked into the content. So "target decodes from
slot content" may just mean "slot attention routes to the target region," NOT that
pure self-prediction paid to represent target position usably.

Discriminator: decode target (and arm) from three representations of the SAME frozen
SLOT_C0 encoder, then compare px error:
  content   : flattened slot content [B, K*D]      (the 4.84 source)
  attn_com  : per-slot attention center-of-mass [B, K*2]   (pure routing)
  attn_full : full flattened attention map [B, N*K]        (richest routing readout)

Read:
  attn_com/attn_full target px  ~<=  content px  -> the PASS is a ROUTING readout
                                                    (weak/possibly-useless property)
  content px  <<  attn_com/attn_full px          -> content grounds target beyond routing

    python -m system1_motion.slot_probe_ablate --ckpt runs/dissoc_slot/substrate_SLOT_C0_s0.pt \
        --data runs/reacher_transitions.npz
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from system1_motion.slot_encoder import SlotEncoder
from gates.gate0_precommit import decode_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--out", default="runs/slot_ablate.json")
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or args.device in ("cpu", "mps")) else "cpu"

    ck = torch.load(args.ckpt, map_location=dev); a = ck["args"]; S = a["frame_stack"]; in_ch = 3 * S
    enc = SlotEncoder(in_channels=in_ch, vit_dim=a.get("vit_dim", 192), patch=a["patch"],
                      depth=a["enc_depth"], n_slots=a.get("n_slots", 6), slot_dim=a.get("slot_dim", 64)).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()

    d = np.load(args.data); frames = d["frames"]; HW = frames.shape[2]; img_px = int(d["img_px"])
    ep = d["ep_id"].astype(np.int64); N = len(frames)
    pairs = np.array([j for j in range(S - 1, N - S) if ep[j - S + 1] == ep[j + S]])
    ends = pairs[int((1 - args.eval_frac) * len(pairs)):]
    past = np.stack([frames[e - S + 1:e + 1].reshape(-1, HW, HW) for e in ends]).astype(np.uint8)
    pos = d["pos"][ends]; target = d["target"][ends]

    # collect the three representations over the eval set
    content, com, full = [], [], []
    with torch.no_grad():
        for i in range(0, len(past), 256):
            x = torch.from_numpy(past[i:i+256]).float().to(dev) / 255.0
            slots, attn, gh, gw = enc.encode_with_attn(x)       # slots[B,K,D], attn[B,N,K]
            B, Np, K = attn.shape
            yx = torch.stack(torch.meshgrid(torch.arange(gh, device=dev).float(),
                                            torch.arange(gw, device=dev).float(), indexing="ij"), -1).reshape(-1, 2)
            w = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)   # normalize over patches per slot
            com_b = torch.einsum("bnk,nc->bkc", w, yx)          # [B,K,2] center-of-mass
            content.append(slots.reshape(B, -1).cpu().numpy())
            com.append(com_b.reshape(B, -1).cpu().numpy())
            full.append(attn.reshape(B, -1).cpu().numpy())
    reps = {"content": np.concatenate(content), "attn_com": np.concatenate(com), "attn_full": np.concatenate(full)}

    print(f"=== SLOT_C0 content-vs-routing ({args.ckpt}) ===", flush=True)
    print(f"{'representation':12} {'dim':>6} {'TARGET px':>11} {'ARM px':>9}", flush=True)
    out = {}
    for name, R in reps.items():
        gt = decode_gate(R, target, img_px, steps=5000, threshold_px=5.0, device=dev)
        ga = decode_gate(R, pos, img_px, steps=5000, threshold_px=5.0, device=dev)
        out[name] = {"dim": R.shape[1], "target_px": gt["mean_l2_px"], "target_pass": gt["pass"],
                     "arm_px": ga["mean_l2_px"], "arm_pass": ga["pass"]}
        print(f"{name:12} {R.shape[1]:6d} {gt['mean_l2_px']:8.2f} {'PASS' if gt['pass'] else 'FAIL':>3} "
              f"{ga['mean_l2_px']:6.2f} {'PASS' if ga['pass'] else 'FAIL':>3}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)

    ct, rt = out["content"]["target_px"], min(out["attn_com"]["target_px"], out["attn_full"]["target_px"])
    print("\nVERDICT:", flush=True)
    if rt <= ct + 0.5:
        print(f"  routing alone decodes target ({rt:.2f}px) ~as well as content ({ct:.2f}px)"
              f" -> SLOT_C0 target PASS is a ROUTING/center-of-mass readout, not content grounding.", flush=True)
    else:
        print(f"  content ({ct:.2f}px) decodes target far better than routing ({rt:.2f}px)"
              f" -> content grounds target BEYOND routing; the PASS is real.", flush=True)


if __name__ == "__main__":
    main()
