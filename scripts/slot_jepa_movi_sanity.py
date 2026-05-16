"""Phase 7 (Kubric/MOVi) probe sanity-check.

Frozen ConvNeXt-T + SlotAttention pass over 32 MOVi episodes; fit and
evaluate the identity-aware probe. The point isn't to get good numbers
here (no training) — it's to verify the metric pipeline computes
end-to-end on real rendered video and that nothing's broken.

Usage:
    python scripts/slot_jepa_movi_sanity.py \\
        --cache /workspace/movi_a_local/validation \\
        --n-episodes 32 --image-size 128 --n-slots 12 \\
        --out /workspace/movi_sanity_run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.convnext_encoder import ConvNeXtEncoderConfig, ConvNeXtSlotEncoder
from system1_jepa.identity_probe import (
    ProbeFitConfig,
    identity_aware_probe_eval,
)
from system1_jepa.movi_data import MoviDataset, MoviSpec
from system1_jepa.slot import SlotAttention, SlotAttentionConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--n-episodes", type=int, default=32)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=12)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = args.device
    ds = MoviDataset(MoviSpec(cache_dir=args.cache, image_size=args.image_size,
                                max_entities=10))
    n = min(args.n_episodes, len(ds))
    print(f"Loaded MOVi cache: {len(ds)} episodes total, using {n}.")

    enc = ConvNeXtSlotEncoder(ConvNeXtEncoderConfig(
        input_size=args.image_size, slot_dim=args.slot_dim,
        pretrained=False, freeze_early_stages=0,
    )).to(device).eval()

    slot_attn = SlotAttention(
        input_dim=args.slot_dim,
        cfg=SlotAttentionConfig(
            n_slots=args.n_slots, slot_dim=args.slot_dim, n_iters=3,
        ),
    ).to(device).eval()

    # ----- Forward pass: collect slot states + GT -----
    all_states = []
    all_positions = []
    all_attrs = []
    all_visible = []
    all_ids = []
    all_ep = []
    all_frame = []
    all_hidden = []

    with torch.no_grad():
        for ep_idx in range(n):
            sample = ds[ep_idx]
            video = sample["video"].to(device)            # [T, 3, H, W]
            T = video.shape[0]
            # Per-frame slot encoding.
            tokens = enc(video)                            # [T, n_patches, D]
            slots = slot_attn(tokens)                      # [T, n_slots, D]

            # Accumulate per-frame rows.
            for t in range(T):
                all_states.append(slots[t])               # [n_slots, D]
                all_positions.append(sample["positions"][t])     # [E_max, 2]
                all_attrs.append(sample["attrs"])                # [E_max, A]
                all_visible.append(sample["visibility"][t])      # [E_max]
                all_ids.append(sample["entity_ids"])             # [E_max]
                all_ep.append(ep_idx)
                all_frame.append(t)
                all_hidden.append(0)

    states = torch.stack(all_states).cpu()                       # [N, n_slots, D]
    gt_pos = torch.stack(all_positions)                           # [N, E_max, 2]
    gt_attr = torch.stack(all_attrs)                              # [N, E_max, A]
    gt_visible = torch.stack(all_visible)                         # [N, E_max] bool
    gt_ids = torch.stack(all_ids)                                 # [N, E_max] long
    ep_ids = torch.tensor(all_ep, dtype=torch.long)
    frame_idx = torch.tensor(all_frame, dtype=torch.long)
    hidden_step = torch.tensor(all_hidden, dtype=torch.long)

    print(f"States: {tuple(states.shape)}; gt_pos: {tuple(gt_pos.shape)}; "
          f"gt_visible: {tuple(gt_visible.shape)}.")

    # ----- Run identity probe -----
    cfg = ProbeFitConfig(epochs=300, lr=5e-3, batch_size=128, attr_weight=1.0)
    result = identity_aware_probe_eval(
        states=states, gt_pos=gt_pos, gt_attr=gt_attr,
        gt_visible=gt_visible, gt_entity_ids=gt_ids,
        ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
        J=0, cfg=cfg,
    )

    summary = {
        "n_episodes": n,
        "n_rows": states.shape[0],
        "n_slots": args.n_slots,
        "slot_dim": args.slot_dim,
        "image_size": args.image_size,
        "encoder_pretrained": False,
        "encoder_trained": False,
        "metrics": {
            "visible_position_mse": result.visible_position_mse,
            "hidden_position_mse": result.hidden_position_mse,
            "identity_switch_rate": result.identity_switch_rate,
            "mean_slot_diversity": result.mean_slot_diversity,
            "n_visible_rows": result.n_visible,
            "n_hidden_rows": result.n_hidden,
        },
        "interpretation": {
            "switch_rate_chance": (args.n_slots - 1) / args.n_slots,
            "switch_rate_perfect": 0.0,
            "note": "Frozen random-init encoder — switches near chance are EXPECTED. "
                    "This only validates that the metric pipeline works end-to-end.",
        },
    }
    print(json.dumps(summary, indent=2))
    with open(out / "sanity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out / 'sanity_summary.json'}")


if __name__ == "__main__":
    main()
