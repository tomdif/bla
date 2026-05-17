"""Phase 13 CLEVRER training: OF-JEPA v0 + optional relation/collision heads.

Mirrors `slot_jepa_movi_train.py` but consumes `ClevrerDataset`. Key
differences from MOVi:

  - 32 frames per episode (CLEVRER 128 @ stride=4) vs 24 (MOVi-A)
  - 6-10 objects with attribute uniqueness per scene
  - GT collision events available → train a CollisionHead
  - GT in_out events available → train a separate readout (Phase 14+)

Three modes:
  of_jepa_v0                — baseline OF-JEPA training
  of_jepa_v0_relations       — + relation graph head + collision head
  slot_delta                 — control (exchangeable slots)

Usage:
    python scripts/slot_jepa_clevrer_train.py \\
        --cache /workspace/clevrer_local/train \\
        --modes of_jepa_v0 --seeds 0 \\
        --max-steps 3000 --jepa-stride 4 \\
        --out /workspace/phase13_run1
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.clevrer_data import ClevrerDataset, ClevrerSpec
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.of_jepa.metrics import Evaluator, ProbeFitConfig
from system1_jepa.id_consistency import cosine_diagnostic
from system1_jepa.identity_probe import hungarian_assign


def train_one_run(model, dataset, train_idx, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time(); step = 0
    for epoch in range(200):
        for batch in loader:
            if step >= args.max_steps: break
            video = batch["video"][0].to(device)
            T = video.shape[0]
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()

            opt.zero_grad(set_to_none=True)
            slot_states, _ = model.encode_video_grad(video)
            id_dim = model.cfg.id_dim
            state_only = slot_states[..., id_dim:]
            stride = args.jepa_stride
            jepa_loss = 0.0
            for t in range(T - stride):
                jepa_loss = jepa_loss + F.mse_loss(state_only[t], state_only[t+stride].detach())
            jepa_loss = jepa_loss / max(T - stride, 1)

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

            loss = jepa_loss + 10.0 * pos_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                print(f"  step {step}/{args.max_steps} loss={float(loss):.4f} t={time.time()-t0:.0f}s", flush=True)
        if step >= args.max_steps: break
    print(f"  training done at step {step} in {time.time()-t0:.0f}s", flush=True)
    return step


def eval_run(model, dataset, eval_idx, args, device):
    """Standard OF-JEPA eval: identity-conditioned + anonymous metrics."""
    model.eval()
    all_states, all_pos, all_attr, all_vis, all_ids, all_ep, all_frame, all_hidden = ([], [], [], [], [], [], [], [])
    cos_records = []
    with torch.no_grad():
        for ep_off, ep_i in enumerate(eval_idx):
            s = dataset[ep_i]
            video = s["video"].to(device)
            slot_seq, _ = model.encode_video(video)
            T = video.shape[0]
            E = s["visibility"].shape[-1]
            gt_pos_e = s["positions"].to(device)
            gt_vis_e = s["visibility"].to(device).bool()
            cos = cosine_diagnostic(slot_seq, model.slot_to_pos_aux, gt_pos_e, gt_vis_e, model.id_dim)
            if cos["n_same_pairs"] > 0 and cos["n_diff_pairs"] > 0:
                cos_records.append(cos)
            last_visible = -torch.ones(E, dtype=torch.long)
            for t in range(T):
                cur_vis = s["visibility"][t]
                for e in range(E):
                    if cur_vis[e]: last_visible[e] = t
                h = torch.where(cur_vis, torch.zeros(E, dtype=torch.long), t - last_visible)
                hd = int(h[~cur_vis].max().item()) if (~cur_vis).any() else 0
                all_states.append(slot_seq[t])
                all_pos.append(s["positions"][t])
                all_attr.append(s["attrs"])
                all_vis.append(cur_vis)
                all_ids.append(s["entity_ids"])
                all_ep.append(ep_off); all_frame.append(t); all_hidden.append(hd)

    states = torch.stack(all_states).cpu()
    gt_pos_all = torch.stack(all_pos); gt_attr_all = torch.stack(all_attr)
    gt_vis_all = torch.stack(all_vis); gt_ids_all = torch.stack(all_ids)
    ep_ids = torch.tensor(all_ep, dtype=torch.long)
    frame_idx = torch.tensor(all_frame, dtype=torch.long)
    hidden_step = torch.tensor(all_hidden, dtype=torch.long)

    evaluator = Evaluator(cfg=ProbeFitConfig(epochs=300, lr=5e-3, batch_size=128, attr_weight=1.0))
    result = evaluator.run(
        states=states, gt_pos=gt_pos_all, gt_attr=gt_attr_all,
        gt_visible=gt_vis_all, gt_entity_ids=gt_ids_all,
        ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
    )
    primary = result["primary"]
    secondary = result["secondary"]
    if cos_records:
        secondary["same_object_cos"] = float(np.mean([c["same_cos"] for c in cos_records]))
        secondary["diff_object_cos"] = float(np.mean([c["diff_cos"] for c in cos_records]))
        secondary["cos_gap"] = secondary["same_object_cos"] - secondary["diff_object_cos"]
    return {**primary, **secondary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=12)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    dataset = ClevrerDataset(ClevrerSpec(cache_dir=args.cache, image_size=args.image_size,
                                          max_entities=10, normalize_positions=False))
    n = len(dataset)
    print(f"Episodes: {n}", flush=True)
    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    train_idx, eval_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n=== seed={seed} ===", flush=True)
        cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                            state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim,
                            id_ema_alpha=0.05, state_delta_scale=0.2,
                            sinkhorn_iters=20, sinkhorn_temperature=0.1)
        model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(args.device)
        steps = train_one_run(model, dataset, train_idx, args, args.device)
        metrics = eval_run(model, dataset, eval_idx, args, args.device)
        metrics["steps"] = steps
        with open(out / f"seed{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nseed={seed} metrics: {json.dumps(metrics, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
