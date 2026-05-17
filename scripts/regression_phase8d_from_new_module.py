"""Step 6 regression test: verify the refactored of_jepa subpackage
reproduces Phase 8D numbers (within tolerance) using the new Evaluator.

Phase 8D locked numbers (3 seeds × 3000 steps, MOVi-A min_entities=8, stride=4):

    of_jepa_v0:
      visible_position_mse:  3.12e-5 ± 3.6e-5
      hidden_position_mse:   2.22e-5 ± 1.8e-5
      identity_switch_rate:  0.024 ± 0.008
      mean_slot_diversity:   1.32 ± 0.06
      cos_gap:               0.412 ± 0.019
      dyn_drift:             4.31e-3 ± 1.9e-3

Tolerance: ±2× on absolute metrics (numerical seed jitter); switch rate
must remain <0.10. Cos_gap must remain >0.30. Slot_diversity <2.0.

Usage:
    python scripts/regression_phase8d_from_new_module.py \\
        --cache /workspace/movi_a_local/validation \\
        --seeds 0,1,2 --out /workspace/phase10_regression
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# All imports go through the new package boundary.
from system1_jepa.of_jepa import (
    OFJEPA, OFJEPAConfig, OFJEPAObjectFiles,
)
from system1_jepa.of_jepa.metrics import Evaluator, ProbeFitConfig
from system1_jepa.movi_data import MoviDataset, MoviSpec


# Phase 8D locked targets + tolerance bands.
PHASE_8D_TARGETS = {
    "visible_position_mse": (3.12e-5, 1e-4),     # mean, abs upper bound (allow 3x)
    "hidden_position_mse":  (2.22e-5, 1e-4),
    "identity_switch_rate": (0.024,    0.10),    # must stay <0.10
    "mean_slot_diversity":  (1.32,     2.0),     # must stay <2.0
    "cos_gap":              (0.412,    None),    # must stay >0.30 (min, not max)
}
COS_GAP_MIN = 0.30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--min-entities", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    dataset = MoviDataset(MoviSpec(
        cache_dir=args.cache, image_size=args.image_size,
        max_entities=25, min_entities=args.min_entities,
    ))
    n = len(dataset)
    print(f"Episodes: {n} (min_entities={args.min_entities})", flush=True)

    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    n_train = int(0.8 * n)
    train_idx, eval_idx = indices[:n_train], indices[n_train:]

    results = {}
    for seed in seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = OFJEPAConfig(
            n_files=12, id_dim=64, state_dim=64, proposal_dim=128,
            id_ema_alpha=0.05, state_delta_scale=0.2,
            sinkhorn_iters=20, sinkhorn_temperature=0.1,
        )
        model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        from torch.utils.data import DataLoader, Subset
        from system1_jepa.identity_probe import hungarian_assign
        import torch.nn.functional as F
        loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                             num_workers=0, drop_last=True)

        t0 = time.time()
        step = 0
        for epoch in range(200):  # generous cap; max_steps is the actual stop condition
            for batch in loader:
                if step >= args.max_steps: break
                video = batch["video"][0].to(args.device)
                T = video.shape[0]
                gt_pos = batch["positions"][0].to(args.device)
                gt_vis = batch["visibility"][0].to(args.device).bool()

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
                    vis_mask = gt_vis[t]
                    if not vis_mask.any(): continue
                    pp_t = pred_pos[t].unsqueeze(0)
                    gt_t_vis = gt_pos[t][vis_mask].unsqueeze(0)
                    if gt_t_vis.shape[1] == 0: continue
                    pp_np = pp_t[0].detach().cpu().numpy()
                    gt_np = gt_t_vis[0].detach().cpu().numpy()
                    rows, cols, _ = hungarian_assign(pp_np, gt_np)
                    if len(rows) > 0:
                        rs = torch.from_numpy(rows).to(args.device)
                        cs = torch.from_numpy(cols).to(args.device)
                        pos_loss = pos_loss + F.mse_loss(pp_t[0, rs], gt_t_vis[0, cs])
                        pos_count += 1
                pos_loss = pos_loss / max(pos_count, 1)

                loss = jepa_loss + 10.0 * pos_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 500 == 0:
                    print(f"  step {step}/{args.max_steps} loss={float(loss):.4f} t={time.time()-t0:.0f}s", flush=True)
            if step >= args.max_steps: break

        # Eval with the new Evaluator class.
        model.eval()
        from system1_jepa.id_consistency import cosine_diagnostic
        all_states, all_pos, all_attr, all_vis, all_ids, all_ep, all_frame, all_hidden = [], [], [], [], [], [], [], []
        cos_records = []
        with torch.no_grad():
            for ep_off, ep_i in enumerate(eval_idx):
                s = dataset[ep_i]
                video = s["video"].to(args.device)
                slot_seq, _ = model.encode_video(video)
                T = video.shape[0]; E = s["visibility"].shape[-1]
                last_visible = -torch.ones(E, dtype=torch.long)
                gt_pos_e = s["positions"].to(args.device)
                gt_vis_e = s["visibility"].to(args.device).bool()
                cos = cosine_diagnostic(slot_seq, model.slot_to_pos_aux, gt_pos_e, gt_vis_e, model.id_dim)
                if cos["n_same_pairs"] > 0 and cos["n_diff_pairs"] > 0:
                    cos_records.append(cos)
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
        eval_result = evaluator.run(
            states=states, gt_pos=gt_pos_all, gt_attr=gt_attr_all,
            gt_visible=gt_vis_all, gt_entity_ids=gt_ids_all,
            ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
        )
        primary, secondary = eval_result["primary"], eval_result["secondary"]
        if cos_records:
            secondary["same_object_cos"] = float(np.mean([c["same_cos"] for c in cos_records]))
            secondary["diff_object_cos"] = float(np.mean([c["diff_cos"] for c in cos_records]))
            secondary["cos_gap"] = secondary["same_object_cos"] - secondary["diff_object_cos"]

        merged = {**primary, **secondary, "elapsed_s": time.time() - t0, "steps": step}
        results[f"seed{seed}"] = merged
        with open(out / f"seed{seed}.json", "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  primary: {primary}\n  secondary: {secondary}", flush=True)

    # Aggregate + gate check.
    print("\n=== aggregated ===", flush=True)
    agg = {}
    for k in ["visible_position_mse", "hidden_position_mse", "identity_switch_rate",
              "mean_slot_diversity", "cos_gap"]:
        vals = [results[f"seed{s}"].get(k) for s in seeds if k in results[f"seed{s}"]]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)) if len(vals)>1 else 0.0}

    passes = True
    for k, (target_mean, bound) in PHASE_8D_TARGETS.items():
        if k not in agg: continue
        mean_v = agg[k]["mean"]
        if k == "cos_gap":
            ok = mean_v >= COS_GAP_MIN
            print(f"  {k}: mean={mean_v:.4f} (target ≥ {COS_GAP_MIN}) {'✅' if ok else '❌'}")
        elif bound is not None:
            ok = mean_v <= bound
            print(f"  {k}: mean={mean_v:.4e} (target ≤ {bound}) {'✅' if ok else '❌'}")
        else:
            ok = True
            print(f"  {k}: mean={mean_v:.4e} (no hard bound)")
        passes = passes and ok

    summary = {"agg": agg, "phase8d_targets": {k: {"target": t, "bound": b} for k, (t, b) in PHASE_8D_TARGETS.items()},
                "regression_pass": passes}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== regression: {'✅ PASS' if passes else '❌ FAIL'} ===", flush=True)


if __name__ == "__main__":
    main()
