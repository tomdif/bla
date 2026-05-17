"""Step 7 / Phase 12: relation graph prototype training on MOVi.

Trains a pairwise "near" relation predictor on top of frozen OF-JEPA
object-file states. Uses MOVi's GT image_positions to derive the
ground-truth "near" labels (pairwise distance < threshold).

Per the Phase 12 plan in the JEPA arc roadmap, this validates the
hypothesis:

  relation prediction over persistent object files generalizes from
  raw position info and improves over a slot-content-only baseline.

Reports:
  - AUC for the "near" relation on held-out frames
  - precision @ recall=0.5
  - per-entity-count breakdown

Usage:
    python scripts/train_relations_movi.py \\
        --cache /workspace/movi_a_local/validation \\
        --of-jepa-steps 1500 --rel-epochs 50 \\
        --out /workspace/phase12_relations
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
from system1_jepa.movi_data import MoviDataset, MoviSpec
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.of_jepa.interfaces import ObjectFileBatch
from system1_jepa.of_jepa.relations import (
    RelationConfig, RelationGraphPredictor,
    near_relation_labels, relation_loss,
)
from system1_jepa.identity_probe import hungarian_assign


def train_of_jepa_short(model, dataset, train_idx, args, device):
    """Brief OF-JEPA warm-up so slot states carry meaningful info."""
    print(f"\n--- Warming up OF-JEPA for {args.of_jepa_steps} steps ---", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time(); step = 0
    for epoch in range(200):
        for batch in loader:
            if step >= args.of_jepa_steps: break
            video = batch["video"][0].to(device)
            T = video.shape[0]
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()

            opt.zero_grad(set_to_none=True)
            slot_states, _ = model.encode_video_grad(video)
            id_dim = model.cfg.id_dim
            state_only = slot_states[..., id_dim:]
            jepa_loss = 0.0
            for t in range(T - 1):
                jepa_loss = jepa_loss + F.mse_loss(state_only[t], state_only[t+1].detach())
            jepa_loss = jepa_loss / max(T - 1, 1)

            pred_pos = model.slot_to_pos_aux(slot_states)
            pos_loss = 0.0; pos_count = 0
            for t in range(T):
                vis_mask = gt_vis[t]
                if not vis_mask.any(): continue
                pp_t = pred_pos[t].unsqueeze(0)
                gt_t_vis = gt_pos[t][vis_mask].unsqueeze(0)
                if gt_t_vis.shape[1] == 0: continue
                rows, cols, _ = hungarian_assign(pp_t[0].detach().cpu().numpy(),
                                                  gt_t_vis[0].detach().cpu().numpy())
                if len(rows) > 0:
                    rs = torch.from_numpy(rows).to(device)
                    cs = torch.from_numpy(cols).to(device)
                    pos_loss = pos_loss + F.mse_loss(pp_t[0, rs], gt_t_vis[0, cs])
                    pos_count += 1
            pos_loss = pos_loss / max(pos_count, 1)

            loss = jepa_loss + 10.0 * pos_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                print(f"  warmup step {step}/{args.of_jepa_steps} loss={float(loss):.4f} t={time.time()-t0:.0f}s", flush=True)
        if step >= args.of_jepa_steps: break
    print(f"  warmup done at step {step} in {time.time()-t0:.0f}s", flush=True)


def train_relation_head(model, predictor, dataset, train_idx, args, device):
    """Train relation predictor with FROZEN OF-JEPA encoder."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.rel_lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time()
    for epoch in range(args.rel_epochs):
        losses = []
        for batch in loader:
            video = batch["video"][0].to(device)
            gt_pos = batch["positions"][0].to(device)              # [T, E_max, 2]
            gt_vis = batch["visibility"][0].to(device).bool()      # [T, E_max]
            entity_ids = batch["entity_ids"][0].to(device)         # [E_max]
            n_inst = int(batch["n_instances"][0])

            with torch.no_grad():
                slot_states, _ = model.encode_video(video)         # [T, N_files, full]
                slot_to_pos = model.slot_to_pos_aux(slot_states)   # [T, N_files, 2]

            T = video.shape[0]
            ep_loss = 0.0; n_frames = 0
            for t in range(T):
                # Hungarian-match slots to visible entities at frame t.
                vis_idx = torch.where(gt_vis[t])[0]
                if vis_idx.numel() < 2: continue
                pp = slot_to_pos[t].detach().cpu().numpy()
                gt_t = gt_pos[t][vis_idx].detach().cpu().numpy()
                rows, cols, _ = hungarian_assign(pp, gt_t)
                if len(rows) < 2: continue
                # Build per-slot position prediction at this frame (subset that matched).
                # For matched slots, predict pairwise relations.
                matched_slots = torch.tensor(rows, device=device)
                matched_entities = vis_idx[torch.tensor(cols, device=device)]
                slot_states_t = slot_states[t][matched_slots].unsqueeze(0)  # [1, K, full]

                # Build the GT near-relation matrix over the matched-entity SUBSET.
                positions_for_pairs = gt_pos[t][matched_entities]  # [K, 2]
                visibility_for_pairs = torch.ones(matched_entities.numel(), dtype=torch.bool, device=device)
                labels = near_relation_labels(
                    positions_for_pairs, visibility_for_pairs,
                    threshold=args.near_threshold,
                ).unsqueeze(0).unsqueeze(-1).float()                # [1, K, K, 1]

                # Wrap matched slots into an ObjectFileBatch-shaped tensor
                # for the predictor.
                id_dim = model.cfg.id_dim
                ofb = ObjectFileBatch(
                    id_keys=slot_states_t[..., :id_dim],
                    state_values=slot_states_t[..., id_dim:],
                    confidences=torch.ones(1, slot_states_t.shape[1], device=device),
                    frame_idx=t,
                )
                logits = predictor(ofb, mask_self=True)             # [1, K, K, n_rel]
                # Mask out diagonal so loss doesn't count self-pairs.
                K = slot_states_t.shape[1]
                pair_mask = torch.ones(1, K, K, dtype=torch.bool, device=device)
                pair_mask &= ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
                ep_loss = ep_loss + relation_loss(logits, labels.bool(), mask=pair_mask)
                n_frames += 1

            if n_frames == 0: continue
            ep_loss = ep_loss / n_frames
            opt.zero_grad(set_to_none=True)
            ep_loss.backward()
            opt.step()
            losses.append(float(ep_loss))
        if losses:
            print(f"  rel epoch {epoch+1}/{args.rel_epochs} mean_loss={np.mean(losses):.4f} t={time.time()-t0:.0f}s", flush=True)


def eval_relation_auc(model, predictor, dataset, eval_idx, args, device):
    """Compute AUC of "near" relation predictor on held-out episodes."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    model.eval(); predictor.eval()
    all_scores = []; all_labels = []
    with torch.no_grad():
        for ep_i in eval_idx:
            s = dataset[ep_i]
            video = s["video"].to(device)
            gt_pos = s["positions"].to(device)
            gt_vis = s["visibility"].to(device).bool()
            slot_states, _ = model.encode_video(video)
            slot_to_pos = model.slot_to_pos_aux(slot_states)
            T = video.shape[0]
            id_dim = model.cfg.id_dim
            for t in range(T):
                vis_idx = torch.where(gt_vis[t])[0]
                if vis_idx.numel() < 2: continue
                pp = slot_to_pos[t].cpu().numpy()
                gt_t = gt_pos[t][vis_idx].cpu().numpy()
                rows, cols, _ = hungarian_assign(pp, gt_t)
                if len(rows) < 2: continue
                matched_slots = torch.tensor(rows, device=device)
                matched_entities = vis_idx[torch.tensor(cols, device=device)]
                slot_states_t = slot_states[t][matched_slots].unsqueeze(0)
                positions_for_pairs = gt_pos[t][matched_entities]
                visibility_for_pairs = torch.ones(matched_entities.numel(), dtype=torch.bool, device=device)
                labels = near_relation_labels(
                    positions_for_pairs, visibility_for_pairs,
                    threshold=args.near_threshold,
                ).unsqueeze(0).unsqueeze(-1).float()
                ofb = ObjectFileBatch(
                    id_keys=slot_states_t[..., :id_dim],
                    state_values=slot_states_t[..., id_dim:],
                    confidences=torch.ones(1, slot_states_t.shape[1], device=device),
                    frame_idx=t,
                )
                logits = predictor(ofb, mask_self=True)
                K = slot_states_t.shape[1]
                eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
                upper = (~eye).unsqueeze(-1)  # [1, K, K, 1] keep off-diagonal pairs
                all_scores.append(logits[upper].cpu().numpy())
                all_labels.append(labels.bool()[upper].cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auc": float("nan"), "n_pairs": int(len(labels)),
                 "positive_rate": float(labels.mean())}
    auc = float(roc_auc_score(labels, scores))
    p, r, _ = precision_recall_curve(labels, scores)
    # Precision at recall ≈ 0.5
    if (r >= 0.5).any():
        p_at_r50 = float(p[(r >= 0.5).nonzero()[0][-1]])
    else:
        p_at_r50 = float("nan")
    return {"auc": auc, "p_at_recall50": p_at_r50,
             "n_pairs": int(len(labels)),
             "positive_rate": float(labels.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--out", required=True)
    p.add_argument("--of-jepa-steps", type=int, default=1500)
    p.add_argument("--rel-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rel-lr", type=float, default=3e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--near-threshold", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    dataset = MoviDataset(MoviSpec(cache_dir=args.cache, image_size=args.image_size,
                                     max_entities=25, normalize_positions=False))
    n = len(dataset)
    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    n_train = int(0.8 * n)
    train_idx, eval_idx = indices[:n_train], indices[n_train:]
    print(f"Episodes: {n}, train={len(train_idx)}, eval={len(eval_idx)}", flush=True)

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n=== seed={seed} ===", flush=True)

        cfg = OFJEPAConfig(n_files=12, id_dim=64, state_dim=64, proposal_dim=128)
        model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(args.device)
        train_of_jepa_short(model, dataset, train_idx, args, args.device)

        rel_cfg = RelationConfig(file_dim=cfg.id_dim + cfg.state_dim,
                                  hidden_dim=128, n_relations=1, asymmetric=False)
        predictor = RelationGraphPredictor(rel_cfg).to(args.device)

        train_relation_head(model, predictor, dataset, train_idx, args, args.device)
        metrics = eval_relation_auc(model, predictor, dataset, eval_idx, args, args.device)
        print(f"\nseed={seed} metrics: {metrics}", flush=True)
        with open(out / f"seed{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
