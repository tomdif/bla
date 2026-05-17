"""Phase 13.4: does the relation graph help predict CLEVRER collision events?

The key question: in a regime where collisions actually drive dynamics
(CLEVRER), does adding relation features to a future-state / collision-
event readout improve prediction over an object-only baseline?

Two comparisons:

(A) Future position prediction at k steps ahead (same as Phase 12B,
    repeated here for CLEVRER):
        baseline: slot_state[i, t] -> pos[i, t+k]
        +relations: slot_state[i, t] + rel-weighted neighbor msgs -> pos[i, t+k]

(B) Collision event prediction:
        Per (frame, pair), predict whether the pair will collide
        within the next k frames. Baseline = MLP on concat(slot_i, slot_j).
        +Relations = same MLP but with rel-weight-aggregated neighbor info.

Reports AUC + AP (average precision) for collision prediction, and
MSE delta for future-position prediction.

Usage:
    python scripts/clevrer_collision_predict.py \\
        --cache /workspace/clevrer_local/train \\
        --of-jepa-steps 1500 --rel-epochs 10 --head-epochs 25 \\
        --k-frames 4 --out /workspace/phase13b_collision
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
from system1_jepa.clevrer_data import ClevrerDataset, ClevrerSpec
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.of_jepa.interfaces import ObjectFileBatch
from system1_jepa.of_jepa.relations import (
    RelationConfig, RelationGraphPredictor, near_relation_labels,
)
from system1_jepa.id_consistency import _assign_slots_to_entities
from system1_jepa.identity_probe import hungarian_assign


class CollisionHead(nn.Module):
    """Predict P(collision within k frames) for each object pair from
    object-file states + optional relation weights."""

    def __init__(self, slot_dim: int, use_relations: bool = False,
                 hidden_dim: int = 128):
        super().__init__()
        self.use_relations = use_relations
        # Input: (slot_i + slot_j)  and  |slot_i - slot_j|.
        # Plus relation features if active.
        in_dim = slot_dim * 2 + (slot_dim if use_relations else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slots: torch.Tensor,
                 rel_weights: torch.Tensor = None) -> torch.Tensor:
        """slots: [B, S, D]; rel_weights (if use_relations): [B, S, S].
        Returns [B, S, S] collision logits."""
        B, S, D = slots.shape
        a = slots.unsqueeze(2).expand(-1, -1, S, -1)  # [B, S, S, D]
        b = slots.unsqueeze(1).expand(-1, S, -1, -1)
        pair = torch.cat([a + b, (a - b).abs()], dim=-1)  # [B, S, S, 2D]
        if self.use_relations:
            # Neighbor message — aggregate ALL slots weighted by relation weights
            # then concat to each pair's own neighborhood summary.
            neighbor = torch.einsum("bij,bjd->bid", rel_weights, slots)  # [B, S, D]
            # Use the avg of pair members' neighbor messages.
            n = (neighbor.unsqueeze(2) + neighbor.unsqueeze(1)) * 0.5  # [B, S, S, D]
            pair = torch.cat([pair, n], dim=-1)
        logits = self.head(pair).squeeze(-1)  # [B, S, S]
        return logits


def warmup_of_jepa(model, dataset, train_idx, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    step = 0; t0 = time.time()
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
            stride = 4
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
            (jepa_loss + 10.0 * pos_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 500 == 0:
                print(f"  warmup step {step}/{args.of_jepa_steps} t={time.time()-t0:.0f}s", flush=True)
        if step >= args.of_jepa_steps: break


def train_relation_head_clevrer(model, predictor, dataset, train_idx, args, device):
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.rel_lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time()
    for epoch in range(args.rel_epochs):
        losses = []
        for batch in loader:
            video = batch["video"][0].to(device)
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()
            with torch.no_grad():
                slot_states, _ = model.encode_video(video)
                slot_to_pos = model.slot_to_pos_aux(slot_states)
            T = video.shape[0]
            ep_loss = 0.0; n_frames = 0
            for t in range(T):
                vis_idx = torch.where(gt_vis[t])[0]
                if vis_idx.numel() < 2: continue
                pp = slot_to_pos[t].detach().cpu().numpy()
                gt_t = gt_pos[t][vis_idx].detach().cpu().numpy()
                rows, cols, _ = hungarian_assign(pp, gt_t)
                if len(rows) < 2: continue
                matched_slots = torch.tensor(rows, device=device)
                matched_entities = vis_idx[torch.tensor(cols, device=device)]
                slot_states_t = slot_states[t][matched_slots].unsqueeze(0)
                positions = gt_pos[t][matched_entities]
                vis_for_pairs = torch.ones(matched_entities.numel(), dtype=torch.bool, device=device)
                labels = near_relation_labels(positions, vis_for_pairs, threshold=0.10
                ).unsqueeze(0).unsqueeze(-1).float()
                id_dim = model.cfg.id_dim
                ofb = ObjectFileBatch(
                    id_keys=slot_states_t[..., :id_dim],
                    state_values=slot_states_t[..., id_dim:],
                    confidences=torch.ones(1, slot_states_t.shape[1], device=device),
                    frame_idx=t,
                )
                logits = predictor(ofb, mask_self=True)
                K = slot_states_t.shape[1]
                pair_mask = (~torch.eye(K, dtype=torch.bool, device=device)).unsqueeze(0)
                from system1_jepa.of_jepa.relations import relation_loss
                ep_loss = ep_loss + relation_loss(logits, labels.bool(), mask=pair_mask)
                n_frames += 1
            if n_frames == 0: continue
            ep_loss = ep_loss / n_frames
            opt.zero_grad(set_to_none=True)
            ep_loss.backward()
            opt.step()
            losses.append(float(ep_loss))
        if losses:
            print(f"  rel epoch {epoch+1}/{args.rel_epochs} loss={np.mean(losses):.4f} t={time.time()-t0:.0f}s", flush=True)


def train_collision_head(coll_head, model, predictor, dataset, train_idx,
                          args, device, use_relations=False):
    model.eval()
    if predictor is not None: predictor.eval()
    for p in model.parameters(): p.requires_grad = False
    if predictor is not None:
        for p in predictor.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(coll_head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time()
    for epoch in range(args.head_epochs):
        losses = []
        for batch in loader:
            video = batch["video"][0].to(device)
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()
            collisions = batch["collisions"]  # list of (frame, oa_id, ob_id)
            T = video.shape[0]; n_inst = int(batch["n_instances"][0])

            # Build per-frame collision-pair label tensor [T, n_inst, n_inst].
            coll_table = torch.zeros(T, n_inst, n_inst, dtype=torch.float32, device=device)
            for c in collisions:
                # `collisions` came back as a list-of-tuples or list-of-lists per batch.
                if isinstance(c[0], (list, tuple)):  # nested batch dim
                    for cc in c:
                        f, a, b = int(cc[0]), int(cc[1]), int(cc[2])
                        if 0 <= f < T and 0 <= a < n_inst and 0 <= b < n_inst:
                            coll_table[f, a, b] = 1.0; coll_table[f, b, a] = 1.0
                    continue
                f, a, b = int(c[0]), int(c[1]), int(c[2])
                if 0 <= f < T and 0 <= a < n_inst and 0 <= b < n_inst:
                    coll_table[f, a, b] = 1.0; coll_table[f, b, a] = 1.0
            # Future-window collisions per (t, pair): does pair collide in [t, t+k)?
            k = args.k_frames
            future_coll = torch.zeros_like(coll_table)
            for t in range(T):
                end = min(t + k, T)
                future_coll[t] = coll_table[t:end].sum(0).clamp(max=1.0)

            with torch.no_grad():
                slot_states, _ = model.encode_video(video)

            id_dim = model.cfg.id_dim
            ep_loss = 0.0; n_used = 0
            for t in range(T - 1):
                vis_idx = torch.where(gt_vis[t])[0]
                if vis_idx.numel() < 2: continue
                pp = model.slot_to_pos_aux(slot_states[t]).detach().cpu().numpy()
                gt_t = gt_pos[t][vis_idx].detach().cpu().numpy()
                rows, cols, _ = hungarian_assign(pp, gt_t)
                if len(rows) < 2: continue
                matched_slots = torch.tensor(rows, device=device)
                matched_entities = vis_idx[torch.tensor(cols, device=device)]
                slot_t = slot_states[t][matched_slots].unsqueeze(0)  # [1, K, D]
                rel_weights = None
                if use_relations and predictor is not None:
                    ofb = ObjectFileBatch(
                        id_keys=slot_t[..., :id_dim],
                        state_values=slot_t[..., id_dim:],
                        confidences=torch.ones(1, slot_t.shape[1], device=device),
                        frame_idx=t,
                    )
                    rel_logits = predictor(ofb, mask_self=True)
                    rel_weights = torch.softmax(rel_logits[..., 0], dim=-1)

                pred_logits = coll_head(slot_t, rel_weights)  # [1, K, K]

                # Build target — using matched_entities as the entity IDs.
                K = matched_entities.numel()
                target = future_coll[t][matched_entities][:, matched_entities].unsqueeze(0)
                # Mask the diagonal.
                eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
                mask = ~eye
                if not mask.any(): continue
                loss = F.binary_cross_entropy_with_logits(
                    pred_logits[mask], target[mask],
                    pos_weight=torch.tensor(10.0, device=device),  # collisions are rare
                )
                ep_loss = ep_loss + loss
                n_used += 1
            if n_used == 0: continue
            ep_loss = ep_loss / n_used
            opt.zero_grad(set_to_none=True)
            ep_loss.backward()
            opt.step()
            losses.append(float(ep_loss))
        if losses:
            print(f"  coll({'rel' if use_relations else 'base'}) epoch {epoch+1}/{args.head_epochs} loss={np.mean(losses):.4f} t={time.time()-t0:.0f}s", flush=True)


def eval_collision(coll_head, model, predictor, dataset, eval_idx,
                    args, device, use_relations=False):
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval(); coll_head.eval()
    if predictor is not None: predictor.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for ep_i in eval_idx:
            s = dataset[ep_i]
            video = s["video"].to(device)
            gt_pos = s["positions"].to(device)
            gt_vis = s["visibility"].to(device).bool()
            collisions = s["collisions"]
            T = video.shape[0]; n_inst = int(s["n_instances"])
            coll_table = torch.zeros(T, n_inst, n_inst, dtype=torch.float32, device=device)
            for c in collisions:
                f, a, b = int(c[0]), int(c[1]), int(c[2])
                if 0 <= f < T and 0 <= a < n_inst and 0 <= b < n_inst:
                    coll_table[f, a, b] = 1.0; coll_table[f, b, a] = 1.0
            k = args.k_frames
            future_coll = torch.zeros_like(coll_table)
            for t in range(T):
                end = min(t + k, T)
                future_coll[t] = coll_table[t:end].sum(0).clamp(max=1.0)

            slot_states, _ = model.encode_video(video)
            id_dim = model.cfg.id_dim
            for t in range(T - 1):
                vis_idx = torch.where(gt_vis[t])[0]
                if vis_idx.numel() < 2: continue
                pp = model.slot_to_pos_aux(slot_states[t]).cpu().numpy()
                gt_t = gt_pos[t][vis_idx].cpu().numpy()
                rows, cols, _ = hungarian_assign(pp, gt_t)
                if len(rows) < 2: continue
                matched_slots = torch.tensor(rows, device=device)
                matched_entities = vis_idx[torch.tensor(cols, device=device)]
                slot_t = slot_states[t][matched_slots].unsqueeze(0)
                rel_weights = None
                if use_relations and predictor is not None:
                    ofb = ObjectFileBatch(
                        id_keys=slot_t[..., :id_dim],
                        state_values=slot_t[..., id_dim:],
                        confidences=torch.ones(1, slot_t.shape[1], device=device),
                        frame_idx=t,
                    )
                    rel_logits = predictor(ofb, mask_self=True)
                    rel_weights = torch.softmax(rel_logits[..., 0], dim=-1)
                pred_logits = coll_head(slot_t, rel_weights)
                K = matched_entities.numel()
                target = future_coll[t][matched_entities][:, matched_entities].unsqueeze(0)
                eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
                mask = ~eye
                all_scores.append(pred_logits[mask].cpu().numpy())
                all_labels.append(target[mask].cpu().numpy())

    if not all_scores:
        return {"auc": float("nan"), "ap": float("nan"), "n_pairs": 0, "positive_rate": float("nan")}
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels) > 0.5
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auc": float("nan"), "ap": float("nan"), "n_pairs": len(labels),
                 "positive_rate": float(labels.mean())}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap":  float(average_precision_score(labels, scores)),
        "n_pairs": int(len(labels)),
        "positive_rate": float(labels.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--of-jepa-steps", type=int, default=1500)
    p.add_argument("--rel-epochs", type=int, default=10)
    p.add_argument("--head-epochs", type=int, default=20)
    p.add_argument("--k-frames", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rel-lr", type=float, default=3e-3)
    p.add_argument("--head-lr", type=float, default=3e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset = ClevrerDataset(ClevrerSpec(cache_dir=args.cache, image_size=args.image_size,
                                          max_entities=10, normalize_positions=False))
    n = len(dataset); indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    train_idx, eval_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]
    print(f"Episodes: {n}, train={len(train_idx)}, eval={len(eval_idx)}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = OFJEPAConfig(n_files=12, id_dim=64, state_dim=64, proposal_dim=128)
    model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(args.device)
    print("--- warmup OF-JEPA ---")
    warmup_of_jepa(model, dataset, train_idx, args, args.device)

    slot_dim = cfg.id_dim + cfg.state_dim
    rel_cfg = RelationConfig(file_dim=slot_dim, hidden_dim=128, n_relations=1, asymmetric=False)
    predictor = RelationGraphPredictor(rel_cfg).to(args.device)
    print("--- train relation head ---")
    train_relation_head_clevrer(model, predictor, dataset, train_idx, args, args.device)

    # Baseline collision head
    print("--- train baseline collision head ---")
    base_head = CollisionHead(slot_dim, use_relations=False).to(args.device)
    train_collision_head(base_head, model, None, dataset, train_idx, args, args.device, use_relations=False)
    base_eval = eval_collision(base_head, model, None, dataset, eval_idx, args, args.device, use_relations=False)
    print(f"baseline collision: {base_eval}", flush=True)

    print("--- train +relations collision head ---")
    rel_head = CollisionHead(slot_dim, use_relations=True).to(args.device)
    train_collision_head(rel_head, model, predictor, dataset, train_idx, args, args.device, use_relations=True)
    rel_eval = eval_collision(rel_head, model, predictor, dataset, eval_idx, args, args.device, use_relations=True)
    print(f"+relations collision: {rel_eval}", flush=True)

    print("\n=== Collision-prediction comparison ===")
    for metric in ["auc", "ap"]:
        bv = base_eval[metric]; rv = rel_eval[metric]
        if bv != bv or rv != rv:
            print(f"  {metric}: NaN"); continue
        delta = (rv - bv) / max(bv, 1e-9) * 100
        verdict = "✅ improved" if rv > bv * 1.05 else ("⚠ no clear effect" if 0.95 <= rv / max(bv, 1e-9) <= 1.05 else "❌ regressed")
        print(f"  {metric}: baseline={bv:.4f}  +rel={rv:.4f}  ({delta:+.1f}%)  {verdict}")

    with open(out / f"seed{args.seed}.json", "w") as f:
        json.dump({"baseline": base_eval, "with_relations": rel_eval,
                    "k_frames": args.k_frames, "seed": args.seed}, f, indent=2)


if __name__ == "__main__":
    main()
