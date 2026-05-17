"""Phase 12 level-2 eval: does the relation graph add PREDICTIVE VALUE
beyond OF-JEPA's object files?

Per the user's framing:
  level 1: relation head learns nontrivial relations (done by train_relations_movi)
  level 2: relation features improve future state prediction (THIS)
  level 3: relation graph doesn't regress identity-conditioned metrics

The comparison:

  baseline:    predict entity future position at frame t+k from frame-t
               slot state alone (single-file readout).
  +relations:  predict entity future position from frame-t slot state
               aggregated with relation-weighted neighbor messages.

Both readouts are frozen-OF-JEPA + small trained heads. The relation
graph is loaded from a Phase 12 level-1 checkpoint.

If `future_hpm_with_relations ≤ 0.95 × future_hpm_baseline` (≥5%
improvement) AND `identity_conditioned_hpm` stays within tolerance,
relation graph passes level 2.

Usage:
    python scripts/eval_relations_predictive_value.py \\
        --cache /workspace/movi_a_local/validation \\
        --of-jepa-steps 1500 --readout-epochs 30 --k-steps 4 \\
        --out /workspace/phase12b_predictive
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
from system1_jepa.movi_data import MoviDataset, MoviSpec
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.of_jepa.interfaces import ObjectFileBatch
from system1_jepa.of_jepa.relations import (
    RelationConfig, RelationGraphPredictor, near_relation_labels,
)
from system1_jepa.identity_probe import hungarian_assign


class FutureReadout(nn.Module):
    """Per-slot readout: slot_state → predicted future position.

    If `use_relations=True`, also aggregates relation-weighted neighbor
    state into the input via a single relation-message pass.
    """
    def __init__(self, slot_dim: int, hidden_dim: int = 128, use_relations: bool = False):
        super().__init__()
        self.use_relations = use_relations
        in_dim = slot_dim * 2 if use_relations else slot_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, slot_states: torch.Tensor,
                 rel_weights: torch.Tensor = None) -> torch.Tensor:
        """slot_states: [B, S, D]; rel_weights (if use_relations): [B, S, S]
        with softmax-normalized rows summing to 1.

        Returns: [B, S, 2] predicted positions.
        """
        if self.use_relations:
            # Neighbor message: sum_j w_ij * slot_j
            neighbor = torch.einsum("bij,bjd->bid", rel_weights, slot_states)
            x = torch.cat([slot_states, neighbor], dim=-1)
        else:
            x = slot_states
        return self.head(x)


def warmup_of_jepa(model, dataset, train_idx, args, device):
    """Same as in train_relations_movi.py — short OF-JEPA warm-up."""
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
            for t in range(T - 1):
                jepa_loss = jepa_loss + F.mse_loss(state_only[t], state_only[t+1].detach())
            jepa_loss = jepa_loss / max(T - 1, 1)

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


def train_readout(readout, model, predictor, dataset, train_idx,
                   args, device, use_relations=False, k_steps=4):
    """Train future-position readout. Frozen OF-JEPA + optional relation graph."""
    model.eval()
    if predictor is not None:
        predictor.eval()
    for p in model.parameters(): p.requires_grad = False
    if predictor is not None:
        for p in predictor.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(readout.parameters(), lr=args.readout_lr, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)
    t0 = time.time()
    for epoch in range(args.readout_epochs):
        losses = []
        for batch in loader:
            video = batch["video"][0].to(device)
            T = video.shape[0]
            gt_pos = batch["positions"][0].to(device)
            gt_vis = batch["visibility"][0].to(device).bool()

            with torch.no_grad():
                slot_states, _ = model.encode_video(video)
            id_dim = model.cfg.id_dim
            ep_loss = 0.0; n_pairs = 0
            for t in range(T - k_steps):
                # Predict GT entity positions at t+k from slot states at t.
                # Hungarian-assign slots to entities at t (for visible).
                vm_t = gt_vis[t]; vm_tk = gt_vis[t + k_steps]
                vm = vm_t & vm_tk
                if vm.sum() < 1: continue
                vis_idx = torch.where(vm)[0]
                # Use slot-to-pos aux to match.
                pp_now = model.slot_to_pos_aux(slot_states[t]).detach().cpu().numpy()
                gt_now = gt_pos[t][vis_idx].detach().cpu().numpy()
                rows, cols, _ = hungarian_assign(pp_now, gt_now)
                if len(rows) < 1: continue
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
                    rel_logits = predictor(ofb, mask_self=True)  # [1, K, K, n_rel]
                    rel_weights = torch.softmax(rel_logits[..., 0], dim=-1)  # [1, K, K]

                pred_future = readout(slot_t, rel_weights)        # [1, K, 2]
                target_future = gt_pos[t + k_steps][matched_entities].unsqueeze(0)
                loss = F.mse_loss(pred_future, target_future)
                ep_loss = ep_loss + loss
                n_pairs += 1

            if n_pairs == 0: continue
            ep_loss = ep_loss / n_pairs
            opt.zero_grad(set_to_none=True)
            ep_loss.backward()
            opt.step()
            losses.append(float(ep_loss))
        if losses:
            print(f"  readout({'rel' if use_relations else 'base'}) epoch {epoch+1}/{args.readout_epochs} loss={np.mean(losses):.5f} t={time.time()-t0:.0f}s", flush=True)


def eval_readout(readout, model, predictor, dataset, eval_idx,
                  args, device, use_relations=False, k_steps=4):
    """Evaluate future-position prediction MSE on held-out episodes.

    Reports both AVERAGE over all entities AND the **interaction-heavy
    subset**: entities that have at least one other near-neighbor
    (distance < args.near_threshold) at the prediction time t. This is
    the slice where the relation graph SHOULD help — if it helps
    nowhere else.
    """
    model.eval(); readout.eval()
    if predictor is not None: predictor.eval()
    visible_errors = []
    hidden_errors = []
    inter_visible_errors = []  # interaction-heavy subset
    inter_hidden_errors = []
    with torch.no_grad():
        for ep_i in eval_idx:
            s = dataset[ep_i]
            video = s["video"].to(device)
            gt_pos = s["positions"].to(device)
            gt_vis = s["visibility"].to(device).bool()
            slot_states, _ = model.encode_video(video)
            T = video.shape[0]; id_dim = model.cfg.id_dim
            for t in range(T - k_steps):
                vm = gt_vis[t]
                if not vm.any(): continue
                vis_idx = torch.where(vm)[0]
                pp_now = model.slot_to_pos_aux(slot_states[t]).cpu().numpy()
                gt_now = gt_pos[t][vis_idx].cpu().numpy()
                rows, cols, _ = hungarian_assign(pp_now, gt_now)
                if len(rows) < 1: continue
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
                pred_future = readout(slot_t, rel_weights)
                target_future = gt_pos[t + k_steps][matched_entities].unsqueeze(0)
                err = ((pred_future - target_future) ** 2).sum(-1)  # [1, K]
                future_vis = gt_vis[t + k_steps][matched_entities]
                # Interaction-prone mask: per-entity, true iff entity has at least
                # one matched-entity neighbor at frame t with dist < threshold.
                pos_now = gt_pos[t][matched_entities]  # [K, 2]
                dists = torch.cdist(pos_now.unsqueeze(0), pos_now.unsqueeze(0)).squeeze(0)
                K = dists.shape[0]
                dists.fill_diagonal_(float("inf"))
                interaction_prone = (dists < args.near_threshold).any(dim=-1)  # [K]
                for i in range(err.shape[1]):
                    e = float(err[0, i])
                    if future_vis[i]:
                        visible_errors.append(e)
                        if interaction_prone[i]: inter_visible_errors.append(e)
                    else:
                        hidden_errors.append(e)
                        if interaction_prone[i]: inter_hidden_errors.append(e)
    return {
        "future_visible_mse": float(np.mean(visible_errors)) if visible_errors else float("nan"),
        "future_hidden_mse": float(np.mean(hidden_errors)) if hidden_errors else float("nan"),
        "n_visible": len(visible_errors),
        "n_hidden": len(hidden_errors),
        # Interaction-heavy subset:
        "interaction_visible_mse": float(np.mean(inter_visible_errors)) if inter_visible_errors else float("nan"),
        "interaction_hidden_mse":  float(np.mean(inter_hidden_errors))  if inter_hidden_errors  else float("nan"),
        "n_interaction_visible":   len(inter_visible_errors),
        "n_interaction_hidden":    len(inter_hidden_errors),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--of-jepa-steps", type=int, default=1500)
    p.add_argument("--readout-epochs", type=int, default=20)
    p.add_argument("--k-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--readout-lr", type=float, default=3e-3)
    p.add_argument("--rel-epochs", type=int, default=10)
    p.add_argument("--rel-lr", type=float, default=3e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--near-threshold", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    dataset = MoviDataset(MoviSpec(cache_dir=args.cache, image_size=args.image_size,
                                     max_entities=25, normalize_positions=False))
    n = len(dataset); indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    train_idx, eval_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]
    print(f"Episodes: {n}, train={len(train_idx)}, eval={len(eval_idx)}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = OFJEPAConfig(n_files=12, id_dim=64, state_dim=64, proposal_dim=128)
    model = OFJEPA(image_size=args.image_size, cfg=cfg, version="v0").to(args.device)
    warmup_of_jepa(model, dataset, train_idx, args, args.device)

    # Train relation graph (so it has SOMETHING in it).
    rel_cfg = RelationConfig(file_dim=cfg.id_dim + cfg.state_dim,
                              hidden_dim=128, n_relations=1, asymmetric=False)
    predictor = RelationGraphPredictor(rel_cfg).to(args.device)
    print("\n--- Quick relation head training ---", flush=True)
    from scripts.train_relations_movi import train_relation_head
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    train_relation_head(model, predictor, dataset, train_idx, args, args.device)

    slot_dim = cfg.id_dim + cfg.state_dim

    # --- Baseline readout ---
    print(f"\n--- Training BASELINE readout (slot_state → pos_t+{args.k_steps}) ---", flush=True)
    baseline = FutureReadout(slot_dim, use_relations=False).to(args.device)
    train_readout(baseline, model, None, dataset, train_idx, args, args.device, use_relations=False, k_steps=args.k_steps)
    base_eval = eval_readout(baseline, model, None, dataset, eval_idx, args, args.device, use_relations=False, k_steps=args.k_steps)
    print(f"\n  baseline: {base_eval}", flush=True)

    # --- +Relations readout ---
    print(f"\n--- Training +RELATIONS readout ---", flush=True)
    relout = FutureReadout(slot_dim, use_relations=True).to(args.device)
    train_readout(relout, model, predictor, dataset, train_idx, args, args.device, use_relations=True, k_steps=args.k_steps)
    rel_eval = eval_readout(relout, model, predictor, dataset, eval_idx, args, args.device, use_relations=True, k_steps=args.k_steps)
    print(f"\n  +relations: {rel_eval}", flush=True)

    # Compare on AVERAGE + on interaction-heavy subset.
    print("\n=== Predictive value of relation graph ===")
    for axis in ["future_visible_mse", "future_hidden_mse",
                  "interaction_visible_mse", "interaction_hidden_mse"]:
        bv = base_eval[axis]; rv = rel_eval[axis]
        if bv != bv or rv != rv:  # NaN check
            print(f"  {axis}: NaN (no samples)"); continue
        delta = (rv - bv) / max(bv, 1e-9) * 100
        verdict = "✅ improved" if rv < bv * 0.95 else ("⚠ no clear effect" if 0.95 <= rv / max(bv, 1e-9) <= 1.05 else "❌ regressed")
        marker = " [INTERACTION-HEAVY]" if axis.startswith("interaction") else ""
        print(f"  {axis}{marker}: baseline={bv:.4e}  +rel={rv:.4e}  ({delta:+.1f}%)  {verdict}")

    out_data = {"baseline": base_eval, "with_relations": rel_eval,
                 "k_steps": args.k_steps, "seed": args.seed}
    with open(out / f"seed{args.seed}.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote {out / f'seed{args.seed}.json'}", flush=True)


if __name__ == "__main__":
    main()
