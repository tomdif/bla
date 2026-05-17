"""Phase 7 (Kubric/MOVi) training + eval orchestrator.

Trains ConvNeXt-T → SlotAttention → SlotDeltaPredictor (in one of four
modes) by self-supervised JEPA loss on MOVi-A episodes, then evaluates
with the identity-aware Hungarian probe.

MOVi-A has no agent action — objects fall under gravity. We use a
zero action vector throughout and let the predictor learn dynamics.

Modes:
  slot_delta         — sparse mask × tanh(delta) update
  slot_dense_update  — dense delta update (no mask)
  dense_jepa         — flat JEPA over patch tokens, no slot mechanism
  copy               — identity baseline (predict next == current)

Six metrics reported per sub-run:
  visible_position_mse  — probe MSE on visible frames (calibration)
  hidden_position_mse   — probe MSE on naturally-occluded frames (key)
  identity_switch_rate  — slot ↔ entity stability across consecutive frames
  mean_slot_diversity   — distinct entities each slot was matched to
  hidden_visible_ratio  — bounded-forgetting indicator
  occlusion_degradation — slope of hidden MSE over hidden depth

Usage:
    python scripts/slot_jepa_movi_train.py \\
        --cache /workspace/movi_a_local/validation \\
        --modes slot_delta,slot_dense_update,dense_jepa,copy \\
        --seeds 0,1,2 --out /workspace/phase7_run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.convnext_encoder import ConvNeXtEncoderConfig, ConvNeXtSlotEncoder
from system1_jepa.id_consistency import (
    SlotPosAuxHead, cosine_diagnostic, drift_diagnostic,
    identity_consistency_loss, identity_contrastive_loss,
)
from system1_jepa.of_jepa import OFJEPA, OFJEPAConfig
from system1_jepa.identity_probe import ProbeFitConfig, identity_aware_probe_eval
from system1_jepa.movi_data import MoviDataset, MoviSpec, ATTR_DIM
from system1_jepa.sigreg import sigreg_lewm
from system1_jepa.slot import SlotAttention, SlotAttentionConfig
from system1_jepa.slot_predictor import (
    SlotDeltaPredictor,
    SlotPredictorConfig,
)


# ---------- Model assembly ---------------------------------------------------

class MoviJEPA(nn.Module):
    """Encoder + SlotAttention + (Predictor) wired for MOVi episodes."""

    def __init__(self, image_size: int, slot_dim: int, n_slots: int,
                 mode: str, mask_bias_init: float = 0.0,
                 target_active_slots: int = 0):
        super().__init__()
        # Phase 7C / 8A: mode can carry suffixes indicating which auxiliary
        # losses are active. Suffixes can stack in any order.
        #   "_idcons"      → encoder identity consistency loss + aux pos head
        #   "_idcontrast"  → identity InfoNCE contrastive loss on id_key
        self.use_id_cons = "_idcons" in mode
        self.use_id_contrast = "_idcontrast" in mode
        for suffix in ("_idcons", "_idcontrast"):
            mode = mode.replace(suffix, "")
        self.mode = mode
        self.slot_dim = slot_dim
        self.n_slots = n_slots

        self.enc = ConvNeXtSlotEncoder(ConvNeXtEncoderConfig(
            input_size=image_size, slot_dim=slot_dim,
            pretrained=False, freeze_early_stages=0,
        ))
        self.slot_attn = SlotAttention(
            input_dim=slot_dim,
            cfg=SlotAttentionConfig(n_slots=n_slots, slot_dim=slot_dim, n_iters=3),
        )

        if mode in ("slot_delta", "slot_dense_update", "dynamic", "id_dyn_split"):
            update_mode = {
                "slot_delta": "delta",
                "slot_dense_update": "dense",
                "dynamic": "dynamic",
                "id_dyn_split": "id_dyn_split",
            }[mode]
            self.predictor: Optional[nn.Module] = SlotDeltaPredictor(
                SlotPredictorConfig(
                    slot_dim=slot_dim, obs_dim=slot_dim, action_dim=4,
                    n_layers=2, n_heads=4, mlp_ratio=4,
                    delta_scale=0.2, dropout=0.0,
                    mask_bias_init=mask_bias_init,
                    update_mode=update_mode,
                    target_active_slots=target_active_slots,
                    id_dim=slot_dim // 2,
                    id_ema_alpha=0.05,
                )
            )
        elif mode == "dense_jepa":
            # Identical encoder (SlotAttention) — only the PREDICTOR differs.
            # Dense predictor: flatten [B, S, D] → [B, S*D] → MLP → [B, S*D] → reshape.
            # No slot-structured attention, no change mask, no per-slot delta.
            # Same parameter budget as slot_delta predictor (within ±20%).
            self.predictor = nn.Sequential(
                nn.LayerNorm(n_slots * slot_dim),
                nn.Linear(n_slots * slot_dim, n_slots * slot_dim * 2),
                nn.GELU(),
                nn.Linear(n_slots * slot_dim * 2, n_slots * slot_dim),
            )
            self._dense_jepa_S = n_slots
            self._dense_jepa_D = slot_dim
        elif mode == "copy":
            self.predictor = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Phase 7C: auxiliary slot→pos head, used both to train an oracle
        # binding signal and to drive Hungarian matching for the identity
        # consistency loss. Tiny, single Linear.
        self.id_dim = slot_dim // 2 if mode == "id_dyn_split" else slot_dim // 2
        self.slot_to_pos_aux = SlotPosAuxHead(slot_dim)

    @staticmethod
    def _norm_slots(slots: torch.Tensor) -> torch.Tensor:
        """LayerNorm-style normalize per-slot. Prevents the slot-persistence
        recurrence from compounding magnitudes across 24 frames × N training
        steps (caused exponential loss blowup on seeds 0,2 at LR=3e-4)."""
        return F.layer_norm(slots, slots.shape[-1:])

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """video: [T, 3, H, W] → (slot_states [T, S, D], patch_tokens [T, N, D])
        Uses persistent slot init across frames (slot system carries state)."""
        T = video.shape[0]
        tokens = self.enc(video)
        slots = self._norm_slots(self.slot_attn(tokens[0:1]))
        slot_states = [slots]
        for t in range(1, T):
            slots = self._norm_slots(self.slot_attn(tokens[t:t+1], init_slots=slots))
            slot_states.append(slots)
        return torch.cat(slot_states, dim=0), tokens

    def encode_video_grad(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as encode_video but tracks grads. Used in training pass."""
        T = video.shape[0]
        tokens = self.enc(video)
        slots = self._norm_slots(self.slot_attn(tokens[0:1]))
        slot_states = [slots]
        for t in range(1, T):
            slots = self._norm_slots(self.slot_attn(tokens[t:t+1], init_slots=slots))
            slot_states.append(slots)
        return torch.cat(slot_states, dim=0), tokens

    def predict_next(self, slots_t: torch.Tensor, obs_t: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """Predict slots_{t+1} from slots_t, obs features, action."""
        if self.mode == "copy":
            return slots_t
        if self.mode in ("slot_delta", "slot_dense_update", "dynamic", "id_dyn_split"):
            out = self.predictor(slots_t, obs_t, action)
            return out["next_slots"]
        if self.mode == "dense_jepa":
            # Flat dense predictor over the slot block.
            B, S, D = slots_t.shape
            flat = slots_t.reshape(B, S * D)
            pred_flat = self.predictor(flat)
            return pred_flat.reshape(B, S, D)
        return self.predictor(slots_t, obs_t, action)


# ---------- Training -----------------------------------------------------

def train_one_run(
    train_indices: List[int],
    eval_indices: List[int],
    dataset: MoviDataset,
    mode: str,
    seed: int,
    args,
    out: Path,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = args.device

    if mode == "of_jepa_v0":
        # Phase 8C: Object-File JEPA — persistent learned id_keys, memory-anchored
        # Sinkhorn matching, sparse-delta state_value. Slot_dim becomes id+state.
        of_cfg = OFJEPAConfig(
            n_files=args.n_slots,
            id_dim=args.slot_dim // 2,
            state_dim=args.slot_dim // 2,
            proposal_dim=args.slot_dim,
            id_ema_alpha=0.05,
            state_delta_scale=0.2,
            sinkhorn_iters=20,
            sinkhorn_temperature=0.1,
        )
        model = OFJEPA(image_size=args.image_size, cfg=of_cfg).to(device)
    else:
        model = MoviJEPA(
            image_size=args.image_size, slot_dim=args.slot_dim,
            n_slots=args.n_slots, mode=mode,
            mask_bias_init=args.mask_bias_init,
            target_active_slots=0,
        ).to(device)

    if mode != "copy":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        opt = None

    train_subset = Subset(dataset, train_indices)
    loader = DataLoader(train_subset, batch_size=1, shuffle=True,
                        num_workers=0, drop_last=True)

    log: List[Dict] = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break
            video = batch["video"][0].to(device)        # [T, 3, H, W]
            T = video.shape[0]

            if mode == "copy":
                step += 1
                continue

            opt.zero_grad(set_to_none=True)
            slot_states, tokens = model.encode_video_grad(video)

            if mode == "of_jepa_v0":
                # Phase 8C: OF-JEPA training signal.
                # 1. JEPA self-prediction on state_value via the memory's own
                #    internal step (slot_states[t+1] from slot_states[t]).
                # 2. Supervised position loss via aux_pos head — this is what
                #    drives the encoder + state_value to encode location info,
                #    and provides a clean training signal grounded in MOVi GT.
                # The Sinkhorn assignment in memory.step is trained INDIRECTLY
                # via these two losses + the persistent id_proto parameters.
                gt_pos_t = batch["positions"][0].to(device)
                gt_vis_t = batch["visibility"][0].to(device).bool()

                # JEPA-style internal consistency on STATE_VALUE only.
                # id_key updates via EMA, not predictor — including it in the JEPA
                # loss makes the loss punish EMA drift, which is wrong (and was
                # the source of the v0-first-attempt divergence).
                id_dim = model.cfg.id_dim
                state_dim = model.cfg.state_dim
                state_only = slot_states[..., id_dim:]                 # [T, N, state_dim]
                jepa_loss = 0.0
                for t in range(T - 1):
                    target = state_only[t+1].detach()
                    pred = state_only[t]
                    jepa_loss = jepa_loss + F.mse_loss(pred, target)
                jepa_loss = jepa_loss / max(T - 1, 1)

                # Supervised position loss via slot_to_pos_aux head.
                pred_positions = model.slot_to_pos_aux(slot_states)  # [T, N_files, 2]
                pos_loss = 0.0
                pos_count = 0
                for t in range(T):
                    vis_mask = gt_vis_t[t]
                    if not vis_mask.any():
                        continue
                    pred_t = pred_positions[t].unsqueeze(0)
                    gt_t_vis = gt_pos_t[t][vis_mask].unsqueeze(0)
                    if gt_t_vis.shape[1] == 0:
                        continue
                    # Hungarian-MSE matching predicted slots to visible GT entities.
                    from system1_jepa.identity_probe import hungarian_assign
                    pp_np = pred_t[0].detach().cpu().numpy()
                    gt_np = gt_t_vis[0].detach().cpu().numpy()
                    rows, cols, _ = hungarian_assign(pp_np, gt_np)
                    if len(rows) > 0:
                        rs = torch.from_numpy(rows).to(device)
                        cs = torch.from_numpy(cols).to(device)
                        pos_loss = pos_loss + F.mse_loss(
                            pred_t[0, rs], gt_t_vis[0, cs]
                        )
                        pos_count += 1
                pos_loss = pos_loss / max(pos_count, 1)
                loss = args.of_jepa_w * jepa_loss + args.of_pos_w * pos_loss

            else:
                # Existing JEPA loss for slot_delta-family modes.
                zero_action = torch.zeros(1, 4, device=device)
                loss = 0.0
                count = 0
                for t in range(T - 1):
                    pred = model.predict_next(
                        slot_states[t:t+1], tokens[t:t+1], zero_action,
                    )
                    target = slot_states[t+1:t+2].detach()
                    loss = loss + F.mse_loss(pred, target)
                    count += 1
                loss = loss / max(count, 1)

            # SIGReg on slot states to prevent collapse.
            if args.sigreg_w > 0:
                sr = sigreg_lewm(slot_states.reshape(-1, slot_states.shape[-1]))
                loss = loss + args.sigreg_w * sr

            # Phase 7C: identity consistency loss + aux head training.
            # slot_states is [T, S, D]; gt_pos and gt_vis are [T, E_max, ...]
            mode_uses_idcons = getattr(model, "use_id_cons", False)
            eff_id_cons_w = args.id_consistency_w if mode_uses_idcons else 0.0
            eff_aux_pos_w = args.aux_pos_w if mode_uses_idcons else 0.0
            if eff_id_cons_w > 0 or eff_aux_pos_w > 0:
                gt_pos_t = batch["positions"][0].to(device)        # [T, E_max, 2]
                gt_vis_t = batch["visibility"][0].to(device).bool() # [T, E_max]
                cons_loss, aux_loss = identity_consistency_loss(
                    slot_states, model.slot_to_pos_aux,
                    gt_pos_t, gt_vis_t, model.id_dim,
                )
                loss = loss + eff_id_cons_w * cons_loss + eff_aux_pos_w * aux_loss

            # Phase 8A: identity-contrastive loss on the id-key subspace.
            mode_uses_contrast = getattr(model, "use_id_contrast", False)
            if mode_uses_contrast and args.id_contrastive_w > 0:
                gt_pos_t = batch["positions"][0].to(device)
                gt_vis_t = batch["visibility"][0].to(device).bool()
                contrast_loss = identity_contrastive_loss(
                    slot_states, model.slot_to_pos_aux,
                    gt_pos_t, gt_vis_t, model.id_dim,
                    temperature=args.id_contrastive_temp,
                )
                loss = loss + args.id_contrastive_w * contrast_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                log.append({"step": step, "loss": float(loss.item()),
                             "elapsed_s": time.time() - t0})
                print(f"[seed={seed} mode={mode}] step {step}/{args.max_steps} "
                       f"loss={float(loss.item()):.4f} t={time.time()-t0:.1f}s",
                       flush=True)
        if step >= args.max_steps:
            break

    # ---- Eval with identity probe ----
    model.eval()
    eval_subset = Subset(dataset, eval_indices)
    all_states = []
    all_positions = []
    all_attrs = []
    all_visible = []
    all_ids = []
    all_ep = []
    all_frame = []
    all_hidden = []

    with torch.no_grad():
        for ep_offset, ep_idx in enumerate(eval_indices):
            sample = dataset[ep_idx]
            video = sample["video"].to(device)
            slot_states, _ = model.encode_video(video)
            T = video.shape[0]
            # Hidden step = number of frames since the entity was last visible.
            # We compute it per (frame, entity) pair below in gt loop.
            visibility_te = sample["visibility"]   # [T, E_max]
            E_max = visibility_te.shape[-1]
            last_visible = -torch.ones(E_max, dtype=torch.long)
            for t in range(T):
                hidden_d = torch.zeros(1, dtype=torch.long)
                # Per-entity hidden depth at time t:
                #  count of frames since the entity was last seen visible.
                # We summarize at frame-level by taking max across entities
                # currently invisible — that's the "deepest hidden depth"
                # any entity has reached at this frame.
                cur_vis = visibility_te[t]
                for e in range(E_max):
                    if cur_vis[e]:
                        last_visible[e] = t
                hidden_per_e = torch.where(cur_vis, torch.zeros(E_max, dtype=torch.long),
                                            t - last_visible)
                # Frame-level summary: max over invisible entities, else 0.
                if (~cur_vis).any():
                    hidden_d = hidden_per_e[~cur_vis].max().unsqueeze(0)
                all_states.append(slot_states[t])
                all_positions.append(sample["positions"][t])
                all_attrs.append(sample["attrs"])
                all_visible.append(cur_vis)
                all_ids.append(sample["entity_ids"])
                all_ep.append(ep_offset)
                all_frame.append(t)
                all_hidden.append(int(hidden_d.item()))

    states = torch.stack(all_states).cpu()
    gt_pos = torch.stack(all_positions)
    gt_attr = torch.stack(all_attrs)
    gt_visible = torch.stack(all_visible)
    gt_ids = torch.stack(all_ids)
    ep_ids = torch.tensor(all_ep, dtype=torch.long)
    frame_idx = torch.tensor(all_frame, dtype=torch.long)
    hidden_step = torch.tensor(all_hidden, dtype=torch.long)

    cfg = ProbeFitConfig(epochs=args.probe_epochs, lr=5e-3,
                         batch_size=128, attr_weight=1.0)
    result = identity_aware_probe_eval(
        states=states, gt_pos=gt_pos, gt_attr=gt_attr,
        gt_visible=gt_visible, gt_entity_ids=gt_ids,
        ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
        J=0, cfg=cfg,
    )

    # Phase 7C: id_drift / dyn_drift diagnostic across eval episodes.
    # Skip for the 'copy' baseline (no slot_to_pos_aux meaningfully trained).
    drift_records = []
    cosine_records = []
    with torch.no_grad():
        for ep_idx in eval_indices[:16]:  # first 16 eval eps for speed
            sample = dataset[ep_idx]
            video = sample["video"].to(device)
            slot_seq, _ = model.encode_video(video)        # [T, S, D]
            gt_pos_e = sample["positions"].to(device)
            gt_vis_e = sample["visibility"].to(device).bool()
            drift = drift_diagnostic(slot_seq, gt_pos_e, gt_vis_e,
                                      model.slot_to_pos_aux, model.id_dim)
            if drift["n_pairs"] > 0:
                drift_records.append(drift)
            cos = cosine_diagnostic(slot_seq, model.slot_to_pos_aux,
                                      gt_pos_e, gt_vis_e, model.id_dim)
            if cos["n_same_pairs"] > 0 and cos["n_diff_pairs"] > 0:
                cosine_records.append(cos)

    # 6 metrics for the decision doc.
    metrics = {
        "visible_position_mse": result.visible_position_mse,
        "hidden_position_mse": result.hidden_position_mse,
        "identity_switch_rate": result.identity_switch_rate,
        "mean_slot_diversity": result.mean_slot_diversity,
        "hidden_visible_ratio": (
            result.hidden_position_mse / max(result.visible_position_mse, 1e-9)
        ),
        "occlusion_degradation_per_step": result.per_step_position_mse,
        "n_visible": result.n_visible,
        "n_hidden": result.n_hidden,
    }
    if drift_records:
        metrics["id_drift"] = float(np.mean([d["id_drift"] for d in drift_records]))
        metrics["dyn_drift"] = float(np.mean([d["dyn_drift"] for d in drift_records]))
        metrics["drift_ratio"] = metrics["id_drift"] / max(metrics["dyn_drift"], 1e-9)
    if cosine_records:
        metrics["same_object_cos"] = float(np.mean([c["same_cos"] for c in cosine_records]))
        metrics["diff_object_cos"] = float(np.mean([c["diff_cos"] for c in cosine_records]))
        metrics["cos_gap"] = metrics["same_object_cos"] - metrics["diff_object_cos"]

    # Phase 7D: id-subspace probe — re-fit a fresh linear probe on the id-half
    # of slot states only. Tells us whether the id subspace is doing the
    # assignment-stability work even when the full-slot Hungarian shuffles.
    id_dim = model.id_dim
    if model.mode in ("id_dyn_split",) or model.use_id_cons or model.use_id_contrast:
        result_id = identity_aware_probe_eval(
            states=states, gt_pos=gt_pos, gt_attr=gt_attr,
            gt_visible=gt_visible, gt_entity_ids=gt_ids,
            ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
            J=0, cfg=cfg, subspace_dims=slice(0, id_dim),
        )
        result_dyn = identity_aware_probe_eval(
            states=states, gt_pos=gt_pos, gt_attr=gt_attr,
            gt_visible=gt_visible, gt_entity_ids=gt_ids,
            ep_ids=ep_ids, frame_idx=frame_idx, hidden_step=hidden_step,
            J=0, cfg=cfg, subspace_dims=slice(id_dim, None),
        )
        metrics["switch_rate_id_subspace"] = result_id.identity_switch_rate
        metrics["switch_rate_dyn_subspace"] = result_dyn.identity_switch_rate
        metrics["hidden_pos_mse_id_subspace"] = result_id.hidden_position_mse
        metrics["hidden_pos_mse_dyn_subspace"] = result_dyn.hidden_position_mse
    record = {
        "seed": seed,
        "mode": mode,
        "n_train": len(train_indices),
        "n_eval": len(eval_indices),
        "steps": step,
        "elapsed_s": time.time() - t0,
        "metrics": metrics,
        "loss_log": log,
    }
    return record


# ---------- Main -----------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--modes", default="slot_delta,slot_dense_update,dense_jepa,copy")
    p.add_argument("--seeds", default="0")
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=12)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sigreg-w", type=float, default=0.01)
    p.add_argument("--id-consistency-w", type=float, default=0.0,
                   help="Phase 7C: weight on encoder identity consistency loss")
    p.add_argument("--aux-pos-w", type=float, default=0.0,
                   help="Phase 7C: weight on aux slot→pos head training loss")
    p.add_argument("--id-contrastive-w", type=float, default=0.0,
                   help="Phase 8A: weight on InfoNCE identity contrastive loss")
    p.add_argument("--id-contrastive-temp", type=float, default=0.1,
                   help="Phase 8A: contrastive temperature τ")
    p.add_argument("--of-jepa-w", type=float, default=1.0,
                   help="Phase 8C OF-JEPA: weight on internal JEPA temporal smoothness")
    p.add_argument("--of-pos-w", type=float, default=10.0,
                   help="Phase 8C OF-JEPA: weight on supervised position loss")
    p.add_argument("--mask-bias-init", type=float, default=0.0)
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    dataset = MoviDataset(MoviSpec(cache_dir=args.cache, image_size=args.image_size,
                                     max_entities=10))
    n = len(dataset)
    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)  # stable split across seeds
    n_train = int(args.train_frac * n)
    train_indices = indices[:n_train]
    eval_indices = indices[n_train:]
    print(f"Episodes: {n} total, {n_train} train, {n - n_train} eval.", flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    records = []
    for seed in seeds:
        for mode in modes:
            print(f"\n=== seed={seed} mode={mode} ===", flush=True)
            rec = train_one_run(train_indices, eval_indices, dataset,
                                mode, seed, args, out)
            records.append(rec)
            with open(out / f"sub_seed{seed}_{mode}.json", "w") as f:
                json.dump(rec, f, indent=2)

    with open(out / "all_records.json", "w") as f:
        json.dump(records, f, indent=2)

    # Aggregate: mean & std per mode.
    agg: Dict[str, List[Dict]] = {}
    for r in records:
        agg.setdefault(r["mode"], []).append(r["metrics"])

    summary = {}
    for mode, rs in agg.items():
        per_metric = {}
        for k in rs[0].keys():
            if isinstance(rs[0][k], (int, float)):
                vals = [r[k] for r in rs]
                per_metric[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }
        summary[mode] = per_metric
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
