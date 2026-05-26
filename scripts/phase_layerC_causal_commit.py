"""Phase Layer-C smoke: CausalRelationHead + CommitmentEncoder live together.

Exercises both new modules on the existing moving-patch synthetic data:

  frames -> JEPA target encoder -> patch features
  patch features per frame -> SlotAttention -> slots[t], slots[t+1]
  SlotDeltaPredictor(slots[t], obs[t], action) -> next_slots_pred, change_mask[t]
  SlotDeltaPredictor(slots[t+1], obs[t+1], action_next) -> ..., change_mask[t+1]
  CausalRelationHead(slots[t]) -> EdgeAnnotations
  causal_edge_loss(edges, change_mask[t], change_mask[t+1])

  CommitmentEncoder(claim=pooled_slots, world=pooled_obs)
    c_past at t, c_now at t+1
    surprise = mean-sq error between predicted next slots and actual slots[t+1]
  commitment_consistency_loss(c_past, c_now, surprise)

Total loss = slot_delta + edge_weight * causal_edge + commit_weight * commitment.
Prints per-step JSON like phase1e_train_temporal.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from system1_jepa import (
    BLAJEPAModel,
    CausalRelationConfig,
    CausalRelationHead,
    JEPAConfig,
    MovingPatchSpec,
    SlotAttention,
    SlotAttentionConfig,
    SlotDeltaPredictor,
    SlotPredictorConfig,
    causal_edge_loss,
    make_moving_patch_episodes,
    slot_delta_loss,
)
from verification import (
    CommitmentEncoder,
    CommitmentEncoderConfig,
    commitment_consistency_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-C: causal relations + commitment training smoke.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--commit-dim", type=int, default=16)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---- data + JEPA encoder (frozen target features) ----
    spec = MovingPatchSpec(
        image_size=args.image_size, patch_size=4,
        horizon=args.horizon, history=args.history,
    )
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.action_dim = spec.action_dim
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    d_jepa = jepa_cfg.d_jepa

    # ---- trainable Layer-C stack ----
    slot_attn = SlotAttention(
        input_dim=d_jepa,
        cfg=SlotAttentionConfig(n_slots=args.n_slots, slot_dim=args.slot_dim, n_iters=2),
    ).to(device)
    predictor = SlotDeltaPredictor(SlotPredictorConfig(
        slot_dim=args.slot_dim, obs_dim=d_jepa, action_dim=spec.action_dim,
        n_layers=1, n_heads=2,
    )).to(device)
    relations = CausalRelationHead(CausalRelationConfig(
        slot_dim=args.slot_dim, n_edge_types=4, hidden=64,
    )).to(device)
    commit = CommitmentEncoder(CommitmentEncoderConfig(
        claim_dim=args.slot_dim, world_dim=d_jepa, out_dim=args.commit_dim,
    )).to(device)

    trainable = (
        list(slot_attn.parameters()) + list(predictor.parameters())
        + list(relations.parameters()) + list(commit.parameters())
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    def encode_frames(frames_btchw: torch.Tensor) -> torch.Tensor:
        b, t = frames_btchw.shape[:2]
        flat = frames_btchw.reshape(b * t, *frames_btchw.shape[2:])
        with torch.no_grad():
            z, _, _ = jepa.target_encoder(flat)
        return z.reshape(b, t, *z.shape[1:])  # [B, T, N_patches, d_jepa]

    for step in range(args.steps):
        history, actions, future = make_moving_patch_episodes(
            spec, args.batch_size, device=device,
        )
        # Use the first two future frames as our (t, t+1) pair so both have actions.
        feats_future = encode_frames(future)            # [B, horizon, N, d_jepa]
        obs_t = feats_future[:, 0]                       # [B, N, d_jepa]
        obs_t1 = feats_future[:, 1]
        act_t = actions[:, 0, 0]                         # [B, action_dim]
        act_t1 = actions[:, 1, 0]

        slots_t = slot_attn(obs_t)                       # [B, S, slot_dim]
        slots_t1 = slot_attn(obs_t1)

        # Slot dynamics + change masks at both timesteps.
        pred_t = predictor(slots_t, obs_t, act_t)
        pred_t1 = predictor(slots_t1, obs_t1, act_t1)
        change_t = pred_t["change_mask"]
        change_t1 = pred_t1["change_mask"]

        # (1) Predictor's own loss: predict next slots ~ slots_t1 (detached).
        delta_metrics = slot_delta_loss(
            pred_t["next_slots"], slots_t1.detach(),
            change_t, sparsity_weight=1e-3,
        )

        # (2) Causal relations on slots_t, supervised by co-firing of change masks.
        edges = relations(slots_t)
        edge_metrics = causal_edge_loss(edges, change_t, change_t1)

        # (3) Commitment training. claim = pooled slots, world = pooled obs features.
        # surprise = how badly the predictor missed the actual next slots.
        with torch.no_grad():
            surprise = (pred_t["next_slots"] - slots_t1).pow(2).mean(dim=(1, 2))
        claim = slots_t.mean(dim=1)                      # [B, slot_dim]
        c_past = commit(claim, obs_t.mean(dim=1))
        c_now = commit(claim, obs_t1.mean(dim=1))
        commit_metrics = commitment_consistency_loss(c_past, c_now, surprise)

        total = (
            delta_metrics["loss"]
            + args.edge_weight * edge_metrics["loss"]
            + args.commit_weight * commit_metrics["loss"]
        )
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        print(json.dumps({
            "step": step,
            "total": float(total.detach()),
            "slot_delta": float(delta_metrics["prediction"]),
            "mask_mean": float(delta_metrics["mask_mean"]),
            "edge_bce": float(edge_metrics["bce"]),
            "edge_target_mean": float(edge_metrics["target_mean"]),
            "commit_loss": float(commit_metrics["loss"].detach()),
            "revision_rate": float(commit_metrics["revision_rate"]),
            "drift": float(commit_metrics["drift"]),
            "surprise_mean": float(surprise.mean()),
        }))


if __name__ == "__main__":
    main()
