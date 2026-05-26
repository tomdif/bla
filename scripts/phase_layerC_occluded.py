"""Phase Layer-C on the occluded-navigate environment.

This is the *signal* test for Causal Commitment Training: the env cycles
visible_steps frames where targets render normally, then hidden_steps
frames where targets vanish from the observation. The right behavior:

  visible window  -> low surprise -> low revision_rate -> drift penalized
  occlusion onset -> surprise spike (predictor can't see the target)
                  -> revision_rate jumps -> drift authorized
  target reappears -> second surprise spike -> revision_rate jumps again

This script does not train the agent — it rolls out random actions and
reports per-step metrics so the user can see whether the CCT mechanism
actually fires at occlusion boundaries.

Outputs JSON per step with: visibility flag, slot_delta loss, edge_bce,
revision_rate, drift, surprise_mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn

from system1_jepa import (
    BLAJEPAModel,
    CausalRelationConfig,
    CausalRelationHead,
    JEPAConfig,
    OccludedMultiTargetNavigateEnv,
    OccludedNavigateSpec,
    SlotAttention,
    SlotAttentionConfig,
    SlotDeltaPredictor,
    SlotPredictorConfig,
    causal_edge_loss,
    slot_delta_loss,
)
from verification import (
    CommitmentEncoder,
    CommitmentEncoderConfig,
    commitment_consistency_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-C: causal relations + CCT on occluded navigate.")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--visible-steps", type=int, default=4)
    p.add_argument("--hidden-steps", type=int, default=6)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--commit-dim", type=int, default=16)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.2)
    p.add_argument("--revision-temp", type=float, default=1.0,
                   help="lower temp -> sharper revision-rate response to z-scored surprise")
    p.add_argument("--surprise-ema", type=float, default=0.05,
                   help="EMA rate for surprise mean/var; surprise is z-scored before revision-rate")
    p.add_argument("--commit-horizon", type=int, default=3,
                   help="frames between c_past and c_now")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env_spec = OccludedNavigateSpec(
        image_size=args.image_size,
        n_targets=args.n_targets,
        visible_steps=args.visible_steps,
        hidden_steps=args.hidden_steps,
    )
    env = OccludedMultiTargetNavigateEnv(
        spec=env_spec, batch_size=args.batch_size, device=device, seed=args.seed,
    )

    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.image_size = args.image_size   # honored only if model uses it; safe to set
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    d_jepa = jepa_cfg.d_jepa
    pred_action_dim = jepa_cfg.action_dim

    slot_attn = SlotAttention(
        input_dim=d_jepa,
        cfg=SlotAttentionConfig(n_slots=args.n_slots, slot_dim=args.slot_dim, n_iters=2),
    ).to(device)
    predictor = SlotDeltaPredictor(SlotPredictorConfig(
        slot_dim=args.slot_dim, obs_dim=d_jepa, action_dim=pred_action_dim,
        n_layers=1, n_heads=2,
    )).to(device)
    relations = CausalRelationHead(CausalRelationConfig(
        slot_dim=args.slot_dim, n_edge_types=4, hidden=64,
    )).to(device)
    commit = CommitmentEncoder(CommitmentEncoderConfig(
        claim_dim=args.slot_dim, world_dim=d_jepa, out_dim=args.commit_dim,
    )).to(device)
    # 2-D env action -> predictor's higher-dim action embedding.
    action_embed = nn.Linear(2, pred_action_dim).to(device)

    trainable = (
        list(slot_attn.parameters()) + list(predictor.parameters())
        + list(relations.parameters()) + list(commit.parameters())
        + list(action_embed.parameters())
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    def encode(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z, _, _ = jepa.target_encoder(obs)
        return z   # [B, N_patches, d_jepa]

    env.reset()
    obs = env.observe()
    feats = encode(obs)
    slots_prev = slot_attn(feats)
    obs_feats_prev = feats

    # commitment_horizon-step ring of past commitment encodings keyed by the
    # claim at that time. We only need (c_past, claim_past, world_past).
    past_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # Running EMA of surprise mean and variance — z-scores the raw L2 error
    # so revision_rate responds to *relative* spikes, not the absolute scale.
    # Without this, revision_rate saturates near 0 or 1 and CCT enters a
    # degenerate flat regime.
    surprise_mean = None
    surprise_var = None

    for step in range(args.steps):
        # Random uniform action in env action space. We're probing
        # representations, not learning a policy.
        dxy = (torch.rand(args.batch_size, 2, device=device) * 2 - 1)
        env.step(dxy * env_spec.move_max)
        obs_now = env.observe()
        feats_now = encode(obs_now)
        slots_now = slot_attn(feats_now)
        visible = env.visibility_mask()    # [B] bool

        # Embed env action into the predictor's expected action_dim.
        act_emb = action_embed(dxy)

        pred = predictor(slots_prev, obs_feats_prev, act_emb)
        change_mask = pred["change_mask"]
        # Use the slot_attn re-binding of the current frame as the
        # supervision target for slots_prev's prediction.
        delta_metrics = slot_delta_loss(
            pred["next_slots"], slots_now.detach(),
            change_mask, sparsity_weight=1e-3,
        )

        # Causal relations need a SECOND change-mask sample for t+1 — use
        # the predictor's own forward on the current frame.
        pred_now = predictor(slots_now, feats_now, act_emb)
        edges = relations(slots_prev)
        edge_metrics = causal_edge_loss(edges, change_mask, pred_now["change_mask"])

        # Surprise = how badly the predictor missed the actual rebinding.
        # Detached because we don't want CCT gradients pulling the predictor.
        with torch.no_grad():
            surprise_raw = (pred["next_slots"] - slots_now).pow(2).mean(dim=(1, 2))
            if surprise_mean is None:
                surprise_mean = surprise_raw.mean()
                surprise_var = surprise_raw.var().clamp_min(1e-6)
            else:
                a = args.surprise_ema
                surprise_mean = (1 - a) * surprise_mean + a * surprise_raw.mean()
                surprise_var = (1 - a) * surprise_var + a * surprise_raw.var().clamp_min(1e-6)
            surprise = (surprise_raw - surprise_mean) / surprise_var.sqrt()

        # Commitment: c_past from commit_horizon frames ago, c_now from now,
        # both built from the same claim (slots at the past time).
        claim_now = slots_prev.mean(dim=1)
        c_now_for_past = commit(claim_now, feats_now.mean(dim=1))
        commit_metrics = None
        if len(past_buffer) >= args.commit_horizon:
            past_claim, past_world, past_c = past_buffer.pop(0)
            # Re-encode the OLD claim against the NEW world: drift =
            # 1 - cosine(c_past, c_now_at_old_claim_under_new_world).
            c_now = commit(past_claim, feats_now.mean(dim=1))
            commit_metrics = commitment_consistency_loss(
                past_c.detach(), c_now, surprise.detach(),
                revision_temp=args.revision_temp,
            )

        # Push a new entry to the past buffer for future evaluation.
        past_buffer.append((
            claim_now.detach(),
            feats_now.mean(dim=1).detach(),
            c_now_for_past.detach(),
        ))

        total = (
            delta_metrics["loss"]
            + args.edge_weight * edge_metrics["loss"]
        )
        if commit_metrics is not None:
            total = total + args.commit_weight * commit_metrics["loss"]

        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        record = {
            "step": step,
            "visible": bool(visible[0].item()),
            "total": float(total.detach()),
            "slot_delta": float(delta_metrics["prediction"]),
            "mask_mean": float(delta_metrics["mask_mean"]),
            "edge_bce": float(edge_metrics["bce"]),
            "surprise_raw": float(surprise_raw.mean()),
            "surprise_z": float(surprise.mean()),
        }
        if commit_metrics is not None:
            record.update({
                "commit_loss": float(commit_metrics["loss"].detach()),
                "revision_rate": float(commit_metrics["revision_rate"]),
                "drift": float(commit_metrics["drift"]),
            })
        print(json.dumps(record))

        slots_prev = slots_now.detach()
        obs_feats_prev = feats_now.detach()


if __name__ == "__main__":
    main()
