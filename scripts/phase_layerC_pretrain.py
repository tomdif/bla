"""Phase Layer-C pretrain + eval: validate CCT signal at occlusion boundaries.

Two phases:

  A. PRETRAIN  — visible-only navigate (hidden_steps=0). Train every
                 component (slot_attn, predictor, relations, commit,
                 action_embed) until predictor surprise is meaningful
                 rather than uniformly bad.

  B. EVAL      — freeze every learned component, switch to occluded env
                 (hidden_steps > 0), measure surprise + revision_rate
                 across visibility boundaries. The hypothesis being
                 tested:

                    surprise_hidden > surprise_visible
                    revision_rate_hidden > revision_rate_visible

                 If both hold, CCT does the right thing: the model knows
                 when it has lost the target and authorizes revision of
                 its commitments. If neither holds, the pretrained
                 predictor doesn't differentiate the two regimes.

Prints per-step JSON during both phases, then a summary block with the
visibility-conditioned means of surprise and revision_rate (and the
deltas between regimes — the actual headline number).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn

import torch.nn.functional as F

from system1_jepa import (
    BLAJEPAModel,
    CausalRelationConfig,
    CausalRelationHead,
    JEPAConfig,
    OccludedMultiTargetNavigateEnv,
    OccludedNavigateSpec,
    SceneContentHead,
    SlotAttention,
    SlotAttentionConfig,
    SlotDeltaPredictor,
    SlotPredictorConfig,
    causal_edge_loss,
    scene_content_signal,
    scene_content_surprise,
    slot_delta_loss,
)
from verification import (
    CommitmentEncoder,
    CommitmentEncoderConfig,
    commitment_consistency_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-C pretrain + occluded-eval CCT validator.")
    p.add_argument("--pretrain-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--visible-steps", type=int, default=4)
    p.add_argument("--hidden-steps", type=int, default=6)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--slot-dim", type=int, default=32)
    p.add_argument("--commit-dim", type=int, default=16)
    p.add_argument("--commit-horizon", type=int, default=3)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.2)
    p.add_argument("--existence-weight", type=float, default=0.5)
    p.add_argument("--revision-temp", type=float, default=1.0)
    p.add_argument("--surprise-ema", type=float, default=0.05)
    p.add_argument("--surprise-weighting", choices=["absolute", "asymmetric"],
                   default="asymmetric",
                   help="asymmetric: only spike when predicted-active slot becomes inactive (occlusion direction)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--print-every", type=int, default=20,
                   help="print one in N pretrain steps; eval always prints every step")
    return p.parse_args()


def build_models(args, device, env_spec):
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.image_size = args.image_size
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
    # Scene-content head: pooled-slots + action -> predicted next-frame
    # content scalar. Replaces the per-slot-existence head because per-slot
    # binding mass is uninformative at this scale (slots haven't specialized
    # enough for binding_mass to encode visibility).
    content_head = SceneContentHead(
        slot_dim=args.slot_dim, action_dim=pred_action_dim, hidden=32,
    ).to(device)
    action_embed = nn.Linear(2, pred_action_dim).to(device)
    return jepa, slot_attn, predictor, relations, commit, content_head, action_embed


def encode_factory(jepa):
    @torch.no_grad()
    def encode(obs: torch.Tensor) -> torch.Tensor:
        z, _, _ = jepa.target_encoder(obs)
        return z
    return encode


def step_loop(
    *, args, env, encode, slot_attn, predictor, relations, commit, content_head,
    action_embed, optim, surprise_state, past_buffer, training: bool,
    phase: str, step_offset: int, print_each: bool,
):
    """One env step + one loss step (training=True) or pure measurement.

    Returns a record dict suitable for downstream summary.
    """
    dxy = (torch.rand(args.batch_size, 2, device=env.device) * 2 - 1) * env.spec.move_max
    env.step(dxy)
    obs_now = env.observe()
    # Scene-content scalar — ground truth for next-frame content level.
    # Captured BEFORE feature encoding so it's robust to slot specialization.
    gt_content_now = scene_content_signal(obs_now).detach()
    feats_now = encode(obs_now)
    slots_now = slot_attn(feats_now)
    visible = env.visibility_mask()

    slots_prev = surprise_state["slots_prev"]
    obs_feats_prev = surprise_state["obs_feats_prev"]
    act_emb = action_embed(dxy / env.spec.move_max)

    pred = predictor(slots_prev, obs_feats_prev, act_emb)
    change_mask = pred["change_mask"]
    delta_metrics = slot_delta_loss(
        pred["next_slots"], slots_now.detach(),
        change_mask, sparsity_weight=1e-3,
    )

    pred_now = predictor(slots_now, feats_now, act_emb)
    edges = relations(slots_prev)
    edge_metrics = causal_edge_loss(edges, change_mask, pred_now["change_mask"])

    # Content head: predict next-frame content level from PREVIOUS slots +
    # action. The head learns "what will the world look like next" and
    # surprise = disagreement with what actually arrived.
    pred_content_next = content_head(slots_prev.mean(dim=1), act_emb)
    existence_metrics = {
        "loss": F.smooth_l1_loss(pred_content_next, gt_content_now),
        "pred_mean": pred_content_next.detach().mean(),
        "gt_mean": gt_content_now.mean(),
    }

    with torch.no_grad():
        # Surprise: disagreement on predicted vs actual scene content.
        # Asymmetric: only fires when predictor expected MORE content than
        # arrived (the occlusion direction). Symmetric also fires on
        # "world simpler than expected → world richer than expected".
        surprise_raw = scene_content_surprise(
            pred_content_next.detach(), gt_content_now,
            weighting=args.surprise_weighting,
        )
        if surprise_state["mean"] is None:
            surprise_state["mean"] = surprise_raw.mean()
            surprise_state["var"] = surprise_raw.var().clamp_min(1e-6)
        else:
            a = args.surprise_ema
            surprise_state["mean"] = (1 - a) * surprise_state["mean"] + a * surprise_raw.mean()
            surprise_state["var"] = (1 - a) * surprise_state["var"] + a * surprise_raw.var().clamp_min(1e-6)
        surprise_z = (surprise_raw - surprise_state["mean"]) / surprise_state["var"].sqrt()

    claim_now = slots_prev.mean(dim=1)
    c_now_for_past = commit(claim_now, feats_now.mean(dim=1))
    commit_metrics = None
    if len(past_buffer) >= args.commit_horizon:
        past_claim, _, past_c = past_buffer.pop(0)
        c_now = commit(past_claim, feats_now.mean(dim=1))
        commit_metrics = commitment_consistency_loss(
            past_c.detach(), c_now, surprise_z.detach(),
            revision_temp=args.revision_temp,
        )
    past_buffer.append((
        claim_now.detach(),
        feats_now.mean(dim=1).detach(),
        c_now_for_past.detach(),
    ))

    if training:
        total = (
            delta_metrics["loss"]
            + args.edge_weight * edge_metrics["loss"]
            + args.existence_weight * existence_metrics["loss"]
        )
        if commit_metrics is not None:
            total = total + args.commit_weight * commit_metrics["loss"]
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in optim.param_groups for p in g["params"]], 1.0,
        )
        optim.step()
        total_val = float(total.detach())
    else:
        total_val = float(
            delta_metrics["loss"].detach()
            + args.edge_weight * edge_metrics["loss"].detach()
            + args.existence_weight * existence_metrics["loss"].detach()
            + (args.commit_weight * commit_metrics["loss"].detach()
                if commit_metrics is not None else 0.0)
        )

    surprise_state["slots_prev"] = slots_now.detach()
    surprise_state["obs_feats_prev"] = feats_now.detach()

    record = {
        "phase": phase,
        "step": step_offset,
        "visible": bool(visible[0].item()),
        "total": total_val,
        "slot_delta": float(delta_metrics["prediction"]),
        "edge_bce": float(edge_metrics["bce"]),
        "content_loss": float(existence_metrics["loss"].detach()),
        "pred_content_mean": float(existence_metrics["pred_mean"]),
        "gt_content_mean": float(existence_metrics["gt_mean"]),
        "surprise_raw": float(surprise_raw.mean()),
        "surprise_z": float(surprise_z.mean()),
    }
    if commit_metrics is not None:
        record.update({
            "commit_loss": float(commit_metrics["loss"].detach()),
            "revision_rate": float(commit_metrics["revision_rate"]),
            "drift": float(commit_metrics["drift"]),
        })
    if print_each:
        print(json.dumps(record))
    return record


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---- PHASE A: pretrain on visible-only env ----
    visible_spec = OccludedNavigateSpec(
        image_size=args.image_size, n_targets=args.n_targets,
        visible_steps=args.visible_steps + args.hidden_steps,   # full cycle visible
        hidden_steps=0,
    )
    env_train = OccludedMultiTargetNavigateEnv(
        spec=visible_spec, batch_size=args.batch_size, device=device, seed=args.seed,
    )
    (jepa, slot_attn, predictor, relations, commit, content_head, action_embed) = build_models(
        args, device, visible_spec,
    )
    encode = encode_factory(jepa)
    trainable = (
        list(slot_attn.parameters()) + list(predictor.parameters())
        + list(relations.parameters()) + list(commit.parameters())
        + list(content_head.parameters()) + list(action_embed.parameters())
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    env_train.reset()
    obs0 = env_train.observe()
    feats0 = encode(obs0)
    surprise_state = {
        "slots_prev": slot_attn(feats0).detach(),
        "obs_feats_prev": feats0.detach(),
        "mean": None, "var": None,
    }
    past_buffer = []
    print(json.dumps({"event": "phase_begin", "phase": "pretrain",
                       "steps": args.pretrain_steps}))
    for step in range(args.pretrain_steps):
        step_loop(
            args=args, env=env_train, encode=encode,
            slot_attn=slot_attn, predictor=predictor, relations=relations,
            commit=commit, content_head=content_head, action_embed=action_embed,
            optim=optim, surprise_state=surprise_state, past_buffer=past_buffer,
            training=True, phase="pretrain", step_offset=step,
            print_each=(step % args.print_every == 0),
        )

    # ---- PHASE B: freeze, swap to occluded env, measure ----
    for m in (slot_attn, predictor, relations, commit, content_head, action_embed):
        for p in m.parameters():
            p.requires_grad_(False)

    occluded_spec = OccludedNavigateSpec(
        image_size=args.image_size, n_targets=args.n_targets,
        visible_steps=args.visible_steps, hidden_steps=args.hidden_steps,
    )
    env_eval = OccludedMultiTargetNavigateEnv(
        spec=occluded_spec, batch_size=args.batch_size, device=device, seed=args.seed + 1,
    )
    env_eval.reset()
    obs0 = env_eval.observe()
    feats0 = encode(obs0)
    eval_state = {
        "slots_prev": slot_attn(feats0).detach(),
        "obs_feats_prev": feats0.detach(),
        # Carry surprise stats from pretrain — that's the calibration we
        # built up; resetting them would hide the visibility signal.
        "mean": surprise_state["mean"], "var": surprise_state["var"],
    }
    eval_past_buffer = []
    eval_records = []
    print(json.dumps({"event": "phase_begin", "phase": "eval",
                       "steps": args.eval_steps,
                       "visible_steps": args.visible_steps,
                       "hidden_steps": args.hidden_steps}))
    for step in range(args.eval_steps):
        rec = step_loop(
            args=args, env=env_eval, encode=encode,
            slot_attn=slot_attn, predictor=predictor, relations=relations,
            commit=commit, content_head=content_head, action_embed=action_embed,
            optim=None, surprise_state=eval_state, past_buffer=eval_past_buffer,
            training=False, phase="eval", step_offset=step,
            print_each=True,
        )
        eval_records.append(rec)

    # ---- summary: visibility-conditioned signal split ----
    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    vis_surprise = [r["surprise_raw"] for r in eval_records if r["visible"]]
    hid_surprise = [r["surprise_raw"] for r in eval_records if not r["visible"]]
    vis_rev = [r["revision_rate"] for r in eval_records
                if r["visible"] and "revision_rate" in r]
    hid_rev = [r["revision_rate"] for r in eval_records
                if not r["visible"] and "revision_rate" in r]
    vis_drift = [r["drift"] for r in eval_records
                  if r["visible"] and "drift" in r]
    hid_drift = [r["drift"] for r in eval_records
                  if not r["visible"] and "drift" in r]
    print(json.dumps({
        "event": "summary",
        "n_visible_steps": len(vis_surprise),
        "n_hidden_steps": len(hid_surprise),
        "surprise_visible": mean(vis_surprise),
        "surprise_hidden": mean(hid_surprise),
        "surprise_delta_hidden_minus_visible": mean(hid_surprise) - mean(vis_surprise),
        "revision_rate_visible": mean(vis_rev),
        "revision_rate_hidden": mean(hid_rev),
        "revision_rate_delta_hidden_minus_visible": mean(hid_rev) - mean(vis_rev),
        "drift_visible": mean(vis_drift),
        "drift_hidden": mean(hid_drift),
        "passes_surprise_test": mean(hid_surprise) > mean(vis_surprise),
        "passes_revision_test": mean(hid_rev) > mean(vis_rev),
    }))


if __name__ == "__main__":
    main()
