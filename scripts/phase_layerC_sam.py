"""Phase Layer-C with SAM-shaped binding source — the BLA-Forge bridge.

Same pretrain + eval structure as phase_layerC_ofjepa.py, but the
binding source is `SAMObjectFiles` driven by a `SyntheticOcclusionPerception`
that mocks SAM 2.1 output from env state. Confidence drops to 0.0 for
fully-occluded entities (matching real SAM 2.1 behavior — empty masks
emit zero-confidence detections, per bla/forge/sam_perception.py).

The hardware swap on BLA-Forge is one line:

    perception = SyntheticOcclusionPerception(env)           # before
    perception = SAMPerception(video_path=…, seeds=…,        # after
                                 backend="sam2.1")

Everything downstream — SAMObjectFiles bridge, predictor, relations,
existence head, commitment encoder — is identical.

batch_size=1 by design: SAM 2.1 is single-video, BLA-Forge is single-
camera, and the real-world version of this loop runs at hardware FPS
through the rolling K=5 tracker (BF-0.3).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import (
    CausalRelationConfig,
    CausalRelationHead,
    OccludedMultiTargetNavigateEnv,
    OccludedNavigateSpec,
    SlotDeltaPredictor,
    SlotExistenceHead,
    SlotPredictorConfig,
    causal_edge_loss,
    slot_delta_loss,
    slot_existence_loss,
    visibility_disagreement_surprise,
)
from verification import (
    CommitmentEncoder,
    CommitmentEncoderConfig,
    commitment_consistency_loss,
)
from bla.forge.sam_object_files import (
    SAMObjectFiles, SyntheticOcclusionPerception,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-C with SAM-shaped binding — BLA-Forge bridge.")
    p.add_argument("--pretrain-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=80)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--visible-steps", type=int, default=4)
    p.add_argument("--hidden-steps", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=64)
    p.add_argument("--commit-dim", type=int, default=16)
    p.add_argument("--commit-horizon", type=int, default=3)
    p.add_argument("--action-emb-dim", type=int, default=32)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.2)
    p.add_argument("--existence-weight", type=float, default=1.0)
    p.add_argument("--revision-temp", type=float, default=1.0)
    p.add_argument("--surprise-ema", type=float, default=0.05)
    p.add_argument("--surprise-weighting", choices=["absolute", "asymmetric"],
                   default="asymmetric")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--print-every", type=int, default=40)
    return p.parse_args()


def build_models(args, env, device):
    obj_ids = [0] + [i + 1 for i in range(args.n_targets)]   # agent + targets
    perception = SyntheticOcclusionPerception(env, mock_box_half=8.0)
    binding = SAMObjectFiles(
        perception=perception, obj_ids=obj_ids,
        image_size=args.image_size, slot_dim=args.slot_dim,
    ).to(device)
    pred_cfg = SlotPredictorConfig(
        slot_dim=args.slot_dim, obs_dim=args.slot_dim,
        action_dim=args.action_emb_dim, n_layers=1, n_heads=2,
    )
    predictor = SlotDeltaPredictor(pred_cfg).to(device)
    relations = CausalRelationHead(CausalRelationConfig(
        slot_dim=args.slot_dim, n_edge_types=4, hidden=64,
    )).to(device)
    commit = CommitmentEncoder(CommitmentEncoderConfig(
        claim_dim=args.slot_dim, world_dim=args.slot_dim, out_dim=args.commit_dim,
    )).to(device)
    existence = SlotExistenceHead(slot_dim=args.slot_dim, hidden=32).to(device)
    action_embed = nn.Linear(2, args.action_emb_dim).to(device)
    return binding, predictor, relations, commit, existence, action_embed


def step_loop(
    *, args, env, perception, binding, predictor, relations, commit, existence,
    action_embed, optim, surprise_state, past_buffer, frame_counter,
    training: bool, phase: str, step_offset: int, print_each: bool,
):
    dxy = (torch.rand(1, 2, device=env.device) * 2 - 1) * env.spec.move_max
    env.step(dxy)
    # Snap perception from CURRENT env state, then have the binding adapter
    # consume it at the same frame_idx.
    fi = frame_counter[0]
    perception.snap(fi)
    ofb_now = binding.observe(fi)
    frame_counter[0] += 1

    slots_now = ofb_now.full_slot                              # [1, N, slot_dim]
    gt_exists_now = ofb_now.confidences.detach().clamp(0.0, 1.0)
    visible = env.visibility_mask()

    slots_prev = surprise_state["slots_prev"]
    act_emb = action_embed(dxy / env.spec.move_max)

    pred = predictor(slots_prev, slots_prev, act_emb)
    change_mask = pred["change_mask"]
    delta_metrics = slot_delta_loss(
        pred["next_slots"], slots_now.detach(),
        change_mask, sparsity_weight=1e-3,
    )

    pred_now = predictor(slots_now, slots_now, act_emb)
    edges = relations(slots_prev)
    edge_metrics = causal_edge_loss(edges, change_mask, pred_now["change_mask"])

    pred_exists_next = existence(pred["next_slots"])
    existence_metrics = slot_existence_loss(pred_exists_next, gt_exists_now)

    with torch.no_grad():
        surprise_raw = visibility_disagreement_surprise(
            pred_exists_next.detach(), gt_exists_now,
            weighting=args.surprise_weighting,
        )
        # batch_size=1 makes within-batch .var() NaN; use temporal EMA of
        # squared deviation instead (also a more honest scale estimator).
        a = args.surprise_ema
        if surprise_state["mean"] is None:
            surprise_state["mean"] = surprise_raw.mean()
            surprise_state["var"] = torch.tensor(1.0, device=surprise_raw.device)
        else:
            new_mean = (1 - a) * surprise_state["mean"] + a * surprise_raw.mean()
            dev_sq = (surprise_raw.mean() - surprise_state["mean"]) ** 2
            surprise_state["var"] = (
                (1 - a) * surprise_state["var"] + a * dev_sq
            ).clamp_min(1e-6)
            surprise_state["mean"] = new_mean
        surprise_z = (surprise_raw - surprise_state["mean"]) / surprise_state["var"].sqrt()

    claim_now = slots_prev.mean(dim=1)
    c_now_for_past = commit(claim_now, slots_now.mean(dim=1))
    commit_metrics = None
    if len(past_buffer) >= args.commit_horizon:
        past_claim, _, past_c = past_buffer.pop(0)
        c_now = commit(past_claim, slots_now.mean(dim=1))
        commit_metrics = commitment_consistency_loss(
            past_c.detach(), c_now, surprise_z.detach(),
            revision_temp=args.revision_temp,
        )
    past_buffer.append((
        claim_now.detach(),
        slots_now.mean(dim=1).detach(),
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

    record = {
        "phase": phase, "step": step_offset,
        "visible": bool(visible[0].item()),
        "total": total_val,
        "slot_delta": float(delta_metrics["prediction"]),
        "edge_bce": float(edge_metrics["bce"]),
        "existence_bce": float(existence_metrics["loss"].detach()),
        "pred_exists_mean": float(existence_metrics["pred_mean"]),
        "gt_exists_mean": float(existence_metrics["gt_mean"]),
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

    visible_spec = OccludedNavigateSpec(
        image_size=args.image_size, n_targets=args.n_targets,
        visible_steps=args.visible_steps + args.hidden_steps, hidden_steps=0,
    )
    env_train = OccludedMultiTargetNavigateEnv(
        spec=visible_spec, batch_size=1, device=device, seed=args.seed,
    )
    (binding, predictor, relations, commit, existence, action_embed) = build_models(
        args, env_train, device,
    )

    trainable = (
        list(binding.parameters()) + list(predictor.parameters())
        + list(relations.parameters()) + list(commit.parameters())
        + list(existence.parameters()) + list(action_embed.parameters())
    )
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    env_train.reset()
    binding.reset_episode(batch_size=1)
    frame_counter = [0]
    binding.perception.snap(0)
    ofb0 = binding.observe(0)
    frame_counter[0] = 1
    surprise_state = {
        "slots_prev": ofb0.full_slot.detach(),
        "mean": None, "var": None,
    }
    past_buffer = []
    print(json.dumps({"event": "phase_begin", "phase": "pretrain",
                       "steps": args.pretrain_steps}))
    for step in range(args.pretrain_steps):
        step_loop(
            args=args, env=env_train, perception=binding.perception,
            binding=binding, predictor=predictor, relations=relations,
            commit=commit, existence=existence, action_embed=action_embed,
            optim=optim, surprise_state=surprise_state, past_buffer=past_buffer,
            frame_counter=frame_counter, training=True,
            phase="pretrain", step_offset=step,
            print_each=(step % args.print_every == 0),
        )

    for m in (binding, predictor, relations, commit, existence, action_embed):
        for p in m.parameters():
            p.requires_grad_(False)

    occluded_spec = OccludedNavigateSpec(
        image_size=args.image_size, n_targets=args.n_targets,
        visible_steps=args.visible_steps, hidden_steps=args.hidden_steps,
    )
    env_eval = OccludedMultiTargetNavigateEnv(
        spec=occluded_spec, batch_size=1, device=device, seed=args.seed + 1,
    )
    # Rebuild perception against the new env (its detect() reads env state).
    eval_perception = SyntheticOcclusionPerception(env_eval, mock_box_half=8.0)
    binding.perception = eval_perception
    binding.reset_episode(batch_size=1)
    env_eval.reset()
    frame_counter = [0]
    eval_perception.snap(0)
    ofb0 = binding.observe(0)
    frame_counter[0] = 1
    eval_state = {
        "slots_prev": ofb0.full_slot.detach(),
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
            args=args, env=env_eval, perception=eval_perception,
            binding=binding, predictor=predictor, relations=relations,
            commit=commit, existence=existence, action_embed=action_embed,
            optim=None, surprise_state=eval_state, past_buffer=eval_past_buffer,
            frame_counter=frame_counter, training=False,
            phase="eval", step_offset=step, print_each=True,
        )
        eval_records.append(rec)

    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    vis_surprise = [r["surprise_raw"] for r in eval_records if r["visible"]]
    hid_surprise = [r["surprise_raw"] for r in eval_records if not r["visible"]]
    vis_rev = [r["revision_rate"] for r in eval_records
                if r["visible"] and "revision_rate" in r]
    hid_rev = [r["revision_rate"] for r in eval_records
                if not r["visible"] and "revision_rate" in r]
    vis_gt = [r["gt_exists_mean"] for r in eval_records if r["visible"]]
    hid_gt = [r["gt_exists_mean"] for r in eval_records if not r["visible"]]
    print(json.dumps({
        "event": "summary",
        "n_visible_steps": len(vis_surprise),
        "n_hidden_steps": len(hid_surprise),
        "gt_confidence_visible": mean(vis_gt),
        "gt_confidence_hidden": mean(hid_gt),
        "gt_confidence_delta": mean(hid_gt) - mean(vis_gt),
        "surprise_visible": mean(vis_surprise),
        "surprise_hidden": mean(hid_surprise),
        "surprise_delta_hidden_minus_visible": mean(hid_surprise) - mean(vis_surprise),
        "revision_rate_visible": mean(vis_rev),
        "revision_rate_hidden": mean(hid_rev),
        "revision_rate_delta_hidden_minus_visible": mean(hid_rev) - mean(vis_rev),
        "passes_surprise_test": mean(hid_surprise) > mean(vis_surprise),
        "passes_revision_test": mean(hid_rev) > mean(vis_rev),
    }))


if __name__ == "__main__":
    main()
