"""Phase 3b: behavioural transfer via replay-buffer BC + DAGGER-lite.

Loads a *frozen* encoder from a Phase 4B `final.pt`, trains a small
policy head on top with DAGGER-style expert mixing + a replay buffer
of (state, expert_action) pairs, then evaluates policy success rate
at multiple J values.

The earlier Phase-2 BC attempt failed because vanilla BC on rollouts
that follow the expert produces a distribution-shift collapse at
inference (the policy never saw its own mistakes during training).
DAGGER-lite fixes this by mixing the policy's own actions into the
training-time rollouts, with the expert providing the *label* at
every step.

Compares slot_delta vs dense_jepa_flatten with identical policy
architecture; only the encoder + state representation differ.

Usage:

    python scripts/slot_jepa_phase3b_bc.py \
        --encoder-ckpt /workspace/phase4b_run1/seed_0/runs/seed=0_mode=slot_delta_K=5_nt=3_nd=5/final.pt \
        --encoder-mode slot_delta \
        --bc-episodes 300 \
        --eval-J 20,40,80 \
        --out /workspace/phase3b_run1/slot_delta_seed0/
"""
from __future__ import annotations
import argparse, collections, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_jepa import (
    JEPAConfig, BLAJEPAModel,
    OccludedMultiTargetNavigateEnv, OccludedNavigateSpec,
    SlotAttention, SlotAttentionConfig,
    SlotDeltaPredictor, SlotPredictorConfig,
    pool_patch_tokens,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", required=True,
                   help="Path to a Phase 4B final.pt (slot_delta or dense_jepa).")
    p.add_argument("--encoder-mode", choices=["slot_delta", "dense_jepa_flatten"],
                   required=True)

    # Env knobs (must match the env the encoder was trained on, otherwise
    # the encoder's feature distribution will be wrong).
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--n-distractors", type=int, default=5)
    p.add_argument("--visible-steps", type=int, default=5)
    p.add_argument("--hidden-steps", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=24)
    p.add_argument("--d-jepa", type=int, default=64)
    p.add_argument("--n-slots", type=int, default=16)
    p.add_argument("--moving-distractors", action="store_true", default=True)
    p.add_argument("--partial-observability", action="store_true", default=True)
    p.add_argument("--obs-radius", type=float, default=8.0)
    p.add_argument("--perceptual-noise", type=float, default=0.1)
    p.add_argument("--color-randomization", action="store_true", default=True)
    p.add_argument("--background-randomization", action="store_true", default=True)

    # BC + DAGGER knobs
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bc-episodes", type=int, default=300,
                   help="Number of DAGGER-lite rollout episodes.")
    p.add_argument("--policy-updates-per-episode", type=int, default=4)
    p.add_argument("--policy-batch", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--policy-lr", type=float, default=3e-3)
    p.add_argument("--expert-mix-init", type=float, default=1.0,
                   help="Fraction of expert actions at iter 0 (1.0 = pure BC).")
    p.add_argument("--expert-mix-final", type=float, default=0.0,
                   help="Fraction of expert actions at last iter (0.0 = full policy).")
    p.add_argument("--seed", type=int, default=0)

    # Eval
    p.add_argument("--eval-J", default="20,40,80",
                   help="Comma-separated J values to eval at.")
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--out", required=True)

    # Phase 3b attempt-2: expose the visited-target mask to the policy.
    # Without this, env.expert_action depends on env.visited (hidden
    # task state not in any encoder's observation) so the policy
    # cannot match the expert regardless of encoder quality. The flag
    # concatenates env.visited.float() ([B, n_targets]) to the state
    # vector, turning the task into a representation-transfer test
    # rather than a hidden-state-inference test.
    p.add_argument("--use-visited-mask", action="store_true",
                   help="Append env.visited mask (n_targets bits) to the "
                         "policy input. Required for a valid behavioural eval.")
    p.add_argument("--oracle-state", action="store_true",
                   help="DIAGNOSTIC: bypass the encoder entirely and feed "
                         "(agent_xy, target_xy_per_target, visited_mask) "
                         "directly to the policy. If this fails, the BC "
                         "pipeline itself is broken; if it succeeds, any "
                         "encoder failure is real.")
    p.add_argument("--include-agent-position", action="store_true",
                   help="Append normalized agent (x, y) to the policy input. "
                         "Models proprioception: realistic for any embodied "
                         "agent and isolates 'did the encoder preserve target "
                         "info' from 'can the policy decode agent position'.")
    return p.parse_args()


# ---------- encoder loading ----------

def load_encoder(args, device):
    """Returns (encode_fn, state_dim, slot_attn_or_none, slot_predictor_or_none).

    encode_fn(obs) → [B, state_dim]. For slot_delta we also need the
    slot state across timesteps, so the caller maintains it; the encode
    fn here just gives per-frame patch tokens. The 'state' the policy
    sees is built outside.
    """
    state = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    saved_args = state.get("args", {})

    cfg = JEPAConfig.tiny()
    cfg.d_jepa = saved_args.get("d_jepa", args.d_jepa)
    cfg.patch_size = saved_args.get("patch_size", args.patch_size)
    cfg.action_dim = cfg.d_jepa
    cfg.dtype = "float32"
    jepa = BLAJEPAModel(cfg).to(device)
    # Both slot and dense modes train context_encoder via the
    # self-supervised loss. The state_dict was saved by
    # slot_jepa_train.py with the key "context_encoder".
    jepa.context_encoder.load_state_dict(state["context_encoder"])
    jepa.context_encoder.eval()
    for p in jepa.context_encoder.parameters():
        p.requires_grad = False

    slot_attn = None
    slot_predictor = None
    if args.encoder_mode == "slot_delta":
        slot_cfg = SlotAttentionConfig(
            n_slots=saved_args.get("n_slots", args.n_slots),
            slot_dim=cfg.d_jepa,
            n_iters=saved_args.get("slot_iters", 3),
        )
        slot_attn = SlotAttention(input_dim=cfg.d_jepa, cfg=slot_cfg).to(device)
        slot_attn.load_state_dict(state["slot_attn"])
        slot_attn.eval()
        pred_cfg = SlotPredictorConfig(
            slot_dim=cfg.d_jepa, obs_dim=cfg.d_jepa, action_dim=cfg.d_jepa,
            n_layers=2, n_heads=4,
            delta_scale=saved_args.get("delta_scale", 0.1),
            mask_bias_init=saved_args.get("mask_bias_init", 0.0),
            update_mode=saved_args.get("update_mode", "delta"),
        )
        slot_predictor = SlotDeltaPredictor(pred_cfg).to(device)
        slot_predictor.load_state_dict(state["slot_predictor"])
        slot_predictor.eval()
        for m in (slot_attn, slot_predictor):
            for p in m.parameters():
                p.requires_grad = False
        state_dim = slot_cfg.n_slots * cfg.d_jepa
    else:  # dense_jepa_flatten
        n_patches = (args.image_size // args.patch_size) ** 2
        state_dim = n_patches * cfg.d_jepa
    return jepa.context_encoder, state_dim, slot_attn, slot_predictor, cfg.d_jepa


def build_env(args, device, hidden_steps_override=None):
    spec = OccludedNavigateSpec(
        image_size=args.image_size, patch_size=args.patch_size,
        n_targets=args.n_targets, n_distractors=args.n_distractors,
        visible_steps=args.visible_steps,
        hidden_steps=hidden_steps_override or args.hidden_steps,
        moving_distractors=args.moving_distractors,
        partial_observability=args.partial_observability,
        obs_radius=args.obs_radius,
        perceptual_noise=args.perceptual_noise,
        color_randomization=args.color_randomization,
        background_randomization=args.background_randomization,
        max_steps=args.episode_length,
        action_dim=args.d_jepa,
    )
    return OccludedMultiTargetNavigateEnv(spec, batch_size=args.batch_size,
                                            device=device, seed=args.seed)


def make_policy(state_dim: int, action_dim: int = 2,
                hidden: int = 512) -> nn.Module:
    """3-layer MLP. Hidden defaults to 512 because the slot/patch state
    is ~1024-d and a 128-wide bottleneck at the input projection was the
    Phase-3b-run3 limiter (BC loss plateaued ≫ zero)."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, action_dim),
    )


@torch.no_grad()
def encode_state(args, ctx_enc, slot_attn, slot_predictor,
                  obs, slots, prev_action_vec, device, visited=None,
                  agent_xy=None):
    """Returns (state_vec [B, state_dim], updated_slots_or_None).

    PatchViTEncoder returns (tokens, grid_h, grid_w); the policy only
    needs `tokens`. If `args.use_visited_mask` and `visited` is given,
    the visited mask is appended to the state vector so the policy can
    compute "which target is next".
    """
    tokens, _, _ = ctx_enc(obs)  # [B, N, D]
    if args.encoder_mode == "slot_delta":
        if slots is None:
            slots = slot_attn(tokens)
        else:
            pred = slot_predictor(slots, tokens, prev_action_vec)
            slots = pred["next_slots"]
        state = slots.reshape(slots.size(0), -1)
        out_slots = slots
    else:
        state = tokens.reshape(tokens.size(0), -1)
        out_slots = None
    if args.use_visited_mask and visited is not None:
        state = torch.cat([state, visited.float().to(state.device)], dim=-1)
    if args.include_agent_position and agent_xy is not None:
        state = torch.cat([state, agent_xy.to(state.device)], dim=-1)
    return state, out_slots


def oracle_state(env, image_size: float) -> torch.Tensor:
    """Diagnostic state: ground-truth (agent_xy, target_xy_flat, visited_mask),
    normalized to [-1, 1] for agent_xy and target_xy."""
    b = env.batch_size
    s = image_size
    agent = torch.stack([env.x, env.y], dim=-1) / (s / 2) - 1.0       # [B, 2]
    tgt = torch.stack([env.tx, env.ty], dim=-1) / (s / 2) - 1.0       # [B, n_t, 2]
    visited = env.visited.float()                                       # [B, n_t]
    return torch.cat([agent, tgt.reshape(b, -1), visited], dim=-1)


def oracle_state_dim(args) -> int:
    return 2 + 2 * args.n_targets + args.n_targets


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.oracle_state:
        # Diagnostic mode: skip encoder loading entirely.
        ctx_enc = slot_attn = slot_predictor = None
        d_jepa = args.d_jepa
        state_dim = oracle_state_dim(args)
        print(json.dumps({"event": "oracle_mode",
                           "state_dim": state_dim}), flush=True)
    else:
        ctx_enc, state_dim, slot_attn, slot_predictor, d_jepa = load_encoder(args, device)
        if args.use_visited_mask:
            # Policy input now also carries the visited mask (n_targets bits).
            state_dim = state_dim + args.n_targets
        if args.include_agent_position:
            state_dim = state_dim + 2
    policy = make_policy(state_dim).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr,
                                weight_decay=1e-4)
    buffer = collections.deque(maxlen=args.buffer_size)

    print(json.dumps({"event": "init", "encoder_mode": args.encoder_mode,
                       "state_dim": state_dim, "ckpt": args.encoder_ckpt}),
          flush=True)

    env = build_env(args, device)
    t0 = time.time()

    for ep_idx in range(args.bc_episodes):
        obs = env.reset()
        slots = None
        prev_action_vec = torch.zeros(args.batch_size, d_jepa, device=device)
        # Linear schedule from expert_mix_init → expert_mix_final.
        frac = ep_idx / max(args.bc_episodes - 1, 1)
        expert_mix = args.expert_mix_init + frac * (args.expert_mix_final - args.expert_mix_init)

        for t in range(args.episode_length):
            with torch.no_grad():
                if args.oracle_state:
                    state = oracle_state(env, args.image_size).to(device)
                else:
                    agent_xy = (torch.stack([env.x, env.y], dim=-1)
                                 / (args.image_size / 2) - 1.0)
                    state, slots = encode_state(
                        args, ctx_enc, slot_attn, slot_predictor,
                        obs, slots, prev_action_vec, device,
                        visited=env.visited, agent_xy=agent_xy,
                    )
                policy_act = policy(state).clamp(-2.0, 2.0)
                expert_act = env.expert_action()
            # Buffer: always (state, expert) — DAGGER core idea.
            for b in range(args.batch_size):
                buffer.append((state[b].cpu(), expert_act[b].cpu()))
            # Decide which action to execute this step.
            use_expert = torch.rand(args.batch_size, device=device) < expert_mix
            action = torch.where(use_expert.unsqueeze(-1), expert_act, policy_act)
            # Pad action for predictor.
            if d_jepa > 2:
                pad = torch.zeros(args.batch_size, d_jepa - 2, device=device)
                prev_action_vec = torch.cat([action, pad], dim=-1)
            obs, _, done = env.step(action)
            if bool(done.all().item()):
                break

        # Train policy on the buffer.
        if len(buffer) >= args.policy_batch:
            for _ in range(args.policy_updates_per_episode):
                batch = random.sample(buffer, args.policy_batch)
                states = torch.stack([b[0] for b in batch]).to(device)
                experts = torch.stack([b[1] for b in batch]).to(device)
                preds = policy(states)
                loss = F.mse_loss(preds, experts)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()

        if (ep_idx + 1) % 20 == 0:
            print(json.dumps({
                "event": "bc", "ep": ep_idx + 1,
                "expert_mix": round(expert_mix, 3),
                "buffer_size": len(buffer),
                "loss": round(float(loss), 4) if len(buffer) >= args.policy_batch else None,
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)

    # ---- Eval policy at multiple J values ----
    eval_J = [int(j) for j in args.eval_J.split(",")]
    policy.eval()
    results = []
    for J in eval_J:
        env_eval = build_env(args, device, hidden_steps_override=J)
        env_eval.gen.manual_seed(args.seed + 9999 + J)
        n_done = 0
        n_success = 0
        while n_done < args.eval_episodes:
            obs = env_eval.reset()
            slots = None
            prev_action_vec = torch.zeros(args.batch_size, d_jepa, device=device)
            success = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
            for t in range(args.episode_length):
                with torch.no_grad():
                    if args.oracle_state:
                        state = oracle_state(env_eval, args.image_size).to(device)
                    else:
                        agent_xy = (torch.stack([env_eval.x, env_eval.y], dim=-1)
                                     / (args.image_size / 2) - 1.0)
                        state, slots = encode_state(
                            args, ctx_enc, slot_attn, slot_predictor,
                            obs, slots, prev_action_vec, device,
                            visited=env_eval.visited, agent_xy=agent_xy,
                        )
                    action = policy(state).clamp(-2.0, 2.0)
                if d_jepa > 2:
                    pad = torch.zeros(args.batch_size, d_jepa - 2, device=device)
                    prev_action_vec = torch.cat([action, pad], dim=-1)
                obs, _, done = env_eval.step(action)
                success = success | env_eval.success_mask()
                if bool(done.all().item()):
                    break
            n_done += args.batch_size
            n_success += int(success.sum().item())
        rate = n_success / max(n_done, 1)
        out = {"J": J, "success_rate": rate, "n_episodes": n_done,
                "n_success": n_success}
        results.append(out)
        print(json.dumps({"event": "eval", **out}), flush=True)

    summary = {
        "encoder_mode": args.encoder_mode,
        "encoder_ckpt": args.encoder_ckpt,
        "results": results,
        "args": vars(args),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "bc_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"event": "done", **{k: summary[k] for k in
                                              ("encoder_mode", "elapsed_s")},
                       "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
