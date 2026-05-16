"""Slot-JEPA prototype trainer on the occluded multi-target navigate env.

Three modes, sharing all other knobs:
  --mode slot_delta   slot attention + SlotDeltaPredictor (the proposal)
  --mode dense        baseline: pool encoder tokens → predict next pooled embedding
  --mode copy         trivial: slots/embeds don't change; measures the
                       persistence prior alone

Loss in all modes is a JEPA-style next-step prediction against an EMA
target encoder so we test representation quality, not pixel reconstruction.

What's logged each step:
  prediction loss
  (slot mode only) change_mask mean + max
  (slot mode only) slot cosine stability across hidden→visible boundaries
  (slot mode only) identity diagnostic: stable slot↔target match through
                   the hidden window (no training signal — eval only)
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import (
    JEPAConfig,
    BLAJEPAModel,
    OccludedMultiTargetNavigateEnv,
    OccludedNavigateSpec,
    SlotAttention,
    SlotAttentionConfig,
    SlotDeltaPredictor,
    SlotPredictorConfig,
    pool_patch_tokens,
    slot_delta_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["slot_delta", "dense", "dense_jepa", "copy"],
                   default="slot_delta",
                   help="slot_delta: 16-slot + delta predictor (the proposal). "
                         "dense_jepa: full patch-level JEPA with masked prediction "
                         "(BLAJEPAModel) — the fair dense baseline. "
                         "dense: naive pooled next-step (legacy, kept only for "
                         "the original sanity comparison). "
                         "copy: encoder same-frame consistency, no predictor.")
    p.add_argument("--probe-pool", choices=["flatten", "mean"], default="flatten",
                   help="How to pool patch/slot tokens into a state vector for "
                         "the linear probe. Default 'flatten' preserves position-"
                         "ish information; 'mean' is the standard pooling that "
                         "tends to erase object locations.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--episode-length", type=int, default=24)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--visible-steps", type=int, default=5)
    p.add_argument("--hidden-steps", type=int, default=10)
    p.add_argument("--n-distractors", type=int, default=2)
    p.add_argument("--moving-distractors", action="store_true",
                   help="Distractors random-walk each env step. Phase-3 stress flag.")
    p.add_argument("--distractor-move-max", type=float, default=1.0)
    p.add_argument("--partial-observability", action="store_true",
                   help="Mask observation outside a circle around the agent. "
                         "Phase-3 stress flag.")
    p.add_argument("--obs-radius", type=float, default=8.0)
    p.add_argument("--perceptual-noise", type=float, default=0.0,
                   help="Gaussian pixel noise σ. Phase-4A flag: turns the "
                         "rendered-patch observation into a noisier perception "
                         "channel. Default 0.0 (Phase-3 behaviour).")
    p.add_argument("--color-randomization", action="store_true",
                   help="Phase-4B: sample random RGB per entity at each reset; "
                         "breaks the channel-as-label shortcut.")
    p.add_argument("--background-randomization", action="store_true",
                   help="Phase-4B: low-magnitude random per-pixel background "
                         "canvas sampled at each reset.")
    p.add_argument("--d-jepa", type=int, default=64)
    p.add_argument("--n-slots", type=int, default=16)
    p.add_argument("--slot-iters", type=int, default=3)
    p.add_argument("--delta-scale", type=float, default=0.1)
    p.add_argument("--sparsity-weight", type=float, default=1e-3)
    p.add_argument("--bimodal-weight", type=float, default=0.0)
    p.add_argument("--mask-bias-init", type=float, default=-1.0)
    p.add_argument("--update-mode", choices=["delta", "dense", "dynamic"],
                   default="delta",
                   help="Slot update mechanism. 'delta' = sparse + bounded "
                         "(Phase 2/3/4 default). 'dense' = predictor outputs "
                         "full next_slots directly (Phase 2 ablation). "
                         "'dynamic' = top-K active gate over a larger slot "
                         "pool; only K slots receive delta updates each step.")
    p.add_argument("--target-active-slots", type=int, default=0,
                   help="Phase 5B: in --update-mode dynamic, this many slots "
                         "of the n_slots pool are active each step. Inactive "
                         "slots are frozen.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema-tau", type=float, default=0.996)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--probe-episodes", type=int, default=0,
                   help="If > 0, after self-supervised training, run "
                         "linear-probe eval at each J in --eval-J-values. "
                         "Trains a linear regression state -> target_xy on "
                         "visible frames only, reports MSE on hidden frames.")
    p.add_argument("--probe-epochs", type=int, default=200,
                   help="Number of full passes over the visible-frame buffer "
                         "when fitting the linear probe.")
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--eval-J-values", type=str, default="",
                   help="Comma-separated list of hidden_steps values for eval (e.g. '5,10,20,40')")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_env(args, device):
    spec = OccludedNavigateSpec(
        image_size=args.image_size,
        patch_size=args.patch_size,
        n_targets=args.n_targets,
        visible_steps=args.visible_steps,
        hidden_steps=args.hidden_steps,
        n_distractors=args.n_distractors,
        moving_distractors=args.moving_distractors,
        distractor_move_max=args.distractor_move_max,
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


def build_modules(args, device):
    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.d_jepa = args.d_jepa
    jepa_cfg.patch_size = args.patch_size
    jepa_cfg.action_dim = args.d_jepa
    jepa_cfg.dtype = "float32"
    # Build a JEPAModel just to recycle its encoder + EMA-target wiring.
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    context_encoder = jepa.context_encoder
    target_encoder = jepa.target_encoder

    slot_attn = None
    slot_predictor = None
    if args.mode == "slot_delta":
        slot_cfg = SlotAttentionConfig(
            n_slots=args.n_slots, slot_dim=args.d_jepa, n_iters=args.slot_iters,
        )
        slot_attn = SlotAttention(input_dim=args.d_jepa, cfg=slot_cfg).to(device)
        pred_cfg = SlotPredictorConfig(
            slot_dim=args.d_jepa, obs_dim=args.d_jepa, action_dim=args.d_jepa,
            n_layers=2, n_heads=4, delta_scale=args.delta_scale,
            mask_bias_init=args.mask_bias_init,
            update_mode=args.update_mode,
            target_active_slots=args.target_active_slots,
        )
        slot_predictor = SlotDeltaPredictor(pred_cfg).to(device)

    return jepa, context_encoder, target_encoder, slot_attn, slot_predictor


def trainable_params(mode, jepa, slot_attn, slot_predictor):
    """Phase 1 trainables. Note: dense_jepa needs the predictor + context
    encoder; the EMA target encoder is never in the optimizer."""
    params = list(jepa.context_encoder.parameters())
    if mode == "slot_delta":
        params += list(slot_attn.parameters()) + list(slot_predictor.parameters())
    elif mode == "dense_jepa":
        params += list(jepa.predictor.parameters())
    return params


def build_policy_head(in_dim: int, action_dim: int, hidden: int = 128) -> nn.Module:
    """BC head: state → 2D (dx, dy). Same architecture across all modes so
    the comparison is fair on the policy side; only the input dim — which
    is the state representation — differs."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, action_dim),
    )


def slots_to_state(slots: torch.Tensor) -> torch.Tensor:
    """Flatten the slot tensor [B, S, D] → [B, S·D]. Mean-pooling would
    destroy per-slot information (the whole point of having slots) so we
    feed the full concatenated slot vector to the policy head and let it
    learn which slot encodes which entity."""
    return slots.reshape(slots.size(0), -1)


def policy_input_dim(args) -> int:
    """Dimension of the state vector that feeds the policy head, which
    differs by mode: flat slots vs pooled embedding."""
    if args.mode == "slot_delta":
        return args.n_slots * args.d_jepa
    return args.d_jepa


def make_state(args, tokens: torch.Tensor, slots: torch.Tensor | None,
                pooled: torch.Tensor) -> torch.Tensor:
    """Build the state vector that the linear probe consumes.

    slot_delta: flatten 16 slots ([B, S·D]).
    dense_jepa: flatten or mean-pool patch tokens depending on --probe-pool.
    dense / copy: pooled embedding (legacy, kept for the original baseline).
    """
    if args.mode == "slot_delta":
        # Always flatten — mean-pooling slots would erase the whole point.
        return slots.reshape(slots.size(0), -1)
    if args.mode == "dense_jepa":
        if args.probe_pool == "flatten":
            return tokens.reshape(tokens.size(0), -1)
        return tokens.mean(dim=1)
    return pooled


def _collect_probe_rollouts(args, j_value, mode, ctx_enc, slot_attn,
                              slot_predictor, device, n_episodes):
    """Roll out n_episodes with EXPERT actions; record per-frame
    (state, target_xy_concat, is_visible, hidden_step_idx). Returns
    tensors stacked across all frames.

    Using expert actions avoids the BC distribution-shift problem — the
    point of this experiment is representational, not behavioural."""
    spec = OccludedNavigateSpec(
        image_size=args.image_size, patch_size=args.patch_size,
        n_targets=args.n_targets, visible_steps=args.visible_steps,
        hidden_steps=j_value, n_distractors=args.n_distractors,
        moving_distractors=args.moving_distractors,
        distractor_move_max=args.distractor_move_max,
        partial_observability=args.partial_observability,
        obs_radius=args.obs_radius,
        perceptual_noise=args.perceptual_noise,
        color_randomization=args.color_randomization,
        background_randomization=args.background_randomization,
        max_steps=args.episode_length, action_dim=args.d_jepa,
    )
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=args.batch_size,
                                          device=device,
                                          seed=args.seed + 17 + j_value)
    states, target_xys, visibles, hidden_steps = [], [], [], []
    episode_ids = []  # which episode each sample came from (held-out probe split)
    n_done = 0
    while n_done < n_episodes:
        obs = env.reset()
        if mode == "slot_delta":
            with torch.no_grad():
                tokens0, _ = encode_frame(ctx_enc, obs)
                slots = slot_attn(tokens0)
        else:
            slots = None
        # Track time since the most recent visible→hidden transition.
        hidden_since = torch.full((args.batch_size,), -1, dtype=torch.long,
                                     device=device)
        for t in range(args.episode_length):
            with torch.no_grad():
                tokens, pooled = encode_frame(ctx_enc, obs)
                state = make_state(args, tokens, slots, pooled)
            is_visible_now = env.visibility_mask()
            # update hidden_since: 0 on the first hidden frame, +1 each step
            # in the same hidden window, reset to -1 when visible.
            hidden_since = torch.where(
                is_visible_now,
                torch.full_like(hidden_since, -1),
                torch.where(hidden_since < 0,
                              torch.zeros_like(hidden_since),
                              hidden_since + 1),
            )
            # target_xy concatenated as [B, 2 * n_targets]
            tgt_xy = torch.stack([env.tx, env.ty], dim=-1)  # [B, n_t, 2]
            tgt_xy_flat = tgt_xy.reshape(args.batch_size, -1)

            states.append(state.detach().cpu())
            target_xys.append(tgt_xy_flat.detach().cpu())
            visibles.append(is_visible_now.detach().cpu())
            hidden_steps.append(hidden_since.detach().cpu())
            # Tag each batch-element with a unique episode index so we can
            # later split episodes into train/test for the probe.
            ep_ids = torch.arange(
                n_done, n_done + args.batch_size, device=device, dtype=torch.long,
            )
            episode_ids.append(ep_ids.cpu())

            action_xy = env.expert_action()
            obs, _, done = env.step(action_xy)

            if mode == "slot_delta":
                with torch.no_grad():
                    tokens_next, _ = encode_frame(ctx_enc, obs)
                pad = torch.zeros(args.batch_size, args.d_jepa - 2, device=device)
                act_vec = torch.cat([action_xy, pad], dim=-1)
                with torch.no_grad():
                    out = slot_predictor(slots, tokens_next, act_vec)
                    slots = out["next_slots"]

            if bool(done.all().item()):
                break
        n_done += args.batch_size

    return (
        torch.cat(states, dim=0),
        torch.cat(target_xys, dim=0),
        torch.cat(visibles, dim=0),
        torch.cat(hidden_steps, dim=0),
        torch.cat(episode_ids, dim=0),
    )


def _hungarian_mse(pred_xy: torch.Tensor, true_xy: torch.Tensor) -> float:
    """Permutation-invariant MSE: for each example, match predicted points
    to ground-truth points via min-cost assignment (Hungarian algorithm),
    then compute the mean of matched squared distances.

    pred_xy, true_xy: [B, n_targets, 2]. Returns mean over batch and targets
    of the matched MSE. Robust to slot-permutation: a probe that outputs
    the right *set* of positions scores zero regardless of slot ordering.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback: greedy matching by closest distance. O(n²) — fine for
        # n_targets ≤ 32 used here.
        linear_sum_assignment = None
    B, n_t, _ = pred_xy.shape
    total = 0.0
    pred_np = pred_xy.detach().cpu().numpy()
    true_np = true_xy.detach().cpu().numpy()
    for b in range(B):
        diff = pred_np[b][:, None, :] - true_np[b][None, :, :]
        cost = (diff ** 2).sum(axis=-1)  # [n_t, n_t]
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
        else:
            # Greedy fallback.
            rows = list(range(n_t)); cols = []
            used = set()
            for r in rows:
                ordered = sorted(range(n_t), key=lambda c: cost[r, c])
                for c in ordered:
                    if c not in used:
                        cols.append(c); used.add(c); break
        matched = float(cost[rows, cols].sum() / n_t)
        total += matched
    return total / B


def _train_linear_probe(states_train, targets_train, lr, epochs,
                          weight_decay: float = 1e-3):
    """Fit a single nn.Linear from state → target_xy on visible frames.

    Weight decay is on by default — defends against probe memorization
    when the state has large constant-per-episode components (Phase
    5B-attempt-1 lesson: zero-weight-decay + held-out probe set is
    what we need; the train-test split also goes on the *episode* axis
    in the caller)."""
    in_dim = states_train.size(-1)
    out_dim = targets_train.size(-1)
    probe = nn.Linear(in_dim, out_dim)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    bs = min(256, states_train.size(0))
    for _ in range(epochs):
        idx = torch.randperm(states_train.size(0))[:bs]
        x = states_train[idx]
        y = targets_train[idx]
        pred = probe(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probe


def linear_probe_eval(args, j_values, mode, ctx_enc, slot_attn,
                       slot_predictor, device):
    """For each J: collect rollouts, fit probe on visible frames, eval on
    hidden frames. Returns a dict with per-J metrics + per-hidden-step
    breakdown."""
    results = []
    for j in j_values:
        states, targets, visibles, hidden_step, ep_ids = _collect_probe_rollouts(
            args, j, mode, ctx_enc, slot_attn, slot_predictor,
            device, args.probe_episodes,
        )
        # Held-out probe split: train probe on visible frames from a
        # subset of episodes; test on hidden frames from the OTHER
        # episodes. Without this split, an episode-specific-constant
        # state (Phase 5B-attempt-1 issue) lets the probe memorize
        # (state-signature → targets) and score 0 MSE everywhere.
        unique_eps = ep_ids.unique()
        n_eps = unique_eps.numel()
        train_eps = unique_eps[: int(n_eps * 0.8)]
        train_mask = torch.isin(ep_ids, train_eps)
        test_mask = ~train_mask

        v_train = visibles & train_mask
        h_test = (~visibles) & test_mask
        v_test = visibles & test_mask
        n_vis = int(v_train.sum().item())
        n_hid = int(h_test.sum().item())
        if n_vis < 10 or n_hid < 10:
            results.append({"J": j, "n_visible": n_vis, "n_hidden": n_hid,
                              "visible_mse": None, "hidden_mse": None})
            continue
        probe = _train_linear_probe(states[v_train], targets[v_train],
                                       args.probe_lr, args.probe_epochs)
        with torch.no_grad():
            v_mse = float(F.mse_loss(probe(states[v_test]), targets[v_test])) \
                    if v_test.any() else float("nan")
            h_mse = float(F.mse_loss(probe(states[h_test]), targets[h_test]))
        # Permutation-invariant Hungarian-matched MSE on test split.
        n_targets = targets.size(-1) // 2
        if n_targets > 1:
            with torch.no_grad():
                if v_test.any():
                    v_pred = probe(states[v_test]).reshape(-1, n_targets, 2)
                    v_true = targets[v_test].reshape(-1, n_targets, 2)
                    v_hung = _hungarian_mse(v_pred, v_true)
                else:
                    v_hung = None
                h_pred = probe(states[h_test]).reshape(-1, n_targets, 2)
                h_true = targets[h_test].reshape(-1, n_targets, 2)
                h_hung = _hungarian_mse(h_pred, h_true)
        else:
            v_hung = v_mse
            h_hung = h_mse
        # Per-hidden-step breakdown on TEST episodes only.
        per_step = []
        for h in range(j):
            mask = (hidden_step == h) & test_mask
            n_h = int(mask.sum().item())
            if n_h < 5:
                per_step.append(None)
                continue
            with torch.no_grad():
                step_mse = float(F.mse_loss(probe(states[mask]), targets[mask]))
            per_step.append(step_mse)
        # Degradation slope: (MSE at last hidden step − MSE at first hidden step)
        first_vals = [v for v in per_step[:max(1, j // 4)] if v is not None]
        last_vals = [v for v in per_step[-max(1, j // 4):] if v is not None]
        slope = None
        if first_vals and last_vals:
            slope = (sum(last_vals) / len(last_vals)
                      - sum(first_vals) / len(first_vals))
        results.append({
            "J": j, "n_visible": n_vis, "n_hidden": n_hid,
            "visible_mse": round(v_mse, 5),
            "hidden_mse": round(h_mse, 5),
            "visible_mse_hungarian": round(v_hung, 5) if v_hung is not None else None,
            "hidden_mse_hungarian": round(h_hung, 5),
            "hidden_visible_ratio": round(h_mse / max(v_mse, 1e-9), 3),
            "per_hidden_step_mse": [round(v, 5) if v is not None else None for v in per_step],
            "degradation_slope": round(slope, 5) if slope is not None else None,
        })
    return results


@torch.no_grad()
def eval_success_rate(
    args, env_spec_overrides, mode, ctx_enc, slot_attn, slot_predictor,
    policy_head, device, n_episodes: int = 50,
) -> dict:
    """Roll out `n_episodes` with the learned policy and measure success.

    env_spec_overrides override fields on OccludedNavigateSpec for the
    eval (e.g. different hidden_steps). The policy is purely from the
    learned head; the env reward signal is ignored — we only check
    whether the agent visited all targets within max_steps."""
    spec = OccludedNavigateSpec(
        image_size=args.image_size, patch_size=args.patch_size,
        n_targets=args.n_targets, visible_steps=args.visible_steps,
        hidden_steps=env_spec_overrides.get("hidden_steps", args.hidden_steps),
        n_distractors=args.n_distractors, max_steps=args.episode_length,
        action_dim=args.d_jepa,
    )
    # Use a separate seed so eval doesn't overlap the train rollouts.
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=args.batch_size,
                                          device=device,
                                          seed=args.seed + 12345)
    n_done = 0
    n_success = 0
    while n_done < n_episodes:
        obs = env.reset()
        # Initial slot binding from obs_0 (no predictor call yet).
        if mode == "slot_delta":
            tokens, _ = encode_frame(ctx_enc, obs)
            slots = slot_attn(tokens)
        else:
            slots = None
        success_at_done = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        for _ in range(args.episode_length):
            if mode == "slot_delta":
                state = slots_to_state(slots)
            else:
                _, state = encode_frame(ctx_enc, obs)
            policy_out = policy_head(state)
            action_xy = policy_out.clamp(-spec.move_max, spec.move_max)
            obs, _, done_step = env.step(action_xy)
            # After env.step, obs is the post-action observation. Update
            # slots with (slots_t, action_t, tokens_{t+1}) per training
            # convention.
            if mode == "slot_delta":
                tokens_next, _ = encode_frame(ctx_enc, obs)
                pad = torch.zeros(
                    args.batch_size, args.d_jepa - 2, device=device
                )
                act_vec = torch.cat([action_xy, pad], dim=-1)
                out = slot_predictor(slots, tokens_next, act_vec)
                slots = out["next_slots"]
            success_at_done = success_at_done | env.success_mask()
            if bool(done_step.all().item()):
                break
        n_done += args.batch_size
        n_success += int(success_at_done.sum().item())
    return {"success_rate": n_success / max(n_done, 1),
            "n_episodes": n_done,
            "n_success": n_success,
            "hidden_steps": spec.hidden_steps}


@torch.no_grad()
def update_ema(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(tau).add_(sp.detach().to(dtype=tp.dtype), alpha=1.0 - tau)


def encode_frame(encoder, img):
    """Returns (tokens [B, N, D], pooled [B, D]). Encoder operates in float32."""
    tokens, _, _ = encoder(img)
    return tokens, pool_patch_tokens(tokens)


def step_slot_delta(slots, prev_action_vec, context_tokens_t, target_tokens_t,
                    slot_attn, slot_predictor, sparsity_weight, bimodal_weight):
    """One self-supervised step for the slot-delta mode.

    slots:               [B, S, D]   slots from the previous timestep
    prev_action_vec:     [B, D]      action taken from t-1 → t
    context_tokens_t:    [B, N, D]   current frame, online encoder
    target_tokens_t:     [B, N, D]   current frame, EMA target encoder

    The model predicts the slots that *the EMA target slot-binder would
    have produced for the current frame* given (slots_{t-1}, action_{t-1},
    online_tokens_t). The detached target slot binding is the supervision
    signal.
    """
    out = slot_predictor(slots, context_tokens_t, prev_action_vec)
    with torch.no_grad():
        target_slots = slot_attn(target_tokens_t, init_slots=slots.detach())
    metrics = slot_delta_loss(out["next_slots"], target_slots, out["change_mask"],
                                sparsity_weight=sparsity_weight,
                                bimodal_weight=bimodal_weight)
    return out["next_slots"], metrics


def step_dense(prev_pooled, prev_action_vec, target_pooled_t, predictor_dense):
    """One self-supervised step for the dense baseline: predict next pooled
    embedding from the previous one + action."""
    inp = torch.cat([prev_pooled, prev_action_vec], dim=-1)
    pred = predictor_dense(inp)
    target = target_pooled_t.detach()
    loss = F.smooth_l1_loss(pred, target)
    return pred, {"loss": loss, "prediction": loss.detach()}


def step_copy(pooled_ctx_t, pooled_tgt_t):
    """Copy baseline: there is no learned predictor. The encoder still
    trains via a same-frame JEPA consistency loss (online ↔ EMA target),
    so the encoder learns a representation; only the *prediction* of
    'next frame given prev frame' is replaced with the persistence prior.

    This gives a fair representation-quality control against slot_delta:
    same encoder training signal, no predictor module."""
    target = pooled_tgt_t.detach()
    loss = F.smooth_l1_loss(pooled_ctx_t, target)
    return pooled_ctx_t, {"loss": loss, "prediction": loss.detach()}


def identity_diagnostic(slots, env):
    """Return: for each target in the env, which slot is currently closest
    to that target in slot-embedding space. Used to track whether the
    same slot stays bound to the same target across an occlusion window.

    slots: [B, S, D]
    env:   the env, with .tx, .ty target positions [B, n_targets]
    returns slot_idx_for_each_target: [B, n_targets] long
    """
    b, s, d = slots.shape
    n_t = env.spec.n_targets
    # Encode each target as a (very small) sinusoidal position embed in
    # the same D as slots — that gives us a deterministic, training-free
    # target → slot-space mapping for the diagnostic.
    target_emb = torch.zeros(b, n_t, d, device=slots.device, dtype=slots.dtype)
    pos = torch.stack([env.tx, env.ty], dim=-1)             # [B, n_t, 2]
    target_emb[..., 0] = pos[..., 0] / env.spec.image_size  # x in [0,1]
    target_emb[..., 1] = pos[..., 1] / env.spec.image_size  # y in [0,1]
    # Compute per-batch cosine between each target embed and every slot,
    # take the argmax slot index per target.
    sim = F.cosine_similarity(
        target_emb.unsqueeze(2), slots.unsqueeze(1), dim=-1
    )  # [B, n_t, S]
    return sim.argmax(dim=-1)                                 # [B, n_t]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(json.dumps({"event": "init", "mode": args.mode, "device": str(device),
                       "config": vars(args)}), flush=True)

    env = build_env(args, device)
    jepa, ctx_enc, tgt_enc, slot_attn, slot_predictor = build_modules(args, device)

    predictor_dense = None
    if args.mode == "dense":
        predictor_dense = nn.Sequential(
            nn.Linear(args.d_jepa * 2, args.d_jepa * 4),
            nn.GELU(),
            nn.Linear(args.d_jepa * 4, args.d_jepa),
        ).to(device)

    # Build phase-1 optimizer *before* the policy head exists, so any
    # RNG consumed by policy-head init doesn't shift the slot module's
    # parameter initialization (we observed this shift dropping the model
    # into a degenerate mask=0 basin).
    params = trainable_params(args.mode, jepa, slot_attn, slot_predictor)
    if args.mode == "dense":
        params += list(predictor_dense.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Shared policy head architecture across all modes — only the input
    # state dim differs. Action is 2D (dx, dy). Build under a saved RNG
    # state so the slot/predictor modules' inits are *byte-identical* to
    # a run without a policy head (observed: a single extra RNG draw was
    # enough to flip the slot trainer into a degenerate mask=0 basin).
    rng_state = torch.get_rng_state()
    policy_head = build_policy_head(policy_input_dim(args), action_dim=2).to(device)
    torch.set_rng_state(rng_state)

    t0 = time.time()
    log_buf = {"prediction": 0.0, "mask_mean": 0.0, "mask_max": 0.0,
                "stability_cos": 0.0, "id_diag_stable": 0.0,
                "n_log": 0, "n_stab": 0}

    # Rolling per-step state (slot / pooled embedding) carried across the
    # episode. Initialised on each episode reset.
    obs = env.reset()
    slots_state = None
    prev_pooled = None
    prev_action_vec = torch.zeros(args.batch_size, args.d_jepa, device=device)
    # For the stability + identity diagnostics: snapshots at the last
    # visible frame and at the first visible frame after a hidden window.
    last_visible_slots = None
    last_visible_target_id = None

    for step in range(1, args.steps + 1):
        action_xy = env.expert_action()
        action_vec = env.encode_action(action_xy)
        # Pad encoded action to match d_jepa (encode_action only fills first two slots).
        if action_vec.shape[-1] < args.d_jepa:
            pad = torch.zeros(
                args.batch_size, args.d_jepa - action_vec.shape[-1], device=device
            )
            action_vec = torch.cat([action_vec, pad], dim=-1)

        # Encode this frame through both encoders.
        tokens_ctx, pooled_ctx = encode_frame(ctx_enc, obs)
        with torch.no_grad():
            tokens_tgt, pooled_tgt = encode_frame(tgt_enc, obs)

        if args.mode == "slot_delta":
            if slots_state is None:
                # Initialize slots at episode start by binding context tokens.
                slots_state = slot_attn(tokens_ctx)
                step_metrics = None
            else:
                slots_state, step_metrics = step_slot_delta(
                    slots_state, prev_action_vec, tokens_ctx, tokens_tgt,
                    slot_attn, slot_predictor,
                    args.sparsity_weight, args.bimodal_weight,
                )
        elif args.mode == "dense_jepa":
            # Fair dense baseline: standard patch-level masked JEPA. The
            # JEPAModel handles both encoders (context masked, target full)
            # and the SIGReg objective.
            step_metrics = jepa.training_loss(obs, action_vec)
        elif args.mode == "dense":
            if prev_pooled is None:
                step_metrics = None
            else:
                _, step_metrics = step_dense(
                    prev_pooled, prev_action_vec, pooled_tgt, predictor_dense
                )
            prev_pooled = pooled_ctx
        else:  # copy
            # Train encoder via same-frame consistency; no predictor.
            _, step_metrics = step_copy(pooled_ctx, pooled_tgt)
            prev_pooled = pooled_ctx


        # Diagnostics for slot mode — captured at visibility-window boundaries.
        if args.mode == "slot_delta" and slots_state is not None:
            visible_now = env.visibility_mask()[0].item()
            if visible_now:
                if last_visible_slots is not None:
                    # Cosine stability between the slot state at the previous
                    # visible frame and the current visible frame, after a
                    # hidden window. Computed per-slot and averaged.
                    cos = F.cosine_similarity(
                        slots_state.detach(), last_visible_slots, dim=-1
                    ).mean().item()
                    log_buf["stability_cos"] += cos
                    new_id = identity_diagnostic(slots_state.detach(), env)
                    same = (new_id == last_visible_target_id).float().mean().item()
                    log_buf["id_diag_stable"] += same
                    log_buf["n_stab"] += 1
                last_visible_slots = slots_state.detach().clone()
                last_visible_target_id = identity_diagnostic(slots_state.detach(), env)

        if step_metrics is not None:
            optim.zero_grad(set_to_none=True)
            step_metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            update_ema(tgt_enc, ctx_enc, args.ema_tau)

            log_buf["prediction"] += float(step_metrics["prediction"])
            if "mask_mean" in step_metrics:
                log_buf["mask_mean"] += float(step_metrics["mask_mean"])
                log_buf["mask_max"] += float(step_metrics["mask_max"])
            log_buf["n_log"] += 1

        # Detach state carried into the next step so we don't backprop
        # through previous timesteps' graphs. Training is single-step
        # at this stage; multi-step BPTT is a later upgrade.
        if slots_state is not None:
            slots_state = slots_state.detach()
        if prev_pooled is not None:
            prev_pooled = prev_pooled.detach()

        # Take env step.
        obs, _, done = env.step(action_xy)
        prev_action_vec = action_vec

        if bool(done.all().item()):
            obs = env.reset()
            slots_state = None
            prev_pooled = None
            prev_action_vec = torch.zeros(args.batch_size, args.d_jepa, device=device)
            last_visible_slots = None
            last_visible_target_id = None

        if step % args.log_every == 0 and log_buf["n_log"] > 0:
            n = log_buf["n_log"]
            n_stab = max(1, log_buf["n_stab"])
            payload = {
                "step": step,
                "elapsed_s": round(time.time() - t0, 1),
                "prediction": round(log_buf["prediction"] / n, 5),
            }
            if args.mode == "slot_delta":
                payload["mask_mean"] = round(log_buf["mask_mean"] / n, 4)
                payload["mask_max"] = round(log_buf["mask_max"] / n, 4)
                payload["stability_cos"] = round(log_buf["stability_cos"] / n_stab, 4)
                payload["id_stable"] = round(log_buf["id_diag_stable"] / n_stab, 4)
                payload["n_stab"] = log_buf["n_stab"]
            print(json.dumps(payload), flush=True)
            log_buf = {"prediction": 0.0, "mask_mean": 0.0, "mask_max": 0.0,
                        "stability_cos": 0.0, "id_diag_stable": 0.0,
                        "n_log": 0, "n_stab": 0}

    # ---------- (BC phase removed; see linear_probe_eval below) ----------
    if False:
        print(json.dumps({"event": "phase2_start",
                           "steps": args.phase2_steps}), flush=True)
        # Freeze everything except the policy head.
        for p in ctx_enc.parameters():
            p.requires_grad = False
        if slot_attn is not None:
            for p in slot_attn.parameters():
                p.requires_grad = False
        if slot_predictor is not None:
            for p in slot_predictor.parameters():
                p.requires_grad = False
        if predictor_dense is not None:
            for p in predictor_dense.parameters():
                p.requires_grad = False
        bc_optim = torch.optim.AdamW(policy_head.parameters(), lr=args.bc_lr)

        obs = env.reset()
        if args.mode == "slot_delta":
            with torch.no_grad():
                tokens0, _ = encode_frame(ctx_enc, obs)
                slots = slot_attn(tokens0)
        else:
            slots = None

        for step in range(1, args.phase2_steps + 1):
            with torch.no_grad():
                tokens, pooled = encode_frame(ctx_enc, obs)
                if args.mode == "slot_delta":
                    state = slots_to_state(slots)
                else:
                    state = pooled
            policy_out = policy_head(state)
            expert = env.expert_action()
            bc_loss = F.mse_loss(policy_out, expert)
            bc_optim.zero_grad(set_to_none=True)
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_head.parameters(), 1.0)
            bc_optim.step()

            with torch.no_grad():
                obs, _, done = env.step(expert)
                if args.mode == "slot_delta":
                    tokens_next, _ = encode_frame(ctx_enc, obs)
                    pad = torch.zeros(args.batch_size, args.d_jepa - 2, device=device)
                    act_vec = torch.cat([expert, pad], dim=-1)
                    out = slot_predictor(slots, tokens_next, act_vec)
                    slots = out["next_slots"]
                if bool(done.all().item()):
                    obs = env.reset()
                    if args.mode == "slot_delta":
                        tokens0, _ = encode_frame(ctx_enc, obs)
                        slots = slot_attn(tokens0)

            if step % args.log_every == 0:
                print(json.dumps({"event": "phase2", "step": step,
                                    "bc_loss": round(float(bc_loss), 5)}),
                        flush=True)

    final_path = os.path.join(args.output, "final.pt")
    state = {
        "mode": args.mode,
        "args": vars(args),
        "context_encoder": ctx_enc.state_dict(),
        "target_encoder": tgt_enc.state_dict(),
        "policy_head": policy_head.state_dict(),
    }
    if args.mode == "slot_delta":
        state["slot_attn"] = slot_attn.state_dict()
        state["slot_predictor"] = slot_predictor.state_dict()
    if args.mode == "dense":
        state["predictor_dense"] = predictor_dense.state_dict()
    torch.save(state, final_path)
    print(json.dumps({"event": "final", "path": final_path,
                       "elapsed_s": round(time.time() - t0, 1)}, indent=2), flush=True)

    # ---- Linear-probe eval under varying J ----
    if args.probe_episodes > 0 and args.eval_J_values:
        eval_J_list = [int(j) for j in args.eval_J_values.split(",")]
        ctx_enc.eval()
        if slot_attn is not None:
            slot_attn.eval()
        if slot_predictor is not None:
            slot_predictor.eval()
        results = linear_probe_eval(
            args, eval_J_list, args.mode, ctx_enc, slot_attn,
            slot_predictor, device,
        )
        for r in results:
            print(json.dumps({"event": "probe", **r}), flush=True)
        with open(os.path.join(args.output, "probe_eval.json"), "w") as f:
            json.dump({"mode": args.mode, "results": results,
                        "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
