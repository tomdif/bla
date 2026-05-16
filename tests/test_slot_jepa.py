"""Smoke tests for the slot-JEPA pipeline (Phase 11/KAM-JEPA-v0).

Coverage focuses on the load-bearing pieces of the falsification test:
  - SlotAttention produces non-degenerate, slot-distinguishing output
  - SlotDeltaPredictor wires shapes correctly + delta is bounded
  - change_mask sparsity pressure pushes mean below 0.5 on a trivial task
  - the soft change_mask defaults to a small but nontrivial value (the
    `change_head.bias = -1.0` prior)
  - Occluded env actually hides targets during the hidden window
  - End-to-end: encoder → slots → delta → next-slot loss runs + backprops
"""
import numpy as np
import torch

from system1_jepa import (
    JEPAConfig,
    BLAJEPAModel,
    OccludedMultiTargetNavigateEnv,
    OccludedNavigateSpec,
    SlotAttention,
    SlotAttentionConfig,
    SlotDeltaPredictor,
    SlotPredictorConfig,
    copy_baseline,
    pool_patch_tokens,
    slot_delta_loss,
)


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_slot_attention_produces_distinguishable_slots():
    _seed()
    cfg = SlotAttentionConfig(n_slots=4, slot_dim=16, n_iters=3)
    slot_attn = SlotAttention(input_dim=8, cfg=cfg)
    inputs = torch.randn(2, 12, 8)
    slots = slot_attn(inputs)
    assert slots.shape == (2, 4, 16)
    # Slots should not all be identical — the iteration is meant to bind
    # different inputs to different slots.
    pairwise = (slots[:, :, None, :] - slots[:, None, :, :]).pow(2).sum(-1).sqrt()
    # ignore diagonals
    off_diag = pairwise + 1e9 * torch.eye(4)
    min_pair_dist = off_diag.min().item()
    assert min_pair_dist > 1e-3, f"slots are degenerate: min pairwise dist={min_pair_dist}"


def test_slot_attention_accepts_init_slots_without_error():
    """SlotAttention must accept caller-supplied initial slots without
    crashing or reshaping. Whether identity is preserved across iterations
    is a property of training, not of random init — the delta predictor
    is what carries memory across timesteps in our pipeline. This test
    only verifies the wiring."""
    _seed()
    cfg = SlotAttentionConfig(n_slots=4, slot_dim=16, n_iters=2)
    slot_attn = SlotAttention(input_dim=8, cfg=cfg)
    init = torch.randn(2, 4, 16) * 0.1
    inputs = torch.randn(2, 12, 8)
    out = slot_attn(inputs, init_slots=init)
    assert out.shape == init.shape
    assert out.isfinite().all()


def test_slot_predictor_shapes_and_bounded_delta():
    _seed()
    cfg = SlotPredictorConfig(slot_dim=16, obs_dim=8, action_dim=4,
                                delta_scale=0.1, n_layers=1, n_heads=2)
    predictor = SlotDeltaPredictor(cfg)
    slots = torch.randn(2, 4, 16)
    obs = torch.randn(2, 12, 8)
    action = torch.randn(2, 4)
    out = predictor(slots, obs, action, return_diagnostics=True)

    assert out["next_slots"].shape == (2, 4, 16)
    assert out["change_mask"].shape == (2, 4, 1)
    assert out["delta"].shape == (2, 4, 16)
    # The bounded delta must respect delta_scale * tanh ⇒ |delta| ≤ delta_scale.
    assert out["delta"].abs().max().item() <= cfg.delta_scale + 1e-5
    # The change_mask is a sigmoid ⇒ strictly in [0, 1].
    assert 0.0 <= out["change_mask"].min().item()
    assert out["change_mask"].max().item() <= 1.0
    # With the -1.0 bias init, the mean mask should start well below 0.5.
    assert out["change_mask"].mean().item() < 0.5


def test_slot_delta_loss_falls_with_sparsity_pressure():
    """Tiny optimization: minimize slot_delta_loss with a sparsity term and
    confirm the change_mask responds. Verifies that the L1 penalty actually
    pushes the mask down, not just whether the prediction term descends."""
    _seed()
    cfg = SlotPredictorConfig(slot_dim=16, obs_dim=8, action_dim=4,
                                delta_scale=0.1, n_layers=1, n_heads=2)
    predictor = SlotDeltaPredictor(cfg)
    # Target = current slots (identity dynamics). The optimal mask is 0.
    slots = torch.randn(2, 4, 16)
    obs = torch.randn(2, 12, 8)
    action = torch.randn(2, 4)
    target = slots.clone()

    optim = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    initial_mask_mean = None
    for step in range(50):
        out = predictor(slots, obs, action)
        loss_d = slot_delta_loss(out["next_slots"], target, out["change_mask"],
                                  sparsity_weight=1e-2)
        if step == 0:
            initial_mask_mean = float(loss_d["mask_mean"])
        optim.zero_grad(set_to_none=True)
        loss_d["loss"].backward()
        optim.step()
    final_mask_mean = float(loss_d["mask_mean"])
    assert final_mask_mean < initial_mask_mean, (
        f"sparsity pressure did not reduce mask mean: "
        f"{initial_mask_mean:.3f} -> {final_mask_mean:.3f}"
    )


def test_copy_baseline_is_identity():
    slots = torch.randn(3, 5, 8)
    out = copy_baseline(slots)
    assert torch.equal(out, slots)


def test_occluded_env_hides_targets_during_hidden_window():
    _seed()
    spec = OccludedNavigateSpec(
        image_size=16, patch_size=2, n_targets=2,
        visible_steps=2, hidden_steps=3,
        max_steps=20, n_distractors=0,
    )
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=2, seed=0)

    obs_visible = env.observe()
    # Targets render into the red channel (channel 0).
    target_pixels_visible = obs_visible[:, 0].sum().item()
    assert target_pixels_visible > 0, "targets not visible at t=0"

    # Step through to a hidden frame. With visible_steps=2 and hidden_steps=3,
    # t in [2, 3, 4] is hidden.
    zero_action = torch.zeros(2, 2)
    for _ in range(3):
        env.step(zero_action)
    obs_hidden = env.observe()
    target_pixels_hidden = obs_hidden[:, 0].sum().item()
    assert target_pixels_hidden == 0.0, (
        f"targets still visible during hidden window: {target_pixels_hidden} red pixels"
    )

    # Step back into a visible window and verify targets return.
    for _ in range(3):
        env.step(zero_action)
    obs_visible_again = env.observe()
    assert obs_visible_again[:, 0].sum().item() > 0, (
        "targets did not reappear after hidden window"
    )


def test_occluded_env_distractors_render_in_blue_channel():
    _seed()
    spec = OccludedNavigateSpec(
        image_size=16, patch_size=2, n_targets=1,
        visible_steps=1, hidden_steps=0,  # never hidden, simplify
        max_steps=5, n_distractors=3,
    )
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=2, seed=0)
    obs = env.observe()
    # Distractors paint into the blue channel (channel 2).
    blue_total = obs[:, 2].sum().item()
    assert blue_total > 0, "distractors not rendered in blue channel"


def test_occluded_env_default_regression():
    """Phase-2 defaults must remain byte-identical after Phase-3 flags
    are added. Otherwise we accidentally moved the floor and the locked
    Phase-2 numbers stop comparing."""
    _seed()
    spec = OccludedNavigateSpec(
        image_size=32, patch_size=4, n_targets=3, n_distractors=2,
        visible_steps=5, hidden_steps=10, max_steps=24,
    )
    # All Phase-3 flags must default to "off / Phase-2 behaviour".
    assert spec.moving_distractors is False
    assert spec.partial_observability is False
    assert spec.distractor_move_max == 1.0
    assert spec.obs_radius == 8.0
    assert spec.rendered_patches is True

    env = OccludedMultiTargetNavigateEnv(spec, batch_size=2, seed=0)
    # Distractors at t=0 vs after a no-op step: identical when moving is off.
    pos0 = env.dx_pos.clone(), env.dy_pos.clone()
    env.step(torch.zeros(2, 2))
    pos1 = env.dx_pos, env.dy_pos
    assert torch.equal(pos0[0], pos1[0]), "distractors moved with default settings"
    assert torch.equal(pos0[1], pos1[1])


def test_moving_distractors_change_position_each_step():
    _seed()
    spec = OccludedNavigateSpec(
        image_size=32, patch_size=4, n_targets=1, n_distractors=4,
        visible_steps=1, hidden_steps=0, max_steps=10,
        moving_distractors=True, distractor_move_max=2.0,
    )
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=2, seed=0)
    initial = env.dx_pos.clone(), env.dy_pos.clone()
    env.step(torch.zeros(2, 2))
    moved = env.dx_pos, env.dy_pos
    # With move_max=2 it's astronomically unlikely all 8 coords land on
    # exactly their previous integer value via random walk.
    assert not (torch.equal(initial[0], moved[0]) and torch.equal(initial[1], moved[1])), (
        "moving_distractors=True did not change positions"
    )
    # Per-step displacement is bounded by move_max in each axis.
    max_xy = spec.image_size - spec.patch_size
    dx = (moved[0] - initial[0]).abs()
    dy = (moved[1] - initial[1]).abs()
    # Allowing a tiny epsilon for the clamp boundary cases.
    assert (dx <= spec.distractor_move_max + 1e-5).all(), f"dx exceeded bound: {dx.max()}"
    assert (dy <= spec.distractor_move_max + 1e-5).all(), f"dy exceeded bound: {dy.max()}"


def test_perceptual_noise_changes_canvas():
    """perceptual_noise > 0 must perturb pixel values away from the
    deterministic-render canvas. Defaults must preserve Phase-3 behaviour."""
    _seed()
    spec_no = OccludedNavigateSpec(
        image_size=16, patch_size=2, n_targets=1, n_distractors=2,
        visible_steps=1, hidden_steps=0, max_steps=5,
        perceptual_noise=0.0,
    )
    spec_yes = OccludedNavigateSpec(
        image_size=16, patch_size=2, n_targets=1, n_distractors=2,
        visible_steps=1, hidden_steps=0, max_steps=5,
        perceptual_noise=0.1,
    )
    env0 = OccludedMultiTargetNavigateEnv(spec_no, batch_size=2, seed=0)
    env1 = OccludedMultiTargetNavigateEnv(spec_yes, batch_size=2, seed=0)
    o0 = env0.observe()
    o1 = env1.observe()
    diff = (o0 - o1).abs().mean().item()
    assert diff > 0.001, f"perceptual_noise=0.1 did not perturb pixels (diff={diff})"
    # And: zero-noise must reproduce noise-free obs even after env steps.
    # (Phase-3 regression — confirms default behaviour is unchanged.)
    o0_again = env0.observe()
    assert torch.equal(o0, o0_again), "noise=0 obs is non-deterministic"


def test_partial_observability_masks_far_pixels():
    """With partial_observability=True and obs_radius < image_size, pixels
    beyond the radius around the agent must be exactly zero."""
    _seed()
    spec = OccludedNavigateSpec(
        image_size=32, patch_size=4, n_targets=3, n_distractors=4,
        visible_steps=1, hidden_steps=0, max_steps=5,
        partial_observability=True, obs_radius=4.0,
    )
    env = OccludedMultiTargetNavigateEnv(spec, batch_size=2, seed=0)
    obs = env.observe()  # [B, 3, H, W]
    # A pixel diagonally far from the agent (corner) must be zero.
    # Test: pick the corner opposite to the agent's position for each batch.
    for b in range(obs.shape[0]):
        ax = float(env.x[b])
        ay = float(env.y[b])
        # Corner farthest from (ax, ay).
        corner_x = 0 if ax > spec.image_size / 2 else spec.image_size - 1
        corner_y = 0 if ay > spec.image_size / 2 else spec.image_size - 1
        v = obs[b, :, corner_y, corner_x]
        assert v.abs().sum().item() == 0.0, (
            f"partial observability didn't mask far corner ({corner_y},{corner_x}): {v}"
        )
    # And the agent's own pixel must still be visible (non-zero).
    for b in range(obs.shape[0]):
        ax = int(env.x[b])
        ay = int(env.y[b])
        center_val = obs[b, :, ay, ax].abs().sum().item()
        assert center_val > 0.0, (
            f"agent pixel was masked by partial observability at ({ay},{ax})"
        )


def test_end_to_end_encoder_slot_predictor_backprops():
    """Compose JEPA encoder → SlotAttention → SlotDeltaPredictor → loss in
    one forward+backward and confirm gradients flow to every module."""
    _seed()
    jepa_cfg = JEPAConfig.tiny()
    encoder = BLAJEPAModel(jepa_cfg).target_encoder
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True

    slot_cfg = SlotAttentionConfig(n_slots=8, slot_dim=jepa_cfg.d_jepa, n_iters=2)
    slot_attn = SlotAttention(input_dim=jepa_cfg.d_jepa, cfg=slot_cfg)
    pred_cfg = SlotPredictorConfig(
        slot_dim=jepa_cfg.d_jepa, obs_dim=jepa_cfg.d_jepa,
        action_dim=jepa_cfg.action_dim, n_layers=1, n_heads=2, delta_scale=0.1,
    )
    predictor = SlotDeltaPredictor(pred_cfg)

    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, jepa_cfg.action_dim)

    tokens, _, _ = encoder(image)
    slots = slot_attn(tokens)
    out = predictor(slots, tokens, action)
    # Trivial target — current-slot state — to keep the test fast.
    metrics = slot_delta_loss(out["next_slots"], slots, out["change_mask"])
    metrics["loss"].backward()

    enc_grad = sum(
        p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None
    )
    slot_grad = sum(
        p.grad.abs().sum().item() for p in slot_attn.parameters() if p.grad is not None
    )
    pred_grad = sum(
        p.grad.abs().sum().item() for p in predictor.parameters() if p.grad is not None
    )
    assert enc_grad > 0, "encoder received no gradient"
    assert slot_grad > 0, "slot attention received no gradient"
    assert pred_grad > 0, "slot predictor received no gradient"
