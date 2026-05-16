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
