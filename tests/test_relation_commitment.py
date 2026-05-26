"""Tests for Layer C (CausalRelationHead) and Causal Commitment Training.

Style-matched to tests/test_smoke.py: torch + numpy seeding, shape +
finite + gradient + overfit assertions, no fixtures beyond pytest's
built-ins.
"""
import numpy as np
import pytest
import torch

from system1_jepa.causal_relations import (
    CausalRelationConfig,
    CausalRelationHead,
    EdgeAnnotations,
    causal_edge_loss,
)
from system1_jepa.slot import SlotAttention, SlotAttentionConfig
from system1_jepa.slot_existence import (
    SlotExistenceHead,
    binding_mass,
    slot_existence_loss,
    visibility_disagreement_surprise,
)
from system1_jepa.slot_predictor import SlotDeltaPredictor, SlotPredictorConfig
from verification.commitment_loss import (
    CommitmentEncoder,
    CommitmentEncoderConfig,
    commitment_consistency_loss,
)


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# -------------------- CausalRelationHead --------------------


def test_causal_relation_head_shapes():
    _seed()
    cfg = CausalRelationConfig(slot_dim=16, n_edge_types=4, hidden=32)
    head = CausalRelationHead(cfg)
    slots = torch.randn(2, 5, cfg.slot_dim)
    edges = head(slots)
    assert isinstance(edges, EdgeAnnotations)
    assert edges.edge_type_logits.shape == (2, 5, 5, cfg.n_edge_types)
    assert edges.confidence.shape == (2, 5, 5)
    assert edges.causal_strength.shape == (2, 5, 5)
    assert edges.refined_slots.shape == slots.shape


def test_causal_relation_head_sigmoid_outputs_in_unit_interval():
    _seed()
    cfg = CausalRelationConfig(slot_dim=16)
    head = CausalRelationHead(cfg)
    slots = torch.randn(2, 5, cfg.slot_dim) * 10.0  # large activations
    edges = head(slots)
    assert (edges.confidence >= 0).all() and (edges.confidence <= 1).all()
    assert (edges.causal_strength >= 0).all() and (edges.causal_strength <= 1).all()


def test_causal_relation_head_refined_slots_are_relation_aware():
    """If the GNN msg pass is wired correctly, refined_slots should differ
    from input slots — else the message passing is dead."""
    _seed()
    cfg = CausalRelationConfig(slot_dim=16)
    head = CausalRelationHead(cfg)
    slots = torch.randn(2, 5, cfg.slot_dim)
    edges = head(slots)
    assert not torch.allclose(edges.refined_slots, slots, atol=1e-4)


def test_causal_edge_loss_target_is_detached_from_change_mask():
    """change_mask is *supervision*, not a backprop path — we must not
    let the relation head's loss tug on the slot predictor's masks."""
    _seed()
    cfg = CausalRelationConfig(slot_dim=16)
    head = CausalRelationHead(cfg)
    slots = torch.randn(2, 5, cfg.slot_dim)
    mask_t = torch.rand(2, 5, 1, requires_grad=True)
    mask_t1 = torch.rand(2, 5, 1, requires_grad=True)
    edges = head(slots)
    out = causal_edge_loss(edges, mask_t, mask_t1)
    out["loss"].backward()
    assert mask_t.grad is None or mask_t.grad.abs().sum() == 0
    assert mask_t1.grad is None or mask_t1.grad.abs().sum() == 0


def test_causal_edge_loss_gradient_flows_to_relation_head():
    _seed()
    cfg = CausalRelationConfig(slot_dim=16)
    head = CausalRelationHead(cfg)
    slots = torch.randn(2, 5, cfg.slot_dim)
    mask_t = torch.rand(2, 5, 1)
    mask_t1 = torch.rand(2, 5, 1)
    edges = head(slots)
    out = causal_edge_loss(edges, mask_t, mask_t1)
    out["loss"].backward()
    grad_sum = sum(
        p.grad.abs().sum().item()
        for p in head.parameters() if p.grad is not None
    )
    assert grad_sum > 0, "no gradient reached CausalRelationHead parameters"


def test_causal_edge_loss_learns_synthetic_cofiring():
    """Overfit: a fixed (slots, mask_t, mask_t1) co-firing pattern should
    train the head's edge_pred toward the implied i->j target."""
    _seed()
    cfg = CausalRelationConfig(slot_dim=16, hidden=32)
    head = CausalRelationHead(cfg)
    optim = torch.optim.AdamW(head.parameters(), lr=5e-3)
    slots = torch.randn(2, 4, cfg.slot_dim)
    # Two fixed "firing" patterns; this implies a specific co-firing target
    # under the outer product m_t.unsqueeze(2) * m_t1.unsqueeze(1).
    mask_t = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0]]).unsqueeze(-1)
    mask_t1 = torch.tensor([[0.0, 1.0, 0.0, 1.0],
                             [1.0, 0.0, 1.0, 0.0]]).unsqueeze(-1)

    losses = []
    for _ in range(80):
        edges = head(slots)
        out = causal_edge_loss(edges, mask_t, mask_t1, diag_penalty=0.0)
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        optim.step()
        losses.append(float(out["bce"]))
    assert losses[-1] < losses[0] * 0.5, (
        f"causal_edge_loss did not halve on a fixed pattern: "
        f"{losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_causal_edge_loss_diagonal_penalty_punishes_self_edges():
    """With diag_penalty > 0 the head should learn to push self-edge
    strength below the baseline of having no diag penalty."""
    _seed()
    cfg = CausalRelationConfig(slot_dim=16)
    slots = torch.randn(2, 4, cfg.slot_dim)
    mask_t = torch.rand(2, 4, 1)
    mask_t1 = torch.rand(2, 4, 1)

    def train(diag_penalty):
        _seed()
        head = CausalRelationHead(cfg)
        optim = torch.optim.AdamW(head.parameters(), lr=5e-3)
        for _ in range(40):
            edges = head(slots)
            out = causal_edge_loss(edges, mask_t, mask_t1, diag_penalty=diag_penalty)
            optim.zero_grad(set_to_none=True)
            out["loss"].backward()
            optim.step()
        with torch.no_grad():
            edges = head(slots)
            eye = torch.eye(4)
            return float((edges.causal_strength * eye).sum())

    self_with_penalty = train(diag_penalty=0.5)
    self_no_penalty = train(diag_penalty=0.0)
    assert self_with_penalty < self_no_penalty, (
        f"diag penalty did not reduce self-edge strength: "
        f"with={self_with_penalty:.4f}, without={self_no_penalty:.4f}"
    )


def test_causal_relation_composes_with_slot_attention_and_predictor():
    """End-to-end: input tokens -> SlotAttention -> CausalRelationHead ->
    SlotDeltaPredictor on refined slots, gradients flow to every module."""
    _seed()
    slot_cfg = SlotAttentionConfig(n_slots=4, slot_dim=16, n_iters=2)
    slot_attn = SlotAttention(input_dim=16, cfg=slot_cfg)
    rel_cfg = CausalRelationConfig(slot_dim=16, hidden=32)
    rel_head = CausalRelationHead(rel_cfg)
    pred_cfg = SlotPredictorConfig(
        slot_dim=16, obs_dim=16, action_dim=8, n_layers=1, n_heads=2,
    )
    predictor = SlotDeltaPredictor(pred_cfg)

    inputs = torch.randn(2, 8, 16)
    obs = torch.randn(2, 8, 16)
    action = torch.randn(2, 8)

    slots = slot_attn(inputs)
    edges = rel_head(slots)
    pred_out = predictor(edges.refined_slots, obs, action)
    # Target slots are just a perturbed copy — we only care about gradient flow.
    target = slots.detach() + 0.01 * torch.randn_like(slots)
    loss = torch.nn.functional.smooth_l1_loss(pred_out["next_slots"], target)
    loss.backward()

    for name, mod in [("slot_attn", slot_attn), ("rel_head", rel_head), ("pred", predictor)]:
        g = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert g > 0, f"{name} received no gradient (g={g})"


# -------------------- CommitmentEncoder + loss --------------------


def test_commitment_encoder_returns_unit_norm():
    _seed()
    cfg = CommitmentEncoderConfig(claim_dim=16, world_dim=32, out_dim=8)
    enc = CommitmentEncoder(cfg)
    claim = torch.randn(4, cfg.claim_dim)
    world = torch.randn(4, cfg.world_dim)
    c = enc(claim, world)
    assert c.shape == (4, cfg.out_dim)
    norms = c.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_commitment_loss_zero_surprise_penalizes_drift():
    """Under zero surprise, drifted commitments should produce a loss
    dominated by the consistency term."""
    _seed()
    c_past = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    c_now = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    surprise = torch.zeros(8)
    out = commitment_consistency_loss(c_past, c_now, surprise, margin=0.1)
    assert out["revision_rate"] < 0.5
    # consistency is the dominant term
    assert float(out["consistency"]) > float(out["contradiction"])


def test_commitment_loss_high_surprise_relaxes_consistency():
    """Same drift, but high surprise: revision_rate should rise and the
    consistency term should shrink relative to zero-surprise."""
    _seed()
    c_past = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    c_now = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)

    low = commitment_consistency_loss(c_past, c_now, torch.zeros(8))
    high = commitment_consistency_loss(c_past, c_now, torch.full((8,), 10.0))
    assert float(high["revision_rate"]) > float(low["revision_rate"])
    assert float(high["consistency"]) < float(low["consistency"])


def test_commitment_loss_collapse_defense_punishes_identical_under_surprise():
    """If we cheat by making c_past == c_now (drift = 0), the loss should
    *rise* under high surprise via the contradiction term — otherwise the
    encoder has a trivial collapse solution."""
    _seed()
    c = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    identical_no_surprise = commitment_consistency_loss(c, c, torch.zeros(8))
    identical_high_surprise = commitment_consistency_loss(
        c, c, torch.full((8,), 10.0)
    )
    assert float(identical_high_surprise["loss"]) > float(identical_no_surprise["loss"]), (
        "contradiction term failed: identical commitments under surprise "
        "should be MORE costly, not less"
    )
    assert float(identical_high_surprise["contradiction"]) > 0


def test_commitment_loss_gradient_flows_to_encoder():
    _seed()
    cfg = CommitmentEncoderConfig(claim_dim=16, world_dim=32, out_dim=8)
    enc = CommitmentEncoder(cfg)
    claim = torch.randn(4, cfg.claim_dim)
    world_past = torch.randn(4, cfg.world_dim)
    world_now = torch.randn(4, cfg.world_dim)
    c_past = enc(claim, world_past)
    c_now = enc(claim, world_now)
    surprise = torch.rand(4)
    out = commitment_consistency_loss(c_past, c_now, surprise)
    out["loss"].backward()
    g = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    assert g > 0, "CommitmentEncoder received no gradient"


# -------------------- SlotExistenceHead + visibility-disagreement --------------------


def test_slot_attention_returns_attention_when_requested():
    _seed()
    cfg = SlotAttentionConfig(n_slots=4, slot_dim=16, n_iters=2)
    attn_mod = SlotAttention(input_dim=16, cfg=cfg)
    inputs = torch.randn(2, 9, 16)
    slots = attn_mod(inputs)
    slots2, attn = attn_mod(inputs, return_attention=True)
    assert slots.shape == slots2.shape == (2, 4, 16)
    assert attn.shape == (2, 9, 4)
    # Softmax over slots -> rows sum to 1.
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_binding_mass_is_per_slot_in_unit_interval():
    _seed()
    cfg = SlotAttentionConfig(n_slots=4, slot_dim=16, n_iters=2)
    attn_mod = SlotAttention(input_dim=16, cfg=cfg)
    inputs = torch.randn(2, 9, 16)
    _, attn = attn_mod(inputs, return_attention=True)
    mass = binding_mass(attn)
    assert mass.shape == (2, 4)
    assert (mass >= 0).all() and (mass <= 1).all()
    # Per-batch slot masses sum to 1 (since softmax-over-slots rows sum to 1,
    # then we divide by N_inputs and sum back).
    totals = mass.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_slot_existence_head_output_shape_and_range():
    _seed()
    head = SlotExistenceHead(slot_dim=16, hidden=8)
    slots = torch.randn(2, 5, 16) * 5.0
    exists = head(slots)
    assert exists.shape == (2, 5)
    assert (exists >= 0).all() and (exists <= 1).all()


def test_visibility_disagreement_surprise_spikes_on_synthetic_occlusion():
    """The core unit test for the FIX: when predicted-active slots become
    actually-inactive (the occlusion regime), surprise should be HIGH;
    when both agree, surprise should be ~0."""
    pred_all_active = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    gt_two_occluded = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    gt_none_occluded = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    surprise_occluded = visibility_disagreement_surprise(
        pred_all_active, gt_two_occluded,
    )
    surprise_visible = visibility_disagreement_surprise(
        pred_all_active, gt_none_occluded,
    )
    assert float(surprise_occluded) > float(surprise_visible)
    assert float(surprise_visible) < 1e-6
    assert float(surprise_occluded) == pytest.approx(0.5, abs=1e-5)


def test_visibility_disagreement_surprise_asymmetric_ignores_new_appearances():
    """Asymmetric weighting should fire ONLY on 'expected but absent',
    not on 'unexpected but present'. Useful for narrowing CCT to the
    occlusion regime."""
    pred_quiet = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    gt_new_object = torch.tensor([[1.0, 1.0, 1.0, 1.0]])      # new appearances
    gt_occlusion = torch.tensor([[0.0, 0.0, 0.0, 0.0]])       # expected slots gone

    sym_new = float(visibility_disagreement_surprise(pred_quiet, gt_new_object))
    sym_occl = float(visibility_disagreement_surprise(pred_quiet, gt_occlusion))
    asym_new = float(visibility_disagreement_surprise(
        pred_quiet, gt_new_object, weighting="asymmetric"))
    asym_occl = float(visibility_disagreement_surprise(
        pred_quiet, gt_occlusion, weighting="asymmetric"))

    # Symmetric: both directions register equally.
    assert sym_new == pytest.approx(sym_occl, abs=1e-5)
    # Asymmetric: only occlusion fires, new appearances ignored.
    assert asym_new == pytest.approx(0.0, abs=1e-5)
    assert asym_occl > 0


def test_slot_existence_head_learns_binding_mass_on_synthetic_data():
    """Overfit: fix slots + binding-mass target, train the head, BCE drops."""
    _seed()
    head = SlotExistenceHead(slot_dim=16, hidden=32)
    optim = torch.optim.AdamW(head.parameters(), lr=5e-3)
    slots = torch.randn(4, 6, 16)
    # Two "always-bound" slots and four "never-bound" slots, as a target.
    target = torch.tensor([[0.9, 0.9, 0.05, 0.05, 0.05, 0.05]] * 4)

    losses = []
    for _ in range(60):
        pred = head(slots)
        out = slot_existence_loss(pred, target)
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        optim.step()
        losses.append(float(out["loss"]))
    assert losses[-1] < losses[0] * 0.5, (
        f"existence head did not halve BCE: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_end_to_end_existence_pipeline_runs_with_slot_attention():
    """SlotAttention -> binding_mass + SlotExistenceHead -> existence loss,
    gradients flow to the head (binding_mass target is detached)."""
    _seed()
    attn_mod = SlotAttention(
        input_dim=16,
        cfg=SlotAttentionConfig(n_slots=5, slot_dim=16, n_iters=2),
    )
    head = SlotExistenceHead(slot_dim=16, hidden=16)
    inputs = torch.randn(2, 9, 16)
    slots, attn = attn_mod(inputs, return_attention=True)
    gt = binding_mass(attn).detach()
    pred = head(slots)
    out = slot_existence_loss(pred, gt)
    out["loss"].backward()
    g = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    assert g > 0
    # Surprise can be computed from the trained head's predictions.
    s = visibility_disagreement_surprise(pred.detach(), gt)
    assert s.shape == (2,)


def test_commitment_loss_revision_rate_monotonic_in_surprise():
    """revision_rate should be a monotone non-decreasing function of surprise."""
    _seed()
    c_past = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1)
    c_now = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1)
    rates = []
    for s in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]:
        out = commitment_consistency_loss(
            c_past, c_now, torch.full((64,), s)
        )
        rates.append(float(out["revision_rate"]))
    for a, b in zip(rates, rates[1:]):
        assert b >= a - 1e-6, f"revision_rate not monotone: {rates}"
    assert rates[0] < 0.5 < rates[-1]
