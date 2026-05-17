"""Smoke tests for the Phase 12 relation graph prototype."""
import torch
import pytest

from system1_jepa.of_jepa import OFJEPAObjectFiles
from system1_jepa.of_jepa.relations import (
    RelationConfig, RelationHead, RelationGraphPredictor,
    near_relation_labels, relation_loss,
)


def test_relation_head_shape_and_symmetry():
    cfg = RelationConfig(file_dim=32, hidden_dim=16, n_relations=3, asymmetric=False)
    head = RelationHead(cfg)
    files = torch.randn(2, 5, 32)
    logits = head(files)
    assert logits.shape == (2, 5, 5, 3)
    # Symmetric: logit(i, j, r) == logit(j, i, r)
    assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-5)


def test_relation_head_asymmetric_breaks_symmetry():
    cfg = RelationConfig(file_dim=32, hidden_dim=16, n_relations=1, asymmetric=True)
    head = RelationHead(cfg)
    files = torch.randn(1, 5, 32)
    logits = head(files)
    # Should NOT be symmetric (almost surely with random files + init).
    assert not torch.allclose(logits, logits.transpose(1, 2), atol=1e-3)


def test_near_relation_labels():
    pos = torch.tensor([
        [0.10, 0.10],   # close to entity 1
        [0.15, 0.12],   # close to entity 0
        [0.80, 0.80],   # far from both
    ])
    vis = torch.tensor([True, True, True])
    labels = near_relation_labels(pos, vis, threshold=0.10)
    assert labels[0, 1].item() is True
    assert labels[1, 0].item() is True
    assert labels[0, 2].item() is False
    assert labels[2, 0].item() is False
    assert labels[0, 0].item() is False  # self-pair masked


def test_near_relation_masks_invisible_entities():
    pos = torch.tensor([[0.1, 0.1], [0.15, 0.12]])
    vis = torch.tensor([True, False])
    labels = near_relation_labels(pos, vis, threshold=0.10)
    assert labels.sum().item() == 0  # invisible entity → no relations


def test_relation_predictor_from_object_file_batch():
    """End-to-end: instantiate OFJEPAObjectFiles, observe a frame, run
    RelationGraphPredictor on the ObjectFileBatch."""
    torch.manual_seed(0)
    substrate = OFJEPAObjectFiles(
        image_size=64, n_files=6, slot_dim=32, device="cpu",
    )
    substrate.reset_episode(batch_size=1)
    ofb = substrate.observe(torch.randn(3, 64, 64))

    rg_cfg = RelationConfig(file_dim=32, hidden_dim=16, n_relations=1, asymmetric=False)
    predictor = RelationGraphPredictor(rg_cfg)
    logits = predictor(ofb, mask_self=True)
    assert logits.shape == (1, 6, 6, 1)
    # Diagonal should be zero (masked).
    for i in range(6):
        assert logits[0, i, i, 0].item() == 0.0


def test_relation_loss_runs():
    torch.manual_seed(1)
    pred = torch.randn(2, 4, 4, 2, requires_grad=True)
    gt = torch.zeros(2, 4, 4, 2).bool()
    gt[0, 0, 1, 0] = True
    gt[0, 1, 0, 0] = True
    loss = relation_loss(pred, gt)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None


def test_relation_loss_with_mask():
    pred = torch.randn(1, 3, 3, 1, requires_grad=True)
    gt = torch.zeros(1, 3, 3, 1).bool()
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1] = True
    mask[0, 1, 0] = True
    loss = relation_loss(pred, gt, mask=mask)
    assert loss.item() >= 0
    loss.backward()
