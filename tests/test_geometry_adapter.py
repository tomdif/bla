import torch
import numpy as np

from system1_jepa.geometry_adapter import (
    ObjectFileGeometryAdapter,
    train_adapter_supervised,
)


def test_adapter_shape():
    torch.manual_seed(0)
    adapter = ObjectFileGeometryAdapter(slot_dim=128, hidden=32, n_hidden=2)
    slot = torch.randn(5, 128)
    goal = torch.randn(5, 2)
    out = adapter(slot, goal)
    assert out.shape == (5, 10)
    # Single example
    out1 = adapter(torch.randn(128), torch.randn(2))
    assert out1.shape == (1, 10)


def test_adapter_recovers_linear_mapping():
    torch.manual_seed(0)
    n = 200
    slot_dim = 32
    goal_dim = 2
    out_dim = 10
    slots = torch.randn(n, slot_dim)
    goals = torch.randn(n, goal_dim)
    # Synthetic teacher: target_geo = linear projection of (slot, goal)
    W = torch.randn(slot_dim + goal_dim, out_dim) * 0.2
    inputs = torch.cat([slots, goals], dim=-1)
    target_geo = inputs @ W

    adapter = ObjectFileGeometryAdapter(slot_dim=slot_dim, goal_dim=goal_dim,
                                          out_dim=out_dim, hidden=64,
                                          n_hidden=2)
    stats = train_adapter_supervised(
        adapter, slots, goals, target_geo,
        steps=500, batch_size=32, lr=1e-3, val_split=0.2, seed=0,
    )
    assert stats.final_train_mse < stats.initial_train_mse * 0.2
    assert stats.final_val_mse < stats.initial_val_mse * 0.4
    assert stats.mean_val_pearson > 0.7  # should learn the linear mapping well


def test_adapter_constant_features_handled():
    """Constant target features should give NaN corr without crashing."""
    torch.manual_seed(0)
    slots = torch.randn(50, 16)
    goals = torch.randn(50, 2)
    # Target with one constant column
    target = torch.randn(50, 10)
    target[:, 3] = 1.0
    adapter = ObjectFileGeometryAdapter(slot_dim=16, hidden=32, n_hidden=2)
    stats = train_adapter_supervised(
        adapter, slots, goals, target,
        steps=100, batch_size=16, lr=1e-3, val_split=0.2, seed=0,
    )
    # nans appear for constant feature; remaining mean ignores them
    assert not (stats.mean_val_pearson != stats.mean_val_pearson)
