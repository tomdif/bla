import torch
import numpy as np

from system1_jepa.geometry_adapter import (
    End2EndAdapterValue,
    train_end2end_supervised,
)


def test_end2end_shape():
    torch.manual_seed(0)
    m = End2EndAdapterValue(
        slot_dim=128, latent_dim=10, plan_horizon=4, action_dim=3,
        adapter_hidden=32, adapter_n_hidden=2,
        value_hidden=32, value_n_hidden=2,
    )
    slot = torch.randn(5, 128)
    goal = torch.randn(5, 2)
    actions = torch.randn(5, 4, 3)
    out = m(slot, goal, actions)
    assert out.shape == (5,)


def test_end2end_training_reduces_loss():
    torch.manual_seed(0)
    n = 200
    slot_dim = 32
    horizon = 3
    action_dim = 2
    slots = torch.randn(n, slot_dim)
    goals = torch.randn(n, 2)
    plans = torch.randn(n, horizon, action_dim)
    # Synthetic teacher: label = nonlinear combo of slot dims + plan sum
    labels = torch.tanh(slots[:, 0] + slots[:, 1] * 0.5
                          + plans.sum(dim=(1, 2)) * 0.2).abs()

    m = End2EndAdapterValue(
        slot_dim=slot_dim, latent_dim=10, plan_horizon=horizon,
        action_dim=action_dim,
        adapter_hidden=64, adapter_n_hidden=2,
        value_hidden=64, value_n_hidden=2,
    )
    stats = train_end2end_supervised(
        m, slots, goals, plans, labels,
        steps=400, batch_size=32, lr=1e-3, val_split=0.2, seed=0,
    )
    assert stats.final_loss < stats.initial_loss * 0.35
    # Val loss should also decrease (some generalization)
    assert stats.final_val_loss < stats.initial_val_loss * 0.8


def test_end2end_config_roundtrip():
    from system1_jepa.geometry_adapter import (
        end2end_config, build_end2end_from_config,
    )
    m1 = End2EndAdapterValue(slot_dim=64, latent_dim=10, plan_horizon=5,
                                action_dim=3, adapter_hidden=32,
                                adapter_n_hidden=2, value_hidden=32,
                                value_n_hidden=2)
    cfg = end2end_config(m1)
    m2 = build_end2end_from_config(cfg)
    m2.load_state_dict(m1.state_dict())  # should match shapes
    x = torch.randn(3, 64)
    g = torch.randn(3, 2)
    a = torch.randn(3, 5, 3)
    out1 = m1(x, g, a)
    out2 = m2(x, g, a)
    assert torch.allclose(out1, out2)
