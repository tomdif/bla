import numpy as np
import torch

from system1_jepa.value_head import (
    GoalProgressValueHead,
    train_value_head_supervised,
    combine_scores,
    normalize_score,
)


def test_value_head_shape():
    torch.manual_seed(0)
    head = GoalProgressValueHead(state_dim=10, action_dim=7, plan_horizon=10,
                                   hidden=32, n_hidden=2)
    state = torch.randn(5, 10)
    goal = torch.randn(5, 2)
    actions = torch.randn(5, 10, 7)
    out = head(state, goal, actions)
    assert out.shape == (5,)
    # Single example
    out1 = head(torch.randn(10), torch.randn(2), torch.randn(10, 7))
    assert out1.shape == (1,)


def test_value_head_rejects_wrong_action_shape():
    head = GoalProgressValueHead(state_dim=10, plan_horizon=10)
    try:
        head(torch.randn(2, 10), torch.randn(2, 2), torch.randn(2, 5, 7))
        raise AssertionError("expected ValueError for wrong horizon")
    except ValueError:
        pass


def test_value_head_training_reduces_loss():
    torch.manual_seed(0)
    n = 200
    state_dim = 6
    horizon = 3
    action_dim = 2
    states = torch.randn(n, state_dim)
    goals = torch.randn(n, 2)
    actions = torch.randn(n, horizon, action_dim)
    # Synthetic teacher: label = nonlinear fn of state[0] and actions sum
    labels = torch.tanh(states[:, 0] + actions.sum(dim=(1, 2)) * 0.3).abs()

    head = GoalProgressValueHead(state_dim=state_dim, action_dim=action_dim,
                                   plan_horizon=horizon, hidden=64, n_hidden=2)
    stats = train_value_head_supervised(
        head, states, goals, actions, labels,
        steps=400, batch_size=32, lr=1e-3, val_split=0.2, seed=0,
    )
    assert stats.final_loss < stats.initial_loss * 0.5
    # Validation loss should also reduce (no leakage requirement, just learning)
    assert stats.final_val_loss < stats.initial_val_loss


def test_normalize_score_safe_on_constant():
    arr = np.ones(10)
    out = normalize_score(arr)
    assert np.allclose(out, 0.0)


def test_combine_scores_modes():
    rng = np.random.RandomState(0)
    p = rng.randn(8)
    v = rng.randn(8)
    s_sum = combine_scores(p, v, mode="sum", lam=0.5)
    s_max = combine_scores(p, v, mode="max")
    s_only = combine_scores(p, v, mode="value_only")
    assert s_sum.shape == (8,) and s_max.shape == (8,) and s_only.shape == (8,)
    # value_only is just normalize_score(v)
    assert np.allclose(s_only, normalize_score(v))
