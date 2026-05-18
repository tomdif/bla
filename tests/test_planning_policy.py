import torch

from system1_jepa.planning_policy import (
    PlanProposalPolicy,
    plan_weighted_mse,
    train_plan_policy_supervised,
)


def test_plan_proposal_policy_shape_and_bounds():
    torch.manual_seed(0)
    policy = PlanProposalPolicy(in_dim=10, hidden=32, plan_horizon=4, action_dim=7)
    x = torch.randn(5, 10)
    plan = policy(x)
    assert plan.shape == (5, 4, 7)
    assert torch.all(plan <= 1.0)
    assert torch.all(plan >= -1.0)

    single = policy.propose(torch.randn(10))
    assert single.shape == (4, 7)


def test_plan_weighted_mse_prioritizes_early_actions():
    target = torch.zeros(1, 4, 2)
    early_error = target.clone()
    early_error[:, 0] = 1.0
    late_error = target.clone()
    late_error[:, -1] = 1.0

    early_loss = plan_weighted_mse(early_error, target, step_decay=0.5)
    late_loss = plan_weighted_mse(late_error, target, step_decay=0.5)
    assert early_loss > late_loss


def test_supervised_policy_distillation_reduces_loss():
    torch.manual_seed(0)
    n = 64
    in_dim = 6
    horizon = 3
    action_dim = 2
    x = torch.randn(n, in_dim)
    teacher = torch.randn(in_dim, horizon * action_dim) * 0.4
    y = torch.tanh(x @ teacher).view(n, horizon, action_dim)

    policy = PlanProposalPolicy(in_dim=in_dim, hidden=64, plan_horizon=horizon, action_dim=action_dim)
    stats = train_plan_policy_supervised(
        policy,
        x,
        y,
        steps=200,
        batch_size=32,
        lr=1e-3,
        seed=0,
    )
    assert stats.final_loss < stats.initial_loss * 0.35
