"""Unit tests for the Goal-Conditioned Inverse Dynamics Model (gc_idm.py).

Robosuite-free: validates shapes, AdaLN-Zero init, both modes, learnability of a
known inverse map, and config round-trip. The end-to-end env comparison vs the
locked recipe lives in a scripts/ phase runner (see docs/GC_IDM_OFJEPA.md).
"""
import torch

from system1_jepa.gc_idm import (
    GCInverseDynamics, train_gc_idm_supervised, act,
    gc_idm_config, build_gc_idm_from_config, sinusoidal_embed,
)


def test_flatten_shapes_and_horizon_zero_init():
    head = GCInverseDynamics(state_dim=24, action_dim=7, hidden=128, mode="flatten")
    z_t, z_g = torch.randn(4, 24), torch.randn(4, 24)
    out = head(z_t, z_g, torch.tensor([1, 5, 10, 50]))
    assert out.shape == (4, 7)
    assert out.abs().max() <= 1.0 + 1e-5  # tanh-bounded
    # AdaLN-Zero: at init the horizon path is zero, so changing horizon must not
    # change the output (gamma=beta=0 => identity modulation). eval() to disable
    # dropout, which would otherwise mask this.
    head.eval()
    o1 = head(z_t, z_g, torch.full((4,), 1.0))
    o2 = head(z_t, z_g, torch.full((4,), 49.0))
    assert torch.allclose(o1, o2, atol=1e-6)


def test_perfile_shapes_and_confidence():
    head = GCInverseDynamics(state_dim=16, action_dim=7, hidden=128, mode="perfile")
    st, gl = torch.randn(3, 12, 16), torch.randn(3, 12, 16)
    conf = torch.rand(3, 12)
    out = head(st, gl, torch.tensor([2, 8, 30]), conf)
    assert out.shape == (3, 7)
    # confidence changes the pooled action
    out2 = head(st, gl, torch.tensor([2, 8, 30]), torch.rand(3, 12))
    assert not torch.allclose(out, out2)


def test_learns_known_inverse_map():
    torch.manual_seed(0)
    D, A, N = 16, 7, 2000
    W = torch.randn(A, 2 * D) * 0.1
    z_t, z_g = torch.randn(N, D), torch.randn(N, D)
    h = torch.randint(1, 50, (N,)).float()
    a = torch.tanh(torch.cat([z_t, z_g], -1) @ W.T)
    head = GCInverseDynamics(state_dim=D, action_dim=A, hidden=128, mode="flatten")
    st = train_gc_idm_supervised(head, z_t, z_g, h, a, steps=600, batch_size=128)
    assert st.final_val_loss < 0.5 * st.initial_val_loss


def test_act_single_step_and_config_roundtrip():
    head = GCInverseDynamics(state_dim=20, action_dim=7, mode="flatten")
    a = act(head, torch.randn(20), torch.randn(20), 5)
    assert a.shape == (7,)
    rebuilt = build_gc_idm_from_config(gc_idm_config(head))
    assert rebuilt.state_dim == 20 and rebuilt.mode == "flatten"


def test_sinusoidal_embed():
    e = sinusoidal_embed(torch.tensor([1.0, 5.0, 10.0]), dim=64)
    assert e.shape == (3, 64)
