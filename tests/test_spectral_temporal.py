import torch

from system1_jepa.spectral_temporal import (
    SpectralAugmentedTemporalPredictor,
    SpectralBlendTemporalPredictor,
    SpectralFeatureTemporalPredictor,
    SpectralResidualTemporalPredictor,
    SpectralTemporalConfig,
    carrier_prior,
)
from system1_jepa.temporal import TemporalConfig


def test_affine_carrier_prior_extrapolates_linear_history():
    base = torch.randn(3, 5)
    slope = torch.randn(3, 5) * 0.1
    t = torch.arange(4, dtype=base.dtype).view(1, 4, 1)
    history = base[:, None, :] + slope[:, None, :] * t

    pred = carrier_prior(history, "affine")

    assert torch.allclose(pred, base + slope * 4.0, atol=1e-5)


def test_spectral_residual_temporal_predictor_shape():
    temporal_cfg = TemporalConfig.tiny()
    model = SpectralResidualTemporalPredictor(
        SpectralTemporalConfig(temporal=temporal_cfg, prior_kind="affine")
    )
    z = torch.randn(2, 4, temporal_cfg.d)
    actions = torch.randn(2, temporal_cfg.chunk_size, temporal_cfg.action_dim)

    out = model(z, actions)

    assert out.shape == (2, temporal_cfg.d)


def test_spectral_blend_temporal_predictor_shape():
    temporal_cfg = TemporalConfig.tiny()
    model = SpectralBlendTemporalPredictor(
        SpectralTemporalConfig(temporal=temporal_cfg, prior_kind="last")
    )
    z = torch.randn(2, 4, temporal_cfg.d)
    actions = torch.randn(2, temporal_cfg.chunk_size, temporal_cfg.action_dim)

    out = model(z, actions)

    assert out.shape == (2, temporal_cfg.d)


def test_spectral_feature_temporal_predictor_starts_at_prior():
    temporal_cfg = TemporalConfig.tiny()
    model = SpectralFeatureTemporalPredictor(
        SpectralTemporalConfig(temporal=temporal_cfg, prior_kind="last", residual_scale=1.0)
    )
    z = torch.randn(2, 4, temporal_cfg.d)
    actions = torch.randn(2, temporal_cfg.chunk_size, temporal_cfg.action_dim)

    out = model(z, actions)

    assert torch.allclose(out, z[:, -1], atol=1e-6)


def test_spectral_augmented_temporal_predictor_shape():
    temporal_cfg = TemporalConfig.tiny()
    model = SpectralAugmentedTemporalPredictor(
        SpectralTemporalConfig(temporal=temporal_cfg, prior_kind="last")
    )
    z = torch.randn(2, 4, temporal_cfg.d)
    actions = torch.randn(2, temporal_cfg.chunk_size, temporal_cfg.action_dim)

    out = model(z, actions)

    assert out.shape == (2, temporal_cfg.d)
