import numpy as np
import torch

from latent_bus import EntropyRouter, TokenlessLatentBus, contrastive_infonce
from system1_jepa import BLAJEPAModel, JEPAConfig
from system2_dca import DCAConfig, DCAEngine
from tensor_ram import FaissTensorRAM, quantize_fp4, quantize_int8


def test_jepa_loss_and_ema_target_is_frozen():
    config = JEPAConfig.tiny()
    model = BLAJEPAModel(config)
    masked = torch.randn(2, 3, 16, 16)
    unmasked = torch.randn_like(masked)
    action = torch.randn(2, config.action_dim)

    metrics = model.training_loss(masked, unmasked, action)
    metrics["loss"].backward()

    assert metrics["loss"].isfinite()
    assert all(param.grad is None for param in model.target_encoder.parameters())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.1)
    model.update_target_ema(tau=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_dca_diffusion_and_decoder_shapes():
    config = DCAConfig.tiny()
    model = DCAEngine(config)
    query = torch.randn(2, config.d_core)
    facts = torch.randn(2, 3, config.d_ram)
    canvas = torch.randn(2, 5, config.d_core)
    t = torch.rand(2)

    out = model(query, facts, canvas, t)
    assert out["memory"].shape == (2, config.d_core)
    assert out["eps_pred"].shape == canvas.shape
    tokens = model.decoder.decode(canvas.to(dtype=config.torch_dtype))
    assert tokens.shape == (2, 5)


def test_latent_bus_alignment_and_router():
    bus = TokenlessLatentBus(d_jepa=32, d_core=64)
    z = torch.randn(2, 4, 32)
    core = bus.forward_up(z)
    down = bus.forward_down(core)
    assert core.shape == (2, 4, 64)
    assert down.shape == z.shape
    assert contrastive_infonce(core, core.detach()).isfinite()

    router = EntropyRouter(threshold_tau=0.00001)
    decision = router.decide([z, z + 0.1, z - 0.1])
    assert decision.wake_dca.shape == (2,)
    assert decision.power_state == ["WAKE", "WAKE"]


def test_tensor_ram_and_quantization():
    vectors = np.random.randn(32, 16).astype(np.float32)
    ram = FaissTensorRAM(d_ram=16, use_faiss=False)
    ram.add(vectors)
    got = ram.weighted_retrieve(vectors[:2], k=4)
    assert got.vectors.shape == (2, 4, 16)
    assert got.weighted.shape == (2, 16)

    q8 = quantize_int8(vectors)
    q4 = quantize_fp4(vectors)
    assert q8.dequantize().shape == vectors.shape
    assert q4.dequantize().shape == vectors.shape
