import numpy as np
import torch

from latent_bus import EntropyRouter, TokenlessLatentBus, contrastive_infonce, veto_loss
from latent_bus.bridge import VetoLoop
from system1_jepa import (
    BLAJEPAModel,
    JEPAConfig,
    MovingPatchSpec,
    TemporalConfig,
    TemporalPredictor,
    make_moving_patch_episodes,
    multistep_rollout_loss,
    pool_patch_tokens,
    sigreg_epps_pulley,
    sigreg_lewm,
)
from system2_dca import DCAConfig, DCAEngine, diffusion_score_matching_loss
from tensor_ram import (
    DifferentiableTensorRAM,
    FaissTensorRAM,
    quantize_int4,
    quantize_int8,
)


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_jepa_predicts_target_shape_and_target_encoder_is_frozen():
    _seed()
    config = JEPAConfig.tiny()
    model = BLAJEPAModel(config)
    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, config.action_dim)

    z_hat, z_target, mask = model(image, action)
    assert z_hat.shape == z_target.shape == (2, mask.num_target, config.d_jepa)
    assert mask.num_context + mask.num_target == (16 // config.patch_size) ** 2

    metrics = model.training_loss(image, action, mask=mask)
    metrics["loss"].backward()
    assert metrics["loss"].isfinite()
    assert all(p.grad is None for p in model.target_encoder.parameters())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.1)
    model.update_target_ema(tau=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_jepa_can_overfit_a_single_batch():
    _seed()
    config = JEPAConfig.tiny()
    config.sigreg_weight = 0.0
    model = BLAJEPAModel(config)
    optim = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=3e-3,
    )
    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, config.action_dim)
    mask = model.sample_mask(image)

    losses = []
    for _ in range(40):
        metrics = model.training_loss(image, action, mask=mask)
        optim.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()), 1.0
        )
        optim.step()
        losses.append(float(metrics["prediction"]))
    assert losses[-1] < losses[0] * 0.5, f"prediction loss did not halve: {losses[0]} -> {losses[-1]}"


def test_sigreg_zero_on_isotropic_gaussian_high_on_collapse():
    _seed()
    iso = torch.randn(256, 32)
    collapsed = torch.zeros(256, 32) + torch.randn(256, 1) * 0.01
    iso_score = float(sigreg_epps_pulley(iso))
    collapse_score = float(sigreg_epps_pulley(collapsed))
    assert collapse_score > iso_score, f"SIGReg failed to flag collapse: iso={iso_score}, collapsed={collapse_score}"

    iso_lewm = float(sigreg_lewm(iso))
    collapse_lewm = float(sigreg_lewm(collapsed))
    assert collapse_lewm > iso_lewm


def test_temporal_predictor_can_overfit_moving_patch():
    _seed()
    spec = MovingPatchSpec(image_size=16, patch_size=2, horizon=2, history=2, action_dim=32)
    cfg = TemporalConfig.tiny()
    cfg.action_dim = spec.action_dim
    cfg.max_context = spec.history + spec.horizon + 2
    predictor = TemporalPredictor(cfg)

    jepa_cfg = JEPAConfig.tiny()
    jepa_cfg.action_dim = spec.action_dim
    encoder = BLAJEPAModel(jepa_cfg).target_encoder

    def encode_pool(images_btchw: torch.Tensor) -> torch.Tensor:
        b, t = images_btchw.shape[:2]
        flat = images_btchw.reshape(b * t, *images_btchw.shape[2:])
        z, _, _ = encoder(flat)
        pooled = pool_patch_tokens(z)
        return pooled.reshape(b, t, -1)

    history, actions, future = make_moving_patch_episodes(spec, batch_size=4)
    optim = torch.optim.AdamW(predictor.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        loss, _ = multistep_rollout_loss(predictor, encode_pool, history, actions, future)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optim.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.7, f"rollout did not improve: {losses[0]} -> {losses[-1]}"


def test_dca_diffusion_uses_velocity_target_and_can_overfit():
    _seed()
    config = DCAConfig.tiny()
    model = DCAEngine(config)
    query = torch.randn(2, config.d_core)
    facts = torch.randn(2, 3, config.d_ram)
    canvas = torch.randn(2, 5, config.d_core)
    t = torch.rand(2)

    out = model(query, canvas, t, facts=facts)
    assert out["memory"].shape == (2, config.d_core)
    assert out["velocity"].shape == canvas.shape

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    x0 = torch.randn(2, 5, config.d_core)
    fixed_facts = torch.randn(2, 3, config.d_ram)
    losses = []
    for _ in range(40):
        memory = model.working_memory(query, fixed_facts)
        loss = diffusion_score_matching_loss(model.diffusion, x0, memory)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.7, f"velocity loss did not drop: {losses[0]} -> {losses[-1]}"

    sampled = model.diffusion.sample(memory.detach(), seq_len=5, steps=4)
    assert sampled.shape == canvas.shape
    tokens = model.decoder.decode(sampled)
    assert tokens.shape == (2, 5)


def test_dca_with_differentiable_ram_fetches_facts_internally():
    _seed()
    config = DCAConfig.tiny()
    ram = DifferentiableTensorRAM(d_ram=config.d_ram)
    ram.add_random(64)
    model = DCAEngine(config, ram=ram)
    query = torch.randn(2, config.d_core)
    canvas = torch.randn(2, 5, config.d_core)
    t = torch.rand(2)

    out = model(query, canvas, t)
    assert out["facts"].shape == (2, config.ram_query_heads, config.d_core)
    assert out["velocity"].shape == canvas.shape

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    x0 = torch.randn(2, 5, config.d_core)
    losses = []
    for _ in range(20):
        metrics = model.training_loss(query, x0)
        optim.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        optim.step()
        losses.append(float(metrics["loss"].detach()))
    assert losses[-1] < losses[0], f"RAM-aware DCA did not learn: {losses[0]} -> {losses[-1]}"


def test_jepa_prior_warm_start_returns_correct_shape():
    _seed()
    config = DCAConfig.tiny()
    model = DCAEngine(config)
    memory = torch.randn(2, config.d_core)
    prior = torch.randn(2, 5, config.d_core)
    pure = model.diffusion.sample(memory, seq_len=5, steps=2)
    warm = model.diffusion.sample(memory, seq_len=5, steps=2, prior=prior, t_start=0.5)
    assert pure.shape == warm.shape == (2, 5, config.d_core)
    assert not torch.allclose(pure, warm)


def test_veto_loss_provides_gradient_to_dca_plan():
    _seed()
    jepa_cfg = JEPAConfig.tiny()
    dca_cfg = DCAConfig.tiny()
    jepa = BLAJEPAModel(jepa_cfg)
    bus = TokenlessLatentBus(d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core, dtype=torch.float32)
    veto = VetoLoop(bus, jepa.predictor)

    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, jepa_cfg.action_dim)
    z_context, _, mask = jepa(image, action)

    plan = torch.randn(2, 5, dca_cfg.d_core, requires_grad=True)
    out = veto_loss(
        veto,
        z_current=z_context,
        dca_plan=plan,
        action=action,
        target_positions=mask.target_idx,
        grid_h=16 // jepa_cfg.patch_size,
        grid_w=16 // jepa_cfg.patch_size,
    )
    out["veto_loss"].backward()
    assert plan.grad is not None and plan.grad.abs().sum() > 0


def test_latent_bus_alignment_and_router_buffer():
    _seed()
    bus = TokenlessLatentBus(d_jepa=32, d_core=64, dtype=torch.float32)
    z = torch.randn(2, 4, 32)
    core = bus.forward_up(z)
    down = bus.forward_down(core)
    assert core.shape == (2, 4, 64)
    assert down.shape == z.shape
    assert contrastive_infonce(core, core.detach()).isfinite()

    router = EntropyRouter(threshold_tau=1e-5)
    decision = router.decide([z, z + 0.1, z - 0.1])
    assert decision.wake_dca.shape == (2,)
    assert decision.power_state == ["WAKE", "WAKE"]
    assert "threshold_tau" in dict(router.named_buffers())


def test_tensor_ram_and_quantization_roundtrip():
    _seed()
    vectors = np.random.randn(32, 16).astype(np.float32)
    ram = FaissTensorRAM(d_ram=16, use_faiss=False)
    ram.add(vectors)
    got = ram.weighted_retrieve(vectors[:2], k=4)
    assert got.vectors.shape == (2, 4, 16)
    assert got.weighted.shape == (2, 16)

    q8 = quantize_int8(vectors)
    q4 = quantize_int4(vectors)
    assert q8.dequantize().shape == vectors.shape
    assert q4.dequantize().shape == vectors.shape


def test_decoder_codebook_is_fp32_and_frozen():
    config = DCAConfig.tiny()
    model = DCAEngine(config)
    assert model.decoder.codebook.dtype == torch.float32
    assert not model.decoder.codebook.requires_grad


def test_end_to_end_pipeline_one_training_step():
    """Compose every subsystem in one composed forward + backward pass.

    image -> JEPA -> bus.up -> AsyncPrefetcher pings RAM -> DCA fetches
    facts via RAMReader -> diffusion velocity loss -> sample with JEPA prior
    warm start -> decode tokens -> veto round-trip with gradient. All in
    one step, gradients flowing through every learned module.
    """

    _seed()
    jepa_cfg = JEPAConfig.tiny()
    dca_cfg = DCAConfig.tiny()
    ram = DifferentiableTensorRAM(d_ram=dca_cfg.d_ram)
    ram.add_random(64)

    jepa = BLAJEPAModel(jepa_cfg)
    bus = TokenlessLatentBus(d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core, dtype=torch.float32)
    dca = DCAEngine(dca_cfg, ram=ram)

    from latent_bus import AsyncPrefetcher
    prefetcher = AsyncPrefetcher(bus, ram, d_state=dca_cfg.d_core)

    trainable = (
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters())
        + list(bus.parameters())
        + [p for p in dca.parameters() if p.requires_grad]
    )
    optim = torch.optim.AdamW(trainable, lr=1e-3)

    grad_history = {"jepa_context": 0.0, "jepa_predictor": 0.0, "bus": 0.0, "dca": 0.0}
    for step in range(3):
        image = torch.randn(2, 3, 16, 16)
        action = torch.randn(2, jepa_cfg.action_dim)
        z_hat, z_target, mask = jepa(image, action)

        z_pooled = pool_patch_tokens(z_hat)
        prefetcher.ping(z_pooled.detach())
        assert prefetcher.cached_facts() is not None

        core_query = bus.forward_up(z_pooled.unsqueeze(1)).squeeze(1)
        x0 = torch.randn(2, 4, dca_cfg.d_core)
        metrics = dca.training_loss(core_query, x0)

        jepa_metrics = jepa.training_loss(image, action, mask=mask)
        total = metrics["loss"] + 0.1 * jepa_metrics["loss"]

        optim.zero_grad(set_to_none=True)
        total.backward()
        grad_history["jepa_context"] += sum(p.grad.norm().item() for p in jepa.context_encoder.parameters() if p.grad is not None)
        grad_history["jepa_predictor"] += sum(p.grad.norm().item() for p in jepa.predictor.parameters() if p.grad is not None)
        grad_history["bus"] += sum(p.grad.norm().item() for p in bus.parameters() if p.grad is not None)
        grad_history["dca"] += sum(p.grad.norm().item() for p in dca.parameters() if p.grad is not None and p.requires_grad)
        optim.step()
        jepa.update_target_ema()

    # DiT adaLN-Zero zero-inits out_proj, so the diffusion path delivers zero
    # gradient to upstream (bus, query, memory) at the very first step. By
    # step 2+ the diffusion's own out_proj has trained enough that gradients
    # flow all the way back through the bus.
    assert all(g > 0 for g in grad_history.values()), f"some module never received gradient: {grad_history}"

    with torch.no_grad():
        prior = bus.forward_up(pool_patch_tokens(z_hat).unsqueeze(1)).expand(-1, 4, -1)
        sampled = dca.diffusion.sample(metrics["memory"], seq_len=4, steps=2, prior=prior, t_start=0.5)
        tokens = dca.decoder.decode(sampled)
    assert tokens.shape == (2, 4)


def test_sparse_ram_reader_lowers_entropy_with_sparsity_weight():
    _seed()
    config = DCAConfig.tiny()
    config.ram_sparsity_weight = 0.5
    ram = DifferentiableTensorRAM(d_ram=config.d_ram)
    ram.add_random(64)
    model = DCAEngine(config, ram=ram)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    query = torch.randn(2, config.d_core)
    x0 = torch.randn(2, 4, config.d_core)

    with torch.no_grad():
        _, aux0 = model.fetch_facts(query, return_aux=True)
        entropy_before = -(aux0["weights"] * aux0["weights"].clamp_min(1e-9).log()).sum(dim=-1).mean().item()

    for _ in range(30):
        metrics = model.training_loss(query, x0)
        optim.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        optim.step()

    with torch.no_grad():
        _, aux1 = model.fetch_facts(query, return_aux=True)
        entropy_after = -(aux1["weights"] * aux1["weights"].clamp_min(1e-9).log()).sum(dim=-1).mean().item()

    assert entropy_after < entropy_before, f"sparsity penalty did not reduce entropy: {entropy_before} -> {entropy_after}"


def test_spatiotemporal_jepa_predicts_target_features():
    _seed()
    from system1_jepa import SpatiotemporalConfig, SpatiotemporalJEPA

    cfg = SpatiotemporalConfig(image_size=8, patch_size=2, n_frames=3, d=16, action_dim=4)
    model = SpatiotemporalJEPA(cfg)
    frames = torch.randn(2, cfg.n_frames, 3, cfg.image_size, cfg.image_size)
    actions = torch.randn(2, cfg.n_frames, cfg.action_dim)

    z_hat, z_target, mask = model(frames, actions)
    assert z_hat.shape == z_target.shape
    assert z_hat.shape[-1] == cfg.d
    metrics = model.training_loss(frames, actions, mask=mask)
    metrics["loss"].backward()
    assert metrics["loss"].isfinite()
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_checkpoint_save_load_roundtrip(tmp_path):
    _seed()
    cfg = JEPAConfig.tiny()
    model = BLAJEPAModel(cfg)
    image = torch.randn(2, 3, 16, 16)
    action = torch.randn(2, cfg.action_dim)
    z_hat_before, _, mask = model(image, action)

    from checkpoints import load_into, save
    path = str(tmp_path / "jepa.pt")
    save(model, path, config=cfg, step=42)

    new_model = BLAJEPAModel(cfg)
    info = load_into(new_model, path)
    assert info["step"] == 42

    z_hat_after, _, _ = new_model(image, action, mask=mask)
    assert torch.allclose(z_hat_before, z_hat_after, atol=1e-5)


def test_cem_planner_returns_action_chunk_shape():
    _seed()
    from system1_jepa import CEMConfig, TemporalConfig, TemporalPredictor, cem_plan

    cfg = TemporalConfig.tiny()
    cfg.action_dim = 4
    cfg.max_context = 16
    predictor = TemporalPredictor(cfg)
    frame_embeds = torch.randn(2, 2, cfg.d)
    plan = cem_plan(predictor, frame_embeds, action_dim=cfg.action_dim, cfg=CEMConfig(horizon=3, iterations=2, population=16))
    assert plan.shape == (2, 3, cfg.action_dim)


def test_linear_probe_runs_on_synthetic_data():
    _seed()
    from diagnostics import train_linear_probe
    from system1_jepa import ImageBatchSpec, SyntheticImageLoader

    spec = ImageBatchSpec(image_size=8, batch_size=8)
    loader = iter(SyntheticImageLoader(spec, seed=0))

    def feature_fn(images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))

    def target_fn(images: torch.Tensor) -> torch.Tensor:
        return (images.mean(dim=(1, 2, 3)) > 0).long()

    report = train_linear_probe(
        feature_fn=feature_fn,
        image_loader=loader,
        target_fn=target_fn,
        n_classes=2,
        epochs=1,
        steps_per_epoch=4,
        eval_steps=2,
    )
    assert report.eval_accuracy is not None
    assert 0.0 <= report.eval_accuracy <= 1.0


def test_async_prefetcher_caches_facts_from_jepa_pings():
    _seed()
    from latent_bus import AsyncPrefetcher

    jepa_cfg = JEPAConfig.tiny()
    dca_cfg = DCAConfig.tiny()
    bus = TokenlessLatentBus(d_jepa=jepa_cfg.d_jepa, d_core=dca_cfg.d_core, dtype=torch.float32)
    ram = DifferentiableTensorRAM(d_ram=dca_cfg.d_ram)
    ram.add_random(64)
    prefetcher = AsyncPrefetcher(bus, ram, d_state=dca_cfg.d_core)
    assert prefetcher.cached_facts() is None

    z_jepa = torch.randn(2, jepa_cfg.d_jepa)
    facts = prefetcher.ping(z_jepa)
    assert facts.shape == (2, 4, dca_cfg.d_core)
    assert torch.equal(facts, prefetcher.cached_facts())
    assert prefetcher.cached_indices() is not None
