"""Stage C: train TokenlessLatentBus via InfoNCE between JEPA pooled features
and DCA working memory."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from latent_bus import TokenlessLatentBus, contrastive_infonce
from system1_jepa import (
    BLAJEPAModel,
    ImageBatchSpec,
    JEPAConfig,
    make_image_loader,
    pool_patch_tokens,
)
from system2_dca import DCAConfig, DCAEngine
from tensor_ram import DifferentiableTensorRAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data", choices=["cifar10", "synthetic"], default="cifar10")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ram-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ckpt = torch.load(args.jepa_checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    d = cfg_dict["d"]
    image_size = cfg_dict["image_size"]

    jepa_cfg = JEPAConfig(
        d_jepa=d, in_channels=3, patch_size=cfg_dict["patch_size"],
        encoder_depth=cfg_dict["encoder_depth"], encoder_heads=cfg_dict["encoder_heads"],
        predictor_depth=cfg_dict["predictor_depth"], predictor_heads=cfg_dict["predictor_heads"],
        action_dim=d, ema_tau=cfg_dict["ema_tau"], sigreg_weight=cfg_dict["sigreg_weight"],
        target_mask_ratio=cfg_dict["target_ratio"], dtype="float32",
    )
    jepa = BLAJEPAModel(jepa_cfg).to(device)
    jepa.target_encoder.load_state_dict(ckpt["target"])
    encoder = jepa.target_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dca_cfg = DCAConfig(d_core=d, d_ram=d, ssm_layers=2, dit_layers=2, heads=8, vocab_size=512, memory_tokens=4, dtype="float32")
    ram = DifferentiableTensorRAM(d_ram=d).to(device)
    ram.add_random(args.ram_size)
    dca = DCAEngine(dca_cfg, ram=ram).to(device)
    bus = TokenlessLatentBus(d_jepa=d, d_core=d, dtype=torch.float32).to(device)

    optim = torch.optim.AdamW(bus.parameters(), lr=args.lr, weight_decay=0.01)
    spec = ImageBatchSpec(image_size=image_size, batch_size=args.batch_size)
    loader = make_image_loader(spec, source=args.data, seed=0)

    history_log = []
    for step in range(args.steps):
        images = next(loader).to(device, non_blocking=True)
        with torch.no_grad():
            z, _, _ = encoder(images)
            z_pooled = pool_patch_tokens(z)
            core_query = bus.forward_up(z_pooled.unsqueeze(1)).squeeze(1).detach()
            memory = dca.working_memory(core_query, dca.fetch_facts(core_query))

        core = bus.forward_up(z_pooled.unsqueeze(1)).squeeze(1)
        loss = contrastive_infonce(core, memory)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bus.parameters(), 1.0)
        optim.step()
        if (step + 1) % args.log_every == 0:
            entry = {"step": step + 1, "alignment_loss": float(loss.detach())}
            history_log.append(entry)
            print(json.dumps(entry), flush=True)

    torch.save({
        "bus": bus.state_dict(),
        "history": history_log,
        "step": args.steps,
        "d": d,
    }, args.output)
    print(json.dumps({"event": "saved", "path": args.output}), flush=True)


if __name__ == "__main__":
    main()
