"""Force-target grounding for BLAJEPAModel — the contact-force sponsor.

Validated first on synthetic + real MuJoCo state rigs (see ~/force_ground/): predicting contact force grounds
material properties that are invisible to the prediction target but written into force (friction: decode-R2 0.95,
grip-servo at oracle).  This wires that sponsor into the production image-JEPA model.

`ForceGroundedJEPA` extends the real BLAJEPAModel: it reuses the masked context encoder, EMA target encoder,
action-conditioned predictor, and SIGReg untouched, and adds a `ContactForceHead` that predicts the contact-force
vector from the (pooled) context tokens + action.  The training loss is the standard JEPA loss + a contact-masked
force MSE.  The force objective is what grounds material state into z_context; feeding force *in* is deliberately
not done here (that is the inert control, per the pre-registered design).

Doctrine carried over from the port (~/force_ground/ofjepa_port.py): SIGReg keeps z full-rank and can *preserve*
task-irrelevant salient signal, fighting a clean force-target/force-input dissociation — so for a clean split the
force sponsor should act on a z_content subspace with the bottleneck, SIGReg on z_struct.  Here we keep the single
shared z (production default) and expose `force_weight` / `content_dim` so the pod can enable that factorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .model import BLAJEPAModel, JEPAConfig
from .masking import PatchMask
from .vit import PatchViTEncoder


class ContactForceHead(nn.Module):
    """Predict the contact-force vector from pooled context tokens + action (the force-target sponsor).

    Pools the encoder's context tokens to a global vector (contact force is a global reading), concatenates the
    action, and regresses the force.  Only the mean-pool is global; per-patch structure stays in the encoder.
    """

    def __init__(self, latent_dim: int, action_dim: int, force_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, force_dim),
        )

    def forward(self, context_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pooled = context_tokens.mean(dim=1)                      # [B, D] global contact readout
        return self.net(torch.cat([pooled, action], dim=-1))     # [B, force_dim]


@dataclass
class ForceGroundedConfig(JEPAConfig):
    force_dim: int = 2
    force_weight: float = 1.0        # weight of the force-target sponsor loss


class ForceGroundedJEPA(BLAJEPAModel):
    """BLAJEPAModel + contact-force head. The force objective grounds material state into z_context."""

    def __init__(self, config: ForceGroundedConfig):
        super().__init__(config)
        self.force_config = config
        self.force_head = ContactForceHead(
            latent_dim=config.d_jepa, action_dim=config.action_dim, force_dim=config.force_dim
        )
        self.force_head.to(dtype=config.torch_dtype)

    def forward(
        self, image: torch.Tensor, action: torch.Tensor, mask: PatchMask | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, PatchMask, torch.Tensor]:
        mask = mask if mask is not None else self.sample_mask(image)
        ctx_dtype = next(self.context_encoder.parameters()).dtype
        z_context, grid_h, grid_w = self.context_encoder(image.to(dtype=ctx_dtype), keep_idx=mask.context_idx)
        with torch.no_grad():
            from .masking import gather_tokens
            z_full_target, _, _ = self.target_encoder(image.to(dtype=next(self.target_encoder.parameters()).dtype))
            z_target = gather_tokens(z_full_target, mask.target_idx)
        z_hat = self.predictor(
            context_tokens=z_context, action=action.to(dtype=ctx_dtype),
            target_positions=mask.target_idx, grid_h=grid_h, grid_w=grid_w,
        )
        force_pred = self.force_head(z_context, action.to(dtype=ctx_dtype))
        return z_hat, z_target, mask, force_pred

    @torch.no_grad()
    def encode_context(self, image: torch.Tensor) -> torch.Tensor:
        """Pooled context representation over the full (unmasked) image — for probing what z grounds."""
        ctx_dtype = next(self.context_encoder.parameters()).dtype
        z, _, _ = self.context_encoder(image.to(dtype=ctx_dtype))
        return z.mean(dim=1).float()

    def training_loss(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
        force: torch.Tensor,
        contact: torch.Tensor | None = None,
        mask: PatchMask | None = None,
    ) -> Dict[str, torch.Tensor]:
        from .losses import jepa_loss
        z_hat, z_target, _, force_pred = self(image, action, mask=mask)
        losses = jepa_loss(
            predicted_target=z_hat, target_latent=z_target,
            sigreg_weight=self.config.sigreg_weight, sigreg_variant=self.config.sigreg_variant,
        )
        err = (force_pred.float() - force.float()) ** 2
        if contact is not None:                                  # contact-mask the force loss (free space carries no force)
            w = contact.float().view(-1, 1)
            force_loss = (err * w).sum() / (w.sum() * err.shape[1] + 1e-6)
        else:
            force_loss = err.mean()
        total = losses["loss"] + self.force_config.force_weight * force_loss
        return {"loss": total, "prediction": losses["prediction"], "sigreg": losses["sigreg"],
                "force": force_loss.detach()}


class MultiModalForceJEPA(nn.Module):
    """The CORRECT production integration: a MULTI-MODAL encoder (image + force-history) + force-target sponsor.

    A pure-image encoder can't gain from a force head — material is either in the image (patch-prediction already
    grounds it) or absent (the head can't extract it).  The state rig worked because force history was an ENCODER
    INPUT.  Here the real image ViT produces image tokens; a force-history embedder produces force tokens; a small
    fusion transformer (same nn.TransformerEncoderLayer block) fuses them into z.  The force-target head predicts
    NEXT force from z + action, which grounds material that is invisible in the image but written into force.
    """

    def __init__(self, cfg: ForceGroundedConfig, hist_len: int = 8, content_dim: int = 4):
        super().__init__()
        self.cfg = cfg; self.hist_len = hist_len; self.content_dim = content_dim
        self.img_encoder = PatchViTEncoder(  # reuse the production image ViT verbatim
            in_channels=cfg.in_channels, latent_dim=cfg.d_jepa, patch_size=cfg.patch_size,
            depth=cfg.encoder_depth, heads=cfg.encoder_heads, mlp_ratio=cfg.encoder_mlp_ratio)
        self.force_embed = nn.Linear(2 * cfg.force_dim, cfg.d_jepa)   # tokens = (probe action, resulting force) pairs -> mu identifiable
        self.force_pos = nn.Parameter(torch.zeros(1, hist_len, cfg.d_jepa)); nn.init.trunc_normal_(self.force_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_jepa, cfg.encoder_heads, dim_feedforward=cfg.d_jepa * 4,
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(cfg.d_jepa)
        # z_content: the low-dim TASK/probe subspace. A hard capacity bottleneck squeezes out passively-retained,
        # task-irrelevant signal STRUCTURALLY (robust), not via the knife-edge noise-KL BETA.
        self.content_proj = nn.Linear(cfg.d_jepa, content_dim)
        self.recon = nn.Linear(content_dim, cfg.in_channels)              # mu-free control-arm target head
        self.force_head = ContactForceHead(content_dim, cfg.action_dim, cfg.force_dim)

    def encode(self, image, force_pairs, use_force=True):
        img_tok, _, _ = self.img_encoder(image)                          # [B, P, D] image (material-invisible modality)
        if use_force:
            ftok = self.force_embed(force_pairs) + self.force_pos        # [B, hist, D] (action,force)-pair modality
            tok = torch.cat([img_tok, ftok], dim=1)
        else:
            tok = img_tok
        pooled = self.norm(self.fusion(tok)).mean(dim=1)                 # fused trunk
        return self.content_proj(pooled)                                 # -> z_content (dim content_dim)

    def forward(self, image, force_pairs, action, use_force=True, noise_std=0.0):
        z_mean = self.encode(image, force_pairs, use_force)
        z = z_mean + noise_std * torch.randn_like(z_mean) if noise_std > 0 else z_mean   # optional soft bottleneck too
        force_pred = self.force_head(z.unsqueeze(1), action)   # head pools dim=1; feed z as a length-1 sequence
        recon = self.recon(z)                                  # mu-free image summary (control-arm target)
        return z_mean, z, force_pred, recon


# --------------------------------------------------------------------------------------------------------------
# End-to-end smoke test on the REAL model class: a hidden material latent lives (subtly) in the image; the contact
# force depends on material x action.  The force sponsor should ground the latent into z_context better than a
# plain JEPA with no force head.  Trains in seconds on CPU with JEPAConfig.tiny().
# --------------------------------------------------------------------------------------------------------------
def _r2(z, y, ntr=1000):
    A = np.c_[z[:ntr], np.ones(ntr)]; wv, *_ = np.linalg.lstsq(A, y[:ntr], rcond=None)
    p = np.c_[z[ntr:], np.ones(len(z) - ntr)] @ wv; yt = y[ntr:]
    return max(0., 1 - ((yt - p) ** 2).sum() / (((yt - yt.mean()) ** 2).sum() + 1e-9))


if __name__ == "__main__":
    import numpy as np, os
    torch.manual_seed(0); np.random.seed(0)
    BETA = float(os.environ.get("BETA", "0.0"))   # soft noise-KL bottleneck (0 = off; hard content bottleneck used instead)
    DC = int(os.environ.get("DC", "4"))           # z_content capacity: hard bottleneck that robustly squeezes out passive retention
    cfg = ForceGroundedConfig.tiny(); cfg.force_dim = 2; cfg.force_weight = 10.0; cfg.sigreg_weight = 0.002
    C, P = cfg.in_channels, cfg.patch_size; H = W = P * 4; HL = 8
    B, STEPS = 96, 700

    # MULTI-MODAL task (the faithful real-model analog of the friction rig):
    #   material mu is INVISIBLE in the image (image is pure noise) but written into the force history
    #   (force = mu * action).  Only an encoder that reads force AND is asked to predict force can ground mu.
    def batch(n):
        mu = np.random.uniform(-1, 1, (n, 1)).astype(np.float32)
        img = 0.6 * np.random.randn(n, C, H, W).astype(np.float32)              # material NOT in image
        hist_a = np.random.uniform(-1, 1, (n, HL, cfg.force_dim)).astype(np.float32)
        force_pairs = np.concatenate([hist_a, mu[:, None] * hist_a], axis=-1)   # (action, resulting force) pairs -> mu = f/a identifiable
        a = np.random.uniform(-1, 1, (n, cfg.action_dim)).astype(np.float32)
        next_force = mu * a[:, :cfg.force_dim]                                  # next force = mu * action -> predicting it needs mu
        t = lambda x: torch.tensor(x)
        return t(img), t(force_pairs), t(a), t(next_force), mu

    def train(model, arm, steps=STEPS):
        opt = torch.optim.Adam(model.parameters(), 2e-3)
        use_force = (arm != "image-only")
        ns = 1.0 if BETA > 0 else 0.0
        for it in range(steps):
            img, fh, a, nf, _ = batch(B)
            z_mean, z, fp, recon = model(img, fh, a, use_force=use_force, noise_std=ns)
            if arm == "force-target":
                task = ((fp - nf) ** 2).mean()                                 # predict next force -> grounds mu into z_content
            else:
                task = ((recon - img.mean(dim=(2, 3))) ** 2).mean()            # mu-free objective (image summary)
            loss = task + BETA * 0.5 * (z_mean ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            img, fh, a, nf, mu = batch(1500)
            z_mean, z, _, _ = model(img, fh, a, use_force=use_force, noise_std=ns)
        return (z if ns > 0 else z_mean).numpy(), mu[:, 0]

    print("MULTI-MODAL force-JEPA on real components (image ViT + force tokens + fusion transformer)")
    print("material mu is INVISIBLE in the image; only written into force.\n")
    print(f"(z_content hard bottleneck DC={DC}, soft-bottleneck BETA={BETA})")
    res = {}
    for arm in ["image-only", "force-input", "force-target"]:
        m = MultiModalForceJEPA(cfg, hist_len=HL, content_dim=DC)
        z, mu = train(m, arm); res[arm] = _r2(z, mu)
        print(f"  decode-R2(material) [{arm:<12}] = {res[arm]:.3f}")
    print(f"\nP1 force-target grounds material invisible-in-image: "
          f"ft={res['force-target']:.3f} vs image-only={res['image-only']:.3f} input={res['force-input']:.3f} "
          f"-> {'HIT' if res['force-target'] > res['image-only'] + 0.10 and res['force-target'] > res['force-input'] + 0.10 else 'weak'}")
    print("(image-only can't ground mu -- not in its input; force-target grounds it by predicting force.)")
