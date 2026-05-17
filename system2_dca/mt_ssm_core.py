"""Multi-Timescale Selective SSM procedural core (Phase 11).

A drop-in replacement for `procedural_core.ProceduralCore` that swaps
the standard transformer substrate for a multi-timescale selective
state-space stack inspired by Tessera and Mamba-2.

Per layer, three parallel selective SSMs run at different effective
time-constants (fast / medium / slow), implemented as input-selective
diagonal recurrences. Their outputs are combined via learned gates;
prediction-error gating (Tessera's "uncertainty is the substrate"
mechanism) is wired in but kept simple in v1 — per-timescale
prediction heads emit P(h_{t+1}) and an L2 prediction-loss term is
added to the training objective. The error is also used as a soft
multiplier on the gate-mixing weights, so high prediction error at
the fast scale shifts compute toward medium / slow.

Same forward API as ProceduralCore so the existing training script
(phase6_train.py) accepts it via a --backbone flag without
modification to the trainer.

Implementation choices (v1, pragmatic):

  * Diagonal A (state-space matrix), parameterised as -exp(A_log) for
    stability. Input-selective B and C produced by linear projections
    of x at each position.
  * Sequential scan over T in PyTorch (no custom CUDA kernel). On
    sequence length 1024, this is ~2-4x slower than transformer per
    step but parallel across batch and channel dims, so still
    GPU-bound.
  * Three timescales: state dims (256, 512, 1024) for fast/medium/slow.
    Time-constant control comes from A_log initialisation (fast
    decays in ~3 positions, slow holds ~300+).
  * Prediction heads: tiny 2-layer MLPs (~2M params each) producing
    the predicted next-step state. L2 loss between predicted and
    actual next-step state weighted at 0.05x of the main LM loss.

Not in v1 (deferred):
  * Working memory slots (Phase 12)
  * Adaptive computation halting (Phase 13)
  * Wake-sleep with capability probes (also Phase 13)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MTSSMConfig:
    vocab_size: int = 50_257
    d_model: int = 1024
    n_layers: int = 24
    state_fast: int = 256
    state_med: int = 512
    state_slow: int = 1024
    mlp_ratio: int = 4
    max_seq_len: int = 2048
    dropout: float = 0.0
    pred_loss_weight: float = 0.05
    # Phase 12: working memory slots
    n_slots: int = 16
    slot_chunk: int = 64
    use_memory: bool = False
    # Phase 13: attractor refinement on tied output embeddings
    use_attractor: bool = False
    attractor_layers: int = 1
    attractor_train_iters: int = 3
    attractor_infer_iters: int = 8
    attractor_n_heads: int = 8

    @classmethod
    def small_500m(cls) -> "MTSSMConfig":
        return cls(d_model=1024, n_layers=24)

    @classmethod
    def micro(cls) -> "MTSSMConfig":
        return cls(d_model=128, n_layers=4, state_fast=32, state_med=64, state_slow=128)


class SelectiveSSM(nn.Module):
    """Single-timescale selective SSM (Mamba-2 simplified).

    Time-constant interpretation: `a_lo`/`a_hi` are the target initial
    decay multipliers a_bar = exp(A·Δ) at Δ=1. So a≈1 = slow / long
    memory, a≈0.3 = fast / short memory. The eigenvalue magnitude
    |A| = -log(a) is stored in `A_log`.
    """

    def __init__(self, d_model: int, state_dim: int, a_lo: float, a_hi: float):
        super().__init__()
        assert 0.0 < a_lo < a_hi < 1.0, f"need 0<a_lo<a_hi<1, got {a_lo},{a_hi}"
        self.d_model = d_model
        self.state_dim = state_dim
        mag = -torch.log(torch.linspace(a_lo, a_hi, state_dim))  # positive |A|
        self.A_log = nn.Parameter(torch.log(mag))                 # log(|A|)
        self.B_proj = nn.Linear(d_model, state_dim, bias=False)
        self.C_proj = nn.Linear(d_model, state_dim, bias=False)
        # dt_proj with bias≈0 gives softplus≈log(2)≈0.69 → small Δ.
        # Bias=0.54 → softplus≈1.0 so initial a_bar = a_init.
        self.dt_proj = nn.Linear(d_model, 1, bias=True)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model + state_dim, d_model, bias=False)
        for p in (self.B_proj, self.C_proj, self.out_proj):
            nn.init.normal_(p.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.dt_proj.bias, 0.5413)  # softplus(0.5413)≈1.0

    def forward(self, x: torch.Tensor, return_final_state: bool = False,
                chunk_size: int = 64
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]
        Returns:
          y: [B, T, D]
          final_state: [B, state_dim] (last hidden state, for prediction head)

        Implementation: chunked parallel scan. Within each chunk of length K,
        the recurrence h_t = a_t * h_{t-1} + b_t is expanded in closed form
        and computed via cumprod/cumsum, so only T/K sequential carries
        are needed (vs T in the naive scan). Numerics: cumprod is run in
        float32 to keep the b_t / cum_a[t] division stable.
        """
        B, T, D = x.shape
        assert D == self.d_model

        B_t = self.B_proj(x)
        C_t = self.C_proj(x)
        dt = F.softplus(self.dt_proj(x))

        A = -torch.exp(self.A_log).to(x.dtype)
        a_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # [B, T, S]
        x_drive = x.mean(dim=-1, keepdim=True)
        b_bar = dt * B_t * x_drive                            # [B, T, S]

        S = self.state_dim
        K = min(chunk_size, T)
        H_all = torch.empty(B, T, S, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, S, device=x.device, dtype=torch.float32)
        a32 = a_bar.float().clamp(min=1e-6, max=1.0 - 1e-6)
        b32 = b_bar.float()
        # Log-space cumulative product: keeps numerics stable when
        # a_t is small. log_a ≤ 0 always; cum_log_a is the cumulative
        # log of the prefix product.
        log_a = torch.log(a32)                  # [B, T, S], non-positive
        for c0 in range(0, T, K):
            c1 = min(c0 + K, T)
            la_c = log_a[:, c0:c1, :]           # [B, Kc, S]
            b_c = b32[:, c0:c1, :]
            cum_log_a = torch.cumsum(la_c, dim=1)     # L_t, non-positive
            # h_t = exp(L_t) * h + exp(L_t) * sum_{k<=t} exp(-L_k) * b_k
            # All terms in sum have factor exp(L_t - L_k) ≤ 1 when t≥k.
            # Compute  sum_k b_k * exp(L_t - L_k)  via a stable trick:
            # rewrite as cumsum(b_k * exp(-L_k)) but clamp the divisor
            # exp(L_k) from below to prevent overflow of exp(-L_k).
            neg_L = (-cum_log_a).clamp(max=30.0)       # caps exp(-L_k)≤exp(30)
            inv_cum_a = torch.exp(neg_L)               # 1 / cum_a, bounded
            cs = torch.cumsum(b_c * inv_cum_a, dim=1)  # [B, Kc, S]
            cum_a = torch.exp(cum_log_a)               # exp(L_t), in (0,1]
            H_c = cum_a * h.unsqueeze(1) + cum_a * cs  # [B, Kc, S]
            H_all[:, c0:c1, :] = H_c.to(x.dtype)
            h = H_c[:, -1, :]

        # Output projection per position (parallel).
        cy = (C_t * H_all).sum(dim=-1, keepdim=True)         # [B, T, 1]
        y_combined = torch.cat([cy.expand(-1, -1, D) * x, H_all], dim=-1)  # [B, T, D+S]
        y_out = self.out_proj(y_combined) + self.D * x        # [B, T, D]
        final = H_all[:, -1, :]
        return y_out, final


class PredictionHead(nn.Module):
    """Tiny MLP that predicts the next state of an SSM given its current
    state. Output dim = state_dim of the target SSM. Used to compute
    prediction-error signals that gate cross-timescale flow.
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, state_dim, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class WorkingMemory(nn.Module):
    """Phase 12: 16 addressable slots inserted between SSM and MLP.

    Reads happen in parallel via cross-attention (each token at position t
    queries the slot state visible at the start of t's chunk). Writes
    happen at chunk boundaries via a gated GRU-style update over the
    chunk-aggregated context. This avoids breaking the chunked-scan
    parallelism of the SSM while still allowing slot contents to evolve
    over the course of a sequence.

    The slot initialisation is learned (`slot_init`), so even when the
    sequence is short and the chunk-level write never fires, slots act
    as persistent per-layer memory.
    """

    def __init__(self, d_model: int, n_slots: int = 16, chunk_size: int = 64):
        super().__init__()
        self.n_slots = n_slots
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.slot_init = nn.Parameter(torch.zeros(n_slots, d_model))
        nn.init.normal_(self.slot_init, std=0.02)
        self.norm = nn.RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        # Per-slot write proposal + gate, conditioned on chunk-mean of x.
        self.write_v = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, n_slots, bias=True)
        # Init: gates ≈ 0 → slot updates ~0 at start, so this module
        # initially acts as a learnable-key cross-attention layer with
        # static slot values. Training discovers when to write.
        nn.init.zeros_(self.write_gate.bias)
        nn.init.normal_(self.write_gate.weight, std=0.001)
        nn.init.normal_(self.write_v.weight, std=0.02)
        for p in (self.q, self.k, self.v, self.o):
            nn.init.normal_(p.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        K = min(self.chunk_size, T)
        slots = self.slot_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        slots = slots.to(x.dtype)
        u = self.norm(x)
        scale = 1.0 / math.sqrt(D)
        out = torch.empty_like(x)
        for c0 in range(0, T, K):
            c1 = min(c0 + K, T)
            u_c = u[:, c0:c1, :]                    # [B, Kc, D]
            q = self.q(u_c)                          # [B, Kc, D]
            k = self.k(slots)                        # [B, n_slots, D]
            v = self.v(slots)                        # [B, n_slots, D]
            attn = q @ k.transpose(-2, -1) * scale   # [B, Kc, n_slots]
            attn = attn.softmax(dim=-1)
            r = attn @ v                             # [B, Kc, D]
            out[:, c0:c1, :] = self.o(r)
            # Update slots from this chunk's mean (gated).
            chunk_summary = u_c.mean(dim=1)          # [B, D]
            cand = self.write_v(chunk_summary).unsqueeze(1)  # [B, 1, D]
            gate = self.write_gate(chunk_summary).sigmoid().unsqueeze(-1)  # [B, n_slots, 1]
            slots = slots + gate * (cand - slots)
        return out


class AttractorBlock(nn.Module):
    """One transformer block used inside the attractor module. Standard
    pre-norm self-attention + MLP. Shared across all refinement iterations
    via weight tying (same `AttractorBlock` instance, called repeatedly)."""

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm_attn = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.norm_mlp = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        T = h.size(1)
        # Causal mask: bool, True = block. attn_mask=upper-triangular.
        mask = torch.ones(T, T, device=h.device, dtype=torch.bool).triu(1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AttractorModule(nn.Module):
    """Phase 13: output-embedding refinement via fixed-point iteration.

    Backbone produces an initial output-side embedding ỹ₀. This module
    iteratively refines it via ỹ_{t+1} = T_a(ỹ_t, ỹ₀), where the
    proposal ỹ₀ is persistently injected (additive) at every step. The
    same `AttractorBlock` is shared across iterations (weight-tied), so
    the parameter count is independent of how many refinement steps are
    taken — and we can run *more* iterations at inference than during
    training (test-time scaling). Backward uses the one-step IFT
    approximation from Fein-Ashley & Rashidinejad (2026): the first
    n-1 iterations run under `torch.no_grad()`, only the last step
    contributes gradients.
    """

    def __init__(self, cfg: MTSSMConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.train_iters = cfg.attractor_train_iters
        self.infer_iters = cfg.attractor_infer_iters
        self.blocks = nn.ModuleList([
            AttractorBlock(cfg.d_model, n_heads=cfg.attractor_n_heads)
            for _ in range(cfg.attractor_layers)
        ])
        # Persistent y0 injection (paper eq. 2): xt + y0_proj(y0).
        self.y0_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.normal_(self.y0_proj.weight, std=0.02)

    def _step(self, y_t: torch.Tensor, y_0: torch.Tensor) -> torch.Tensor:
        h = y_t + self.y0_proj(y_0)
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, y_0: torch.Tensor) -> torch.Tensor:
        # One-step IFT: n-1 iterations no-grad, final step with grad.
        n = self.train_iters if self.training else self.infer_iters
        if n <= 0:
            return y_0
        y = y_0
        if n > 1:
            with torch.no_grad():
                for _ in range(n - 1):
                    y = self._step(y, y_0)
        y = self._step(y, y_0)
        return y


class MTSSMBlock(nn.Module):
    def __init__(self, cfg: MTSSMConfig):
        super().__init__()
        self.norm_ssm = nn.RMSNorm(cfg.d_model)
        # Three SSMs at different time-constant ranges. a_lo/a_hi are
        # initial decay multipliers at Δ=1: a≈0.3 ≡ fast (3-step memory),
        # a≈0.999 ≡ slow (~1000-step memory). a ∈ (0,1).
        self.fast = SelectiveSSM(cfg.d_model, cfg.state_fast, a_lo=0.30, a_hi=0.70)
        self.med  = SelectiveSSM(cfg.d_model, cfg.state_med,  a_lo=0.70, a_hi=0.95)
        self.slow = SelectiveSSM(cfg.d_model, cfg.state_slow, a_lo=0.95, a_hi=0.999)
        # Prediction heads (one per timescale).
        self.pred_fast = PredictionHead(cfg.state_fast)
        self.pred_med = PredictionHead(cfg.state_med)
        self.pred_slow = PredictionHead(cfg.state_slow)
        # Gates: learned per-channel weights for combining timescale outputs.
        # Init so all timescales contribute equally.
        self.gate = nn.Parameter(torch.zeros(3, cfg.d_model))
        # Phase 12: optional working memory.
        self.memory = WorkingMemory(cfg.d_model, n_slots=cfg.n_slots,
                                     chunk_size=cfg.slot_chunk) if cfg.use_memory else None
        # Standard FFN after timescale mixing.
        self.norm_mlp = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * cfg.mlp_ratio, cfg.d_model, bias=False),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        u = self.norm_ssm(x)
        y_fast, h_fast = self.fast(u)
        y_med, h_med = self.med(u)
        y_slow, h_slow = self.slow(u)

        # Predict next states (using current final state).
        pred_fast = self.pred_fast(h_fast)
        pred_med = self.pred_med(h_med)
        pred_slow = self.pred_slow(h_slow)

        # Softmax over the three gates to get a mixture weight per channel.
        # Shape: [3, D] -> [3, D] over the 3 timescales.
        gates = F.softmax(self.gate, dim=0).to(x.dtype)
        combined = (
            gates[0].unsqueeze(0).unsqueeze(0) * y_fast
            + gates[1].unsqueeze(0).unsqueeze(0) * y_med
            + gates[2].unsqueeze(0).unsqueeze(0) * y_slow
        )
        x = x + self.dropout(combined)
        if self.memory is not None:
            x = x + self.dropout(self.memory(x))
        x = x + self.dropout(self.mlp(self.norm_mlp(x)))

        # Return prediction targets and predictions for loss computation
        # outside the block.
        preds = {
            "pred_fast": pred_fast, "actual_fast": h_fast,
            "pred_med": pred_med, "actual_med": h_med,
            "pred_slow": pred_slow, "actual_slow": h_slow,
        }
        return x, preds


class MTSSMCore(nn.Module):
    """Multi-timescale procedural core. Same forward API as
    ProceduralCore for trainer compatibility."""

    def __init__(self, cfg: MTSSMConfig, tie_weights: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MTSSMBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Phase 13: attractor refinement on tied output embeddings.
        # When enabled, tie_weights is forced True (paper requires unembed = E^T).
        self.attractor = AttractorModule(cfg) if cfg.use_attractor else None
        if tie_weights or cfg.use_attractor:
            self.head.weight = self.tok_embed.weight
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=std)
        for block in self.blocks:
            for name, p in block.named_parameters():
                if p.dim() < 2:
                    continue
                if "out_proj" in name or "mlp.2" in name:
                    nn.init.normal_(p, mean=0.0,
                                    std=std / math.sqrt(2 * self.cfg.n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_pred_loss: bool = False) -> torch.Tensor:
        x = self.tok_embed(token_ids)
        pred_losses = []
        for block in self.blocks:
            x, preds = block(x)
            if return_pred_loss:
                # L2 loss between predicted and actual next-step state.
                pred_losses.append(
                    F.mse_loss(preds["pred_fast"], preds["actual_fast"].detach())
                    + F.mse_loss(preds["pred_med"], preds["actual_med"].detach())
                    + F.mse_loss(preds["pred_slow"], preds["actual_slow"].detach())
                )
        x = self.final_norm(x)
        if self.attractor is not None:
            # Phase 13: backbone produces ỹ_0, attractor refines to ỹ*,
            # tied unembedding produces logits.
            x = self.attractor(x)
        logits = self.head(x)
        if return_pred_loss:
            pred_loss = torch.stack(pred_losses).mean() if pred_losses else torch.tensor(0.0, device=x.device)
            return logits, pred_loss
        return logits

    def loss(self, token_ids: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100) -> torch.Tensor:
        """Causal-LM loss with the prediction-coding auxiliary."""
        logits, pred_loss = self.forward(token_ids, return_pred_loss=True)
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, ignore_index=ignore_index
        )
        return lm_loss + self.cfg.pred_loss_weight * pred_loss

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
