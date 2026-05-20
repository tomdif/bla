"""Phase V1a-G0 — Cosmos-Tokenizer CV4x8x8 token-stability probe.

Counterpart to phase_v1_g0_token_stability.py (V-JEPA 2). Same overall
question, different encoder. Cosmos differs in three ways:

  1. Input shape: [B, 3, T, H, W] (channel-first), not [B, T, 3, H, W].
  2. Temporal compression: 4× (CV4x8x8) — so OFFSET=4 frames is the
     natural 1-latent-position shift between overlapping windows.
     OFFSET=2 (the V-JEPA 2 protocol) gives a fractional shift and
     isn't usable.
  3. Latent shape: [B, 16, T_lat, H_lat, W_lat]; 16 channels at
     T_lat = (T-1)/4 + 1, H_lat = H/8, W_lat = W/8.

K values for Cosmos must satisfy T ≡ 1 (mod 4) — so K ∈ {9, 13, 17, 21}.

Prediction (from `feedback_vjepa2_position_bound_tokens`): Cosmos is a
VAE encoder, not a RoPE positional transformer, so it should NOT exhibit
V-JEPA 2's position-bound drift. Per-token cosine across aligned
overlapping latent positions should be high (≥0.95).

The risk is different: Cosmos features are pixel-reconstruction-
oriented, so identity under occlusion may be weaker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

K_VALUES = (9, 13, 17, 21)      # Cosmos requires T ≡ 1 (mod 4)
OFFSET = 4                       # 1 latent position shift
PASS_COSINE = 0.95
PASS_FRAC = 0.80

# Cosmos CV4x8x8 expects 256×256 input.
COSMOS_RES = 256


def collect_rollout_frames(image_size: int = COSMOS_RES,
                              n_frames: int = 40) -> np.ndarray:
    """Same protocol as V-JEPA 2 probe — random-action robosuite Lift rollout."""
    import robosuite as rs
    env = rs.make("Lift", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=False,
                    camera_names="agentview",
                    camera_heights=image_size, camera_widths=image_size,
                    horizon=n_frames + 10)
    obs = env.reset()
    frames = [obs["agentview_image"]]
    rng = np.random.RandomState(0)
    for _ in range(n_frames - 1):
        a = rng.uniform(-0.1, 0.1, size=env.action_dim).astype(np.float32)
        a[6] = 1.0   # gripper open
        obs, _, _, _ = env.step(a)
        frames.append(obs["agentview_image"])
    return np.stack(frames, axis=0)


def load_cosmos_encoder(ckpt_path: str, device: str = "cuda"):
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    enc = CausalVideoTokenizer(checkpoint_enc=ckpt_path)
    return enc


@torch.no_grad()
def encode_window(encoder, frames_THWC: np.ndarray,
                    device: str = "cuda") -> torch.Tensor:
    """Cosmos input: [B, 3, T, H, W] in [-1, 1] range (bfloat16).

    Returns latent [16, T_lat, H_lat, W_lat] (squeezed batch).
    """
    # frames_THWC: [T, H, W, 3] uint8 in [0, 255]
    x = torch.from_numpy(frames_THWC).float() / 127.5 - 1.0   # → [-1, 1]
    x = x.permute(3, 0, 1, 2).unsqueeze(0)                     # [1, 3, T, H, W]
    x = x.to(device).to(torch.bfloat16)
    (lat,) = encoder.encode(x)
    return lat[0].float()  # [16, T_lat, H_lat, W_lat]


def cosine_per_token(a: torch.Tensor, b: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1)


def probe_K(encoder, frames: np.ndarray, K: int, offset: int,
              device: str) -> dict:
    """Encode windows[0:K] and windows[offset:K+offset]; compare aligned
    overlapping latent positions.

    For Cosmos CV4x8x8 with offset=4:
      window 1 latents: [L0, L1, ..., L_{T_lat-1}]
      window 2 latents: [L'_0, L'_1, ..., L'_{T_lat-1}]
      latent position L_i (window 1) corresponds to input frames
        [4(i-1)+1 .. 4i] (causal); window 2's L'_j corresponds to
        input frames [4(j-1)+5 .. 4j+4].
      So L1 (window 1) and L0' (window 2) cover roughly the same
        source frames. Compare L1:T_lat (window 1) vs L0:T_lat-1 (window 2).
    """
    n_total = frames.shape[0]
    assert n_total >= K + offset, f"need ≥ {K+offset} frames, got {n_total}"
    w1 = frames[0:K]
    w2 = frames[offset:offset + K]

    t0 = time.time()
    lat1 = encode_window(encoder, w1, device=device)   # [16, T_lat, 32, 32]
    lat2 = encode_window(encoder, w2, device=device)
    elapsed = time.time() - t0

    C, T_lat, H_lat, W_lat = lat1.shape

    # offset=4 frames = 1 latent position shift in CV4x8x8
    lat_shift = offset // 4
    overlap_n = T_lat - lat_shift
    if overlap_n <= 0:
        return {"K": K, "skipped": True,
                  "reason": f"no overlap (T_lat={T_lat}, shift={lat_shift})"}

    print(json.dumps({"event": "shapes_inferred",
                       "K": K, "T_lat": T_lat, "H_lat": H_lat,
                       "W_lat": W_lat, "C": C,
                       "lat_shift": lat_shift,
                       "overlap_n": overlap_n,
                       "encode_elapsed_s": elapsed}), flush=True)

    # Align: lat1[lat_shift:] vs lat2[:overlap_n], both shape [16, overlap_n, 32, 32]
    a = lat1[:, lat_shift:lat_shift + overlap_n]
    b = lat2[:, :overlap_n]

    # Per-spatial-token cosine: treat each (T_lat, H_lat, W_lat) location
    # as a 16-D feature vector; flatten C to the last axis for cosine.
    # Shape after permute: [overlap_n, H_lat, W_lat, C]
    a_flat = a.permute(1, 2, 3, 0).reshape(-1, C)
    b_flat = b.permute(1, 2, 3, 0).reshape(-1, C)
    cos = cosine_per_token(a_flat, b_flat).cpu().numpy()

    # Per-latent-frame mean (diagnostic)
    cos_per_frame = cos.reshape(overlap_n, H_lat * W_lat).mean(axis=1).tolist()

    # Mean-pooled-temporal (analogous to V-JEPA 2 pooled variant)
    a_pooled = a.mean(dim=1).permute(1, 2, 0).reshape(-1, C)   # [H*W, C]
    b_pooled = b.mean(dim=1).permute(1, 2, 0).reshape(-1, C)
    cos_pooled = cosine_per_token(a_pooled, b_pooled).cpu().numpy()

    # Clip-summary: mean across all positions
    clip_a = a.flatten().reshape(C, -1).mean(dim=-1)
    clip_b = b.flatten().reshape(C, -1).mean(dim=-1)
    clip_cos = float(cosine_per_token(clip_a.unsqueeze(0),
                                           clip_b.unsqueeze(0))[0].item())

    stats = {
        "K": K,
        "offset_frames": offset,
        "lat_shift": lat_shift,
        "n_overlap_latent_frames": int(overlap_n),
        "n_overlap_spatial_tokens": int(cos.size),
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "p10_cos": float(np.percentile(cos, 10)),
        "p50_cos": float(np.percentile(cos, 50)),
        "p90_cos": float(np.percentile(cos, 90)),
        "frac_above_0p95": float((cos >= PASS_COSINE).mean()),
        "per_frame_mean_cos": cos_per_frame,
        "pooled_temporal_mean_cos": float(cos_pooled.mean()),
        "pooled_temporal_frac_above_0p95": float((cos_pooled >= PASS_COSINE).mean()),
        "clip_summary_cos": clip_cos,
        "passes_mean": bool(cos.mean() >= PASS_COSINE),
        "passes_frac": bool((cos >= PASS_COSINE).mean() >= PASS_FRAC),
    }
    stats["passes_g0"] = stats["passes_mean"] and stats["passes_frac"]
    return stats


def reencode_determinism_check(encoder, frames, device):
    w = frames[:9]
    t1 = encode_window(encoder, w, device=device)
    t2 = encode_window(encoder, w, device=device)
    C = t1.shape[0]
    t1_flat = t1.permute(1, 2, 3, 0).reshape(-1, C)
    t2_flat = t2.permute(1, 2, 3, 0).reshape(-1, C)
    return float(cosine_per_token(t1_flat, t2_flat).mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                    default="/root/cosmos_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit")
    p.add_argument("--out", default="/root/bla/runs/phase_v1a_g0_cosmos")
    p.add_argument("--n-frames", type=int, default=40)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    # 1. Collect rollout
    t0 = time.time()
    frames = collect_rollout_frames(COSMOS_RES, args.n_frames)
    print(json.dumps({"event": "rollout_collected",
                       "frames_shape": list(frames.shape),
                       "elapsed_s": time.time() - t0}), flush=True)

    # 2. Load encoder
    encoder = load_cosmos_encoder(args.ckpt, args.device)
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(json.dumps({"event": "model_loaded",
                       "vram_alloc_gb": round(vram_after_load, 2)}), flush=True)

    # 3. Determinism sanity
    det_cos = reencode_determinism_check(encoder, frames, args.device)
    print(json.dumps({"event": "determinism_check",
                       "mean_cos_same_window_twice": det_cos}), flush=True)

    # 4. Probe each K
    all_stats = []
    for K in K_VALUES:
        try:
            stats = probe_K(encoder, frames, K, OFFSET, args.device)
        except Exception as e:
            stats = {"K": K, "error": str(e)}
        all_stats.append(stats)
        print(json.dumps({"event": "K_done", **stats}), flush=True)

    # 5. Verdict
    k_passes = {s["K"]: s.get("passes_g0", False) for s in all_stats if "K" in s}
    verdict = {
        "k_passes_g0": k_passes,
        "any_k_passes": any(k_passes.values()),
    }
    if any(k_passes.values()):
        verdict["recommendation"] = (
            "PROCEED — Cosmos-Tokenizer features are stable under "
            "rolling-window inference. Greenlight V1a full encoder swap "
            "test (M1-M5 against OF-JEPA baseline).")
    else:
        # Check if it's still useful at clip-summary level
        clip_cos = [s.get("clip_summary_cos") for s in all_stats if "clip_summary_cos" in s]
        if clip_cos and min(clip_cos) >= 0.95:
            verdict["recommendation"] = (
                "REFRAME — Cosmos per-token features unstable but "
                "clip-summary stable. Treat like V-JEPA 2: use as "
                "retrieval-key feature (V1b track), not encoder swap.")
        else:
            verdict["recommendation"] = (
                "DO NOT PROCEED — Cosmos-Tokenizer features unstable at "
                "both per-token and clip-summary levels. Stick with "
                "current OF-JEPA encoder.")

    print(json.dumps({"event": "verdict", **verdict}), flush=True)

    summary = {
        "args": vars(args),
        "K_values": list(K_VALUES),
        "offset_frames": OFFSET,
        "pass_cosine": PASS_COSINE,
        "pass_frac": PASS_FRAC,
        "vram_alloc_gb": round(vram_after_load, 2),
        "determinism_check_cos": det_cos,
        "per_K": all_stats,
        "verdict": verdict,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
