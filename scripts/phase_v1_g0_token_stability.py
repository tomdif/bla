"""Phase V1-G0 — V-JEPA 2 ViT-L token-stability probe on robosuite rollout.

Tests whether V-JEPA 2 produces stable patch tokens under overlapping
K-frame rolling windows. Gate: mean cosine ≥ 0.95 AND ≥ 80% of
frames in the overlap region also ≥ 0.95.

Protocol (matches the spec §V0 § new G0 precommit):

  1. Generate ~64-frame robosuite Lift rollout (one episode is enough).
  2. For each K ∈ {4, 5, 6, 8}:
     - Encode window starting at offset 0: frames[0 : K]
     - Encode window starting at offset 2: frames[2 : K+2]
     - The overlapping region is frames[2 : K]; (K-2) frames.
     - V-JEPA 2 uses tubelet=2, so each window's tokens span T/2
       temporal positions × 16×16 spatial. The OVERLAPPING TEMPORAL
       INDICES in the two windows differ by 1 tubelet (because
       offset=2 frames = 1 tubelet).
     - Compute cosine similarity between aligned overlapping tokens.

If the mean cosine clears 0.95 for K=5 (BLA's deployment default),
the full V1 swap is greenlit. If K=5 fails but K=4 or K=6 passes,
flag that BLA's rolling-window default needs to adapt.

Note on V-JEPA 2's "fpc64" naming: the model was trained on 64-frame
clips, but the docs explicitly state that frames_per_clip "does not
impact inference" — variable T is handled via 3D RoPE positional
encoding. Short clips (K=4-8 frames) are fine.
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

K_VALUES = (4, 5, 6, 8)
OFFSET = 2  # frames; matches tubelet=2 stride
PASS_COSINE = 0.95
PASS_FRAC = 0.80   # ≥ 80% of overlap tokens must clear the threshold


def collect_rollout_frames(image_size: int = 224,
                              n_frames: int = 64) -> np.ndarray:
    """Collect one rollout from robosuite Lift task.

    Random actions (just to generate motion in the scene); we're testing
    encoder stability, not policy quality. Returns [n_frames, H, W, 3] uint8.
    """
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
        a[6] = 1.0   # gripper open (avoid grasping artifacts)
        obs, _, _, _ = env.step(a)
        frames.append(obs["agentview_image"])
    return np.stack(frames, axis=0)


def load_vjepa2_vit_l(device: str = "cuda"):
    from transformers import VJEPA2Model, AutoVideoProcessor
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    print(json.dumps({"event": "loading", "model_id": model_id}), flush=True)
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = VJEPA2Model.from_pretrained(
        model_id, dtype=torch.bfloat16,
    ).to(device).eval()
    return model, processor


@torch.no_grad()
def encode_window(model, processor, frames_TWHC: np.ndarray,
                    device: str = "cuda") -> torch.Tensor:
    """Encode T frames → [N_tokens, D] last_hidden_state.

    V-JEPA 2's processor accepts a list of PIL frames or a numpy/tensor
    sequence. We pass numpy [T, H, W, 3] uint8.
    """
    inputs = processor(videos=list(frames_TWHC), return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(device).to(torch.bfloat16)
    # [B, T, 3, H, W]
    outputs = model(pixel_values, skip_predictor=True)
    return outputs.last_hidden_state[0].float()  # [N_tokens, D]


def expected_token_shape(T_frames: int, H: int = 256, W: int = 256,
                            tubelet: int = 2, patch: int = 16) -> dict:
    """V-JEPA 2 patch grid expectation: tokens = (T/tubelet) * (H/patch) * (W/patch).
    NB: the processor may pad/resize to internal canonical size; report observed."""
    return {
        "expected_T_tokens": T_frames // tubelet,
        "expected_HW_tokens": (H // patch, W // patch),
        "expected_total": (T_frames // tubelet) * (H // patch) * (W // patch),
    }


def cosine_per_token(a: torch.Tensor, b: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    """Cosine per row of [N, D] x [N, D] → [N]."""
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1)


def probe_K(model, processor, frames: np.ndarray, K: int,
              offset: int, device: str) -> dict:
    """Encode two K-frame windows at offsets 0 and `offset`; align overlap;
    return cosine statistics.

    With tubelet=2 and offset=2 frames, the two windows' tubelet-grids
    align with a temporal shift of 1 tubelet. Specifically:

      window 1 spans frames [0, K)            → tubelets {0, 1, ..., K/2-1}
      window 2 spans frames [offset, K+offset) → tubelets shifted by offset/2

    Overlap region in frame space: [offset, K) — K-offset frames.
    In tubelet space (V-JEPA 2 output): window 1 tubelets {offset/2, ..., K/2-1};
                                          window 2 tubelets {0, ..., (K-offset)/2 - 1}.

    For K=5, offset=2: window 1 has T_in=5 (model may pad to next even);
    window 2 same. We expect tubelets that span the same source frames to
    have HIGH cosine similarity.
    """
    n_total = frames.shape[0]
    assert n_total >= K + offset, f"need ≥ {K+offset} frames, got {n_total}"
    w1 = frames[0:K]
    w2 = frames[offset:offset + K]

    t0 = time.time()
    tok1 = encode_window(model, processor, w1, device=device)
    tok2 = encode_window(model, processor, w2, device=device)
    elapsed = time.time() - t0

    # Both tensors are [N_tokens, D]. We need to align the overlap region.
    # Infer the temporal layout: total tokens = T_lat * H_lat * W_lat.
    # For V-JEPA 2 the processor canonicalizes to 256×256 → spatial 16×16 = 256
    # tokens per temporal slice.
    HW_tokens = 256
    N1, D = tok1.shape
    N2, _ = tok2.shape
    T1 = N1 // HW_tokens
    T2 = N2 // HW_tokens
    if T1 * HW_tokens != N1 or T2 * HW_tokens != N2:
        # Fall back: try 14×14 (if model resized to 224)
        for hw in (14 * 14, 12 * 12, 16 * 16):
            if N1 % hw == 0:
                HW_tokens = hw
                T1 = N1 // HW_tokens
                T2 = N2 // HW_tokens
                break
    print(json.dumps({"event": "shapes_inferred",
                       "K": K, "N1": N1, "N2": N2,
                       "T1_tubelets": T1, "T2_tubelets": T2,
                       "HW_tokens": HW_tokens, "D": D,
                       "encode_elapsed_s": elapsed}), flush=True)

    # Reshape to [T_tubelet, HW, D]
    tok1_g = tok1.reshape(T1, HW_tokens, D)
    tok2_g = tok2.reshape(T2, HW_tokens, D)

    # tubelet stride from offset=2 frames: 1 tubelet
    tubelet_stride = offset // 2

    # Number of overlapping tubelets between the two windows
    overlap_tubelets = min(T1 - tubelet_stride, T2)
    if overlap_tubelets <= 0:
        return {"K": K, "skipped": True,
                  "reason": f"no overlap (T1={T1}, T2={T2}, stride={tubelet_stride})"}

    # Align: tok1's tubelets [stride, stride+overlap) vs tok2's [0, overlap)
    a = tok1_g[tubelet_stride: tubelet_stride + overlap_tubelets]   # [O, HW, D]
    b = tok2_g[:overlap_tubelets]                                     # [O, HW, D]

    # Flatten to per-token cosine
    a_flat = a.reshape(-1, D)
    b_flat = b.reshape(-1, D)
    cos = cosine_per_token(a_flat, b_flat).cpu().numpy()

    # Per-temporal-tubelet mean (useful diagnostic)
    cos_per_tubelet = cos.reshape(overlap_tubelets, HW_tokens).mean(axis=1).tolist()

    stats = {
        "K": K,
        "offset_frames": offset,
        "n_overlap_tubelets": int(overlap_tubelets),
        "n_overlap_tokens": int(cos.size),
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "p10_cos": float(np.percentile(cos, 10)),
        "p50_cos": float(np.percentile(cos, 50)),
        "p90_cos": float(np.percentile(cos, 90)),
        "frac_above_0p95": float((cos >= PASS_COSINE).mean()),
        "per_tubelet_mean_cos": cos_per_tubelet,
        "passes_mean": bool(cos.mean() >= PASS_COSINE),
        "passes_frac": bool((cos >= PASS_COSINE).mean() >= PASS_FRAC),
    }
    stats["passes_g0"] = stats["passes_mean"] and stats["passes_frac"]

    # --- DIAGNOSTIC: mean-pool over temporal axis, compare clip-summary cosine
    # If raw-token cosine is low but pooled cosine is high, V-JEPA 2 is
    # position-aware (RoPE) but content-stable — usable downstream via an
    # aggregator. If both are low, the encoder really is unstable.
    a_pooled = a_flat.reshape(overlap_tubelets, HW_tokens, D).mean(dim=0)  # [HW, D]
    b_pooled = b_flat.reshape(overlap_tubelets, HW_tokens, D).mean(dim=0)
    cos_pooled = cosine_per_token(a_pooled, b_pooled).cpu().numpy()
    stats["pooled_temporal_mean_cos"] = float(cos_pooled.mean())
    stats["pooled_temporal_min_cos"] = float(cos_pooled.min())
    stats["pooled_temporal_frac_above_0p95"] = float((cos_pooled >= PASS_COSINE).mean())

    # Also: average ALL tokens in each window → single clip vector cosine
    clip_a = tok1.reshape(-1, D).mean(dim=0)
    clip_b = tok2.reshape(-1, D).mean(dim=0)
    stats["clip_summary_cos"] = float(
        cosine_per_token(clip_a.unsqueeze(0), clip_b.unsqueeze(0))[0].item()
    )

    return stats


def reencode_determinism_check(model, processor, frames, device):
    """Sanity check: encode the same 4-frame window twice; must be ≈ 1.0."""
    w = frames[:4]
    t1 = encode_window(model, processor, w, device=device)
    t2 = encode_window(model, processor, w, device=device)
    cos = cosine_per_token(t1, t2).mean().item()
    return float(cos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/root/bla/runs/phase_v1_g0")
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224,
                    help="robosuite render size; V-JEPA 2 processor will "
                          "resize to its expected 256.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    # 1. Collect rollout
    t0 = time.time()
    frames = collect_rollout_frames(args.image_size, args.n_frames)
    print(json.dumps({"event": "rollout_collected",
                       "frames_shape": list(frames.shape),
                       "elapsed_s": time.time() - t0}), flush=True)

    # 2. Load model
    model, processor = load_vjepa2_vit_l(args.device)
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(json.dumps({"event": "model_loaded",
                       "vram_alloc_gb": round(vram_after_load, 2)}), flush=True)

    # 2b. Determinism sanity check
    det_cos = reencode_determinism_check(model, processor, frames, args.device)
    print(json.dumps({"event": "determinism_check",
                       "mean_cos_same_window_twice": det_cos}), flush=True)

    # 3. Probe each K
    all_stats = []
    for K in K_VALUES:
        try:
            stats = probe_K(model, processor, frames, K, OFFSET, args.device)
        except Exception as e:
            stats = {"K": K, "error": str(e)}
        all_stats.append(stats)
        print(json.dumps({"event": "K_done", **stats}), flush=True)

    # 4. Verdict
    k_passes = {s["K"]: s.get("passes_g0", False) for s in all_stats if "K" in s}
    verdict = {
        "k_passes_g0": k_passes,
        "k_default_5_passes": bool(k_passes.get(5, False)),
        "any_k_passes": any(k_passes.values()),
    }
    if k_passes.get(5, False):
        verdict["recommendation"] = "PROCEED — V1 full encoder swap with V-JEPA 2 ViT-L"
    elif k_passes.get(4, False) or k_passes.get(6, False):
        verdict["recommendation"] = (
            "ADAPT — V-JEPA 2 features stable at K=4 or K=6 but NOT K=5. "
            "BLA rolling-window default needs to shift to encoder-compatible K.")
    else:
        verdict["recommendation"] = (
            "DO NOT PROCEED — V-JEPA 2 features unstable under rolling-window "
            "inference. Consider Cosmos-Tokenizer or stick with current encoder.")

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
