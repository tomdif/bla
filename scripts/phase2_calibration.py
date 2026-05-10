"""Phase 2.5 — closed-loop demo + calibration audit.

Loads the Phase 1 B.L.A. policy, runs episodes on multi-target navigate,
emits a CommitmentObject per episode. The SimulatorAgreement certifier
re-rolls the same policy N times under noise; its success fraction is
the uncertainty estimate written into the commitment.

Then we audit calibration: predicted P(success) vs actual deployed
outcome. Gate: Brier ≤ 0.1, ECE ≤ 0.05.

Usage:
    python3 scripts/phase2_calibration.py \\
        --policy runs/local_bla_phase1_causal/final.pt \\
        --episodes 256 --rollouts 10 --output runs/phase2_calibration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system1_jepa import (
    MultiTargetNavigateEnv,
    MultiTargetNavigateSpec,
    PatchViTEncoder,
)
from verification import CommitmentObject, SimulatorAgreement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="Path to Phase 1 final.pt checkpoint")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=10, help="N simulator rollouts per episode")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="SimulatorAgreement passes if rollout success fraction ≥ threshold")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="Action noise std added during simulator rollouts")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--n-bins", type=int, default=10, help="Bins for ECE")
    return p.parse_args()


def load_policy(checkpoint_path: str, device: torch.device):
    from scripts.bla_multitarget import RecurrentBCPolicy

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    encoder = PatchViTEncoder(
        in_channels=3,
        latent_dim=cfg["d"],
        patch_size=cfg["patch_size"],
        depth=cfg["encoder_depth"],
        heads=cfg["encoder_heads"],
    ).to(device)
    policy = RecurrentBCPolicy(encoder, cfg["d"], ssm_layers=cfg["ssm_layers"]).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, cfg


def run_episode(
    policy,
    spec: MultiTargetNavigateSpec,
    seed: int,
    device: torch.device,
    noise_std: float = 0.0,
) -> dict:
    """One deterministic deployment (with optional action noise) — returns success bool + final state."""
    env = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed)
    obs = env.reset()
    history = [obs]
    with torch.no_grad():
        for _ in range(spec.max_steps):
            obs_seq = torch.stack(history, dim=1)
            actions = policy(obs_seq)
            dxy = actions[:, -1]
            if noise_std > 0:
                dxy = dxy + torch.randn_like(dxy) * noise_std
            obs, _, done = env.step(dxy)
            history.append(obs)
            if done.all():
                break
    return {
        "success": bool(env.success_mask().item()),
        "targets_visited": int(env.visited.sum().item()),
        "n_targets": spec.n_targets,
    }


def brier_score(preds, outcomes) -> float:
    """Mean (predicted_p - actual_outcome)^2 over (p, y) pairs."""
    return float(sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / max(len(preds), 1))


def temperature_scale(preds, outcomes, n_iter: int = 200, lr: float = 0.05):
    """Fit a Platt-scaling-style affine map p_calibrated = sigmoid(a*logit(p) + b).
    Returns (a, b, calibrated_preds). Held-out calibration use is the caller's job."""
    import math
    eps = 1e-4
    logits = [math.log(max(eps, min(1 - eps, p)) / max(eps, 1 - max(eps, min(1 - eps, p))))
              for p in preds]
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        # gradient of cross-entropy wrt a, b
        ga, gb = 0.0, 0.0
        for z, y in zip(logits, outcomes):
            p = 1.0 / (1.0 + math.exp(-(a * z + b)))
            ga += (p - y) * z
            gb += (p - y)
        n = len(preds)
        a -= lr * ga / n
        b -= lr * gb / n
    calibrated = [1.0 / (1.0 + math.exp(-(a * z + b))) for z in logits]
    return a, b, calibrated


def expected_calibration_error(preds, outcomes, n_bins: int) -> float:
    """ECE: bin by predicted prob, compute |confidence - accuracy| per bin, weight by bin size."""
    bins = [[] for _ in range(n_bins)]
    for p, o in zip(preds, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))
    ece = 0.0
    n = len(preds)
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_o = sum(o for _, o in b) / len(b)
        ece += (len(b) / n) * abs(avg_p - avg_o)
    return ece


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    policy, cfg = load_policy(args.policy, device)
    spec = MultiTargetNavigateSpec(
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        n_targets=cfg["n_targets"],
        max_steps=cfg["max_steps_per_ep"],
        action_dim=cfg["d"],
    )
    print(json.dumps({"event": "init", "policy_step": cfg.get("steps", "?"), "spec": spec.__dict__,
                      "episodes": args.episodes, "rollouts": args.rollouts}), flush=True)

    commitments: list[CommitmentObject] = []
    predicted_p: list[float] = []
    actual_outcome: list[float] = []
    t0 = time.time()

    for ep in range(args.episodes):
        # 1) actual deployment
        actual = run_episode(policy, spec, seed=args.seed + ep, device=device, noise_std=0.0)

        # 2) simulator certifier — re-roll the SAME starting state under
        # action noise. seed_offset cycles only the noise realization
        # (via torch's RNG state). The env seed is held fixed at
        # `args.seed + ep` so we estimate THIS episode's robustness, not
        # average policy success across episodes.
        def rollout(seed_offset: int) -> bool:
            torch.manual_seed(args.seed + ep * 10000 + seed_offset)
            r = run_episode(policy, spec,
                            seed=args.seed + ep,
                            device=device, noise_std=args.noise_std)
            return r["success"]

        cert = SimulatorAgreement(rollout, n_rollouts=args.rollouts, threshold=args.threshold)

        commitment = CommitmentObject(
            claim={"episode_seed": args.seed + ep, "task": "multi-target-navigate"},
            evidence=[{"actual_targets_visited": actual["targets_visited"],
                       "n_targets": spec.n_targets}],
            reasoning_trace={"policy_checkpoint": args.policy},
            reproducibility_packet={"seed": args.seed + ep,
                                    "noise_std": args.noise_std,
                                    "rollouts": args.rollouts},
        )
        cert_result = cert.attach(commitment, candidate=None)
        commitment.uncertainty = 1.0 - cert_result.confidence
        commitments.append(commitment)

        predicted_p.append(cert_result.confidence)
        actual_outcome.append(1.0 if actual["success"] else 0.0)

        if (ep + 1) % 32 == 0:
            elapsed = time.time() - t0
            running_brier = brier_score(predicted_p, actual_outcome)
            running_acc = sum(actual_outcome) / len(actual_outcome)
            running_pred = sum(predicted_p) / len(predicted_p)
            print(json.dumps({
                "step": ep + 1,
                "elapsed_s": round(elapsed, 1),
                "running_actual_acc": round(running_acc, 4),
                "running_predicted_p": round(running_pred, 4),
                "running_brier": round(running_brier, 4),
            }), flush=True)

    final_brier = brier_score(predicted_p, actual_outcome)
    final_ece = expected_calibration_error(predicted_p, actual_outcome, args.n_bins)
    actual_success_rate = sum(actual_outcome) / max(len(actual_outcome), 1)
    mean_predicted = sum(predicted_p) / max(len(predicted_p), 1)

    # Post-hoc calibration via Platt scaling on first half, eval on second half.
    half = len(predicted_p) // 2
    if half >= 16:
        a_fit, b_fit, _ = temperature_scale(predicted_p[:half], actual_outcome[:half])
        # apply to held-out half
        import math
        eps = 1e-4
        held_out_logits = [math.log(max(eps, min(1 - eps, p)) / max(eps, 1 - max(eps, min(1 - eps, p))))
                           for p in predicted_p[half:]]
        calibrated = [1.0 / (1.0 + math.exp(-(a_fit * z + b_fit))) for z in held_out_logits]
        cal_brier = brier_score(calibrated, actual_outcome[half:])
        cal_ece = expected_calibration_error(calibrated, actual_outcome[half:], args.n_bins)
    else:
        a_fit, b_fit = 1.0, 0.0
        cal_brier, cal_ece = final_brier, final_ece

    summary = {
        "episodes": args.episodes,
        "rollouts_per_episode": args.rollouts,
        "actual_success_rate": actual_success_rate,
        "mean_predicted_p": mean_predicted,
        "raw_brier_score": final_brier,
        "raw_ece": final_ece,
        "calibrated_brier_score": cal_brier,
        "calibrated_ece": cal_ece,
        "platt_a": a_fit,
        "platt_b": b_fit,
        "gate_brier_threshold": 0.1,
        "gate_ece_threshold": 0.05,
        "brier_passed_raw": final_brier <= 0.1,
        "ece_passed_raw": final_ece <= 0.05,
        "brier_passed_calibrated": cal_brier <= 0.1,
        "ece_passed_calibrated": cal_ece <= 0.05,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)

    with open(os.path.join(args.output, "calibration.json"), "w") as f:
        json.dump({
            "summary": summary,
            "predicted_p": predicted_p,
            "actual_outcome": actual_outcome,
        }, f, indent=2)

    sample_path = os.path.join(args.output, "sample_commitment.json")
    with open(sample_path, "w") as f:
        f.write(commitments[0].to_json())
    print(json.dumps({"event": "saved", "sample_commitment": sample_path}), flush=True)


if __name__ == "__main__":
    main()
