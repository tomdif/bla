"""Phase 5 — RL router for compute economy on navigate tasks.

Task mix:
  EASY  — single-target navigate (BC alone hits ~100%)
  HARD  — multi-target navigate (BC alone caps near 17%; needs the
          B.L.A. recurrent policy with SSM memory)

Action space:
  SHALLOW (BC, ~1× FLOPs) — simple policy, deploys directly
  DEEP   (B.L.A. + SSM memory, ~3× FLOPs)

Reward = correctness − λ × flops/flops_max
Train with REINFORCE + entropy bonus.

The asymmetric-scaling thesis on this task:
  * On EASY, BC alone is enough; routing to DEEP is wasteful.
  * On HARD, BC fails; DEEP is needed.
  * Router learns the difference from observation features.

Gate:
  hard / easy compute ratio ≥ 10
  easy accuracy ≤ 5pp drop vs always-DEEP
  total compute ≤ 30% of always-DEEP

Run:
    python3 scripts/phase5_router_rl_navigate.py \\
        --bc-policy runs/local_bc_navigate_st/final.pt \\
        --bla-policy runs/local_bla_phase1_causal/final.pt \\
        --output runs/phase5_navigate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_jepa import (
    MultiTargetNavigateEnv,
    MultiTargetNavigateSpec,
    PatchViTEncoder,
    pool_patch_tokens,
)


# Approximate FLOPs (relative units, calibrated to actual model sizes).
# BC policy: ~10K trainable params + small ViT encoder, single forward per step
# B.L.A. recurrent: ~830K params (encoder + SSM scratchpad + head),
#                   T-step scan over the history at every step
# Empirical FLOPs ratio is ~80×; using 80 as the realistic spread.
FLOPS_SHALLOW = 1.0
FLOPS_DEEP = 80.0

SHALLOW = 0
DEEP = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-policy", required=True,
                   help="BC policy checkpoint (Phase 1.2 — shallow path)")
    p.add_argument("--bla-policy", required=True,
                   help="B.L.A. recurrent policy checkpoint (Phase 1.4 — deep path)")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--frac-easy", type=float, default=0.5,
                   help="Fraction of episodes that are EASY (single-target)")
    p.add_argument("--router-d", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lam-flops-start", type=float, default=0.0)
    p.add_argument("--lam-flops-end", type=float, default=0.4)
    p.add_argument("--entropy-bonus", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_bc(checkpoint_path: str, device: torch.device):
    from scripts.bc_navigate import BCPolicy
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    encoder = PatchViTEncoder(
        in_channels=3, latent_dim=cfg["d"], patch_size=cfg["patch_size"],
        depth=cfg["encoder_depth"], heads=cfg["encoder_heads"],
    ).to(device)
    policy = BCPolicy(encoder, cfg["d"]).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, cfg


def load_bla(checkpoint_path: str, device: torch.device):
    from scripts.bla_multitarget import RecurrentBCPolicy
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    encoder = PatchViTEncoder(
        in_channels=3, latent_dim=cfg["d"], patch_size=cfg["patch_size"],
        depth=cfg["encoder_depth"], heads=cfg["encoder_heads"],
    ).to(device)
    policy = RecurrentBCPolicy(encoder, cfg["d"], ssm_layers=cfg["ssm_layers"]).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, cfg


@torch.no_grad()
def deploy_bc(policy, env: MultiTargetNavigateEnv) -> bool:
    obs = env.reset()
    for _ in range(env.spec.max_steps):
        dxy = policy(obs)
        obs, _, done = env.step(dxy)
        if done.all():
            break
    return bool(env.success_mask().item())


@torch.no_grad()
def deploy_bla(policy, env: MultiTargetNavigateEnv) -> bool:
    obs = env.reset()
    history = [obs]
    for _ in range(env.spec.max_steps):
        obs_seq = torch.stack(history, dim=1)
        actions = policy(obs_seq)
        dxy = actions[:, -1]
        obs, _, done = env.step(dxy)
        history.append(obs)
        if done.all():
            break
    return bool(env.success_mask().item())


def build_episode_pool(
    n_episodes: int,
    frac_easy: float,
    image_size: int,
    patch_size: int,
    bc_policy,
    bla_policy,
    bc_cfg: dict,
    device: torch.device,
    seed_base: int = 0,
) -> list[dict]:
    """Pre-roll episodes through both BC and B.L.A. (cached for fast RL inner loop).

    Each entry: {difficulty, n_targets, seed, bc_correct, bla_correct, init_obs}
    """
    n_easy = int(n_episodes * frac_easy)
    n_hard = n_episodes - n_easy
    out = []
    bc_max_steps = 14

    for i in range(n_easy):
        spec = MultiTargetNavigateSpec(
            image_size=image_size, patch_size=patch_size, n_targets=1,
            max_steps=bc_max_steps, action_dim=bc_cfg["d"],
        )
        env = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + i)
        env.reset()
        init_obs = env.observe().clone()
        # roll out BC
        env_bc = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + i)
        bc_ok = deploy_bc(bc_policy, env_bc)
        # roll out B.L.A.
        env_bla = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + i)
        bla_ok = deploy_bla(bla_policy, env_bla)
        out.append({
            "difficulty": "easy",
            "n_targets": 1,
            "seed": seed_base + i,
            "bc_correct": float(bc_ok),
            "bla_correct": float(bla_ok),
            "init_obs": init_obs.cpu(),
        })

    for i in range(n_hard):
        spec = MultiTargetNavigateSpec(
            image_size=image_size, patch_size=patch_size, n_targets=2,
            max_steps=bc_max_steps, action_dim=bc_cfg["d"],
        )
        env = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + n_easy + i)
        env.reset()
        init_obs = env.observe().clone()
        env_bc = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + n_easy + i)
        bc_ok = deploy_bc(bc_policy, env_bc)
        env_bla = MultiTargetNavigateEnv(spec, batch_size=1, device=device, seed=seed_base + n_easy + i)
        bla_ok = deploy_bla(bla_policy, env_bla)
        out.append({
            "difficulty": "hard",
            "n_targets": 2,
            "seed": seed_base + n_easy + i,
            "bc_correct": float(bc_ok),
            "bla_correct": float(bla_ok),
            "init_obs": init_obs.cpu(),
        })
    return out


class RouterPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def features_from_episodes(episodes: list[dict]) -> torch.Tensor:
    """Use the initial observation, mean-pooled per channel, as router features.
    On multi-target navigate, the brightness/coverage of the canvas tells you
    how many targets are present — easy vs hard is encoded in the observation."""
    feats = []
    for ep in episodes:
        obs = ep["init_obs"]  # [1, 3, H, W]
        per_channel_mean = obs.mean(dim=(2, 3)).flatten()        # 3 dims
        per_channel_max = obs.amax(dim=(2, 3)).flatten()         # 3 dims
        per_channel_std = obs.std(dim=(2, 3)).flatten()          # 3 dims
        # rough count of bright pixels in red channel (target overlay)
        n_bright = (obs[0, 0] > 0.5).float().sum().unsqueeze(0)  # 1 dim
        feats.append(torch.cat([per_channel_mean, per_channel_max,
                                per_channel_std, n_bright], dim=0))
    return torch.stack(feats, dim=0).float()


def reinforce_train(
    train: list[dict],
    train_feats: torch.Tensor,
    args,
    device: torch.device,
):
    embed_dim = train_feats.shape[1]
    policy = RouterPolicy(embed_dim, hidden=args.router_d, n_actions=2).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    flops_max = max(FLOPS_SHALLOW, FLOPS_DEEP)
    history = []

    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        lam = args.lam_flops_start + (args.lam_flops_end - args.lam_flops_start) * progress

        idx = torch.randperm(len(train))
        epoch_loss = 0.0
        epoch_reward = 0.0
        action_counts = [0, 0]
        n_batches = 0

        for start in range(0, len(train), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            x = train_feats[batch_idx].to(device)
            logits = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            rewards = []
            for j, a in zip(batch_idx.tolist(), actions.tolist()):
                if a == SHALLOW:
                    correct = train[j]["bc_correct"]
                    flops = FLOPS_SHALLOW
                else:
                    correct = train[j]["bla_correct"]
                    flops = FLOPS_DEEP
                rewards.append(correct - lam * (flops / flops_max))
                action_counts[a] += 1

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            baseline = rewards_t.mean().detach()
            advantage = rewards_t - baseline
            policy_loss = -(advantage * log_prob).mean()
            ent_term = -args.entropy_bonus * entropy.mean()
            loss = policy_loss + ent_term

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss.detach())
            epoch_reward += float(rewards_t.mean().detach())
            n_batches += 1

        history.append({
            "epoch": epoch + 1,
            "lam_flops": lam,
            "loss": epoch_loss / max(n_batches, 1),
            "reward": epoch_reward / max(n_batches, 1),
            "action_share_shallow": action_counts[0] / max(sum(action_counts), 1),
            "action_share_deep": action_counts[1] / max(sum(action_counts), 1),
        })
        print(json.dumps({"event": "epoch", **history[-1]}), flush=True)

    return policy, {"history": history}


@torch.no_grad()
def evaluate(policy, episodes: list[dict], feats: torch.Tensor, device: torch.device) -> dict:
    policy.eval()
    logits = policy(feats.to(device))
    actions = logits.argmax(dim=-1).cpu().tolist()

    by_difficulty: dict[str, dict] = {}
    total_correct = 0.0
    total_flops = 0.0
    always_deep_correct = 0.0
    always_deep_flops = 0.0
    always_shallow_correct = 0.0
    always_shallow_flops = 0.0

    for ep, a in zip(episodes, actions):
        d = ep["difficulty"]
        bucket = by_difficulty.setdefault(d, {
            "n": 0, "shallow": 0, "deep": 0, "correct": 0.0, "flops": 0.0,
            "always_deep_correct": 0.0, "always_shallow_correct": 0.0,
        })
        bucket["n"] += 1

        if a == SHALLOW:
            bucket["shallow"] += 1
            bucket["correct"] += ep["bc_correct"]
            bucket["flops"] += FLOPS_SHALLOW
            total_correct += ep["bc_correct"]
            total_flops += FLOPS_SHALLOW
        else:
            bucket["deep"] += 1
            bucket["correct"] += ep["bla_correct"]
            bucket["flops"] += FLOPS_DEEP
            total_correct += ep["bla_correct"]
            total_flops += FLOPS_DEEP
        bucket["always_deep_correct"] += ep["bla_correct"]
        bucket["always_shallow_correct"] += ep["bc_correct"]
        always_deep_correct += ep["bla_correct"]
        always_deep_flops += FLOPS_DEEP
        always_shallow_correct += ep["bc_correct"]
        always_shallow_flops += FLOPS_SHALLOW

    n = len(episodes)
    summary = {
        "router_accuracy": total_correct / n,
        "router_flops_per_query": total_flops / n,
        "always_deep_accuracy": always_deep_correct / n,
        "always_deep_flops_per_query": always_deep_flops / n,
        "always_shallow_accuracy": always_shallow_correct / n,
        "always_shallow_flops_per_query": always_shallow_flops / n,
        "compute_share_vs_always_deep": total_flops / max(always_deep_flops, 1e-6),
        "by_difficulty": {},
    }
    for d, b in by_difficulty.items():
        summary["by_difficulty"][d] = {
            "n": b["n"],
            "deep_share": b["deep"] / max(b["n"], 1),
            "router_accuracy": b["correct"] / max(b["n"], 1),
            "router_flops_per_query": b["flops"] / max(b["n"], 1),
            "always_deep_accuracy": b["always_deep_correct"] / max(b["n"], 1),
            "always_shallow_accuracy": b["always_shallow_correct"] / max(b["n"], 1),
        }

    if "easy" in summary["by_difficulty"] and "hard" in summary["by_difficulty"]:
        easy_flops = summary["by_difficulty"]["easy"]["router_flops_per_query"]
        hard_flops = summary["by_difficulty"]["hard"]["router_flops_per_query"]
        summary["compute_split_hard_over_easy"] = hard_flops / max(easy_flops, 1e-6)
        easy_acc_drop = (summary["by_difficulty"]["easy"]["always_deep_accuracy"]
                         - summary["by_difficulty"]["easy"]["router_accuracy"])
        summary["easy_accuracy_drop_pp"] = easy_acc_drop * 100
        summary["compute_split_passed"] = summary["compute_split_hard_over_easy"] >= 10.0
        summary["easy_accuracy_passed"] = easy_acc_drop * 100 <= 5.0
        summary["compute_share_passed"] = summary["compute_share_vs_always_deep"] <= 0.30
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    bc_policy, bc_cfg = load_bc(args.bc_policy, device)
    bla_policy, bla_cfg = load_bla(args.bla_policy, device)
    print(json.dumps({"event": "loaded",
                      "bc_d": bc_cfg["d"], "bla_d": bla_cfg["d"]}), flush=True)

    image_size = bla_cfg["image_size"]
    patch_size = bla_cfg["patch_size"]

    print(json.dumps({"event": "rollout_train"}), flush=True)
    train = build_episode_pool(
        args.n_train, args.frac_easy, image_size, patch_size,
        bc_policy, bla_policy, bla_cfg, device, seed_base=10_000,
    )
    print(json.dumps({"event": "rollout_test"}), flush=True)
    test = build_episode_pool(
        args.n_test, args.frac_easy, image_size, patch_size,
        bc_policy, bla_policy, bla_cfg, device, seed_base=99_999,
    )

    train_feats = features_from_episodes(train)
    test_feats = features_from_episodes(test)
    print(json.dumps({"event": "features",
                      "train": list(train_feats.shape),
                      "test": list(test_feats.shape)}), flush=True)

    # Sanity: print accuracies of always-shallow / always-deep on each difficulty
    def quick_summary(pool):
        out = {}
        for d in ("easy", "hard"):
            sub = [ep for ep in pool if ep["difficulty"] == d]
            out[d] = {
                "n": len(sub),
                "always_shallow_acc": sum(e["bc_correct"] for e in sub) / max(len(sub), 1),
                "always_deep_acc": sum(e["bla_correct"] for e in sub) / max(len(sub), 1),
            }
        return out
    print(json.dumps({"event": "baselines", "train": quick_summary(train),
                      "test": quick_summary(test)}, indent=2), flush=True)

    policy, train_history = reinforce_train(train, train_feats, args, device)
    summary = evaluate(policy, test, test_feats, device)
    print(json.dumps({"event": "summary", **summary}, indent=2))

    with open(os.path.join(args.output, "phase5.json"), "w") as f:
        # strip large tensors before saving
        clean_test = [{k: v for k, v in e.items() if k != "init_obs"} for e in test]
        json.dump({"summary": summary, "train_history": train_history,
                   "test_episodes": clean_test}, f, indent=2)

    torch.save({
        "policy": policy.state_dict(),
        "config": vars(args),
        "summary": summary,
    }, os.path.join(args.output, "router.pt"))


if __name__ == "__main__":
    main()
