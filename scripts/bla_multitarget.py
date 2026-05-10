"""B.L.A. recurrent policy on multi-target navigate. Phase 1.4.

Same encoder + expert imitation as bc_multitarget.py, but with the
B.L.A. SSM scratchpad maintaining a hidden state over the episode
history. The hidden state encodes "which targets have been visited" —
information that the observation alone does not expose.

Gate: success rate must beat BC by ≥ 20 percentage points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch import nn

from system1_jepa import (
    MultiTargetNavigateEnv,
    MultiTargetNavigateSpec,
    PatchViTEncoder,
    pool_patch_tokens,
)
from system2_dca.ssm import CausalSSMScratchpad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--episodes-per-step", type=int, default=32)
    p.add_argument("--max-steps-per-ep", type=int, default=24)
    p.add_argument("--n-targets", type=int, default=3)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-heads", type=int, default=8)
    p.add_argument("--ssm-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class RecurrentBCPolicy(nn.Module):
    """Encoder + SSM scratchpad over history + linear policy head.

    forward(obs_seq) takes [B, T, C, H, W] history and returns [B, T, 2]
    actions — one per timestep. At eval, we feed the cumulative history
    each step and read the last output.
    """

    def __init__(self, encoder: nn.Module, d: int, ssm_layers: int = 2):
        super().__init__()
        self.encoder = encoder
        self.scratchpad = CausalSSMScratchpad(d_model=d, layers=ssm_layers)
        self.head = nn.Linear(d, 2)

    def encode_frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: [B, T, C, H, W] -> features [B, T, D]"""
        b, t = obs_seq.shape[:2]
        flat = obs_seq.reshape(b * t, *obs_seq.shape[2:])
        z, _, _ = self.encoder(flat)
        pooled = pool_patch_tokens(z)
        return pooled.reshape(b, t, -1)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> [B, T, 2] action prediction at every timestep."""
        per_frame = self.encode_frames(obs_seq)
        memory_seq = self.scratchpad(per_frame)
        return self.head(memory_seq)


@torch.no_grad()
def deploy(
    policy: RecurrentBCPolicy,
    spec: MultiTargetNavigateSpec,
    n_episodes: int,
    device: torch.device,
    seed: int = 99,
) -> dict:
    policy.eval()
    bs = min(n_episodes, 32)
    n_batches = (n_episodes + bs - 1) // bs
    successes = 0
    total = 0
    visit_counts = []
    for b in range(n_batches):
        env = MultiTargetNavigateEnv(spec, batch_size=bs, device=device, seed=seed + b)
        obs = env.reset()
        history = [obs]
        for _ in range(spec.max_steps):
            obs_seq = torch.stack(history, dim=1)
            actions = policy(obs_seq)
            dxy = actions[:, -1]
            obs, _, done = env.step(dxy)
            history.append(obs)
            if done.all():
                break
        ep_successes = env.success_mask()
        successes += int(ep_successes.sum())
        total += bs
        visit_counts.append(env.visited.sum(dim=1).float().mean().item())
    policy.train()
    return {
        "success_rate": successes / max(total, 1),
        "successes": successes,
        "total": total,
        "mean_targets_visited": sum(visit_counts) / max(len(visit_counts), 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    spec = MultiTargetNavigateSpec(
        image_size=args.image_size, patch_size=args.patch_size,
        n_targets=args.n_targets, max_steps=args.max_steps_per_ep,
        action_dim=args.d,
    )
    encoder = PatchViTEncoder(
        in_channels=3, latent_dim=args.d, patch_size=args.patch_size,
        depth=args.encoder_depth, heads=args.encoder_heads,
    ).to(device)
    policy = RecurrentBCPolicy(encoder, args.d, ssm_layers=args.ssm_layers).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in policy.parameters())
    print(json.dumps({"event": "init", "n_params": n_params, "config": vars(args)}), flush=True)

    env = MultiTargetNavigateEnv(spec, batch_size=args.episodes_per_step, device=device, seed=args.seed)
    log = {"loss": 0.0, "n": 0}
    t0 = time.time()

    for step in range(args.steps):
        obs = env.reset()
        observations = [obs]
        expert_actions = []
        for _ in range(args.max_steps_per_ep):
            dxy = env.expert_action()
            expert_actions.append(dxy)
            obs, _, done = env.step(dxy)
            observations.append(obs)
            if done.all():
                break
        observations = torch.stack(observations[:-1], dim=1)  # [B, T, C, H, W]
        expert_actions = torch.stack(expert_actions, dim=1)    # [B, T, 2]

        pred = policy(observations)  # [B, T, 2]
        loss = F.mse_loss(pred, expert_actions)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optim.step()

        log["loss"] += float(loss.detach())
        log["n"] += 1

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(json.dumps({
                "step": step + 1,
                "bc_loss": log["loss"] / log["n"],
                "elapsed_s": round(elapsed, 1),
            }), flush=True)
            log = {"loss": 0.0, "n": 0}

        if (step + 1) % args.eval_every == 0:
            ev = deploy(policy, spec, args.eval_episodes, device)
            print(json.dumps({"event": "eval", "step": step + 1, **ev}), flush=True)

    final_eval = deploy(policy, spec, args.eval_episodes, device)
    print(json.dumps({"event": "final", "elapsed_s": round(time.time() - t0, 1), **final_eval}), flush=True)
    torch.save({
        "policy": policy.state_dict(),
        "config": vars(args),
        "final_eval": final_eval,
    }, os.path.join(args.output, "final.pt"))


if __name__ == "__main__":
    main()
