"""Phase 5 — RL-trained router for compute economy.

The asymmetric-scaling claim that this phase tests:
*every joule of compute should be routed to the substrate that needs it.*
On a multi-difficulty task mix, the router learns to spend cheap
parametric inference on easy questions and expensive retrieval on hard
ones, while preserving accuracy.

Action space (a 2-action subset of the full 7-action B.L.A. space — Phase
2 built the API, this phase trains a real allocator on it):
  * SHALLOW: parametric LLM only (~ small FLOPs)
  * DEEP:    retrieval-augmented LLM (~ larger FLOPs)

Reward = correctness − λ × flops_used / flops_max

Router architecture: question text → sentence-transformer embedding →
2-layer MLP → 2-way logits. Trained with REINFORCE + entropy bonus.

Gate:
  * routed compute on hard tasks ≥ 10× compute on easy tasks
  * accuracy on easy tasks does not drop more than 5pp vs always-DEEP baseline
  * total compute ≤ 30% of always-DEEP

Run:
  python3 scripts/phase5_router_rl.py \\
      --memory-db runs/phase3_facts/symbolic.db \\
      --output runs/phase5
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

from system1_jepa.factual_corpus import (
    COUNTRY_CAPITALS,
    PLANET_ORBITS,
    ELEMENT_ATOMIC_NUMBERS,
)
from system2_dca.symbolic_memory import SymbolicMemory
from verification import CommitmentObject

# Reuse Phase 4 LLM utilities
from scripts.phase4_qa import (
    _NATURAL_LANGUAGE_TEMPLATE,
    generate,
    load_llm,
    retrieve_context,
    score,
)


# Approximate FLOPs (relative units — we just need a consistent ratio)
FLOPS_SHALLOW = 1.0     # parametric: just the LLM forward
FLOPS_DEEP = 3.0        # retrieval + LLM forward with longer context


SHALLOW = 0
DEEP = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-db", required=True)
    p.add_argument("--llm", default="gpt2")
    p.add_argument("--router-d", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lam-flops-start", type=float, default=0.0)
    p.add_argument("--lam-flops-end", type=float, default=0.5)
    p.add_argument("--entropy-bonus", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_dataset() -> list[dict]:
    """Combine corpora into one labeled task mix.

    Difficulty label:
      easy: country_capital_is — GPT-2 small parametric ~47% accurate
      hard: atomic_number     — GPT-2 small parametric ~0% accurate
      med:  orbits             — GPT-2 small parametric ~16% accurate

    Truncating each to 40 to keep balance.
    """
    cases: list[dict] = []
    for country, capital, _ in COUNTRY_CAPITALS[:40]:
        cases.append({
            "predicate": "country_capital_is",
            "subject": country,
            "ground_truth": capital,
            "difficulty": "easy",
            "question": f"Q: What is the capital of {country}?\nA:",
        })
    for body, parent, _ in PLANET_ORBITS[:19]:
        cases.append({
            "predicate": "orbits",
            "subject": body,
            "ground_truth": parent,
            "difficulty": "med",
            "question": f"Q: What does {body} orbit?\nA:",
        })
    for elem, z in ELEMENT_ATOMIC_NUMBERS[:40]:
        cases.append({
            "predicate": "atomic_number",
            "subject": elem,
            "ground_truth": str(z),
            "difficulty": "hard",
            "question": f"Q: What is the atomic number of {elem}?\nA:",
        })
    return cases


def split(cases: list[dict], train_frac: float = 0.7) -> tuple[list, list]:
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(cases))
    n_train = int(len(cases) * train_frac)
    train = [cases[i] for i in perm[:n_train]]
    test = [cases[i] for i in perm[n_train:]]
    return train, test


class RouterPolicy(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def precompute_outcomes(
    cases: list[dict],
    sm: SymbolicMemory,
    tok,
    model,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict]:
    """For each case, run BOTH actions once and cache outcome + flops.
    The router is then trained on cached samples — fast inner loop."""
    out = []
    for i, case in enumerate(cases):
        # SHALLOW: parametric
        ans_s = generate(tok, model, case["question"], max_new_tokens)
        ok_s = score(ans_s, case["ground_truth"])

        # DEEP: retrieval-augmented
        context, evidence = retrieve_context(sm, case, top_k=1)
        prompt_d = f"{context}\n{case['question']}" if context else case["question"]
        ans_d = generate(tok, model, prompt_d, max_new_tokens)
        ok_d = score(ans_d, case["ground_truth"])

        out.append({
            **case,
            "shallow_correct": float(ok_s),
            "deep_correct": float(ok_d),
            "shallow_flops": FLOPS_SHALLOW,
            "deep_flops": FLOPS_DEEP,
            "shallow_answer": ans_s,
            "deep_answer": ans_d,
            "evidence": evidence,
        })
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "precompute", "done": i + 1, "total": len(cases)}), flush=True)
    return out


def embed_questions(cases: list[dict], embedder) -> torch.Tensor:
    texts = [c["question"] for c in cases]
    return torch.from_numpy(embedder.embed(texts)).float()


def reinforce_train(
    train: list[dict],
    train_emb: torch.Tensor,
    args,
    device: torch.device,
) -> tuple[RouterPolicy, dict]:
    embed_dim = train_emb.shape[1]
    policy = RouterPolicy(embed_dim, hidden=args.router_d, n_actions=2).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    flops_max = max(FLOPS_SHALLOW, FLOPS_DEEP)

    for epoch in range(args.epochs):
        # cosine ramp on lambda_flops to start with high accuracy and gradually penalize compute
        progress = epoch / max(args.epochs - 1, 1)
        lam = args.lam_flops_start + (args.lam_flops_end - args.lam_flops_start) * progress

        idx = torch.randperm(len(train))
        epoch_loss = 0.0
        epoch_reward = 0.0
        action_counts = [0, 0]
        n_batches = 0

        for start in range(0, len(train), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            x = train_emb[batch_idx].to(device)
            logits = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            rewards = []
            for j, a in zip(batch_idx.tolist(), actions.tolist()):
                if a == SHALLOW:
                    correct = train[j]["shallow_correct"]
                    flops = FLOPS_SHALLOW
                else:
                    correct = train[j]["deep_correct"]
                    flops = FLOPS_DEEP
                r = correct - lam * (flops / flops_max)
                rewards.append(r)
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
def evaluate(policy: RouterPolicy, cases: list[dict], emb: torch.Tensor, device: torch.device) -> dict:
    policy.eval()
    logits = policy(emb.to(device))
    actions = logits.argmax(dim=-1).cpu().tolist()

    by_difficulty: dict[str, dict] = {}
    total_correct = 0
    total_flops = 0.0
    always_deep_correct = 0
    always_deep_flops = 0.0
    always_shallow_correct = 0
    always_shallow_flops = 0.0

    for case, a in zip(cases, actions):
        d = case["difficulty"]
        bucket = by_difficulty.setdefault(d, {"n": 0, "shallow": 0, "deep": 0,
                                              "correct": 0, "flops": 0.0,
                                              "always_deep_correct": 0,
                                              "always_deep_flops": 0.0,
                                              "always_shallow_correct": 0,
                                              "always_shallow_flops": 0.0})
        bucket["n"] += 1
        if a == SHALLOW:
            bucket["shallow"] += 1
            bucket["correct"] += case["shallow_correct"]
            bucket["flops"] += FLOPS_SHALLOW
        else:
            bucket["deep"] += 1
            bucket["correct"] += case["deep_correct"]
            bucket["flops"] += FLOPS_DEEP

        bucket["always_deep_correct"] += case["deep_correct"]
        bucket["always_deep_flops"] += FLOPS_DEEP
        bucket["always_shallow_correct"] += case["shallow_correct"]
        bucket["always_shallow_flops"] += FLOPS_SHALLOW

        if a == SHALLOW:
            total_correct += case["shallow_correct"]
            total_flops += FLOPS_SHALLOW
        else:
            total_correct += case["deep_correct"]
            total_flops += FLOPS_DEEP

        always_deep_correct += case["deep_correct"]
        always_deep_flops += FLOPS_DEEP
        always_shallow_correct += case["shallow_correct"]
        always_shallow_flops += FLOPS_SHALLOW

    n = len(cases)
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

    # the gate: compute split + accuracy on easy
    if "easy" in summary["by_difficulty"] and "hard" in summary["by_difficulty"]:
        easy_flops = summary["by_difficulty"]["easy"]["router_flops_per_query"]
        hard_flops = summary["by_difficulty"]["hard"]["router_flops_per_query"]
        summary["compute_split_hard_over_easy"] = hard_flops / max(easy_flops, 1e-6)
        easy_acc_drop = summary["by_difficulty"]["easy"]["always_deep_accuracy"] - summary["by_difficulty"]["easy"]["router_accuracy"]
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

    sm = SymbolicMemory(db_path=args.memory_db)
    tok, model = load_llm(args.llm)
    model.to(device)

    print(json.dumps({"event": "loaded", "llm": args.llm,
                      "params": sum(p.numel() for p in model.parameters())}), flush=True)

    cases = build_dataset()
    print(json.dumps({"event": "dataset", "n": len(cases)}), flush=True)

    train, test = split(cases, train_frac=0.7)

    print(json.dumps({"event": "precompute_train"}), flush=True)
    train = precompute_outcomes(train, sm, tok, model, args.max_new_tokens, device)
    print(json.dumps({"event": "precompute_test"}), flush=True)
    test = precompute_outcomes(test, sm, tok, model, args.max_new_tokens, device)

    train_emb = embed_questions(train, sm._memoria.embedder)
    test_emb = embed_questions(test, sm._memoria.embedder)
    print(json.dumps({"event": "embedded", "train": list(train_emb.shape), "test": list(test_emb.shape)}), flush=True)

    policy, train_history = reinforce_train(train, train_emb, args, device)

    test_summary = evaluate(policy, test, test_emb, device)
    print(json.dumps({"event": "summary", **test_summary}, indent=2))

    with open(os.path.join(args.output, "phase5.json"), "w") as f:
        json.dump({"summary": test_summary, "train_history": train_history,
                   "test_cases": [{k: v for k, v in c.items() if k != "evidence"} for c in test]}, f, indent=2)

    torch.save({
        "policy": policy.state_dict(),
        "config": vars(args),
        "summary": test_summary,
    }, os.path.join(args.output, "router.pt"))


if __name__ == "__main__":
    main()
