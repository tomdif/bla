"""Phase 14.4 — Action-ranking eval on robosuite Stack.

The question per Phase 14.3's split result:

  > Does action conditioning improve choosing actions (planner-facing),
    not just predicting state (perception-loop)?

For each (state_t, actual_action a*, actual_next_pos) in eval:
  1. Get predicted_pos(a*) from the model.
  2. For K-1 random alternative actions a_k, get predicted_pos(a_k).
  3. Check whether the model's prediction for a* is CLOSER to the
     actual next position than its predictions for any alternative.

If the model has learned action effects (action-conditioned mode):
  top-1 hit rate > chance (1/K).

If the model ignores actions (baseline mode):
  top-1 hit rate ≈ chance.

Metrics:
  top1_hit_rate     — fraction where a* prediction is closest
  top3_hit_rate     — fraction where a* prediction is in top-3
  rank_of_actual    — mean rank of a* among K candidates (1 = perfect)
  delta_distance    — mean (best_alternative_dist - actual_dist) / actual_dist

Usage:
    python scripts/phase14_action_ranking.py \\
        --cache /workspace/robosuite_local/stack \\
        --modes of_jepa_v0,of_jepa_v0_action --seed 0 \\
        --max-steps 1500 --k-candidates 8 \\
        --out /workspace/phase14b_ranking
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.robosuite_data import RobosuiteDataset, RobosuiteSpec
from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.identity_probe import hungarian_assign
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA, train_one_run, eval_future_mse


def eval_action_ranking(model, dataset, eval_idx, args, device,
                          use_action: bool, k_candidates: int = 8):
    """For each (state, actual_action) pair, rank actual_action against
    K-1 random alternatives.

    Returns:
      top1_hit_rate, top3_hit_rate, rank_of_actual, delta_distance
    """
    model.eval()
    top1 = 0; top3 = 0; n = 0
    sum_rank = 0.0; sum_delta = 0.0
    stride = args.jepa_stride
    with torch.no_grad():
        for ep_i in eval_idx:
            s = dataset[ep_i]
            video = s["video"].to(device)
            actions = s["actions"].to(device)
            gt_pos = s["positions"].to(device)
            slot_states, _ = model.encode_video(video)
            T = video.shape[0]
            id_dim = model.cfg.id_dim
            for t in range(T - stride):
                state_t = slot_states[t:t+1]  # [1, S, slot_dim]
                actual_action = actions[t:t+1]    # [1, action_dim]
                actual_next_pos = gt_pos[t + stride]  # [E, 2]

                # Predict for actual + K-1 random alternatives.
                candidates = [actual_action]
                for _ in range(k_candidates - 1):
                    rand_a = torch.empty_like(actual_action).uniform_(-1, 1)
                    candidates.append(rand_a)

                pred_pos_per_cand = []
                for a in candidates:
                    state_pred = model.predict_state_delta(state_t, a)
                    full_pred = torch.cat([state_t[..., :id_dim], state_pred], dim=-1)
                    pred_pos = model.slot_to_pos_aux(full_pred)[0]
                    pred_pos_per_cand.append(pred_pos)

                # Hungarian-match each candidate's predicted positions to GT.
                # Use the SAME match as for ranking — but a fairer comparison is
                # to compute distance to actual_next_pos for each candidate.
                dists = []
                for pp in pred_pos_per_cand:
                    pp_np = pp.cpu().numpy()
                    gt_np = actual_next_pos.cpu().numpy()
                    rows, cols, _ = hungarian_assign(pp_np, gt_np)
                    if len(rows) > 0:
                        d = float(((pp_np[rows] - gt_np[cols]) ** 2).sum(-1).mean())
                    else:
                        d = float("inf")
                    dists.append(d)

                # Rank candidates by distance to actual next pos (lower=better).
                dists = np.array(dists)
                order = np.argsort(dists)
                rank_of_actual = int(np.where(order == 0)[0][0]) + 1  # 1=best
                top1 += (rank_of_actual == 1)
                top3 += (rank_of_actual <= 3)
                sum_rank += rank_of_actual
                # Delta = (best_alternative - actual) / actual  (negative = actual better)
                best_alt = float(dists[1:].min())
                actual_d = float(dists[0])
                if actual_d > 1e-9:
                    sum_delta += (best_alt - actual_d) / actual_d
                n += 1
    if n == 0:
        return {"top1_hit_rate": float("nan"), "top3_hit_rate": float("nan"),
                 "rank_of_actual": float("nan"), "delta_distance": float("nan"),
                 "n": 0, "k_candidates": k_candidates}
    return {
        "top1_hit_rate":    top1 / n,
        "top3_hit_rate":    top3 / n,
        "rank_of_actual":   sum_rank / n,
        "delta_distance":   sum_delta / n,
        "n":                n,
        "k_candidates":     k_candidates,
        "chance_top1":      1.0 / k_candidates,
        "chance_top3":      3.0 / k_candidates,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", default="of_jepa_v0,of_jepa_v0_action")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--k-candidates", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset = RobosuiteDataset(RobosuiteSpec(cache_dir=args.cache, image_size=args.image_size))
    n = len(dataset); print(f"Episodes: {n}, action_dim={dataset.action_dim}", flush=True)
    indices = list(range(n))
    np.random.RandomState(0).shuffle(indices)
    train_idx, eval_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    modes = [m.strip() for m in args.modes.split(",")]
    results = {}
    for mode in modes:
        use_action = mode.endswith("_action")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        print(f"\n=== mode={mode} use_action={use_action} ===", flush=True)
        cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                            state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
        model = ActionConditionedOFJEPA(
            image_size=args.image_size, cfg=cfg,
            action_dim=dataset.action_dim, use_action=use_action,
        ).to(args.device)
        train_one_run(model, dataset, train_idx, args, args.device, use_action)
        # Joint eval: state-matching MSE (Phase 14.3 gates A,C) + action ranking (gate B).
        state_metrics = eval_future_mse(model, dataset, eval_idx, args, args.device, use_action)
        rank_metrics = eval_action_ranking(model, dataset, eval_idx, args, args.device,
                                             use_action, k_candidates=args.k_candidates)
        metrics = {**state_metrics, **rank_metrics, "mode": mode}
        results[mode] = metrics
        with open(out / f"seed{args.seed}_{mode}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  {mode}: {metrics}", flush=True)

    # Comparison
    print("\n=== Action-ranking comparison ===")
    base = results.get("of_jepa_v0")
    act = results.get("of_jepa_v0_action")
    if base and act:
        print(f"  top1_hit_rate: base={base['top1_hit_rate']:.3f}  +action={act['top1_hit_rate']:.3f}  (chance={base.get('chance_top1', 0):.3f})")
        print(f"  top3_hit_rate: base={base['top3_hit_rate']:.3f}  +action={act['top3_hit_rate']:.3f}")
        print(f"  rank_of_actual: base={base['rank_of_actual']:.3f}  +action={act['rank_of_actual']:.3f}  (perfect=1, chance={(args.k_candidates+1)/2:.1f})")
        improvement = act["top1_hit_rate"] - base["top1_hit_rate"]
        verdict = "✅ improved" if improvement > 0.05 else ("⚠ flat" if abs(improvement) < 0.02 else "❌ worse")
        print(f"  ranking improvement: {improvement:+.3f}  {verdict}")


if __name__ == "__main__":
    main()
