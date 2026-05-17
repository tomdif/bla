"""Phase 14.6 — Train on scripted v3, eval on perturbed-policy held-out sets.

Locks the Phase 14.5 strong-pass claim by asking: did the model learn
**action effects**, or just **the v3 scripted-policy distribution**?

Three perturbed eval sets (each 50 episodes):
  A. strength — OSC gain uniform[4, 16] (v3 was fixed 10)
  B. noise    — action noise sigma uniform[0.05, 0.45] (v3 was 0.20)
  C. horizon  — 120 frames (v3 was 80)

Per-perturbation gates:
  G1. top1_hit_rate(+action) >= 0.25            # 2x chance
  G2. pos_mse(+action) / pos_mse(baseline) <= 0.95

Secondary diagnostic (logged, not gated):
  state_mse(+action) / state_mse(baseline)

Usage:
    python scripts/phase14_generalization.py \\
        --train-cache /workspace/robosuite_local/stack_scripted \\
        --eval-caches /workspace/robosuite_local/stack_perturb_strength,\\
/workspace/robosuite_local/stack_perturb_noise,\\
/workspace/robosuite_local/stack_perturb_horizon \\
        --eval-labels strength,noise,horizon \\
        --modes of_jepa_v0,of_jepa_v0_action \\
        --seed 0 --max-steps 1500 --jepa-stride 4 --k-candidates 8 \\
        --out /workspace/phase14d_generalization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from system1_jepa.robosuite_data import RobosuiteDataset, RobosuiteSpec
from system1_jepa.of_jepa import OFJEPAConfig
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA, train_one_run, eval_future_mse
from scripts.phase14_action_ranking import eval_action_ranking


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--eval-caches", required=True,
                    help="Comma-separated paths to perturbed eval caches.")
    p.add_argument("--eval-labels", required=True,
                    help="Comma-separated labels (one per eval cache).")
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
    eval_caches = [c.strip() for c in args.eval_caches.split(",")]
    eval_labels = [l.strip() for l in args.eval_labels.split(",")]
    assert len(eval_caches) == len(eval_labels), "eval-caches and eval-labels must match"

    # Datasets
    train_dataset = RobosuiteDataset(RobosuiteSpec(cache_dir=args.train_cache,
                                                    image_size=args.image_size))
    print(f"Train cache: {args.train_cache}  episodes={len(train_dataset)}", flush=True)
    eval_datasets = []
    for path, label in zip(eval_caches, eval_labels):
        ds = RobosuiteDataset(RobosuiteSpec(cache_dir=path, image_size=args.image_size))
        eval_datasets.append((label, ds))
        print(f"Eval cache [{label}]: {path}  episodes={len(ds)}", flush=True)

    # 80% of v3 for training (same split as Phase 14.5).
    n_train = len(train_dataset)
    indices = list(range(n_train))
    np.random.RandomState(0).shuffle(indices)
    train_idx = indices[: int(0.8 * n_train)]

    modes = [m.strip() for m in args.modes.split(",")]
    all_results = {}
    for mode in modes:
        use_action = mode.endswith("_action")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        print(f"\n=== mode={mode} use_action={use_action} ===", flush=True)
        cfg = OFJEPAConfig(n_files=args.n_slots, id_dim=args.slot_dim // 2,
                            state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim)
        model = ActionConditionedOFJEPA(image_size=args.image_size, cfg=cfg,
                                          action_dim=7, use_action=use_action).to(args.device)
        train_one_run(model, train_dataset, train_idx, args, args.device, use_action)

        # Evaluate on each perturbed cache.
        all_results[mode] = {}
        for label, eval_ds in eval_datasets:
            eval_idx = list(range(len(eval_ds)))  # use all of perturbed set
            state_m = eval_future_mse(model, eval_ds, eval_idx, args, args.device, use_action)
            rank_m = eval_action_ranking(model, eval_ds, eval_idx, args, args.device,
                                            use_action, k_candidates=args.k_candidates)
            metrics = {**state_m, **rank_m, "mode": mode, "eval_label": label}
            all_results[mode][label] = metrics
            with open(out / f"seed{args.seed}_{mode}_eval_{label}.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  [{mode} / {label}]  state_mse={state_m['future_state_mse']:.4e}  "
                  f"pos_mse={state_m['future_pos_mse']:.4e}  "
                  f"top1={rank_m['top1_hit_rate']:.3f}  rank={rank_m['rank_of_actual']:.3f}",
                  flush=True)

    # Gate evaluation per perturbation.
    print("\n=== Generalization gate verdicts ===")
    summary = {}
    base = all_results.get("of_jepa_v0", {})
    act = all_results.get("of_jepa_v0_action", {})
    for label in eval_labels:
        b = base.get(label); a = act.get(label)
        if not (b and a):
            print(f"  {label}: missing results, skip"); continue
        g1_top1 = a["top1_hit_rate"]
        g1_pass = g1_top1 >= 0.25
        g2_ratio = a["future_pos_mse"] / max(b["future_pos_mse"], 1e-9)
        g2_pass = g2_ratio <= 0.95
        s_ratio = a["future_state_mse"] / max(b["future_state_mse"], 1e-9)
        print(f"  [{label}]  G1 top1={g1_top1:.3f} (>=0.25 {'PASS' if g1_pass else 'FAIL'})"
              f"   G2 pos_ratio={g2_ratio:.3f} (<=0.95 {'PASS' if g2_pass else 'FAIL'})"
              f"   state_ratio={s_ratio:.3f}")
        summary[label] = {
            "g1_top1": g1_top1, "g1_pass": bool(g1_pass),
            "g2_pos_ratio": g2_ratio, "g2_pass": bool(g2_pass),
            "state_ratio": s_ratio,
            "verdict": "PASS" if (g1_pass and g2_pass) else "FAIL",
        }
    n_pass = sum(1 for s in summary.values() if s["verdict"] == "PASS")
    summary["overall"] = {
        "n_pass": n_pass, "n_total": len(summary),
        "verdict": (
            "3/3 clean" if n_pass == 3 else
            "2/3 partial" if n_pass == 2 else
            "1/3 mostly script-bound" if n_pass == 1 else
            "0/3 overfit"
        ),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall: {summary['overall']['verdict']}  ({n_pass}/{len(eval_labels)} pass)")


if __name__ == "__main__":
    main()
