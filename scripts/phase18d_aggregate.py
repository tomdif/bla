"""Phase 18δ aggregator: compute mean ± std across seeds for Phase 17 metrics.

Reads summary.json from three Phase-17-style eval directories and reports
mean/std per mode, then evaluates Phase 18δ gates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


METRICS = ["improvement", "dir_score", "contact_rate", "success_rate",
            "mean_displacement", "pred_actual_corr", "overshoot_rate"]


def load_seed_results(path):
    with open(Path(path) / "summary.json") as f:
        s = json.load(f)
    return {r["mode"]: r for r in s["results"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-0", required=True)
    p.add_argument("--seed-1", required=True)
    p.add_argument("--seed-2", required=True)
    args = p.parse_args()

    seeds = {0: args.seed_0, 1: args.seed_1, 2: args.seed_2}
    per_seed = {s: load_seed_results(p) for s, p in seeds.items()}
    modes = set()
    for s in per_seed.values():
        modes |= set(s.keys())

    print(f"\n=== Phase 18δ multi-seed aggregation (seeds 0,1,2) ===\n")
    print(f"  {'mode':<22s}{'metric':<22s}{'seed0':>10s}{'seed1':>10s}{'seed2':>10s}{'mean':>10s}{'std':>10s}")
    summary = {}
    for mode in sorted(modes):
        print(f"\n  --- {mode} ---")
        summary[mode] = {}
        for metric in METRICS:
            vals = []
            for s in [0, 1, 2]:
                v = per_seed[s].get(mode, {}).get(metric)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append(float("nan"))
                else:
                    vals.append(float(v))
            arr = np.array(vals)
            valid = arr[~np.isnan(arr)]
            mean = float(np.mean(valid)) if len(valid) > 0 else float("nan")
            std = float(np.std(valid)) if len(valid) > 1 else float("nan")
            print(f"  {'':<22s}{metric:<22s}{vals[0]:>10.3f}{vals[1]:>10.3f}{vals[2]:>10.3f}{mean:>10.3f}{std:>10.3f}")
            summary[mode][metric] = {"per_seed": vals, "mean": mean, "std": std}

    # Phase 18δ gates
    print(f"\n=== Phase 18δ Gates ===")
    sp = summary.get("scripted_prior_cem", {})
    gt = summary.get("gt_closed_loop", {})
    if sp and gt:
        sp_imp_mean = sp["improvement"]["mean"]
        gt_imp_mean = gt["improvement"]["mean"]
        sp_dir_mean = sp["dir_score"]["mean"]
        gt_dir_mean = gt["dir_score"]["mean"]
        sp_corr_mean = sp["pred_actual_corr"]["mean"]

        g1 = sp_imp_mean >= gt_imp_mean
        g2 = sp_imp_mean >= 0.18
        g3 = sp_dir_mean >= 0.90 * gt_dir_mean
        g4 = sp_corr_mean > 0

        print(f"  G1 mean(scripted_prior_cem improvement) >= mean(oracle improvement):")
        print(f"     {sp_imp_mean:.3f} vs {gt_imp_mean:.3f}  {'PASS' if g1 else 'FAIL'}")
        print(f"  G2 mean(scripted_prior_cem improvement) >= 0.18:")
        print(f"     {sp_imp_mean:.3f}  {'PASS' if g2 else 'FAIL'}")
        print(f"  G3 mean(scripted_prior_cem dir_score) >= 0.9 × oracle dir_score:")
        print(f"     {sp_dir_mean:.3f} vs {0.9*gt_dir_mean:.3f}  {'PASS' if g3 else 'FAIL'}")
        print(f"  G4 (diagnostic) mean(pred-actual corr) > 0:")
        print(f"     {sp_corr_mean:.3f}  {'PASS' if g4 else 'FAIL'}")

        # "At least 2/3 seeds positive vs oracle"
        per_seed_gap = [sp["improvement"]["per_seed"][s] - gt["improvement"]["per_seed"][s]
                          for s in range(3)]
        n_positive = sum(1 for g in per_seed_gap if g > -0.02)  # within 2pp counts as positive/tied
        g_consistency = n_positive >= 2
        print(f"\n  Per-seed gap (scripted_prior - oracle): {[f'{g:+.3f}' for g in per_seed_gap]}")
        print(f"  Consistency: {n_positive}/3 seeds >= oracle (within 2pp)")
        print(f"     {'PASS' if g_consistency else 'FAIL'}")

        n_pass = int(g1) + int(g2) + int(g3)
        verdict = (
            "3/3 main + G4 positive → Phase 17 ROBUST" if n_pass == 3 and g4 else
            "3/3 main, G4 fail → planning works but predictor calibration fragile" if n_pass == 3 else
            "2/3 main → partial confirmation" if n_pass == 2 else
            "1/3 → lucky seed concern" if n_pass == 1 else
            "0/3 → Phase 17 fluke"
        )
        print(f"\n  OVERALL: {verdict}")

        # Save aggregate
        out = Path(args.seed_1).parent / "aggregate.json"
        with open(out, "w") as f:
            json.dump({
                "per_seed": {s: per_seed[s] for s in [0, 1, 2]},
                "summary": summary,
                "gates": {"g1": g1, "g2": g2, "g3": g3, "g4": g4,
                           "consistency": g_consistency,
                           "n_main_pass": n_pass, "verdict": verdict},
                "per_seed_oracle_gap": per_seed_gap,
            }, f, indent=2)
        print(f"\n  Aggregate written to {out}")


if __name__ == "__main__":
    main()
