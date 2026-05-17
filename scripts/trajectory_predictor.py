"""Trajectory predictor: forecast final val/PAL from first N training steps.

Pulls JSONL training logs from runs/{run_id}/train.log, extracts trajectory
features (val curve, train-loss curve, etc.) over the first `--cut-step`
steps, joins with PAL eval results from runs/{run_id}/pal_eval_*/eval.json,
and trains/evaluates a simple predictor by leave-one-out CV.

Goal: predict whether a new architecture will hit the 3.5% PAL ceiling
or beat it, based on the first ~1500 training steps. With 5 trajectories,
the predictor is more "structured heuristic" than ML model; the real value
is the feature-extraction pipeline that scales as we collect more runs.

Usage:
    python scripts/trajectory_predictor.py \
        --runs run18 run19 run20 run21 run22 \
        --cut-step 1500 \
        --remote root@212.247.220.138:19448
        # or --local for runs/{run_id}/...
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class Trajectory:
    run_id: str
    architecture: str
    n_params: int
    # Curves over the first cut_step training steps
    train_loss_at_step: dict   # step -> train_loss
    val_loss_at_step: dict     # step -> val_loss
    # Best-val-so-far snapshots
    best_val_at_step_500: float
    best_val_at_step_1000: float
    best_val_at_step_1500: float
    # Loss-variance signatures
    train_loss_std_first_1000: float
    val_loss_std_first_1500: float
    # Convergence indicators
    val_drop_step_500_to_1500: float    # negative = improving
    n_val_dips_below_2: int              # how many checkpoints went under 2.0
    # Labels (the things we want to predict)
    final_best_val: float                # min val over all checkpoints
    final_pal_best: Optional[float]      # max PAL over evaluated ckpts
    final_pal_at_best_val_ckpt: Optional[float]


def _shell(cmd: list[str], remote: Optional[str] = None) -> str:
    """Run shell command locally or via ssh. Returns stdout."""
    if remote:
        host, port = remote.split(":")
        full = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", port, host] + [" ".join(cmd)]
    else:
        full = cmd
    r = subprocess.run(full, capture_output=True, text=True, check=True)
    return r.stdout


def parse_train_log(text: str) -> dict:
    """Pull JSON event lines from a train.log. Returns dict with
    val/train trajectories and metadata."""
    val_by_step = {}
    train_by_step = {}
    init = None
    for line in text.split("\n"):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "init":
            init = obj
        elif obj.get("event") == "val":
            val_by_step[int(obj["step"])] = float(obj["val_loss"])
        elif "step" in obj and "loss" in obj and "event" not in obj:
            train_by_step[int(obj["step"])] = float(obj["loss"])
    return {"init": init, "val": val_by_step, "train": train_by_step}


def parse_pal_evals(eval_dir: str, remote: Optional[str] = None) -> dict:
    """For a given runs/{run_id} dir, scan pal_eval_*/eval.json. Returns
    step -> {accuracy, code_ran}."""
    if remote:
        # ls + cat each eval.json
        host_port = remote.split(":")
        try:
            ls = _shell(["ls", f"{eval_dir}/pal_eval_*/eval.json"], remote=remote).strip()
        except subprocess.CalledProcessError:
            return {}
        results = {}
        for path in ls.split("\n"):
            if not path:
                continue
            m = re.search(r"pal_eval_step(\d+)", path)
            if not m:
                # final.pt -> "pal_eval_final"
                m_final = re.search(r"pal_eval_final", path)
                if m_final:
                    step = None
                else:
                    continue
            else:
                step = int(m.group(1))
            try:
                content = _shell(["cat", path], remote=remote)
                obj = json.loads(content)
            except Exception:
                continue
            s = obj.get("summary", {})
            results[step] = {"accuracy": s.get("accuracy", 0.0),
                              "code_ran": s.get("code_ran", 0),
                              "n_tested": s.get("n_tested", 0)}
        return results
    else:
        import glob
        results = {}
        for path in glob.glob(f"{eval_dir}/pal_eval_*/eval.json"):
            m = re.search(r"pal_eval_step(\d+)", path)
            step = int(m.group(1)) if m else None
            try:
                obj = json.load(open(path))
            except Exception:
                continue
            s = obj.get("summary", {})
            results[step] = {"accuracy": s.get("accuracy", 0.0),
                              "code_ran": s.get("code_ran", 0),
                              "n_tested": s.get("n_tested", 0)}
        return results


def build_trajectory(run_id: str, run_root: str, cut_step: int,
                      remote: Optional[str] = None) -> Optional[Trajectory]:
    log_path = f"{run_root}/{run_id}/train.log"
    if remote:
        try:
            text = _shell(["cat", log_path], remote=remote)
        except subprocess.CalledProcessError:
            print(f"  could not read {log_path}", file=sys.stderr)
            return None
    else:
        try:
            text = open(log_path).read()
        except OSError:
            return None

    parsed = parse_train_log(text)
    if not parsed["init"]:
        return None

    init_cfg = parsed["init"]["config"]
    arch_parts = ["mt_ssm"]
    if init_cfg.get("use_memory"):
        arch_parts.append("memory")
    if init_cfg.get("use_attractor"):
        arch_parts.append("attractor")
    architecture = "+".join(arch_parts)
    n_params = parsed["init"]["n_params"]

    val_curve = parsed["val"]
    train_curve = parsed["train"]

    # Best-val-so-far at fixed checkpoints
    def best_up_to(step):
        relevant = [v for s, v in val_curve.items() if s <= step]
        return min(relevant) if relevant else float("nan")

    # Train-loss variance (signal of training noisiness)
    train_first1k = [v for s, v in train_curve.items() if s <= 1000]
    train_std_1k = float(np.std(train_first1k)) if len(train_first1k) >= 5 else float("nan")
    val_first1500 = [v for s, v in val_curve.items() if s <= 1500]
    val_std_1500 = float(np.std(val_first1500)) if len(val_first1500) >= 2 else float("nan")

    val_at_500 = val_curve.get(500, float("nan"))
    val_at_1500 = best_up_to(1500)
    val_drop = val_at_1500 - val_at_500   # negative means improving

    n_val_dips = sum(1 for v in val_curve.values() if v < 2.0)

    # Truncate trajectories to cut_step for input features
    train_in = {s: v for s, v in train_curve.items() if s <= cut_step}
    val_in = {s: v for s, v in val_curve.items() if s <= cut_step}

    # Labels: final best val + best PAL
    final_best_val = min(val_curve.values()) if val_curve else float("nan")

    # PAL evaluation results, if present
    pal_results = parse_pal_evals(f"{run_root}/{run_id}", remote=remote)
    best_pal = max((r["accuracy"] for r in pal_results.values()), default=None)

    # Try to find ckpt at the best-val step
    best_val_step = (min(val_curve.items(), key=lambda kv: kv[1])[0]
                     if val_curve else None)
    nearest_eval_step = None
    if pal_results and best_val_step:
        nearest_eval_step = min(pal_results.keys(),
                                key=lambda s: abs(s - best_val_step) if s else float("inf"))
    pal_at_best_val = (pal_results.get(nearest_eval_step, {}).get("accuracy")
                       if nearest_eval_step else None)

    return Trajectory(
        run_id=run_id, architecture=architecture, n_params=n_params,
        train_loss_at_step=train_in, val_loss_at_step=val_in,
        best_val_at_step_500=best_up_to(500),
        best_val_at_step_1000=best_up_to(1000),
        best_val_at_step_1500=best_up_to(1500),
        train_loss_std_first_1000=train_std_1k,
        val_loss_std_first_1500=val_std_1500,
        val_drop_step_500_to_1500=val_drop,
        n_val_dips_below_2=n_val_dips,
        final_best_val=final_best_val,
        final_pal_best=best_pal,
        final_pal_at_best_val_ckpt=pal_at_best_val,
    )


def feature_vector(t: Trajectory) -> np.ndarray:
    """Compact feature representation for the predictor."""
    return np.array([
        t.n_params / 1e9,
        t.best_val_at_step_500,
        t.best_val_at_step_1000,
        t.best_val_at_step_1500,
        t.train_loss_std_first_1000,
        t.val_loss_std_first_1500,
        t.val_drop_step_500_to_1500,
        float(t.n_val_dips_below_2),
        float("memory" in t.architecture),
        float("attractor" in t.architecture),
    ], dtype=np.float64)


FEATURE_NAMES = ["params_B", "best_val_500", "best_val_1000", "best_val_1500",
                 "train_std_1k", "val_std_1500", "val_drop_500_1500",
                 "n_dips_below_2", "has_memory", "has_attractor"]


def loocv_predict(features: np.ndarray, labels: np.ndarray):
    """Leave-one-out cross-validation with Ridge regression."""
    from sklearn.linear_model import Ridge
    preds = []
    n = len(labels)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = Ridge(alpha=1.0)
        model.fit(features[mask], labels[mask])
        preds.append(float(model.predict(features[i:i+1])[0]))
    return np.array(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--run-root", default="/workspace/bla/runs")
    p.add_argument("--cut-step", type=int, default=1500,
                   help="Truncate trajectories to first N steps for feature extraction")
    p.add_argument("--remote", default=None,
                   help="ssh target as user@host:port")
    p.add_argument("--output", default="runs/trajectory_predictor/trajectories.json")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    trajectories = []
    for run_id in args.runs:
        t = build_trajectory(run_id, args.run_root, args.cut_step, remote=args.remote)
        if t is None:
            print(f"  skipped {run_id}: no train.log or empty curve", file=sys.stderr)
            continue
        trajectories.append(t)
        print(f"loaded {run_id}: params={t.n_params/1e9:.2f}B  arch={t.architecture}  "
              f"best_val_1500={t.best_val_at_step_1500:.3f}  "
              f"final_val={t.final_best_val:.3f}  "
              f"pal={t.final_pal_best}")

    # Save the structured data for future runs
    with open(args.output, "w") as f:
        json.dump([asdict(t) for t in trajectories], f, indent=2)
    print(f"\nsaved {len(trajectories)} trajectories -> {args.output}\n")

    if len(trajectories) < 3:
        print("not enough trajectories for LOOCV; need 3+ runs", file=sys.stderr)
        return

    # Build feature matrix
    feats = np.array([feature_vector(t) for t in trajectories])
    # Two targets: final_best_val and final_pal_best (skip runs missing PAL)
    val_labels = np.array([t.final_best_val for t in trajectories])

    print("=== predict final best val ===")
    val_preds = loocv_predict(feats, val_labels)
    print(f"{'run':<8s} {'true_val':>10s} {'pred_val':>10s} {'err':>8s}")
    for t, p_ in zip(trajectories, val_preds):
        err = p_ - t.final_best_val
        print(f"{t.run_id:<8s} {t.final_best_val:>10.3f} {p_:>10.3f} {err:>+8.3f}")
    mae = float(np.mean(np.abs(val_preds - val_labels)))
    print(f"LOOCV MAE: {mae:.3f}")

    have_pal = [(t, fv) for t, fv in zip(trajectories, feats)
                if t.final_pal_best is not None]
    if len(have_pal) >= 3:
        pal_feats = np.array([fv for _, fv in have_pal])
        pal_labels = np.array([t.final_pal_best for t, _ in have_pal])
        print("\n=== predict final best PAL ===")
        pal_preds = loocv_predict(pal_feats, pal_labels)
        print(f"{'run':<8s} {'true_pal':>10s} {'pred_pal':>10s} {'err':>8s}")
        for (t, _), p_ in zip(have_pal, pal_preds):
            err = p_ - t.final_pal_best
            print(f"{t.run_id:<8s} {t.final_pal_best:>10.4f} {p_:>10.4f} {err:>+8.4f}")
        print(f"LOOCV MAE: {float(np.mean(np.abs(pal_preds - pal_labels))):.4f}")

    # Feature importance from a fit on ALL data
    from sklearn.linear_model import Ridge
    full = Ridge(alpha=1.0).fit(feats, val_labels)
    print("\n=== feature weights (predict final best val) ===")
    for name, w in zip(FEATURE_NAMES, full.coef_):
        print(f"  {name:<20s}: {w:+.4f}")


if __name__ == "__main__":
    main()
