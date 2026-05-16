"""Phase 3 Slot-JEPA stress sweep.

Runs the Phase-2-validated slot-JEPA training across a matrix of
{seed, mode, n_targets, n_distractors, K} configurations, then evaluates
the linear-probe representation metric at every J in --eval-J. Aggregates
per-cell into mean ± stderr ± 95% CI and writes a manifest + raw rows +
aggregate CSV so the run is reproducible and orphan-proof.

Usage (pod target — matches the agreed Phase-3 spec):

    python scripts/slot_jepa_phase3.py \
        --seeds 0,1,2,3,4 \
        --targets 3,5,8 \
        --distractors 2,5,10 \
        --moving-distractors \
        --J 10,20,40,80 \
        --K 3,5,10 \
        --modes slot_delta,slot_dense_update,dense_jepa_flatten,copy \
        --out artifacts/phase3_slot_delta_stress/

For local smoke runs, pass single values:

    python scripts/slot_jepa_phase3.py --seeds 0 --targets 3 \
        --distractors 2 --K 5 --J 5,10 --modes slot_delta,copy \
        --steps 600 --out /tmp/phase3_smoke/
"""
from __future__ import annotations
import argparse, csv, datetime, itertools, json, math, os, subprocess, sys, time

# t critical values for 95% CI two-sided, df = n-1
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t_crit95(n):
    if n < 2:
        return float("nan")
    df = n - 1
    if df in _T95:
        return _T95[df]
    # Fall through for large df: 1.96 normal approx.
    return 1.96


def parse_list(s, type_=int):
    return [type_(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated seeds. Phase-3 default uses 5 seeds.")
    p.add_argument("--targets", default="3,5,8",
                   help="Comma-separated n_targets values.")
    p.add_argument("--distractors", default="2,5,10",
                   help="Comma-separated n_distractors values.")
    p.add_argument("--K", default="3,5,10",
                   help="Comma-separated visible_steps (K) values.")
    p.add_argument("--J", default="10,20,40,80",
                   help="Comma-separated hidden_steps (J) eval values.")
    p.add_argument("--n-slots-list", default="16",
                   help="Comma-separated n_slots values to sweep. Only "
                         "applies to slot_* modes; dense/copy ignore it. "
                         "Phase-5A uses 16,32,64.")
    p.add_argument("--target-active-slots-list", default="0",
                   help="Phase-5B: comma-separated target_active_slots values. "
                         "0 = fixed slots (Phase-5A behaviour); >0 = dynamic "
                         "top-K active gate. Combos where target ≥ n_slots are "
                         "skipped.")
    p.add_argument("--J-train", type=int, default=10,
                   help="Single J value used during training. Eval scans --J.")
    p.add_argument("--modes",
                   default="slot_delta,slot_dense_update,dense_jepa_flatten,copy",
                   help="Comma-separated modes. dense_jepa_flatten and "
                         "dense_jepa_mean are aliases that map to "
                         "--mode dense_jepa --probe-pool {flatten,mean}.")
    p.add_argument("--moving-distractors", action="store_true",
                   help="Pass --moving-distractors to every sub-run.")
    p.add_argument("--partial-observability", action="store_true")
    p.add_argument("--obs-radius", type=float, default=8.0)
    p.add_argument("--perceptual-noise", type=float, default=0.0,
                   help="Phase-4A: Gaussian pixel-noise σ on the rendered "
                         "observation. Default 0.0 = Phase-3 behaviour.")
    p.add_argument("--color-randomization", action="store_true",
                   help="Phase-4B: randomize entity colors per episode.")
    p.add_argument("--background-randomization", action="store_true",
                   help="Phase-4B: low-magnitude random per-pixel background.")
    p.add_argument("--soft-render", action="store_true",
                   help="Phase-6: Gaussian-footprint entity rendering.")
    p.add_argument("--soft-sigma", type=float, default=1.5)

    p.add_argument("--steps", type=int, default=3000,
                   help="Self-supervised training steps per sub-run.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32,
                   help="Phase 5A uses 48 or 64 to fit 16+ entities.")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--episode-length", type=int, default=24)
    p.add_argument("--probe-episodes", type=int, default=32)
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-lr", type=float, default=5e-3)
    p.add_argument("--mask-bias-init", type=float, default=0.0,
                   help="Initial bias of the slot change-mask head. "
                         "Hardware-sensitive: torch 2.4/CPU finds a good "
                         "non-collapsed basin at -2.0 (Phase 2 local); "
                         "torch 2.8/Blackwell GPU needs ~0.0 to escape "
                         "mask=0 collapse. Default 0.0 (pod-friendly).")
    p.add_argument("--device", default="auto",
                   help="'auto' picks cuda if available, else cpu.")

    p.add_argument("--out", required=True, help="Output root directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without launching subprocesses.")
    return p.parse_args()


def mode_to_train_args(mode, mask_bias_init, target_active_slots=0):
    """Translate phase-3 mode aliases to slot_jepa_train.py args.

    mask_bias_init is hardware-dependent — torch 2.4 / CPU lands the
    slot_delta optimization in a non-collapsed basin at bias=-2.0
    (Phase 2 local CPU), torch 2.8 / Blackwell GPU lands in a different
    basin and needs bias≈0.0 to stay out of mask=0 collapse. The
    orchestrator exposes it as a flag and stamps the chosen value in
    the manifest.

    target_active_slots > 0 switches slot_delta to update_mode=dynamic
    (Phase 5B). 0 stays in update_mode=delta (Phase 5A behaviour).
    """
    if mode == "slot_delta":
        args = ["--mode", "slot_delta",
                "--sparsity-weight", "5e-3",
                "--bimodal-weight", "1e-3",
                "--mask-bias-init", str(mask_bias_init)]
        if target_active_slots > 0:
            args += ["--update-mode", "dynamic",
                     "--target-active-slots", str(target_active_slots)]
        return args
    if mode == "slot_dense_update":
        return ["--mode", "slot_delta", "--update-mode", "dense",
                "--sparsity-weight", "5e-3",
                "--bimodal-weight", "1e-3",
                "--mask-bias-init", str(mask_bias_init)]
    if mode == "dense_jepa_flatten":
        return ["--mode", "dense_jepa", "--probe-pool", "flatten"]
    if mode == "dense_jepa_mean":
        return ["--mode", "dense_jepa", "--probe-pool", "mean"]
    if mode == "dense":
        return ["--mode", "dense"]
    if mode == "copy":
        return ["--mode", "copy"]
    raise ValueError(f"unknown mode alias: {mode}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(args, out_root, seeds, modes, n_targets, n_distractors,
                    Ks, Js, phase2_reference):
    manifest = {
        "git_commit": git_commit(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "script_version": "phase3-v1",
        "command": " ".join(sys.argv),
        "modes": modes,
        "seeds": seeds,
        "K_values": Ks,
        "J_values_eval": Js,
        "J_value_train": args.J_train,
        "n_targets": n_targets,
        "n_distractors": n_distractors,
        "moving_distractors": args.moving_distractors,
        "partial_observability": args.partial_observability,
        "obs_radius": args.obs_radius,
        "rendered_patches": True,
        "perceptual_noise": args.perceptual_noise,
        "phase4A_rendered_obs": args.perceptual_noise > 0,
        "color_randomization": args.color_randomization,
        "background_randomization": args.background_randomization,
        "phase4B_appearance_random": args.color_randomization or args.background_randomization,
        # Two slot configs are recorded so future comparisons can't silently
        # mix hardware-specific settings. `phase2_reference` is the locked
        # local-CPU / torch 2.4 config from PHASE_2_JEPA_DECISION.md.
        # `phase3_pod_default` is what this run actually used; the *active*
        # value is mirrored at the top level under `slot_config` to remain
        # backwards-compatible with old parsers.
        "phase2_reference": {
            "mask_bias_init": -2.0,
            "lambda_sparsity": 5e-3,
            "lambda_bimodal": 1e-3,
            "delta_scale": 0.1,
            "n_slots": 16, "slot_iters": 3,
            "hardware": "local CPU, torch 2.4",
        },
        "phase3_pod_default": {
            "mask_bias_init": args.mask_bias_init,
            "lambda_sparsity": 5e-3,
            "lambda_bimodal": 1e-3,
            "delta_scale": 0.1,
            "n_slots": 16, "slot_iters": 3,
            "hardware": "pod GPU, torch ≥ 2.8 — bias=-2.0 collapses to mask=0 here",
        },
        "slot_config": {
            "n_slots": 16, "slot_iters": 3,
            "sparsity_weight": 5e-3, "bimodal_weight": 1e-3,
            "mask_bias_init": args.mask_bias_init, "delta_scale": 0.1,
        },
        "training": {
            "steps": args.steps, "batch_size": args.batch_size,
            "image_size": args.image_size, "patch_size": args.patch_size,
            "episode_length": args.episode_length,
        },
        "probe": {
            "episodes": args.probe_episodes,
            "epochs": args.probe_epochs,
            "lr": args.probe_lr,
            "type": "linear",
            "trained_on": "visible_frames_only",
            "evaluated_on": "hidden_frames",
        },
        "phase2_reference_metrics": phase2_reference,
    }
    path = os.path.join(out_root, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


PHASE2_REFERENCE = {
    "env": "n_targets=3, n_distractors=2, K=5, J_train=10, seed=0",
    "results_at_J20_hidden_mse": {
        "slot_delta": 3.87,
        "slot_dense_update": 68.01,
        "dense_jepa_flatten": 62.23,
        "dense_jepa_mean": 66.87,
        "copy": 66.87,
    },
    "gates": {
        "slot_delta_vs_dense_jepa_flatten_at_J20": "≤ 0.75× threshold passed by 12× margin",
        "slot_delta_vs_slot_dense_update_at_J20": "17.6× margin",
        "slope_J5_to_J40": {"slot_delta": 3.58, "dense_jepa_flatten": 11.35},
    },
}


def run_one(args, run_dir, mode, seed, K, n_targets, n_distractors, J_train,
            eval_Js, n_slots=None, target_active=0):
    os.makedirs(run_dir, exist_ok=True)
    cmd = ["python3", "scripts/slot_jepa_train.py"]
    cmd += mode_to_train_args(mode, args.mask_bias_init,
                                target_active_slots=target_active)
    cmd += [
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--patch-size", str(args.patch_size),
        "--episode-length", str(args.episode_length),
        "--visible-steps", str(K),
        "--hidden-steps", str(J_train),
        "--n-targets", str(n_targets),
        "--n-distractors", str(n_distractors),
        "--probe-episodes", str(args.probe_episodes),
        "--probe-epochs", str(args.probe_epochs),
        "--probe-lr", str(args.probe_lr),
        "--eval-J-values", ",".join(str(j) for j in eval_Js),
        "--seed", str(seed),
        "--log-every", "100000",  # suppress per-step logs in sub-runs
        "--output", run_dir,
    ]
    if n_slots is not None:
        cmd += ["--n-slots", str(n_slots)]
    if args.moving_distractors:
        cmd.append("--moving-distractors")
    if args.partial_observability:
        cmd += ["--partial-observability", "--obs-radius", str(args.obs_radius)]
    if args.perceptual_noise > 0:
        cmd += ["--perceptual-noise", str(args.perceptual_noise)]
    if args.color_randomization:
        cmd.append("--color-randomization")
    if args.background_randomization:
        cmd.append("--background-randomization")
    if args.soft_render:
        cmd += ["--soft-render", "--soft-sigma", str(args.soft_sigma)]
    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return None
    t0 = time.time()
    log_path = os.path.join(run_dir, "stdout.log")
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    elapsed = time.time() - t0
    eval_path = os.path.join(run_dir, "probe_eval.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path) as f:
        return json.load(f), elapsed


def aggregate(rows):
    """Aggregate per-run rows into (mode, K, n_targets, n_distractors, J)
    cells. Returns list of dicts ready to write to CSV."""
    cells = {}
    for r in rows:
        key = (r["mode"], r["K"], r["n_targets"], r["n_distractors"], r["J"])
        cells.setdefault(key, []).append(r)
    out = []
    for key, group in cells.items():
        mode, K, nt, nd, J = key
        hidden = [g["hidden_mse"] for g in group if g["hidden_mse"] is not None]
        visible = [g["visible_mse"] for g in group if g["visible_mse"] is not None]
        ratio = [g["hidden_visible_ratio"] for g in group
                 if g["hidden_visible_ratio"] is not None]

        def stats(xs):
            if not xs:
                return {"mean": None, "stderr": None, "ci95": None, "n": 0}
            n = len(xs)
            mean = sum(xs) / n
            if n == 1:
                return {"mean": mean, "stderr": 0.0, "ci95": 0.0, "n": 1}
            var = sum((x - mean) ** 2 for x in xs) / (n - 1)
            stderr = math.sqrt(var / n)
            return {"mean": mean, "stderr": stderr,
                    "ci95": t_crit95(n) * stderr, "n": n}

        h = stats(hidden); v = stats(visible); rt = stats(ratio)
        out.append({
            "mode": mode, "K": K, "n_targets": nt, "n_distractors": nd, "J": J,
            "n_seeds": h["n"],
            "hidden_mse_mean": h["mean"], "hidden_mse_stderr": h["stderr"],
            "hidden_mse_ci95": h["ci95"],
            "visible_mse_mean": v["mean"], "visible_mse_stderr": v["stderr"],
            "visible_mse_ci95": v["ci95"],
            "ratio_mean": rt["mean"], "ratio_stderr": rt["stderr"],
            "ratio_ci95": rt["ci95"],
        })
    return out


def gate_check(agg):
    """Apply the Phase-3 win/loss gates per (K, n_targets, n_distractors, J)
    cell. For each cell, the gate is:
        hidden_mse[slot_delta] ≤ 0.75 × hidden_mse[dense_jepa_flatten]
        hidden_mse[slot_delta] ≤ 0.75 × hidden_mse[slot_dense_update]"""
    by_cell = {}
    for r in agg:
        ckey = (r["K"], r["n_targets"], r["n_distractors"], r["J"])
        by_cell.setdefault(ckey, {})[r["mode"]] = r
    gates = []
    for ckey, by_mode in by_cell.items():
        sd = by_mode.get("slot_delta")
        if sd is None or sd["hidden_mse_mean"] is None:
            continue
        for control in ("dense_jepa_flatten", "slot_dense_update"):
            ctrl = by_mode.get(control)
            if ctrl is None or ctrl["hidden_mse_mean"] is None:
                continue
            threshold = 0.75 * ctrl["hidden_mse_mean"]
            passed = sd["hidden_mse_mean"] <= threshold
            gates.append({
                "K": ckey[0], "n_targets": ckey[1], "n_distractors": ckey[2],
                "J": ckey[3], "control": control,
                "slot_delta_hidden_mse": sd["hidden_mse_mean"],
                "control_hidden_mse": ctrl["hidden_mse_mean"],
                "threshold_75pct": threshold,
                "passed": passed,
                "margin_x": ctrl["hidden_mse_mean"] / max(sd["hidden_mse_mean"], 1e-9),
            })
    return gates


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = parse_list(args.seeds, int)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    n_targets_l = parse_list(args.targets, int)
    n_distractors_l = parse_list(args.distractors, int)
    Ks = parse_list(args.K, int)
    Js = parse_list(args.J, int)
    n_slots_list = parse_list(args.n_slots_list, int)
    target_active_list = parse_list(args.target_active_slots_list, int)

    manifest_path = write_manifest(
        args, args.out, seeds, modes, n_targets_l, n_distractors_l, Ks, Js,
        PHASE2_REFERENCE,
    )
    print(f"manifest -> {manifest_path}", flush=True)

    rows = []
    raw_path = os.path.join(args.out, "raw_results.jsonl")
    raw_file = open(raw_path, "w")

    def cells_for_mode(mode_name):
        """slot_* modes sweep over n_slots; dense/copy run once."""
        if mode_name.startswith("slot"):
            return n_slots_list
        return [None]

    def n_subruns(mode_name):
        ns_choices = cells_for_mode(mode_name)
        count = 0
        for ns in ns_choices:
            if mode_name == "slot_delta" and ns is not None:
                count += sum(1 for ta in target_active_list if ta < ns)
            else:
                count += 1
        return count

    total = sum(
        len(seeds) * n_subruns(m) * len(n_targets_l) * len(n_distractors_l) * len(Ks)
        for m in modes
    )
    done = 0
    t_start = time.time()

    for seed, mode, nt, nd, K in itertools.product(
            seeds, modes, n_targets_l, n_distractors_l, Ks):
        for n_slots in cells_for_mode(mode):
            # target_active only varies on slot_delta. For other modes,
            # collapse to a single [0] sweep.
            active_choices = (target_active_list
                              if mode == "slot_delta" and n_slots is not None
                              else [0])
            for target_active in active_choices:
                # Skip impossible combos (target_active >= n_slots).
                if n_slots is not None and target_active >= n_slots:
                    continue
                slot_tag = f"_ns={n_slots}" if n_slots is not None else ""
                active_tag = f"_ta={target_active}" if target_active > 0 else ""
                cell = (f"seed={seed}_mode={mode}_K={K}_nt={nt}_nd={nd}"
                        f"{slot_tag}{active_tag}")
                run_dir = os.path.join(args.out, "runs", cell)
                done += 1
                out = run_one(args, run_dir, mode, seed, K, nt, nd,
                                args.J_train, Js,
                                n_slots=n_slots,
                                target_active=target_active)
                if out is None:
                    continue
                eval_payload, elapsed = out
                for r in eval_payload["results"]:
                    row = {
                        "mode": mode, "seed": seed, "K": K,
                        "n_targets": nt, "n_distractors": nd,
                        "n_slots": n_slots,
                        "target_active_slots": target_active,
                        "J": r["J"],
                        "hidden_mse": r.get("hidden_mse"),
                        "visible_mse": r.get("visible_mse"),
                        "hidden_visible_ratio": r.get("hidden_visible_ratio"),
                        "n_visible": r.get("n_visible"),
                        "n_hidden": r.get("n_hidden"),
                        "degradation_slope": r.get("degradation_slope"),
                        "elapsed_s": round(elapsed, 1),
                        "run_dir": run_dir,
                    }
                    rows.append(row)
                    raw_file.write(json.dumps(row) + "\n")
                raw_file.flush()
                eta = (time.time() - t_start) / done * (total - done)
                print(f"[{done}/{total}] {cell}  elapsed={elapsed:.0f}s  "
                      f"eta_remaining={eta:.0f}s", flush=True)
    raw_file.close()

    if args.dry_run or not rows:
        print("(dry-run or no rows) skipping aggregation")
        return

    agg = aggregate(rows)
    agg_path = os.path.join(args.out, "aggregate.csv")
    if agg:
        fieldnames = list(agg[0].keys())
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in agg:
                w.writerow(r)
    gates = gate_check(agg)
    with open(os.path.join(args.out, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    # Concise stdout summary.
    print()
    print("=== gate summary ===")
    for g in gates:
        flag = "PASS" if g["passed"] else "FAIL"
        print(f"[{flag}] K={g['K']} nt={g['n_targets']} nd={g['n_distractors']} "
              f"J={g['J']}  slot_delta {g['slot_delta_hidden_mse']:.3f}  "
              f"vs {g['control']} {g['control_hidden_mse']:.3f}  "
              f"margin={g['margin_x']:.1f}x")
    n_pass = sum(1 for g in gates if g["passed"])
    print(f"\ntotal gates: {len(gates)}  pass: {n_pass}  fail: {len(gates) - n_pass}")
    print(f"raw rows: {raw_path}")
    print(f"aggregate: {agg_path}")


if __name__ == "__main__":
    main()
