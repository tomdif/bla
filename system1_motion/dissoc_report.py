#!/usr/bin/env python3
"""Aggregate the grounding-dissociation runs into the headline 3x2 table.

Reads trace_<COND>_s<SEED>.json files written by train_dissoc.py and prints, per
condition, the final-checkpoint ARM (controllable) and TARGET (uncontrolled) probe
errors (mean +/- std over seeds) and pass-rates. The result to look for is the
INTERACTION: C1 (action grounding) rescues ARM but not TARGET; C2 (target
grounding) rescues TARGET.

    python -m system1_motion.dissoc_report runs/dissoc
"""
import glob, json, os, sys
import numpy as np

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/dissoc"
files = sorted(glob.glob(os.path.join(run_dir, "trace_*.json")))
if not files:
    raise SystemExit(f"no trace_*.json in {run_dir}")

by_cond = {}
for f in files:
    d = json.load(open(f))
    c = d["condition"]; fin = d["final"]
    by_cond.setdefault(c, []).append(fin)

LABEL = {"C0": "baseline (no grounding)",
         "C1": "+action-ground (inv+prior)",
         "C2": "+target-ground (tgt head)"}

print(f"\n=== grounding dissociation @ {run_dir} (probe error in px, lower=better; PASS < 5px) ===")
print(f"{'condition':28} {'ARM (controllable)':>22} {'TARGET (uncontrolled)':>24}")
print("-" * 78)
for c in sorted(by_cond):
    fins = by_cond[c]
    arm = np.array([x["arm_px"] for x in fins]); tgt = np.array([x["target_px"] for x in fins])
    arm_p = np.mean([x["arm_pass"] for x in fins]); tgt_p = np.mean([x["target_pass"] for x in fins])
    print(f"{c} {LABEL.get(c,''):24} "
          f"{arm.mean():6.1f}±{arm.std():3.1f} {'PASS' if arm_p>0.5 else 'FAIL':>4}({arm_p:.0%})   "
          f"{tgt.mean():6.1f}±{tgt.std():3.1f} {'PASS' if tgt_p>0.5 else 'FAIL':>4}({tgt_p:.0%})")
print("-" * 78)
print("dissociation confirmed iff: C1 ARM=PASS & C1 TARGET=FAIL  (action grounding is")
print("selective for the controllable variable), and C2 TARGET=PASS (target is decodable")
print("when the decision-relevance channel grounds it). n_seeds per condition:",
      {c: len(v) for c, v in by_cond.items()})
