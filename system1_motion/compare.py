#!/usr/bin/env python3
"""Aggregate legibility (and, later, plan_eval) JSONs into the encoder head-to-head
table: rows = encoder, cols = objective condition, cell = composite score +/- seed
noise, with a per-variable drill-down. This is the "one table" view of the A x B x C
search — read the noise floor first (seed std), then read winners.

    python -m system1_motion.compare runs/legibility
"""
import glob, json, os, sys
import numpy as np

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/legibility"
files = sorted(glob.glob(os.path.join(run_dir, "*.json")))
if not files:
    raise SystemExit(f"no *.json in {run_dir}")

rows = [json.load(open(f)) for f in files]
# group by (encoder, condition) over seeds
cells = {}
for r in rows:
    cells.setdefault((r["encoder"], r["condition"]), []).append(r)

encoders = sorted({e for e, _ in cells})
conds = sorted({c for _, c in cells})
varset = sorted({k for r in rows for k in r["battery"]})

print(f"\n=== encoder head-to-head @ {run_dir} (composite legibility = mean held-out R^2, higher=better) ===")
print(f"{'encoder':10}" + "".join(f"{c:>16}" for c in conds))
print("-" * (10 + 16 * len(conds)))
for e in encoders:
    line = f"{e:10}"
    for c in conds:
        g = cells.get((e, c))
        if not g:
            line += f"{'-':>16}"; continue
        if any(x.get("collapsed") for x in g):
            line += f"{'COLLAPSED':>16}"; continue
        vals = [x["legibility_meanR2"] for x in g if x.get("legibility_meanR2") is not None]
        m = np.mean(vals); s = np.std(vals)
        line += f"{m:8.3f}±{s:.3f} ".rjust(16)
    print(line)

print("\n--- per-variable drill-down (held-out R^2; px for positions) ---")
for (e, c), g in sorted(cells.items()):
    parts = []
    for v in varset:
        r2 = np.mean([x["battery"][v]["r2"] for x in g if v in x["battery"]])
        pxs = [x["battery"][v]["px"] for x in g if v in x["battery"] and "px" in x["battery"][v]]
        parts.append(f"{v}={r2:.2f}" + (f"/{np.mean(pxs):.1f}px" if pxs else ""))
    print(f"  {e:6} {c:10} (n={len(g)}): " + " | ".join(parts))

print("\nread: pick by the PLANNING end-effect once plan_eval lands; legibility explains why.")
print("absolute state (arm/target px) vs relative state (joint/vel R^2) tells you WHICH")
print("kind of state each encoder+objective actually grounds.")
