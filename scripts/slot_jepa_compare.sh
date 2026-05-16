#!/bin/bash
# Slot-JEPA falsification comparison via linear-probe eval.
#
# For each mode:
#   1. self-supervised training with K=5, J=10
#   2. for each J in {5, 10, 20, 40}:
#        - roll out expert episodes, collect (state, target_xy, visibility)
#        - fit linear probe on visible frames only
#        - eval MSE on hidden frames + per-hidden-step breakdown
#
# Modes:
#   slot_delta              16-slot + sparse delta predictor (the proposal)
#   dense_jepa_flatten      patch-level masked JEPA (BLAJEPAModel), probe on
#                           flattened patch tokens — the FAIR dense control
#   dense_jepa_mean         same encoder, mean-pooled probe state (control
#                           for whether flatten alone is doing the work)
#   dense                   naive pooled next-step prediction (legacy)
#   copy                    encoder same-frame consistency only (persistence)
set -e
OUT=${1:-/tmp/slot_jepa_compare}
STEPS=${2:-3000}
PROBE_EPISODES=${3:-32}
mkdir -p $OUT

COMMON_ARGS=(
  --steps $STEPS
  --batch-size 4
  --probe-episodes $PROBE_EPISODES
  --probe-epochs 300
  --probe-lr 5e-3
  --eval-J-values 5,10,20,40
  --visible-steps 5
  --hidden-steps 10
  --n-targets 3
  --n-distractors 2
  --log-every 1000
)

echo "=== slot_delta ==="
python3 scripts/slot_jepa_train.py --mode slot_delta \
  --sparsity-weight 5e-3 --bimodal-weight 1e-3 --mask-bias-init -2.0 \
  "${COMMON_ARGS[@]}" --output $OUT/slot_delta 2>&1 | tee $OUT/slot_delta.log | grep -E '"event": "probe"|"step": 3000' | head -10

echo "=== dense_jepa (flatten probe) ==="
python3 scripts/slot_jepa_train.py --mode dense_jepa --probe-pool flatten \
  "${COMMON_ARGS[@]}" --output $OUT/dense_jepa_flatten 2>&1 | tee $OUT/dense_jepa_flatten.log | grep -E '"event": "probe"|"step": 3000' | head -10

echo "=== dense_jepa (mean probe) ==="
python3 scripts/slot_jepa_train.py --mode dense_jepa --probe-pool mean \
  "${COMMON_ARGS[@]}" --output $OUT/dense_jepa_mean 2>&1 | tee $OUT/dense_jepa_mean.log | grep -E '"event": "probe"|"step": 3000' | head -10

echo "=== dense (legacy pooled) ==="
python3 scripts/slot_jepa_train.py --mode dense \
  "${COMMON_ARGS[@]}" --output $OUT/dense 2>&1 | tee $OUT/dense.log | grep -E '"event": "probe"|"step": 3000' | head -10

echo "=== copy ==="
python3 scripts/slot_jepa_train.py --mode copy \
  "${COMMON_ARGS[@]}" --output $OUT/copy 2>&1 | tee $OUT/copy.log | grep -E '"event": "probe"|"step": 3000' | head -10

echo
echo "=== summary ==="
python3 - <<PY
import json
out = "$OUT"
mode_dirs = [
    ("slot_delta",         "slot_delta"),
    ("dense_jepa/flatten", "dense_jepa_flatten"),
    ("dense_jepa/mean",    "dense_jepa_mean"),
    ("dense (legacy)",     "dense"),
    ("copy",               "copy"),
]
data = {}
for label, d in mode_dirs:
    try:
        data[label] = json.load(open(f"{out}/{d}/probe_eval.json"))["results"]
    except FileNotFoundError:
        data[label] = []

J_VALUES = (5, 10, 20, 40)

def fmt(v):
    return f"{v:>8.3f}" if v is not None else "    -   "

print()
print(f"{'mode':<22s}  " + "  ".join(f"J={j:>2d}" for j in J_VALUES))
print(f"{'-'*22}  " + "  ".join("-" * 8 for _ in J_VALUES))
for label, results in data.items():
    by_j = {r["J"]: r for r in results}
    cells = [fmt(by_j.get(j, {}).get("hidden_mse")) for j in J_VALUES]
    print(f"{label:<22s}  " + "  ".join(cells) + "  (hidden MSE)")

print()
print(f"{'mode':<22s}  " + "  ".join(f"J={j:>2d}" for j in J_VALUES))
print(f"{'-'*22}  " + "  ".join("-" * 8 for _ in J_VALUES))
for label, results in data.items():
    by_j = {r["J"]: r for r in results}
    cells = [fmt(by_j.get(j, {}).get("visible_mse")) for j in J_VALUES]
    print(f"{label:<22s}  " + "  ".join(cells) + "  (visible MSE)")

print()
print(f"{'mode':<22s}  " + "  ".join(f"J={j:>2d}" for j in J_VALUES))
print(f"{'-'*22}  " + "  ".join("-" * 8 for _ in J_VALUES))
for label, results in data.items():
    by_j = {r["J"]: r for r in results}
    cells = [fmt(by_j.get(j, {}).get("hidden_visible_ratio")) for j in J_VALUES]
    print(f"{label:<22s}  " + "  ".join(cells) + "  (hidden/visible ratio)")
PY
