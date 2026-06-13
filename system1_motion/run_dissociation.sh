#!/usr/bin/env bash
# Grounding-dissociation experiment on the Reacher rig (GPU + MuJoCo render).
# Run from a RunPod GPU pod with dm_control + mujoco + torch installed.
#
#   cd ~/bla && bash system1_motion/run_dissociation.sh
#
# Env knobs:  STEPS (default 30000)  SEEDS (default "0 1 2")  EPISODES (default 350)
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
cd "$(dirname "$0")/.."                      # -> bla root

DATA=runs/reacher_transitions.npz
STEPS=${STEPS:-30000}
SEEDS=${SEEDS:-"0 1 2"}
EPISODES=${EPISODES:-350}
OUT=runs/dissoc

# 1) render WITH target extraction (skip if present). Fails loud if no 'target' geom.
if [ ! -f "$DATA" ]; then
  echo "=== rendering Reacher dataset (with target) ==="
  python -m system1_motion.render_dataset --episodes "$EPISODES" --out "$DATA"
else
  python - <<'PY'
import numpy as np
d=np.load("runs/reacher_transitions.npz")
assert "target" in d.files, "existing dataset has NO 'target' key — delete it and re-render."
print("[ok] dataset has target; frames", d["frames"].shape)
PY
fi

# 2) three conditions x seeds. Same model/objective; only grounding weights differ.
#    C0 baseline | C1 action-grounding (inv+prior) | C2 target-grounding (tgt head)
for SEED in $SEEDS; do
  python -m system1_motion.train_dissoc --data "$DATA" --condition C0 \
      --inv-weight 0   --prior-weight 0   --tgt-weight 0   --steps "$STEPS" --seed "$SEED" --out-dir "$OUT"
  python -m system1_motion.train_dissoc --data "$DATA" --condition C1 \
      --inv-weight 1.0 --prior-weight 0.5 --tgt-weight 0   --steps "$STEPS" --seed "$SEED" --out-dir "$OUT"
  python -m system1_motion.train_dissoc --data "$DATA" --condition C2 \
      --inv-weight 0   --prior-weight 0   --tgt-weight 1.0 --steps "$STEPS" --seed "$SEED" --out-dir "$OUT"
done

# 3) headline table
echo "=== DISSOCIATION RESULT ==="
python -m system1_motion.dissoc_report "$OUT"
