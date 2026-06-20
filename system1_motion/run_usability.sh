#!/usr/bin/env bash
# Closed-loop USABILITY probe across the dissociation conditions (GPU pod).
# Runs AFTER run_dissociation.sh has produced runs/dissoc/substrate_*_{C0,C1,C2}_s*.pt.
# Measures whether the UNCONTROLLED target is *usable* (reachable through the frozen
# latent), and tabulates it against decode(target) — the "decode lies" lie-detector,
# now for the uncontrolled variable (action-cosine cannot see it).
#
#   cd ~/bla && bash system1_motion/run_usability.sh
#
# Env:  SEEDS (default "0 1 2")  ENC (default pool)  OUT (default runs/dissoc)
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}      # only needed if a dataset re-render is triggered upstream
cd "$(dirname "$0")/.."                 # -> bla root
DATA=runs/reacher_transitions.npz
SEEDS=${SEEDS:-"0 1 2"}
ENC=${ENC:-pool}
OUT=${OUT:-runs/dissoc}

[ -f "$DATA" ] || { echo "missing $DATA — run run_dissociation.sh first"; exit 1; }

for SEED in $SEEDS; do
  for C in C0 C1 C2; do
    CK="$OUT/substrate_${ENC}_${C}_s${SEED}.pt"
    [ -f "$CK" ] || { echo "missing $CK — run run_dissociation.sh first"; exit 1; }
    echo "=== usability: $C seed $SEED (arbiter=real-rollout, +ensemble gate) ==="
    # --real-rollout = ARBITER (verdict in real physics, no fitted-dyn confound)
    # --ensemble/--dis-beta = de-confound the cheap imagination proxy (PETS disagreement)
    python -m system1_motion.usability_probe --ckpt "$CK" --data "$DATA" --out-dir "$OUT" \
        --real-rollout --ensemble 3 --dis-beta 5.0
  done
done

echo
echo "=== MINIMAL-RECIPE PAPER TABLE ==="
echo "    conditions: C0=vanilla substrate | C1=+inverse-dynamics | C2=+tiny target decode"
python - "$OUT" "$ENC" <<'PY'
import json, glob, os, sys, collections, statistics as st
out, enc = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(list)
for f in glob.glob(os.path.join(out, f"usability_substrate_{enc}_*.json")):
    r = json.load(open(f)); rows[r["condition"]].append(r)
mean = lambda xs: st.mean(xs) if xs else float('nan')
print(f"{'cond':4} | {'arm_abs':>7} {'arm_vel':>7} | {'tgt_decode_px':>13} {'pass':>5} | {'tgt_USABLE_real':>15} | {'IMAG':>5} | dyn_R2")
for c in ("C0","C1","C2"):
    rs = rows.get(c, [])
    if not rs: print(f"{c:4} | (no runs)"); continue
    aa = mean([x.get('arm_abs_r2', float('nan')) for x in rs])
    av = mean([x.get('arm_vel_r2', float('nan')) for x in rs])
    dpx = mean([x['decode_target_px'] for x in rs]); pas = sum(x.get('decode_target_pass',0) for x in rs)
    ur = mean([x.get('usability_target_real', float('nan')) for x in rs])
    ui = mean([x['usability_target_imag'] for x in rs]); dr = mean([x['dyn_r2'] for x in rs])
    print(f"{c:4} | {aa:>7.2f} {av:>7.2f} | {dpx:>13.2f} {pas}/{len(rs):<2}| {ur:>15.2f} | {ui:>5.2f} | {dr:.2f}")
print()
print("READ — the minimal-recipe result:")
print("  ARM IS FREE  : arm_abs HIGH for C0 already (vanilla substrate grounds the arm; no sponsor needed).")
print("  ID IS A QUOTIENT SHORTCUT : C1 arm_vel >> arm_abs (grounds the controllable delta, not the state) -> why ID was dead.")
print("  TARGET NEEDS C2 + IS IT USABLE : tgt_USABLE_real HIGH only for C2 (decode head). The open question:")
print("        C2 decode-PASS & USABLE HIGH -> decode==usable for the uncontrolled target.")
print("        C2 decode-PASS & USABLE LOW  -> PHASE-MEMORY (decode lies) for the uncontrolled target.")
print("  MINIMAL RECIPE = C0(arm free) + C2(tiny target decode); ID(C1) is REDUNDANT (matches C0 on arm, blind on target).")
print("  ARBITER = USABLE_real (real physics). IMAG is the cheap proxy; |IMAG-real| small -> proxy validated.")
PY
