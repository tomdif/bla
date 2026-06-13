#!/usr/bin/env bash
# Runs the deterministic verification-arc gates (stub modes -- no LLM, no network, no Lean) and asserts
# each prints its expected PASS verdict. Exits nonzero if any gate fails. Used by .github/workflows/ci.yml.
#
# Excluded here (require special setup, run manually / documented in VERIFICATION_ARC.md):
#   - Lean gates (lean_repair_v1, lean_proposer_from_traces, cross_domain_library) -- need the lean toolchain
#   - swebench_adapter -- needs a network git clone
#   - lean_repair_real_v1 -- needs the local collatz-proven repo
#   - *_v1 live-LLM paths -- need ANTHROPIC_API_KEY (the stub paths ARE covered below)
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
while IFS='|' read -r gate pattern; do
  [ -z "$gate" ] && continue
  if python3 "$gate.py" 2>&1 | grep -qF "$pattern"; then
    printf '  PASS  %s\n' "$gate"
  else
    printf '  FAIL  %s  (expected: %s)\n' "$gate" "$pattern"
    fail=1
  fi
done <<'GATES'
code_repair_harness|CODE REPAIR GATE: PASS
code_repair_v1|CODE REPAIR v1: PASS
code_repair_v2|CODE REPAIR v2: PASS
patch_outcome_surrogate|PATCH-OUTCOME SURROGATE GATE: PASS
trace_trained_surrogate|TRACE-TRAINED SURROGATE GATE: PASS
proposer_from_traces|PROPOSER-FROM-TRACES GATE: PASS
technique_discovery_gate|TECHNIQUE DISCOVERY GATE: PASS
rh_proof_harness|RH PROOF HARNESS GATE: PASS
geometry_canvas_gate|GEOMETRY CANVAS GATE: PASS
adversarial_gate|ADVERSARIAL GATE (post-fix): PASS
adversarial_full_gate|FULL ADVERSARIAL GATE: PASS
sql_repair_v1|SQL REPAIR v1: PASS
grid_compose_gate|GRID COMPOSE GATE: PASS
GATES

if [ "$fail" -ne 0 ]; then echo "ONE OR MORE GATES FAILED"; exit 1; fi
echo "ALL DETERMINISTIC GATES PASS"
