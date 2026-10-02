# proofworld trust boundary

**Every belief goes through `proofworld/gate.py`.** Generators (LLM, dreamer, enumerator, decomposer) only
propose. A Lean claim is believed only when the gate returns `PROVED`. Do not write a new
`subprocess.run(["lean", ...])` + "rc == 0 and no 'error:'" check; call `gate.check` / `gate.check_one`.

## Verdicts (closed vocabulary)

| status | meaning |
|---|---|
| `PROVED` | The kernel's own `collectAxioms` reported a footprint ⊆ {propext, Classical.choice, Quot.sound}. |
| `FAILED` | The kernel rejected the proof (an error inside the claim). |
| `REFUSED` | Policy: an escape hatch in untrusted text (sorry, admit, axiom, native_decide, skipKernelTC, implemented_by, injected commands), or sorryAx / non-standard axioms in the footprint. |
| `INCONCLUSIVE` | Timeout, crash, missing footprint marker, or an error outside every claim. **Never a pass.** |

`register` is `lean-kernel`. It is `lean-kernel+native` only when the caller passed `allow_native=True` and the
claim inherits `native_decide` axioms from the library it builds on. For example, RamanujanTau's
`tau_one := by native_decide` sits under every τ(p^k) result. New proof text may never use `native_decide`.

## Rules, and why

Ported from a review of the BootLoops harness, whose gates certified false results in these same ways.

1. **Positive evidence only.** A missing output, a timeout or a failed import used to read as a pass. In the old
   `decompose.try_direct`, a broken import "proved" the goal.
2. **Cheap kill before an expensive check.** LLM proofs *and* LLM-written statements are linted before the
   kernel sees them. The axiom footprint is the backstop.
3. **Controls before trust.** In each (toolchain, project) environment, planted claims run first. A true claim
   must come back `PROVED`, a false one `FAILED`, and sorry / native_decide / injected-axiom claims `REFUSED`.
   Otherwise the gate raises `GateIntegrityError`. Run them with `python3 -m proofworld.gate --selftest [--project DIR]`.
4. **Receipts.** Every verdict is appended to `receipts/ledger.jsonl` under a file lock, with its source sha256,
   toolchain and lake-manifest hash. Set `PROOFWORLD_RECEIPTS` to redirect it.
5. **Batch.** Many claims go in one Lean run, so the Mathlib import is paid once. `conjecture`: 54 s → 10 s.

## Numeric kill-tests (`proofworld/numeric.py`)

Statuses: `SURVIVES | REFUTED | INSUFFICIENT | UNEVALUABLE`. The register is always `numeric`, never `proved`.

- **No evidence is not evidence.** A law with fewer than `min_support` in-domain instances is `INSUFFICIENT`.
- **Independence.** `numeric.split` deals a deterministic held-out set. The proposer sees only `seen`, and
  survival is judged on `held`. A counterexample anywhere still refutes.
- **No `eval` on model output.** `compile_formula` accepts only integer arithmetic over named variables
  (AST whitelist), with caps on exponent and integer size.
- **Planted controls** run before real data: `python3 -m proofworld.numeric`.

## Tests

`python3 -m pytest tests/test_proofworld_gate.py`. The pure-Python decision-rule tests run in CI. The kernel tests
skip without Lean.
