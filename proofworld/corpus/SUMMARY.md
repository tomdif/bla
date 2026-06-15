# proofworld verified corpus (axiom-clean only)

Admitted iff axiom footprint ⊆ {propext, Classical.choice, Quot.sound} (kernel `collectAxioms`).
Rejected = depends on a `sorry` or a project-local unproven `axiom`. Conditional theorems [H]→P kept.

| project | domain | clean (kept) | axiom-dependent (rejected) | sorry (rejected) |
|---|---|---|---|---|
| RamanujanTau | modular forms / Ramanujan tau | 42 | 50 | 0 |
| PlonkLean | PLONK zero-knowledge proof system | 731 | 0 | 0 |

**Total citable facts: 773** in `corpus.jsonl` (name, statement, module, axioms, project, domain).
