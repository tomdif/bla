"""Falsified code paths from the Phase 7-8 JEPA arc.

Code in this package is **NOT canonical**. Each module documents the
phase that falsified it. Kept here for:

  1. Reproducing the falsification sweeps (per the decision docs in
     `docs/phases/`)
  2. Reactivating IF future data regimes flip the verdict (e.g. v1
     lifecycle becomes load-bearing under true streaming data)
  3. Methodology audit trail

Do not import from `_attic` in canonical paths. Canonical OF-JEPA v0
imports stay in `system1_jepa/of_jepa.py`, `of_jepa_api.py`,
`identity_probe.py`, `id_consistency.py`.
"""
