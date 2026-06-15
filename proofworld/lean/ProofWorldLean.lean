-- Portable preamble library for proofworld's Mathlib-backed kernel.
-- Importing this gives nlinarith/linarith (and transitively the tactic stack) plus the demo's recursive def,
-- so research_mathlib can `import ProofWorldLean` against ANY checkout that ran `lake exe cache get`.
import Mathlib.Tactic.Linarith

def oddSum : ℕ → ℕ
  | 0 => 0
  | (n + 1) => oddSum n + (2 * n + 1)
