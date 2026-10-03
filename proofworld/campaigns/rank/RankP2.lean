/-
# Ramanujan's mock theta functions `φ, ψ` inside the Hecke–Rogers series (for `R₀, R₃`)

Class `2k−1` of the orbit `a ≡ ±2k (mod 5)` part of `Σ_a (−1)^a q^{T(a)} PT_a(q)` equals
`q^{1−k}·J_{5,k}·(1 − Φ_k)` (`Φ₁ = 1+φ`, `Φ₂ = 1+ψ`), by an explicit bijection of index sets against the
specialization `RankSpec.spec_coeff` and Jacobi's `J_{5,k}`.
-/
import RamanujanTau.RankSpec

set_option autoImplicit false

namespace RankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5 dis5)

section Coef

/-- `eP a r = T(a) + 3r² + 3ar + r`, `eM a r = T(a) + 3r² + 3ar − r − a` (`r ≥ 1`). -/
def eP (a r : ℕ) : ℕ := a * (a + 1) / 2 + (3 * r ^ 2 + 3 * a * r + r)
def eM (a r : ℕ) : ℕ := a * (a + 1) / 2 + (3 * r ^ 2 + 3 * a * r - r - a)

/-- the coefficient of `PT_a = RHSz a` at `j`, as a signed count. -/
def rhsCoef (a j : ℕ) : ℤ :=
  ∑ r ∈ range (j + 1), ((if 3 * r ^ 2 + 3 * a * r + r = j then 1 else 0)
    - (if 1 ≤ r ∧ 3 * r ^ 2 + 3 * a * r - r - a = j then 1 else 0))

lemma coeff_RHSz (a j : ℕ) : coeff j (RHSz a) = rhsCoef a j := by
  rw [RHSz, coeff_mk, map_sum, rhsCoef]
  refine sum_congr rfl fun r _ => ?_
  rcases r with _ | s
  · simp [αz, Bz, coeff_X_pow, eq_comm]
  · rw [αz, Bz, mul_sub, ← pow_add, ← pow_add, map_sub, coeff_X_pow, coeff_X_pow]
    have hE1 : (s + 1) ^ 2 + a * (s + 1) + (2 * (s + 1) ^ 2 + 2 * a * (s + 1) + (s + 1))
        = 3 * (s + 1) ^ 2 + 3 * a * (s + 1) + (s + 1) := by ring
    have hE2 : (s + 1) ^ 2 + a * (s + 1) + (2 * s ^ 2 + 3 * s + 1 + 2 * a * s + a)
        = 3 * (s + 1) ^ 2 + 3 * a * (s + 1) - (s + 1) - a := by
      rw [Nat.sub_sub, show 3 * (s + 1) ^ 2 + 3 * a * (s + 1)
        = ((s + 1) ^ 2 + a * (s + 1) + (2 * s ^ 2 + 3 * s + 1 + 2 * a * s + a)) + (s + 1 + a) by ring,
        Nat.add_sub_cancel]
    simp only [hE1, hE2, eq_comm (a := j), show 1 ≤ s + 1 from by omega, true_and]

end Coef

section Alg

lemma eP_two (a r : ℕ) : 2 * (eP a r : ℤ) = a * (a + 1) + 2 * (3 * r ^ 2 + 3 * a * r + r) := by
  have h : 2 * (a * (a + 1) / 2) = a * (a + 1) := Nat.mul_div_cancel' (Nat.even_mul_succ_self a).two_dvd
  have : (2 : ℤ) * ((a * (a + 1) / 2 : ℕ) : ℤ) = a * (a + 1) := by exact_mod_cast h
  unfold eP
  generalize a * (a + 1) / 2 = t at this ⊢
  push_cast
  linear_combination this

lemma eM_le (a r : ℕ) (hr : 1 ≤ r) : r + a ≤ 3 * r ^ 2 + 3 * a * r := by nlinarith

lemma eM_two (a r : ℕ) (hr : 1 ≤ r) :
    2 * (eM a r : ℤ) = a * (a + 1) + 2 * (3 * r ^ 2 + 3 * a * r - r - a) := by
  have h : 2 * (a * (a + 1) / 2) = a * (a + 1) := Nat.mul_div_cancel' (Nat.even_mul_succ_self a).two_dvd
  have hle := eM_le a r hr
  have : (2 : ℤ) * ((a * (a + 1) / 2 : ℕ) : ℤ) = a * (a + 1) := by exact_mod_cast h
  unfold eM
  rw [Nat.sub_sub]
  generalize a * (a + 1) / 2 = t at this ⊢
  push_cast [hle]
  linear_combination this

/-- equality of naturals from equality of doubles in `ℤ`. -/
lemma nat_eq_of_two {x y : ℕ} (h : 2 * (x : ℤ) = 2 * y) : x = y := by omega

end Alg

section Maps

lemma two_inj {x y : ℕ} (h : 2 * (x : ℤ) = 2 * (y : ℤ)) : x = y := by omega

lemma alg_Pp {k : ℕ} (hk : k = 1 ∨ k = 2) (b r : ℕ) (hb : 1 ≤ b) :
    eM (5 * b - 2 * k) (5 * r + k + 1) + 5 * (k - 1) = 5 * (5 * eP b r + k * b) + (2 * k - 1) := by
  have h1 := eM_two (5 * b - 2 * k) (5 * r + k + 1) (by omega)
  have h2 := eP_two b r
  apply two_inj
  rcases hk with rfl | rfl <;>
  · push_cast [show 2 * 1 ≤ 5 * b by omega, show 2 * 2 ≤ 5 * b by omega] at h1 ⊢
    linear_combination h1 - 25 * h2

lemma alg_Pm {k : ℕ} (hk : k = 1 ∨ k = 2) (b r : ℕ) (hr : 1 ≤ r) :
    eM (5 * b + 2 * k) (5 * r + 1 - k) + 5 * (k - 1) + 5 * (k * b) = 5 * (5 * eP b r) + (2 * k - 1) := by
  have h1 := eM_two (5 * b + 2 * k) (5 * r + 1 - k) (by omega)
  have h2 := eP_two b r
  apply two_inj
  rcases hk with rfl | rfl <;>
  · push_cast [show 1 ≤ 5 * r + 1 by omega, show 2 ≤ 5 * r + 1 by omega, show 1 ≤ 5 * r by omega] at h1 ⊢
    linear_combination h1 - 25 * h2

lemma alg_Mp {k : ℕ} (hk : k = 1 ∨ k = 2) (b r : ℕ) (hb : 1 ≤ b) (hr : 1 ≤ r) :
    eP (5 * b - 2 * k) (5 * r + k - 1) + 5 * (k - 1) = 5 * (5 * eM b r + k * b) + (2 * k - 1) := by
  have h1 := eP_two (5 * b - 2 * k) (5 * r + k - 1)
  have h2 := eM_two b r hr
  apply two_inj
  rcases hk with rfl | rfl <;>
  · push_cast [show 2 * 1 ≤ 5 * b by omega, show 2 * 2 ≤ 5 * b by omega, show 1 ≤ 5 * r + 1 by omega,
      show 1 ≤ 5 * r + 2 by omega, eM_le b r hr] at h1 h2 ⊢
    linear_combination h1 - 25 * h2

lemma alg_Mm {k : ℕ} (hk : k = 1 ∨ k = 2) (b r : ℕ) (hr : 1 ≤ r) :
    eP (5 * b + 2 * k) (5 * r - k - 1) + 5 * (k - 1) + 5 * (k * b) = 5 * (5 * eM b r) + (2 * k - 1) := by
  have h1 := eP_two (5 * b + 2 * k) (5 * r - k - 1)
  have h2 := eM_two b r hr
  apply two_inj
  rcases hk with rfl | rfl <;>
  · push_cast [show 1 + 1 ≤ 5 * r by omega, show 2 + 1 ≤ 5 * r by omega, Nat.sub_sub, eM_le b r hr] at h1 h2 ⊢
    linear_combination h1 - 25 * h2

lemma alg_J {k : ℕ} (hk : k = 1 ∨ k = 2) (d : ℕ) (hd : 1 ≤ d) :
    eP (5 * d - 2 * k) (k - 1) + 5 * (k - 1) = 5 * (5 * d.choose 2 + k * d) + (2 * k - 1) := by
  have h1 := eP_two (5 * d - 2 * k) (k - 1)
  have h3 := CrankProof.two_choose_two d
  apply two_inj
  rcases hk with rfl | rfl <;>
  · push_cast [show 2 * 1 ≤ 5 * d by omega, show 2 * 2 ≤ 5 * d by omega] at h1 ⊢
    linear_combination h1 - 25 * h3

end Maps

section Sums
variable (k N : ℕ)

def gPn (a r : ℕ) : ℤ := if a % 5 = 5 - 2 * k ∧ eP a r + 5 * (k - 1) = 5 * N + (2 * k - 1) then (-1) ^ a else 0
def gPp (a r : ℕ) : ℤ := if a % 5 = 2 * k ∧ eP a r + 5 * (k - 1) = 5 * N + (2 * k - 1) then (-1) ^ a else 0
def gMn (a r : ℕ) : ℤ :=
  if a % 5 = 5 - 2 * k ∧ 1 ≤ r ∧ eM a r + 5 * (k - 1) = 5 * N + (2 * k - 1) then -(-1) ^ a else 0
def gMp (a r : ℕ) : ℤ :=
  if a % 5 = 2 * k ∧ 1 ≤ r ∧ eM a r + 5 * (k - 1) = 5 * N + (2 * k - 1) then -(-1) ^ a else 0
def mPp (b r : ℕ) : ℤ := if 1 ≤ b ∧ 5 * eP b r + k * b = N then (-1) ^ b else 0
def mPm (b r : ℕ) : ℤ := if 5 * eP b r = N + k * b then (-1) ^ b else 0
def mMp (b r : ℕ) : ℤ := if 1 ≤ b ∧ 1 ≤ r ∧ 5 * eM b r + k * b = N then -(-1) ^ b else 0
def mMm (b r : ℕ) : ℤ := if 1 ≤ r ∧ 5 * eM b r = N + k * b then -(-1) ^ b else 0

lemma neg_one_pow_par {x y : ℕ} (h : x % 2 = y % 2) : (-1 : ℤ) ^ x = (-1) ^ y := by
  rw [neg_one_pow_eq_pow_mod_two, h, ← neg_one_pow_eq_pow_mod_two]

lemma eP_ge (b r : ℕ) : b + r ≤ eP b r := by
  have := MockTheta5.JTP.tri_ge b
  unfold eP; generalize b * (b + 1) / 2 = t at *; generalize 3 * r ^ 2 + 3 * b * r = u; omega

lemma eM_ge (b r : ℕ) (hr : 1 ≤ r) : b + r ≤ eM b r + b := by
  have := MockTheta5.JTP.tri_ge b
  have h2 : r + b ≤ 3 * r ^ 2 + 3 * b * r := eM_le b r hr
  have h3 : 2 * r + b ≤ 3 * r ^ 2 + 3 * b * r := by nlinarith
  unfold eM; generalize b * (b + 1) / 2 = t at *; generalize 3 * r ^ 2 + 3 * b * r = u at *; omega

lemma eM_ge' (b r : ℕ) (hr : 1 ≤ r) : b + r ≤ eM b r := by
  have := MockTheta5.JTP.tri_ge b
  have h3 : 2 * r + b ≤ 3 * r ^ 2 + 3 * b * r := by nlinarith
  unfold eM; generalize b * (b + 1) / 2 = t at *; generalize 3 * r ^ 2 + 3 * b * r = u at *; omega

/-- the class condition in `ZMod 5`, type `M`. -/
lemma zmodM {k N a r : ℕ} (hk : k = 1 ∨ k = 2) (hr : 1 ≤ r)
    (he : eM a r + 5 * (k - 1) = 5 * N + (2 * k - 1)) :
    (a : ZMod 5) * (a + 1) + 2 * (3 * r ^ 2 + 3 * a * r - r - a) = 2 * (2 * k - 1) := by
  have h2 := eM_two a r hr
  have hz := congrArg (Int.cast : ℤ → ZMod 5) h2
  push_cast at hz
  rw [← hz]
  have h5 : (5 : ZMod 5) = 0 := by decide
  rcases hk with rfl | rfl
  · have : (eM a r : ZMod 5) = ((5 * N + 1 : ℕ) : ZMod 5) := by rw [← he]; norm_num
    rw [this]; push_cast; rw [h5]; ring
  · have : (eM a r : ZMod 5) + 5 = ((5 * N + 3 : ℕ) : ZMod 5) := by
      rw [← he]; push_cast; ring
    push_cast at this; rw [h5] at this; linear_combination 2 * this

lemma zmodP {k N a r : ℕ} (hk : k = 1 ∨ k = 2)
    (he : eP a r + 5 * (k - 1) = 5 * N + (2 * k - 1)) :
    (a : ZMod 5) * (a + 1) + 2 * (3 * r ^ 2 + 3 * a * r + r) = 2 * (2 * k - 1) := by
  have h2 := eP_two a r
  have hz := congrArg (Int.cast : ℤ → ZMod 5) h2
  push_cast at hz
  rw [← hz]
  have h5 : (5 : ZMod 5) = 0 := by decide
  rcases hk with rfl | rfl
  · have : (eP a r : ZMod 5) = ((5 * N + 1 : ℕ) : ZMod 5) := by rw [← he]; norm_num
    rw [this]; push_cast; rw [h5]; ring
  · have : (eP a r : ZMod 5) + 5 = ((5 * N + 3 : ℕ) : ZMod 5) := by
      rw [← he]; push_cast; ring
    push_cast at this; rw [h5] at this; linear_combination 2 * this

lemma zmod_res {a v : ℕ} (hv : v < 5) : (a : ZMod 5) = v ↔ a % 5 = v := by
  constructor
  · intro h
    have := (ZMod.natCast_eq_natCast_iff' a v 5).mp h
    rwa [Nat.mod_eq_of_lt hv] at this
  · intro h
    rw [ZMod.natCast_eq_natCast_iff', h, Nat.mod_eq_of_lt hv]

/-- **(i)**: `G_M` with `a ≡ −2k` ↔ `(P, +)` via `(b, r) ↦ (5b − 2k, 5r + k + 1)`. -/
theorem bij_i (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), gMn k N p.1 p.2
      = -∑ p ∈ range (N + 1) ×ˢ range (N + 1), mPp k N p.1 p.2 := by
  rw [← sum_neg_distrib]
  symm
  refine sum_bij_ne_zero (fun p _ _ => (5 * p.1 - 2 * k, 5 * p.2 + k + 1)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hp hne
    simp only [mPp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hb, he⟩, -⟩ := hne
    have h1 := eP_ge b r
    simp only [mem_product, mem_range]
    omega
  · rintro ⟨b, r⟩ _ hne ⟨b', r'⟩ _ hne' h
    simp only [Prod.mk.injEq] at h ⊢
    simp only [mPp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne hne'
    omega
  · rintro ⟨a, r'⟩ hp hne
    simp only [gMn, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨ha, hr, he⟩, -⟩ := hne
    -- residue of `r'`
    have hz := zmodM hk hr he
    have hax : (a : ZMod 5) = ((5 - 2 * k : ℕ) : ZMod 5) := (zmod_res (by omega)).mpr ha
    have hres : r' % 5 = k + 1 := by
      rw [← zmod_res (by omega)]
      rw [hax] at hz
      generalize (r' : ZMod 5) = x at hz ⊢
      rcases hk with rfl | rfl <;> revert x <;> decide
    have hlt : k + 1 ≤ r' := by have := Nat.mod_le r' 5; rcases hk with rfl | rfl <;> omega
    have hb5 : 5 - 2 * k ≤ a := by rcases hk with rfl | rfl <;> omega
    refine ⟨((a + 2 * k) / 5, (r' - (k + 1)) / 5), ?_, ?_, ?_⟩
    · have hal := alg_Pp hk ((a + 2 * k) / 5) ((r' - (k + 1)) / 5) (by rcases hk with rfl | rfl <;> omega)
      rw [show 5 * ((a + 2 * k) / 5) - 2 * k = a by rcases hk with rfl | rfl <;> omega,
        show 5 * ((r' - (k + 1)) / 5) + k + 1 = r' by rcases hk with rfl | rfl <;> omega, he] at hal
      have := eP_ge ((a + 2 * k) / 5) ((r' - (k + 1)) / 5)
      simp only [mem_product, mem_range]
      constructor <;> rcases hk with rfl | rfl <;> omega
    · have hal := alg_Pp hk ((a + 2 * k) / 5) ((r' - (k + 1)) / 5) (by rcases hk with rfl | rfl <;> omega)
      rw [show 5 * ((a + 2 * k) / 5) - 2 * k = a by rcases hk with rfl | rfl <;> omega,
        show 5 * ((r' - (k + 1)) / 5) + k + 1 = r' by rcases hk with rfl | rfl <;> omega, he] at hal
      simp only [mPp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp]
      exact ⟨⟨by rcases hk with rfl | rfl <;> omega, by omega⟩, by positivity⟩
    · simp only [Prod.mk.injEq]
      constructor <;> rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne
    simp only [mPp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hb, he⟩, -⟩ := hne
    have hal := alg_Pp hk b r hb
    rw [he] at hal
    simp only [mPp, gMn, if_pos (And.intro hb he)]
    rw [if_pos ⟨by rcases hk with rfl | rfl <;> omega, by omega, hal⟩, neg_one_pow_par (x := 5 * b - 2 * k) (y := b)
      (by rcases hk with rfl | rfl <;> omega)]

def mPm1 (b r : ℕ) : ℤ := if 1 ≤ r ∧ 5 * eP b r = N + k * b then (-1) ^ b else 0

/-- **(ii)**: `G_M` with `a ≡ 2k` ↔ `(P, −)`, `r ≥ 1`, via `(b, r) ↦ (5b + 2k, 5r + 1 − k)`. -/
theorem bij_ii (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), gMp k N p.1 p.2
      = -∑ p ∈ range (N + 1) ×ˢ range (N + 1), mPm1 k N p.1 p.2 := by
  rw [← sum_neg_distrib]
  symm
  refine sum_bij_ne_zero (fun p _ _ => (5 * p.1 + 2 * k, 5 * p.2 + 1 - k)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hp hne
    simp only [mPm1, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hr, he⟩, -⟩ := hne
    have h1 := eP_ge b r
    simp only [mem_product, mem_range]
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne ⟨b', r'⟩ _ hne' h
    simp only [Prod.mk.injEq] at h ⊢
    simp only [mPm1, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne hne'
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨a, r'⟩ hp hne
    simp only [gMp, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨ha, hr, he⟩, -⟩ := hne
    have hz := zmodM hk hr he
    have hax : (a : ZMod 5) = ((2 * k : ℕ) : ZMod 5) := (zmod_res (by omega)).mpr ha
    have hres : r' % 5 = (6 - k) % 5 := by
      rw [← zmod_res (by omega)]
      rw [hax] at hz
      generalize (r' : ZMod 5) = x at hz ⊢
      rcases hk with rfl | rfl <;> revert x <;> decide
    have hb5 : 2 * k ≤ a := by rcases hk with rfl | rfl <;> omega
    have hrr : 5 ≤ r' + k + 1 := by rcases hk with rfl | rfl <;> omega
    have hal := alg_Pm hk ((a - 2 * k) / 5) ((r' + k - 1) / 5) (by omega)
    rw [show 5 * ((a - 2 * k) / 5) + 2 * k = a by rcases hk with rfl | rfl <;> omega,
      show 5 * ((r' + k - 1) / 5) + 1 - k = r' by rcases hk with rfl | rfl <;> omega] at hal
    refine ⟨((a - 2 * k) / 5, (r' + k - 1) / 5), ?_, ?_, ?_⟩
    · have := eP_ge ((a - 2 * k) / 5) ((r' + k - 1) / 5)
      simp only [mem_product, mem_range]
      constructor <;> rcases hk with rfl | rfl <;> omega
    · simp only [mPm1, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp]
      exact ⟨⟨by omega, by rcases hk with rfl | rfl <;> omega⟩, by positivity⟩
    · simp only [Prod.mk.injEq]
      constructor <;> rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne
    simp only [mPm1, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hr, he⟩, -⟩ := hne
    have hal := alg_Pm hk b r hr
    simp only [mPm1, gMp, if_pos (And.intro hr he)]
    rw [if_pos ⟨by rcases hk with rfl | rfl <;> omega, by rcases hk with rfl | rfl <;> omega,
      by rcases hk with rfl | rfl <;> omega⟩, neg_one_pow_par (x := 5 * b + 2 * k) (y := b) (by omega)]

def gPn1 (a r : ℕ) : ℤ := if r = k - 1 then 0 else gPn k N a r
def gPn0 (a r : ℕ) : ℤ := if r = k - 1 then gPn k N a r else 0

/-- **(iii)**: `G_P` with `a ≡ −2k`, `r ≠ k−1` ↔ `(M, +)` via `(b, r) ↦ (5b − 2k, 5r + k − 1)`. -/
theorem bij_iii (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), gPn1 k N p.1 p.2
      = -∑ p ∈ range (N + 1) ×ˢ range (N + 1), mMp k N p.1 p.2 := by
  rw [← sum_neg_distrib]
  symm
  refine sum_bij_ne_zero (fun p _ _ => (5 * p.1 - 2 * k, 5 * p.2 + k - 1)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hp hne
    simp only [mMp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hb, hr, he⟩, -⟩ := hne
    have h1 := eM_ge' b r hr
    simp only [mem_product, mem_range]
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne ⟨b', r'⟩ _ hne' h
    simp only [Prod.mk.injEq] at h ⊢
    simp only [mMp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne hne'
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨a, r'⟩ hp hne
    simp only [gPn1, gPn, ne_eq, ite_eq_left_iff, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨hr0, ⟨ha, he⟩, -⟩ := hne
    have hz := zmodP hk he
    have hax : (a : ZMod 5) = ((5 - 2 * k : ℕ) : ZMod 5) := (zmod_res (by omega)).mpr ha
    have hres : r' % 5 = k - 1 := by
      rw [← zmod_res (by omega)]
      rw [hax] at hz
      generalize (r' : ZMod 5) = x at hz ⊢
      rcases hk with rfl | rfl <;> revert x <;> decide
    have hb5 : 5 - 2 * k ≤ a := by rcases hk with rfl | rfl <;> omega
    have hrr : k + 4 ≤ r' := by rcases hk with rfl | rfl <;> omega
    have hal := alg_Mp hk ((a + 2 * k) / 5) ((r' + 1 - k) / 5) (by rcases hk with rfl | rfl <;> omega)
      (by rcases hk with rfl | rfl <;> omega)
    rw [show 5 * ((a + 2 * k) / 5) - 2 * k = a by rcases hk with rfl | rfl <;> omega,
      show 5 * ((r' + 1 - k) / 5) + k - 1 = r' by rcases hk with rfl | rfl <;> omega, he] at hal
    refine ⟨((a + 2 * k) / 5, (r' + 1 - k) / 5), ?_, ?_, ?_⟩
    · have := eM_ge' ((a + 2 * k) / 5) ((r' + 1 - k) / 5) (by rcases hk with rfl | rfl <;> omega)
      simp only [mem_product, mem_range]
      constructor <;> rcases hk with rfl | rfl <;> omega
    · simp only [mMp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp]
      exact ⟨⟨by rcases hk with rfl | rfl <;> omega, by rcases hk with rfl | rfl <;> omega,
        by rcases hk with rfl | rfl <;> omega⟩, by positivity⟩
    · simp only [Prod.mk.injEq]
      constructor <;> rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne
    simp only [mMp, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hb, hr, he⟩, -⟩ := hne
    have hal := alg_Mp hk b r hb hr
    rw [he] at hal
    simp only [mMp, gPn1, gPn, if_pos (And.intro hb (And.intro hr he))]
    rw [if_neg (by rcases hk with rfl | rfl <;> omega), if_pos ⟨by rcases hk with rfl | rfl <;> omega, hal⟩,
      neg_one_pow_par (x := 5 * b - 2 * k) (y := b) (by rcases hk with rfl | rfl <;> omega), neg_neg]

def jA (d : ℕ) : ℤ := if 5 * d.choose 2 + k * d = N then (-1) ^ d else 0
def jB (d : ℕ) : ℤ := if 5 * (d + 1).choose 2 + (5 - k) * (d + 1) = N then (-1) ^ (d + 1) else 0
def jA1 (d : ℕ) : ℤ := if 1 ≤ d then jA k N d else 0

/-- **(iii′)**: the leftover `G_P` terms (`r = k − 1`) are the `m ≥ 1` terms of `J_{5,k}`. -/
theorem bij_iii' (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), gPn0 k N p.1 p.2 = ∑ d ∈ range (N + 1), jA1 k N d := by
  symm
  refine sum_bij_ne_zero (fun d _ _ => (5 * d - 2 * k, k - 1)) ?_ ?_ ?_ ?_
  · intro d hd hne
    simp only [jA1, jA, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨hd1, he, -⟩ := hne
    have : d ≤ d.choose 2 + 1 := by
      have := CrankProof.two_choose_two d
      rcases d with _ | d
      · omega
      · have : ((d + 1).choose 2 : ℤ) * 2 = (d + 1) * d := by push_cast at this; linarith
        nlinarith
    simp only [mem_product, mem_range]
    rcases hk with rfl | rfl <;> omega
  · intro d _ hne d' _ hne' h
    dsimp only at h
    simp only [Prod.mk.injEq] at h
    simp only [jA1, jA, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne hne'
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨a, r'⟩ hp hne
    simp only [gPn0, gPn, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨hr0, ⟨ha, he⟩, -⟩ := hne
    subst hr0
    have hb5 : 5 - 2 * k ≤ a := by rcases hk with rfl | rfl <;> omega
    have hal := alg_J hk ((a + 2 * k) / 5) (by rcases hk with rfl | rfl <;> omega)
    rw [show 5 * ((a + 2 * k) / 5) - 2 * k = a by rcases hk with rfl | rfl <;> omega, he] at hal
    have hd1 : 1 ≤ (a + 2 * k) / 5 := by rcases hk with rfl | rfl <;> omega
    have hN : 5 * ((a + 2 * k) / 5).choose 2 + k * ((a + 2 * k) / 5) = N := by
      rcases hk with rfl | rfl <;> omega
    refine ⟨(a + 2 * k) / 5, ?_, ?_, ?_⟩
    · simp only [mem_range]
      have : (a + 2 * k) / 5 ≤ k * ((a + 2 * k) / 5) := Nat.le_mul_of_pos_left _ (by omega)
      omega
    · rw [jA1, if_pos hd1, jA, if_pos hN]; positivity
    · dsimp only; simp only [Prod.mk.injEq]
      exact ⟨by rcases hk with rfl | rfl <;> omega, trivial⟩
  · intro d _ hne
    dsimp only
    simp only [jA1, jA, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨hd1, he, -⟩ := hne
    have hal := alg_J hk d hd1
    rw [he] at hal
    rw [jA1, if_pos hd1, jA, if_pos he, gPn0, if_pos rfl, gPn, if_pos ⟨by rcases hk with rfl | rfl <;> omega, hal⟩,
      neg_one_pow_par (x := 5 * d - 2 * k) (y := d) (by rcases hk with rfl | rfl <;> omega)]

/-- **(iv)**: `G_P` with `a ≡ 2k` ↔ `(M, −)` via `(b, r) ↦ (5b + 2k, 5r − k − 1)`. -/
theorem bij_iv (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), gPp k N p.1 p.2
      = -∑ p ∈ range (N + 1) ×ˢ range (N + 1), mMm k N p.1 p.2 := by
  rw [← sum_neg_distrib]
  symm
  refine sum_bij_ne_zero (fun p _ _ => (5 * p.1 + 2 * k, 5 * p.2 - k - 1)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hp hne
    simp only [mMm, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hr, he⟩, -⟩ := hne
    have h1 := eM_ge' b r hr
    simp only [mem_product, mem_range]
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne ⟨b', r'⟩ _ hne' h
    simp only [Prod.mk.injEq] at h ⊢
    simp only [mMm, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne hne'
    rcases hk with rfl | rfl <;> omega
  · rintro ⟨a, r'⟩ hp hne
    simp only [gPp, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨ha, he⟩, -⟩ := hne
    have hz := zmodP hk he
    have hax : (a : ZMod 5) = ((2 * k : ℕ) : ZMod 5) := (zmod_res (by omega)).mpr ha
    have hres : r' % 5 = 4 - k := by
      rw [← zmod_res (by omega)]
      rw [hax] at hz
      generalize (r' : ZMod 5) = x at hz ⊢
      rcases hk with rfl | rfl <;> revert x <;> decide
    have hb5 : 2 * k ≤ a := by rcases hk with rfl | rfl <;> omega
    have hr1 : 1 ≤ (r' + k + 1) / 5 := by rcases hk with rfl | rfl <;> omega
    have hal := alg_Mm hk ((a - 2 * k) / 5) ((r' + k + 1) / 5) hr1
    rw [show 5 * ((a - 2 * k) / 5) + 2 * k = a by rcases hk with rfl | rfl <;> omega,
      show 5 * ((r' + k + 1) / 5) - k - 1 = r' by rcases hk with rfl | rfl <;> omega] at hal
    refine ⟨((a - 2 * k) / 5, (r' + k + 1) / 5), ?_, ?_, ?_⟩
    · have := eM_ge' ((a - 2 * k) / 5) ((r' + k + 1) / 5) hr1
      simp only [mem_product, mem_range]
      constructor <;> rcases hk with rfl | rfl <;> omega
    · simp only [mMm, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp]
      exact ⟨⟨hr1, by rcases hk with rfl | rfl <;> omega⟩, by positivity⟩
    · simp only [Prod.mk.injEq]
      constructor <;> rcases hk with rfl | rfl <;> omega
  · rintro ⟨b, r⟩ _ hne
    simp only [mMm, neg_eq_zero, ne_eq, ite_eq_right_iff, Classical.not_imp] at hne
    obtain ⟨⟨hr, he⟩, -⟩ := hne
    have hal := alg_Mm hk b r hr
    simp only [mMm, gPp, if_pos (And.intro hr he)]
    rw [if_pos ⟨by rcases hk with rfl | rfl <;> omega, by rcases hk with rfl | rfl <;> omega⟩,
      neg_one_pow_par (x := 5 * b + 2 * k) (y := b) (by omega), neg_neg]

def mPm0 (b r : ℕ) : ℤ := if r = 0 then mPm k N b r else 0

/-- **(v)**: the `r = 0` terms of `(P, −)` are the `m ≤ 0` terms of `J_{5,k}`. -/
theorem bij_v (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (N + 1) ×ˢ range (N + 1), mPm0 k N p.1 p.2 = jA k N 0 + ∑ d ∈ range (N + 1), jB k N d := by
  rw [sum_product]
  have hcol : ∀ b ∈ range (N + 1), ∑ r ∈ range (N + 1), mPm0 k N b r = mPm k N b 0 := fun b _ => by
    rw [sum_eq_single 0 (fun r _ hr => by rw [mPm0, if_neg hr]) (fun h => absurd (mem_range.mpr (by omega)) h),
      mPm0, if_pos rfl]
  rw [sum_congr rfl hcol, sum_range_succ', sum_range_succ (fun d => jB k N d), add_comm (jA k N 0)]
  have hlast : jB k N N = 0 := by
    rw [jB, if_neg]
    have : N + 1 ≤ (5 - k) * (N + 1) := Nat.le_mul_of_pos_left _ (by omega)
    omega
  have h0 : mPm k N 0 0 = jA k N 0 := by
    simp [mPm, jA, eP, eq_comm]
  have hT : ∀ d : ℕ, 5 * eP (d + 1) 0 = 5 * (d + 1).choose 2 + 5 * (d + 1) := fun d => by
    have h1 := eP_two (d + 1) 0
    have h2 := CrankProof.two_choose_two (d + 1)
    apply two_inj; push_cast at h1 h2 ⊢; linear_combination 5 * h1 - 5 * h2
  rw [hlast, add_zero, h0]
  congr 1
  refine sum_congr rfl fun d _ => ?_
  rw [mPm, jB, hT]
  congr 1
  apply propext
  rcases hk with rfl | rfl <;> constructor <;> intro h <;> omega

/-- **the count identity**: `G + M = J` at each exponent `N`. -/
theorem count_identity (hk : k = 1 ∨ k = 2) :
    ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), (gPn k N p.1 p.2 + gPp k N p.1 p.2 + gMn k N p.1 p.2
        + gMp k N p.1 p.2)
      + ∑ p ∈ range (N + 1) ×ˢ range (N + 1), (mPp k N p.1 p.2 + mPm k N p.1 p.2 + mMp k N p.1 p.2
        + mMm k N p.1 p.2)
      = ∑ d ∈ range (N + 1), jA k N d + ∑ d ∈ range (N + 1), jB k N d := by
  have e1 : ∀ a r, gPn k N a r = gPn0 k N a r + gPn1 k N a r := fun a r => by
    unfold gPn0 gPn1; split_ifs <;> simp
  have e2 : ∀ b r, mPm k N b r = mPm0 k N b r + mPm1 k N b r := fun b r => by
    unfold mPm0 mPm1
    rcases Nat.eq_zero_or_pos r with rfl | hr
    · simp
    · rw [if_neg (by omega)]
      by_cases h : 5 * eP b r = N + k * b
      · rw [if_pos ⟨hr, h⟩, mPm, if_pos h, zero_add]
      · rw [if_neg (fun h' => h h'.2), mPm, if_neg h, add_zero]
  have e3 : ∑ d ∈ range (N + 1), jA k N d = jA k N 0 + ∑ d ∈ range (N + 1), jA1 k N d := by
    rw [sum_range_succ', sum_range_succ' (jA1 k N), jA1, if_neg (by omega), add_zero, add_comm]
    simp only [jA1, if_pos (Nat.le_add_left 1 _)]
  simp only [e1, e2, sum_add_distrib]
  rw [bij_i k N hk, bij_ii k N hk, bij_iii k N hk, bij_iii' k N hk, bij_iv k N hk, bij_v k N hk, e3]
  ring

end Sums

section Bridge
open CrankProof (Hblk)

/-- the box indicator of the pair `(a, r)` at exponent `m` (both types). -/
def hb (a r m : ℕ) : ℤ := (if eP a r = m then (-1) ^ a else 0) + (if 1 ≤ r ∧ eM a r = m then -(-1) ^ a else 0)

lemma eP_ge_r (a r : ℕ) : r ≤ eP a r := by have := eP_ge a r; omega
lemma eP_ge_a (a r : ℕ) : a ≤ eP a r := by have := eP_ge a r; omega
lemma eM_ge_r (a r : ℕ) (hr : 1 ≤ r) : r ≤ eM a r := by have := eM_ge' a r hr; omega
lemma eM_ge_a (a r : ℕ) (hr : 1 ≤ r) : a ≤ eM a r := by have := eM_ge' a r hr; omega

lemma coeff_Hblk_box (a m B : ℕ) (hB : m < B) :
    coeff m (Hblk a) = ∑ r ∈ range B, hb a r m := by
  rw [Hblk, mul_assoc, coeff_C_mul, coeff_X_pow_mul']
  split_ifs with hT
  · rw [coeff_RHSz, rhsCoef, mul_sum]
    have hsub : range (m - a * (a + 1) / 2 + 1) ⊆ range B := range_subset_range.mpr (by omega)
    have hext : ∑ r ∈ range B, hb a r m = ∑ r ∈ range (m - a * (a + 1) / 2 + 1), hb a r m := by
      refine (sum_subset hsub fun r _ hr => ?_).symm
      simp only [mem_range, not_lt] at hr
      have hP : a * (a + 1) / 2 + r ≤ eP a r := by unfold eP; omega
      have hM : 1 ≤ r → a * (a + 1) / 2 + r ≤ eM a r := fun h1 => by
        have : r + a + r ≤ 3 * r ^ 2 + 3 * a * r := by nlinarith
        unfold eM; omega
      unfold hb
      rw [if_neg (by omega), if_neg (fun h => by have := hM h.1; omega)]; ring
    rw [hext]
    refine sum_congr rfl fun r _ => ?_
    unfold hb eP eM
    have e1 : (3 * r ^ 2 + 3 * a * r + r = m - a * (a + 1) / 2) ↔ (a * (a + 1) / 2 + (3 * r ^ 2 + 3 * a * r + r) = m) := by
      omega
    have e2 : (1 ≤ r ∧ 3 * r ^ 2 + 3 * a * r - r - a = m - a * (a + 1) / 2)
        ↔ (1 ≤ r ∧ a * (a + 1) / 2 + (3 * r ^ 2 + 3 * a * r - r - a) = m) := by omega
    simp only [e1, e2]
    split_ifs <;> ring
  · rw [mul_zero]; symm
    refine sum_eq_zero fun r _ => ?_
    have := MockTheta5.JTP.tri_ge a
    have hP : a * (a + 1) / 2 ≤ eP a r := by unfold eP; omega
    have hM : a * (a + 1) / 2 ≤ eM a r := by unfold eM; omega
    unfold hb; rw [if_neg (by omega), if_neg (by omega)]; ring

/-- the orbit `a ≡ ±2k (mod 5)` part of `Σ_a (−1)^a q^{T(a)} PT_a`. -/
noncomputable def Gorb (k : ℕ) : PowerSeries ℤ :=
  mk fun m => ∑ a ∈ range (m + 1), if a % 5 = 5 - 2 * k ∨ a % 5 = 2 * k then coeff m (Hblk a) else 0

theorem G_bridge {k : ℕ} (hk : k = 1 ∨ k = 2) (N : ℕ) :
    coeff N (X ^ (k - 1) * dis5 (2 * k - 1) (Gorb k))
      = ∑ p ∈ range (5 * N + 6) ×ˢ range (5 * N + 6), (gPn k N p.1 p.2 + gPp k N p.1 p.2 + gMn k N p.1 p.2
        + gMp k N p.1 p.2) := by
  rw [coeff_X_pow_mul']
  split_ifs with hN
  · rw [MockTheta5.JTP.coeff_dis5, Gorb, coeff_mk]
    set m := 5 * (N - (k - 1)) + (2 * k - 1) with hm
    have hmB : m < 5 * N + 6 := by rcases hk with rfl | rfl <;> omega
    have hext : ∑ a ∈ range (m + 1), (if a % 5 = 5 - 2 * k ∨ a % 5 = 2 * k then coeff m (Hblk a) else 0)
        = ∑ a ∈ range (5 * N + 6), ∑ r ∈ range (5 * N + 6),
          (if a % 5 = 5 - 2 * k ∨ a % 5 = 2 * k then hb a r m else 0) := by
      rw [← sum_subset (range_subset_range.mpr (show m + 1 ≤ 5 * N + 6 by omega)) fun a _ ha => ?_]
      · refine sum_congr rfl fun a _ => ?_
        split_ifs
        · exact coeff_Hblk_box a m _ hmB
        · simp
      · simp only [mem_range, not_lt] at ha
        refine sum_eq_zero fun r _ => ?_
        split_ifs
        · unfold hb
          rw [if_neg (by have := eP_ge_a a r; omega), if_neg (fun h => by have := eM_ge_a a r h.1; omega)]; ring
        · rfl
    rw [hext, sum_product]
    refine sum_congr rfl fun a _ => sum_congr rfl fun r _ => ?_
    have hP : (eP a r + 5 * (k - 1) = 5 * N + (2 * k - 1)) ↔ eP a r = m := by
      rcases hk with rfl | rfl <;> omega
    have hM : (eM a r + 5 * (k - 1) = 5 * N + (2 * k - 1)) ↔ eM a r = m := by
      rcases hk with rfl | rfl <;> omega
    unfold gPn gPp gMn gMp hb
    simp only [hP, hM]
    have hdis : ¬ (a % 5 = 5 - 2 * k ∧ a % 5 = 2 * k) := by rcases hk with rfl | rfl <;> omega
    by_cases h1 : a % 5 = 5 - 2 * k <;> by_cases h2 : a % 5 = 2 * k <;> simp [h1, h2] <;> omega
  · symm
    refine sum_eq_zero fun p _ => ?_
    have hk2 : k = 2 := by rcases hk with rfl | rfl <;> omega
    subst hk2
    unfold gPn gPp gMn gMp
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega)]; ring

/-- `[q^N] q^e E₅(PT_a)` as a signed count. -/
def tC (a e N : ℕ) : ℤ := ∑ r ∈ range (N + 1),
  ((if 5 * (3 * r ^ 2 + 3 * a * r + r) + e = N then 1 else 0)
    - (if 1 ≤ r ∧ 5 * (3 * r ^ 2 + 3 * a * r - r - a) + e = N then 1 else 0))

lemma coeff_Xpow_E5 (e N : ℕ) (f : PowerSeries ℤ) :
    coeff N (X ^ e * E5 f) = if e ≤ N ∧ 5 ∣ N - e then coeff ((N - e) / 5) f else 0 := by
  rw [coeff_X_pow_mul']
  by_cases h : e ≤ N
  · rw [if_pos h, MockTheta5.JTP.coeff_E5]
    by_cases h5 : 5 ∣ N - e
    · rw [if_pos h5, if_pos ⟨h, h5⟩]
    · rw [if_neg h5, if_neg (fun h' => h5 h'.2)]
  · rw [if_neg h, if_neg (fun h' => h h'.1)]

lemma coeff_tC (a e N : ℕ) : coeff N (X ^ e * E5 (RHSz a)) = tC a e N := by
  rw [coeff_Xpow_E5, tC]
  split_ifs with h
  · obtain ⟨he, j, hj⟩ := h
    rw [show (N - e) / 5 = j by omega, coeff_RHSz, rhsCoef]
    rw [← sum_subset (range_subset_range.mpr (show j + 1 ≤ N + 1 by omega)) fun r _ hr => ?_]
    · refine sum_congr rfl fun r _ => ?_
      congr 1
      · congr 1; apply propext; generalize 3 * r ^ 2 + 3 * a * r + r = X; omega
      · congr 1; apply propext; generalize 3 * r ^ 2 + 3 * a * r - r - a = X
        constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨h1, by omega⟩
    · simp only [mem_range, not_lt] at hr
      have : r ≤ 3 * r ^ 2 + 3 * a * r + r := by omega
      rw [if_neg (by omega), if_neg (fun h' => by
        have hU : 2 * r + a ≤ 3 * r ^ 2 + 3 * a * r := by nlinarith [h'.1]
        obtain ⟨-, h2⟩ := h'
        generalize 3 * r ^ 2 + 3 * a * r = U at *
        omega)]; ring
  · symm
    refine sum_eq_zero fun r _ => ?_
    rw [if_neg (fun h' => h ⟨by omega, ⟨3 * r ^ 2 + 3 * a * r + r, by omega⟩⟩),
      if_neg (fun h' => h ⟨by omega, ⟨3 * r ^ 2 + 3 * a * r - r - a, by omega⟩⟩)]; ring

lemma hiE_ge (k a : ℕ) : a ≤ hiE k a := by unfold hiE; have := MockTheta5.JTP.tri_ge a; omega
lemma loE_ge {k : ℕ} (hk5 : k ≤ 4) (a : ℕ) : a ≤ loE k a := by
  unfold loE; have := MockTheta5.JTP.tri_ge a
  have : k * a ≤ 4 * a := Nat.mul_le_mul_right _ hk5
  omega

lemma SpecTr_stable {k : ℕ} (hk5 : k ≤ 4) {c N : ℕ} (h : c ≤ N) :
    coeff c (SpecTr k N) = coeff c (SpecTr k c) := by
  rw [SpecTr, SpecTr, map_sum, map_sum]
  symm
  refine sum_subset (range_subset_range.mpr (by omega)) fun a _ ha => ?_
  simp only [mem_range, not_lt] at ha
  rw [add_mul, add_mul, map_add]
  have h1 : coeff c (X ^ hiE k a * C ((-1 : ℤ) ^ a) * E5 (LHSz a)) = 0 := by
    rw [mul_assoc]; exact coeffZ_Xpow_zero (by have := hiE_ge k a; omega) _
  have h2 : coeff c ((if a = 0 then 0 else X ^ loE k a) * C ((-1 : ℤ) ^ a) * E5 (LHSz a)) = 0 := by
    split_ifs
    · simp
    · rw [mul_assoc]; exact coeffZ_Xpow_zero (by have := loE_ge hk5 a; omega) _
  rw [h1, h2, add_zero]

lemma spec_dvd {k : ℕ} (hk : 1 ≤ k) (hk5 : k ≤ 4) (N : ℕ) :
    (X : PowerSeries ℤ) ^ (N + 1) ∣ Pinf k 5 * Pinf (5 - k) 5 * Phi k - SpecTr k N := by
  rw [X_pow_dvd_iff]; intro c hc
  rw [map_sub, sub_eq_zero, spec_coeff k hk hk5 c, SpecTr_stable hk5 (show c ≤ N by omega)]

theorem M_bridge {k : ℕ} (hk : k = 1 ∨ k = 2) (N : ℕ) :
    coeff N (CrankProof.Jab 5 k * Phi k)
      = ∑ p ∈ range (N + 1) ×ˢ range (N + 1), (mPp k N p.1 p.2 + mPm k N p.1 p.2 + mMp k N p.1 p.2
        + mMm k N p.1 p.2) := by
  have hk1 : 1 ≤ k := by omega
  have hk5 : k ≤ 4 := by omega
  rw [CrankProof.Jab, show Pinf k 5 * Pinf (5 - k) 5 * Pinf 5 5 * Phi k
      = Pinf 5 5 * (Pinf k 5 * Pinf (5 - k) 5 * Phi k) by ring, coeffZ_congr (spec_dvd hk1 hk5 N), SpecTr,
    mul_sum, map_sum, sum_product]
  refine sum_congr rfl fun a _ => ?_
  have hPE : Pinf 5 5 * E5 (LHSz a) = E5 (RHSz a) := by
    rw [Pinf_aa 5 (by norm_num), ← E5_eq_Ea, ← map_mul, bailey_k, ← mul_assoc,
      Ring.mul_inverse_cancel _ isUnit_qfacInf, one_mul]
  have hterm : ∀ e : ℕ, coeff N (Pinf 5 5 * (X ^ e * C ((-1 : ℤ) ^ a) * E5 (LHSz a)))
      = (-1) ^ a * tC a e N := fun e => by
    rw [show Pinf 5 5 * (X ^ e * C ((-1 : ℤ) ^ a) * E5 (LHSz a))
        = C ((-1 : ℤ) ^ a) * (X ^ e * (Pinf 5 5 * E5 (LHSz a))) by ring, hPE, coeff_C_mul, coeff_tC]
  rw [add_mul, add_mul, mul_add, map_add, hterm]
  have hcondP : ∀ r, (5 * (3 * r ^ 2 + 3 * a * r + r) + hiE k a = N) ↔ (5 * eP a r + k * a = N) := fun r => by
    unfold hiE eP; omega
  have hcondM : ∀ r, (1 ≤ r ∧ 5 * (3 * r ^ 2 + 3 * a * r - r - a) + hiE k a = N)
      ↔ (1 ≤ r ∧ 5 * eM a r + k * a = N) := fun r => by
    unfold hiE eM; constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨h1, by omega⟩
  have hka : k * a ≤ 5 * (a * (a + 1) / 2) := by
    have := MockTheta5.JTP.tri_ge a
    have : k * a ≤ 4 * a := Nat.mul_le_mul_right _ hk5
    omega
  have hcondP' : ∀ r, (5 * (3 * r ^ 2 + 3 * a * r + r) + loE k a = N) ↔ (5 * eP a r = N + k * a) := fun r => by
    unfold loE eP; omega
  have hcondM' : ∀ r, (1 ≤ r ∧ 5 * (3 * r ^ 2 + 3 * a * r - r - a) + loE k a = N)
      ↔ (1 ≤ r ∧ 5 * eM a r = N + k * a) := fun r => by
    unfold loE eM; constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨h1, by omega⟩
  rcases Nat.eq_zero_or_pos a with rfl | ha
  · rw [if_pos rfl, zero_mul, zero_mul, mul_zero, map_zero, add_zero, tC, mul_sum]
    refine sum_congr rfl fun r _ => ?_
    have h0 : hiE k 0 = 0 := by simp [hiE]
    simp only [mPp, mMp, mPm, mMm, h0, show ¬ (1 ≤ 0) from by omega, false_and, if_false, zero_add,
      mul_zero, add_zero, pow_zero, one_mul]
    unfold eP eM
    simp only [zero_mul, mul_zero, zero_add, add_zero, Nat.zero_div, Nat.sub_zero]
    generalize r ^ 2 = R
    split_ifs <;> first | ring | omega
  · rw [if_neg (by omega), hterm, tC, tC, mul_sum, mul_sum, ← sum_add_distrib]
    refine sum_congr rfl fun r _ => ?_
    simp only [mPp, mMp, mPm, mMm, hcondP, hcondM, hcondP', hcondM', show 1 ≤ a from ha, true_and]
    split_ifs <;> ring

theorem J_bridge {k : ℕ} (hk : k = 1 ∨ k = 2) (N : ℕ) :
    coeff N (CrankProof.Jab 5 k) = ∑ d ∈ range (N + 1), jA k N d + ∑ d ∈ range (N + 1), jB k N d := by
  rw [CrankProof.Jab, jtp_ab 5 k (by omega) (by omega), thetaS, coeff_mk, thetaTr, map_add, map_sum, map_sum]
  congr 1
  · refine sum_congr rfl fun d _ => ?_
    rw [coeff_C_mul_X_pow, jA]
    split_ifs <;> first | rfl | omega
  · refine sum_congr rfl fun d _ => ?_
    rw [coeff_C_mul_X_pow, jB]
    split_ifs <;> first | rfl | omega

/-- **the mock-theta identity** `q^{k−1}·U_{5,2k−1}(G_{±2k}) = J_{5,k}(1 − Φ_k)`, i.e.
`U_{5,1}(G_{±2}) = −J_{5,1}φ` and `q·U_{5,3}(G_{±1}) = −J_{5,2}ψ`. -/
theorem orbit_mock {k : ℕ} (hk : k = 1 ∨ k = 2) :
    X ^ (k - 1) * dis5 (2 * k - 1) (Gorb k) = CrankProof.Jab 5 k * (1 - Phi k) := by
  ext N
  rw [G_bridge hk, mul_sub, mul_one, map_sub, J_bridge hk, M_bridge hk]
  have := count_identity k N hk
  linarith

end Bridge

end RankProof
