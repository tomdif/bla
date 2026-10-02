/-
# Dyson's rank, step 2: the reduced (2.19) identity as a `q`-series congruence

For every `a ≥ 0` and truncation order `N`,
`Σ_{|b|≤N} q^{(a−b)² + |b|(|b|+1)} PT_{|b|}(q²) ≡ ψ(q) · Σ_{t≤N} q^{a(a+1)/2+3at+t(3t+1)/2}(1 − q^{2a+2t+1})`
modulo `q^{N+1}`, where `PT_k = RHSz k` (Bailey side) and `ψ = psiSum = Σ_{m≥0} q^{m(m+1)/2}`.
Both sides' coefficients are signed counts of lattice points (`RankHR2Lat.param_counts`).
-/
import RamanujanTau.RankHR2Lat
import RamanujanTau.RankBailey
import RamanujanTau.MockTheta5PsiSeries

set_option autoImplicit false

namespace RankProof
open PowerSeries Finset MockTheta5.Bailey MockTheta5.JTP

section Counting

lemma coeff_sum_Xpow {ι : Type*} (B : Finset ι) (f : ι → ℕ) (k : ℕ) :
    coeff k (∑ x ∈ B, (X : PowerSeries ℤ) ^ f x) = ((B.filter fun x => f x = k).card : ℤ) := by
  rw [map_sum]
  simp only [coeff_X_pow]
  rw [sum_boole]
  congr 2
  exact filter_congr fun x _ => eq_comm

lemma coeff_eq_of_dvd {N i : ℕ} {f g : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ (N + 1) ∣ f - g) (hi : i ≤ N) :
    coeff i f = coeff i g := by
  have := (X_pow_dvd_iff.mp h) i (by omega)
  rwa [map_sub, sub_eq_zero] at this

lemma E2_dvd {m : ℕ} {f : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ m ∣ f) : (X : PowerSeries ℤ) ^ (2 * m) ∣ E2 f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨E2 d, by rw [map_mul, map_pow, E2_X, ← pow_mul, mul_comm 2 m]⟩

end Counting

section LSide
variable (a : ℕ)

/-- `(a−b)² + |b|(|b|+1)`. -/
def eLb (b : ℤ) : ℕ := ((a : ℤ) - b).natAbs ^ 2 + b.natAbs * (b.natAbs + 1)

def eLP (b : ℤ) (r : ℕ) : ℕ := eLb a b + 2 * (3 * r ^ 2 + 3 * b.natAbs * r + r)

/-- the `L⁻` exponent at `r = s + 1`. -/
def eLMs (b : ℤ) (s : ℕ) : ℕ :=
  eLb a b + 2 * ((s + 1) ^ 2 + b.natAbs * (s + 1)) + 2 * (2 * s ^ 2 + 3 * s + 1 + 2 * s * b.natAbs + b.natAbs)

lemma term_L (b : ℤ) (r : ℕ) :
    X ^ eLb a b * E2 (X ^ (r ^ 2 + b.natAbs * r) * αz b.natAbs r)
      = X ^ eLP a b r - (if r = 0 then 0 else X ^ eLMs a b (r - 1)) := by
  rcases r with _ | s
  · simp [αz, Bz, eLP]
  · rw [if_neg (by omega), show s + 1 - 1 = s by omega, αz, Bz, mul_sub, map_sub, mul_sub, map_mul, map_mul,
      map_pow, map_pow, map_pow, E2_X, ← pow_mul, ← pow_mul, ← pow_mul, ← pow_add, ← pow_add, ← pow_add, ← pow_add]
    congr 2
    · unfold eLP; ring
    · unfold eLMs; ring

lemma cast_eLP (b : ℤ) (r : ℕ) : (eLP a b r : ℤ)
    = ((a : ℤ) - b) ^ 2 + |b| * (|b| + 1) + 2 * (3 * (r : ℤ) ^ 2 + 3 * |b| * r + r) := by
  unfold eLP eLb; push_cast; rw [sq_abs]

lemma cast_eLMs (b : ℤ) (s : ℕ) : (eLMs a b s : ℤ)
    = ((a : ℤ) - b) ^ 2 + |b| * (|b| + 1) + 2 * (3 * ((s + 1 : ℕ) : ℤ) ^ 2 + 3 * |b| * (s + 1 : ℕ) - (s + 1 : ℕ) - |b|) := by
  unfold eLMs eLb; push_cast; rw [sq_abs]; ring

/-- the box of indices. -/
noncomputable def box (N : ℕ) : Finset (ℤ × ℕ) := Icc (-(N : ℤ)) N ×ˢ range (N + 1)

noncomputable def Lsum (N : ℕ) : PowerSeries ℤ :=
  ∑ b ∈ Icc (-(N : ℤ)) N, X ^ eLb a b * E2 (RHSz b.natAbs)

lemma L_trunc (N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣ Lsum a N
    - (∑ x ∈ box N, X ^ eLP a x.1 x.2 - ∑ x ∈ box N, (if x.2 = 0 then 0 else X ^ eLMs a x.1 (x.2 - 1))) := by
  have e : ∑ x ∈ box N, X ^ eLP a x.1 x.2 - ∑ x ∈ box N, (if x.2 = 0 then 0 else X ^ eLMs a x.1 (x.2 - 1))
      = ∑ b ∈ Icc (-(N : ℤ)) N, X ^ eLb a b
        * E2 (∑ r ∈ range (N + 1), X ^ (r ^ 2 + b.natAbs * r) * αz b.natAbs r) := by
    rw [← sum_sub_distrib, box, sum_product]
    refine sum_congr rfl fun b _ => ?_
    rw [map_sum, mul_sum]
    exact sum_congr rfl fun r _ => (term_L a b r).symm
  rw [e, Lsum, ← sum_sub_distrib]
  refine dvd_sum fun b _ => ?_
  rw [← mul_sub, ← map_sub]
  exact dvd_mul_of_dvd_right ((pow_dvd_pow X (by omega)).trans (E2_dvd (RHSz_dvd _ N))) _

theorem coeff_Lsum (N k : ℕ) (hk : k ≤ N) :
    coeff k (Lsum a N) = (((box N).filter (cLP a k)).card : ℤ) - (((box N).filter (cLM a k)).card : ℤ) := by
  rw [coeff_eq_of_dvd (L_trunc a N) hk, map_sub, coeff_sum_Xpow]
  congr 1
  · congr 2
    refine filter_congr fun x _ => ?_
    rw [cLP, ← cast_eLP]; exact_mod_cast Iff.rfl
  · have : ∀ x : ℤ × ℕ, (if x.2 = 0 then (0 : PowerSeries ℤ) else X ^ eLMs a x.1 (x.2 - 1))
        = if 1 ≤ x.2 then X ^ eLMs a x.1 (x.2 - 1) else 0 := fun x => by split_ifs <;> first | rfl | omega
    simp only [this]
    rw [← sum_filter, coeff_sum_Xpow, filter_filter]
    congr 2
    refine filter_congr fun x _ => ?_
    rw [cLM]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨h1, ?_⟩
      obtain ⟨b, r⟩ := x
      obtain ⟨s, rfl⟩ : ∃ s, r = s + 1 := ⟨r - 1, by simp only at h1; omega⟩
      simp only [show s + 1 - 1 = s by omega] at h2
      have := cast_eLMs a b s
      rw [h2] at this
      push_cast at this ⊢; linarith
    · rintro ⟨h1, h2⟩
      refine ⟨h1, ?_⟩
      obtain ⟨b, r⟩ := x
      obtain ⟨s, rfl⟩ : ∃ s, r = s + 1 := ⟨r - 1, by simp only at h1; omega⟩
      simp only [show s + 1 - 1 = s by omega]
      have h3 : (eLMs a b s : ℤ) = k := by rw [cast_eLMs]; push_cast at h2 ⊢; linarith
      exact_mod_cast h3

end LSide

section RSide
variable (a : ℕ)

lemma tri_even (t : ℕ) : 2 * t * (2 * t + 1) / 2 = 2 * t ^ 2 + t := by
  rw [show 2 * t * (2 * t + 1) = 2 * (2 * t ^ 2 + t) by ring]; omega

lemma tri_odd (t : ℕ) : (2 * t + 1) * (2 * t + 1 + 1) / 2 = (t + 1) * (2 * t + 1) := by
  rw [show (2 * t + 1) * (2 * t + 1 + 1) = 2 * ((t + 1) * (2 * t + 1)) by ring]; omega

lemma toNat_pos (t : ℕ) : (2 * (t : ℤ) ^ 2 + t).toNat = 2 * t ^ 2 + t := by
  rw [show 2 * (t : ℤ) ^ 2 + t = ((2 * t ^ 2 + t : ℕ) : ℤ) by push_cast; ring, Int.toNat_natCast]

lemma toNat_neg (t : ℕ) : (2 * (-((t : ℤ) + 1)) ^ 2 + -((t : ℤ) + 1)).toNat = (t + 1) * (2 * t + 1) := by
  rw [show 2 * (-((t : ℤ) + 1)) ^ 2 + -((t : ℤ) + 1) = (((t + 1) * (2 * t + 1) : ℕ) : ℤ) by push_cast; ring,
    Int.toNat_natCast]

/-- triangular numbers `m(m+1)/2` ↔ `2j² + j`, `j ∈ ℤ`. -/
lemma card_tri (N i : ℕ) (hi : i ≤ N) :
    ((range (N + 1)).filter fun m => m * (m + 1) / 2 = i).card
      = ((Icc (-(N : ℤ)) N).filter fun j => (2 * j ^ 2 + j).toNat = i).card := by
  refine card_nbij' (fun m => if m % 2 = 0 then ((m / 2 : ℕ) : ℤ) else -((m / 2 : ℕ) + 1 : ℤ))
    (fun j => if 0 ≤ j then 2 * j.toNat else 2 * (-j - 1).toNat + 1) ?_ ?_ ?_ ?_
  · intro m hm
    simp only [coe_filter, Set.mem_setOf_eq, mem_range, mem_Icc] at hm ⊢
    obtain ⟨t, rfl | rfl⟩ := Nat.even_or_odd' m
    · rw [if_pos (by omega), show 2 * t / 2 = t by omega, toNat_pos]
      rw [tri_even] at hm
      refine ⟨⟨by omega, by omega⟩, hm.2⟩
    · rw [if_neg (by omega), show (2 * t + 1) / 2 = t by omega, toNat_neg]
      rw [tri_odd] at hm
      refine ⟨⟨by omega, by omega⟩, hm.2⟩
  · intro j hj
    simp only [coe_filter, Set.mem_setOf_eq, mem_range, mem_Icc] at hj ⊢
    rcases le_or_gt 0 j with h | h
    · obtain ⟨t, rfl⟩ := Int.eq_ofNat_of_zero_le h
      rw [if_pos h, Int.toNat_natCast, tri_even]
      rw [toNat_pos] at hj
      refine ⟨by nlinarith, hj.2⟩
    · obtain ⟨t, ht⟩ : ∃ t : ℕ, j = -((t : ℤ) + 1) := ⟨(-j - 1).toNat, by omega⟩
      subst ht
      rw [if_neg (by omega), show -(-((t : ℤ) + 1)) - 1 = t by ring, Int.toNat_natCast, tri_odd]
      rw [toNat_neg] at hj
      refine ⟨by nlinarith, hj.2⟩
  · intro m _
    obtain ⟨t, rfl | rfl⟩ := Nat.even_or_odd' m
    · simp only
      rw [if_pos (by omega), if_pos (by omega)]; omega
    · simp only
      rw [if_neg (by omega), if_neg (by omega)]; omega
  · intro j _
    simp only
    rcases le_or_gt 0 j with h | h
    · rw [if_pos h, if_pos (by omega)]; omega
    · rw [if_neg (show ¬ 0 ≤ j by omega), if_neg (by omega)]; omega

lemma psi_trunc (N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣ psiSum
    - ∑ j ∈ Icc (-(N : ℤ)) N, X ^ (2 * j ^ 2 + j).toNat := by
  rw [X_pow_dvd_iff]
  intro i hi
  rw [map_sub, sub_eq_zero, coeff_sum_Xpow, psiSum, coeff_mtSum _ _ (fun n => tri_ge n) (show i + 1 ≤ N + 1 by omega)]
  simp only [mul_one]
  rw [← map_sum, coeff_sum_Xpow, card_tri N i (by omega)]

/-- `a(a+1)/2 + 3at + t(3t+1)/2 = n(3n+1)/2 − a²` at `n = a + t`. -/
def eRt (t : ℕ) : ℕ := a * (a + 1) / 2 + 3 * a * t + t * (3 * t + 1) / 2

noncomputable def Gsum (N : ℕ) : PowerSeries ℤ :=
  ∑ t ∈ range (N + 1), X ^ eRt a t * (1 - X ^ (2 * a + 2 * t + 1))

def eRP (x : ℤ × ℕ) : ℕ := (2 * x.1 ^ 2 + x.1).toNat + eRt a x.2

lemma two_eRP (x : ℤ × ℕ) :
    2 * (eRP a x : ℤ) = 4 * x.1 ^ 2 + 2 * x.1 + (a : ℤ) * (a + 1) + 6 * a * x.2 + (x.2 : ℤ) * (3 * x.2 + 1) := by
  have h1 : 2 * (a * (a + 1) / 2) = a * (a + 1) := Nat.mul_div_cancel' (Nat.even_mul_succ_self a).two_dvd
  have h2 : 2 * (x.2 * (3 * x.2 + 1) / 2) = x.2 * (3 * x.2 + 1) := by
    apply Nat.mul_div_cancel'
    rw [show x.2 * (3 * x.2 + 1) = x.2 * (x.2 + 1) + 2 * x.2 ^ 2 by ring]
    exact dvd_add (Nat.even_mul_succ_self _).two_dvd (dvd_mul_right _ _)
  have h0 : 0 ≤ 2 * x.1 ^ 2 + x.1 := by nlinarith [sq_nonneg (2 * x.1 + 1)]
  unfold eRP eRt
  push_cast
  rw [Int.toNat_of_nonneg h0]
  have h1' : (2 : ℤ) * ((a * (a + 1) / 2 : ℕ) : ℤ) = a * (a + 1) := by exact_mod_cast h1
  have h2' : (2 : ℤ) * ((x.2 * (3 * x.2 + 1) / 2 : ℕ) : ℤ) = x.2 * (3 * x.2 + 1) := by exact_mod_cast h2
  push_cast at h1' h2'
  linear_combination h1' + h2'

lemma R_trunc (N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣ psiSum * Gsum a N
    - (∑ x ∈ box N, X ^ eRP a x - ∑ x ∈ box N, X ^ (eRP a x + (2 * a + 2 * x.2 + 1))) := by
  have e : ∑ x ∈ box N, X ^ eRP a x - ∑ x ∈ box N, X ^ (eRP a x + (2 * a + 2 * x.2 + 1))
      = (∑ j ∈ Icc (-(N : ℤ)) N, X ^ (2 * j ^ 2 + j).toNat) * Gsum a N := by
    rw [← sum_sub_distrib, box, sum_product, Gsum, sum_mul]
    refine sum_congr rfl fun j _ => ?_
    rw [mul_sum]
    refine sum_congr rfl fun t _ => ?_
    rw [eRP, pow_add, pow_add]; ring
  rw [e, ← sub_mul]
  exact dvd_mul_of_dvd_left (psi_trunc N) _

theorem coeff_Rsum (N k : ℕ) (hk : k ≤ N) :
    coeff k (psiSum * Gsum a N) = (((box N).filter (cRP a k)).card : ℤ) - (((box N).filter (cRM a k)).card : ℤ) := by
  rw [coeff_eq_of_dvd (R_trunc a N) hk, map_sub, coeff_sum_Xpow, coeff_sum_Xpow]
  congr 3
  · refine filter_congr fun x _ => ?_
    rw [cRP, ← two_eRP]; constructor <;> intro h <;> [rw [h]; exact_mod_cast (mul_left_cancel₀ two_ne_zero h :)]
  · refine filter_congr fun x _ => ?_
    rw [cRM, ← two_eRP]
    constructor
    · intro h; rw [← h]; push_cast; ring
    · intro h
      have : (2 : ℤ) * ((eRP a x + (2 * a + 2 * x.2 + 1) : ℕ) : ℤ) = 2 * k := by push_cast; linarith
      exact_mod_cast (mul_left_cancel₀ two_ne_zero this :)

end RSide

/-- **the reduced (2.19), truncated**: `Σ_b q^{(a−b)²+|b|(|b|+1)} PT_{|b|}(q²) ≡ ψ(q)·G_a(q) (mod q^{N+1})`. -/
theorem lattice_trunc (a N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣ Lsum a N - psiSum * Gsum a N := by
  rw [X_pow_dvd_iff]
  intro i hi
  rw [map_sub, sub_eq_zero, coeff_Lsum a N i (by omega), coeff_Rsum a N i (by omega)]
  have := param_counts a i N (by omega)
  simp only [box] at *
  omega

end RankProof
