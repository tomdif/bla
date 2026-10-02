/-
# Dyson's rank, step 1: the Bailey pair behind Garvan's Hecke–Rogers identity

Finite identity (all `k, n`), in any field with `1 − xⁱ ≠ 0`:
  `Σ_{l≤n} xˡ/((x)_l (x)_{l+k}) = Σ_{r≤n} α*_r/((x)_{n−r} (x)_{n+r+k})`,
  `α*_0 = 1`, `α*_r = x^{2r²+2kr+r} − x^{2r²+2kr−r−k}`,
proved by induction on `n` with the telescoping certificate
  `g(r) = −x^{n+1−r}·B_r/((x)_{n+1−r}(x)_{n+r+k})`, `B_r = x^{2r²+2kr−r−k}`.
-/
import Mathlib
import RamanujanTau.MockTheta5DurfeeBase

set_option autoImplicit false

namespace RankProof
open Finset

section Finite
variable {K : Type*} [Field K] (x : K)

/-- `(x;x)_m = ∏_{i<m} (1 − x^{i+1})`. -/
def P (m : ℕ) : K := ∏ i ∈ range m, (1 - x ^ (i + 1))

lemma P_succ (m : ℕ) : P x (m + 1) = P x m * (1 - x ^ (m + 1)) := by
  rw [P, prod_range_succ]; rfl

variable {x}
variable (hx : ∀ i : ℕ, 1 - x ^ (i + 1) ≠ 0)
include hx

lemma P_ne (m : ℕ) : P x m ≠ 0 := prod_ne_zero_iff.mpr fun i _ => hx i

omit hx in
/-- `A_r = x^{2r²+2kr+r}`, `B_{s+1} = x^{2s²+3s+1+2ks+k}` (= `x^{2r²+2kr−r−k}` at `r = s+1`). -/
def A (k r : ℕ) : K := x ^ (2 * r ^ 2 + 2 * k * r + r)
omit hx in
def B (k : ℕ) : ℕ → K
  | 0 => 0
  | s + 1 => x ^ (2 * s ^ 2 + 3 * s + 1 + 2 * k * s + k)

omit hx in
/-- `α*_r = A_r − B_r` (with `B_0 = 0`). -/
def alphaS (k r : ℕ) : K := A (x := x) k r - B (x := x) k r

omit hx in
/-- the Bailey relation's right side `Σ_{r≤n} α*_r/((x)_{n−r}(x)_{n+r+k})`. -/
def SR (k n : ℕ) : K := ∑ r ∈ range (n + 1), alphaS (x := x) k r / (P x (n - r) * P x (n + r + k))

omit hx in
/-- the partial sums `Σ_{l≤n} xˡ/((x)_l(x)_{l+k})`. -/
def SL (k n : ℕ) : K := ∑ l ∈ range (n + 1), x ^ l / (P x l * P x (l + k))

omit hx in
/-- the telescoping certificate at step `n → n+1`. -/
def g (k n r : ℕ) : K := -(x ^ (n + 1 - r) * B (x := x) k r) / (P x (n + 1 - r) * P x (n + r + k))

end Finite

section Local
variable {K : Type*} [Field K] {x : K}

/-- interior step (`r = s+1`, `n = d+r`). -/
lemma local_mid (k d s : ℕ) (p p' : K) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h1 : 1 - x ^ (d + 1) ≠ 0) (h2 : 1 - x ^ (d + 2 * s + 3 + k) ≠ 0) :
    (x ^ (2 * (s + 1) ^ 2 + 2 * k * (s + 1) + (s + 1)) - x ^ (2 * s ^ 2 + 3 * s + 1 + 2 * k * s + k))
        / (p * (1 - x ^ (d + 1)) * (p' * (1 - x ^ (d + 2 * s + 3 + k))))
      - (x ^ (2 * (s + 1) ^ 2 + 2 * k * (s + 1) + (s + 1)) - x ^ (2 * s ^ 2 + 3 * s + 1 + 2 * k * s + k))
        / (p * p')
      = -(x ^ (d + 1) * x ^ (2 * s ^ 2 + 3 * s + 1 + 2 * k * s + k)) / (p * (1 - x ^ (d + 1)) * p')
        - -(x ^ d * x ^ (2 * (s + 1) ^ 2 + 3 * (s + 1) + 1 + 2 * k * (s + 1) + k))
          / (p * (p' * (1 - x ^ (d + 2 * s + 3 + k)))) := by
  field_simp
  ring

/-- bottom step (`r = 0`). -/
lemma local_bot (k n : ℕ) (p p' : K) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h1 : 1 - x ^ (n + 1) ≠ 0) (h2 : 1 - x ^ (n + 1 + k) ≠ 0) :
    1 / (p * (1 - x ^ (n + 1)) * (p' * (1 - x ^ (n + 1 + k)))) - 1 / (p * p')
      = x ^ (n + 1) / (p * (1 - x ^ (n + 1)) * (p' * (1 - x ^ (n + 1 + k))))
        - -(x ^ n * x ^ (1 + k)) / (p * (p' * (1 - x ^ (n + 1 + k)))) := by
  field_simp
  ring

/-- top step (`r = n+1`). -/
lemma local_top (k n : ℕ) (p : K) (hp : p ≠ 0) (h1 : 1 - x ^ (2 * n + 2 + k) ≠ 0) :
    (x ^ (2 * (n + 1) ^ 2 + 2 * k * (n + 1) + (n + 1)) - x ^ (2 * n ^ 2 + 3 * n + 1 + 2 * k * n + k))
        / (1 * (p * (1 - x ^ (2 * n + 2 + k))))
      = -(x ^ 0 * x ^ (2 * n ^ 2 + 3 * n + 1 + 2 * k * n + k)) / (1 * p) := by
  field_simp
  ring

end Local

section Assembly
variable {K : Type*} [Field K] {x : K} (hx : ∀ i : ℕ, 1 - x ^ (i + 1) ≠ 0)
include hx

lemma hx' {m : ℕ} (hm : 0 < m) : 1 - x ^ m ≠ 0 := by
  obtain ⟨i, rfl⟩ : ∃ i, m = i + 1 := ⟨m - 1, by omega⟩
  exact hx i

/-- one step of the Bailey relation: `SR(n+1) − SR(n) = x^{n+1}/((x)_{n+1}(x)_{n+1+k})`. -/
theorem SR_succ (k n : ℕ) :
    SR (x := x) k (n + 1) = SR (x := x) k n + x ^ (n + 1) / (P x (n + 1) * P x (n + 1 + k)) := by
  -- the interior differences telescope
  have hmid : ∀ i ∈ range n,
      alphaS (x := x) k (i + 1) / (P x (n + 1 - (i + 1)) * P x (n + 1 + (i + 1) + k))
        - alphaS (x := x) k (i + 1) / (P x (n - (i + 1)) * P x (n + (i + 1) + k))
      = g (x := x) k n (i + 1) - g (x := x) k n (i + 2) := by
    intro i hi
    have hi' := mem_range.mp hi
    set d := n - 1 - i with hd
    rw [show n + 1 - (i + 1) = d + 1 by omega, show n + 1 + (i + 1) + k = d + 2 * i + 2 + k + 1 by omega,
      show n - (i + 1) = d by omega, show n + (i + 1) + k = d + 2 * i + 2 + k by omega, P_succ, P_succ]
    unfold g
    rw [show n + 1 - (i + 1) = d + 1 by omega, show n + (i + 1) + k = d + 2 * i + 2 + k by omega,
      show n + 1 - (i + 2) = d by omega, show n + (i + 2) + k = d + 2 * i + 2 + k + 1 by omega, P_succ, P_succ]
    unfold alphaS A B
    rw [show d + 2 * i + 2 + k + 1 = d + 2 * i + 3 + k by omega]
    exact local_mid k d i _ _ (P_ne hx _) (P_ne hx _) (hx d) (hx' hx (by omega))
  have hbot : alphaS (x := x) k 0 / (P x (n + 1 - 0) * P x (n + 1 + 0 + k))
      - alphaS (x := x) k 0 / (P x (n - 0) * P x (n + 0 + k))
      = x ^ (n + 1) / (P x (n + 1) * P x (n + 1 + k)) - g (x := x) k n 1 := by
    have hb0 : alphaS (x := x) k 0 = 1 := by simp [alphaS, A, B]
    have hb1 : B (x := x) k 1 = x ^ (1 + k) := by
      show x ^ (2 * 0 ^ 2 + 3 * 0 + 1 + 2 * k * 0 + k) = x ^ (1 + k); ring_nf
    unfold g
    rw [hb0, hb1, show n + 1 - 0 = n + 1 by omega, show n + 1 + 0 + k = n + k + 1 by omega,
      show n - 0 = n by omega, show n + 0 + k = n + k by omega, show n + 1 - 1 = n by omega,
      P_succ x n, P_succ x (n + k), show n + k + 1 = n + 1 + k by omega]
    exact local_bot k n (P x n) (P x (n + k)) (P_ne hx _) (P_ne hx _) (hx n) (hx' hx (by omega))
  have htop : alphaS (x := x) k (n + 1) / (P x (n + 1 - (n + 1)) * P x (n + 1 + (n + 1) + k))
      = g (x := x) k n (n + 1) := by
    unfold g alphaS A B
    rw [show n + 1 - (n + 1) = 0 by omega, show n + 1 + (n + 1) + k = 2 * n + 1 + k + 1 by omega,
      show n + (n + 1) + k = 2 * n + 1 + k by omega, P_succ]
    have := local_top k n (P x (2 * n + 1 + k)) (P_ne hx _) (hx' hx (by omega))
    rw [show P x 0 = 1 by simp [P], show 2 * n + 1 + k + 1 = 2 * n + 2 + k by omega]
    exact this
  have hD : ∑ r ∈ range (n + 1), (alphaS (x := x) k r / (P x (n + 1 - r) * P x (n + 1 + r + k))
      - alphaS (x := x) k r / (P x (n - r) * P x (n + r + k)))
      = x ^ (n + 1) / (P x (n + 1) * P x (n + 1 + k)) - g (x := x) k n (n + 1) := by
    rw [sum_range_succ', sum_congr rfl hmid, sum_range_sub' (fun i => g (x := x) k n (i + 1)) n, hbot]
    ring
  unfold SR
  rw [sum_range_succ, htop]
  rw [sum_sub_distrib] at hD
  linear_combination hD

theorem SL_eq_SR (k n : ℕ) : SL (x := x) k n = SR (x := x) k n := by
  induction n with
  | zero => simp [SL, SR, alphaS, A, B, P]
  | succ n ih => rw [SR_succ hx, ← ih, SL, sum_range_succ, ← SL, show n + 1 + k = n + 1 + k from rfl]

end Assembly

/-! ## Transfer to `ℤ⟦X⟧` and the `a = qᵏ` Bailey transform -/

section Series
open PowerSeries MockTheta5.Bailey MockTheta5.JTP

abbrev K0 := FractionRing (PowerSeries ℤ)
noncomputable abbrev φ : PowerSeries ℤ →+* K0 := algebraMap (PowerSeries ℤ) K0

lemma φ_inj : Function.Injective φ := IsFractionRing.injective _ _

lemma φ_inverse {u : PowerSeries ℤ} (hu : IsUnit u) : φ (Ring.inverse u) = (φ u)⁻¹ :=
  eq_inv_of_mul_eq_one_left (by rw [← map_mul, Ring.inverse_mul_cancel _ hu, map_one])

lemma φ_qfac (m : ℕ) : φ (qfac m) = P (φ X) m := by
  rw [qfac, map_prod, P]
  exact prod_congr rfl fun i _ => by rw [map_sub, map_one, map_pow]

lemma hxφ : ∀ i : ℕ, 1 - (φ X) ^ (i + 1) ≠ 0 := by
  intro i h
  have h1 : φ (1 - X ^ (i + 1)) = φ 0 := by rw [map_sub, map_one, map_pow, map_zero, h]
  have h2 := congrArg (constantCoeff) (φ_inj h1)
  simp at h2

noncomputable def Bz (k : ℕ) : ℕ → PowerSeries ℤ
  | 0 => 0
  | s + 1 => X ^ (2 * s ^ 2 + 3 * s + 1 + 2 * k * s + k)

noncomputable def αz (k r : ℕ) : PowerSeries ℤ := X ^ (2 * r ^ 2 + 2 * k * r + r) - Bz k r

noncomputable def SLz (k n : ℕ) : PowerSeries ℤ :=
  ∑ l ∈ range (n + 1), X ^ l * Ring.inverse (qfac l) * Ring.inverse (qfac (l + k))

noncomputable def SRz (k n : ℕ) : PowerSeries ℤ :=
  ∑ r ∈ range (n + 1), αz k r * Ring.inverse (qfac (n - r)) * Ring.inverse (qfac (n + r + k))

lemma φ_αz (k r : ℕ) : φ (αz k r) = alphaS (x := φ X) k r := by
  cases r <;> simp [αz, alphaS, A, B, Bz, map_sub, map_pow]

/-- **the finite Bailey pair** `(α*, β*)` relative to `a = qᵏ`, in `ℤ⟦X⟧`. -/
theorem SLz_eq_SRz (k n : ℕ) : SLz k n = SRz k n := by
  apply φ_inj
  have h := SL_eq_SR hxφ k n
  unfold SL SR at h
  rw [SLz, SRz, map_sum, map_sum]
  convert h using 2 with l _ r _
  · rw [map_mul, map_mul, map_pow, φ_inverse (isUnit_qfac _), φ_inverse (isUnit_qfac _), φ_qfac, φ_qfac,
      div_eq_mul_inv, mul_inv, mul_assoc]
  · rw [map_mul, map_mul, φ_αz, φ_inverse (isUnit_qfac _), φ_inverse (isUnit_qfac _), φ_qfac, φ_qfac,
      div_eq_mul_inv, mul_inv, mul_assoc]

end Series

section Transform
open PowerSeries MockTheta5.Bailey MockTheta5.JTP

/-- `Σ_j q^{j²+kj} β*_j` (as a coefficientwise limit). -/
noncomputable def LHSz (k : ℕ) : PowerSeries ℤ :=
  mk fun c => coeff c (∑ j ∈ range (c + 1), X ^ (j ^ 2 + k * j) * SLz k j)

/-- `Σ_r q^{r²+kr} α*_r`. -/
noncomputable def RHSz (k : ℕ) : PowerSeries ℤ :=
  mk fun c => coeff c (∑ r ∈ range (c + 1), X ^ (r ^ 2 + k * r) * αz k r)

lemma coeff_Xpow_mul_eq_zero {d c : ℕ} (h : c < d) (f : PowerSeries ℤ) : coeff c (X ^ d * f) = 0 := by
  rw [coeff_X_pow_mul', if_neg (by omega)]

lemma RHSz_dvd (k c : ℕ) :
    (X : PowerSeries ℤ) ^ (c + 1) ∣ RHSz k - ∑ r ∈ range (c + 1), X ^ (r ^ 2 + k * r) * αz k r := by
  rw [X_pow_dvd_iff]; intro i hi
  rw [map_sub, RHSz, coeff_mk, sub_eq_zero, map_sum, map_sum]
  refine sum_subset (range_subset_range.mpr (by omega)) fun r _ hr => ?_
  simp only [mem_range, not_lt] at hr
  exact coeff_Xpow_mul_eq_zero (by nlinarith) _

lemma coeff_mul_congr_right' {c : ℕ} {f g g' : PowerSeries ℤ}
    (h : (X : PowerSeries ℤ) ^ (c + 1) ∣ (g - g')) : coeff c (f * g) = coeff c (f * g') := by
  rw [mul_comm f g, mul_comm f g']; exact coeff_mul_congr_left h

lemma rect_dvd (n c : ℕ) : (X : PowerSeries ℤ) ^ (c + 1) ∣ rectInf n - rectPartial n (c + 1) := by
  rw [X_pow_dvd_iff]; intro i hi
  rw [map_sub, coeff_rectInf n (show i + 1 ≤ c + 1 by omega), sub_self]

/-- **the Bailey transform for `a = qᵏ`** applied to `(α*, β*)`:
`Σ_j q^{j²+kj} β*_j = (1/(q;q)_∞)·Σ_r q^{r²+kr} α*_r` (with `β*` scaled by `(q;q)_k`). -/
noncomputable def Fterm (k c r m : ℕ) : ℤ := coeff c (X ^ (r ^ 2 + k * r) * αz k r * rectTerm (2 * r + k) m)

theorem bailey_k (k : ℕ) : LHSz k = Ring.inverse qfacInf * RHSz k := by
  ext c
  let f := Fterm k c
  have hR : coeff c (Ring.inverse qfacInf * RHSz k) = ∑ r ∈ range (c + 1), ∑ m ∈ range (c + 1), f r m := by
    rw [coeff_mul_congr_right' (RHSz_dvd k c), mul_sum, map_sum]
    refine sum_congr rfl fun r _ => ?_
    rw [← durfee_rect_base (2 * r + k), mul_comm, coeff_mul_congr_right' (rect_dvd _ c), rectPartial,
      mul_sum, map_sum]
    exact sum_congr rfl fun m _ => by show _ = Fterm k c r m; rw [Fterm]
  have hL : coeff c (LHSz k) = ∑ j ∈ range (c + 1), ∑ r ∈ range (j + 1), f r (j - r) := by
    rw [LHSz, coeff_mk, map_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [SLz_eq_SRz, SRz, mul_sum, map_sum]
    refine sum_congr rfl fun r hr => ?_
    obtain ⟨m, rfl⟩ : ∃ m, j = r + m := ⟨j - r, by have := mem_range.mp hr; omega⟩
    show _ = Fterm k c r (r + m - r)
    rw [Fterm, show r + m - r = m by omega, rectTerm]
    congr 1
    rw [show r + m + r + k = 2 * r + k + m by ring]
    ring
  rw [hL, hR, sum_range_diag_flip]
  refine sum_congr rfl fun r hr => ?_
  refine sum_subset (range_subset_range.mpr (by omega)) fun m _ hm => ?_
  simp only [mem_range, not_lt] at hm
  show Fterm k c r m = 0
  rw [Fterm, rectTerm, show X ^ (r ^ 2 + k * r) * αz k r
      * (X ^ (m ^ 2 + (2 * r + k) * m) * Ring.inverse (qfac m) * Ring.inverse (qfac (2 * r + k + m)))
      = X ^ (r ^ 2 + k * r + (m ^ 2 + (2 * r + k) * m))
        * (αz k r * Ring.inverse (qfac m) * Ring.inverse (qfac (2 * r + k + m))) by ring]
  have hr' := mem_range.mp hr
  have h1 : r ≤ r ^ 2 := Nat.le_self_pow two_ne_zero r
  have h2 : m ≤ m ^ 2 := Nat.le_self_pow two_ne_zero m
  have h3 : c + 1 ≤ r + m := by omega
  exact coeff_Xpow_mul_eq_zero (by nlinarith [Nat.zero_le (k * r), Nat.zero_le ((2 * r + k) * m)]) _

end Transform
end RankProof
