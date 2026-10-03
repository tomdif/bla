/-
# Ramanujan's 5-dissection of the rank generating function: the components `R₁, R₂, R₄`

With `R(ζ;q) = Σ_k q^k R_k(q⁵)` at `ζ = ζ₅` (Garvan (4.2), part of the Lost Notebook identity (4.1)):
`R₁ = J₅²/J_{5,1}`, `R₂ = (ζ + ζ⁴) J₅²/J_{5,2}`, `R₄ = 0`.
Read off from the linear system of `RankMod5`, using the product identities `prod_id1`, `prod_id2`.
-/
import RamanujanTau.RankMod5

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5 eQ alphaQ betaQ dis5 ω5 ω5_prim ω5_pow5)
local notation "ψ" => MockTheta5.JTP.ψC

lemma dis5C_two_E2C (f : PowerSeries ℂ) : dis5C 2 (E2C f) = E2C (dis5C 1 f) := by
  ext n
  rw [coeff_dis5C, coeff_E2C, coeff_E2C]
  by_cases h : 2 ∣ n
  · rw [if_pos (by omega), if_pos h, coeff_dis5C, show (5 * n + 2) / 2 = 5 * (n / 2) + 1 by omega]
  · rw [if_neg (by omega), if_neg h]

lemma dis5C_three_E2C (f : PowerSeries ℂ) : dis5C 3 (E2C f) = X * E2C (dis5C 4 f) := by
  ext n
  rw [coeff_dis5C, coeff_E2C]
  rcases n with _ | m
  · rw [if_neg (by omega), coeff_zero_eq_constantCoeff, map_mul, constantCoeff_X, zero_mul]
  · rw [coeff_succ_X_mul, coeff_E2C]
    by_cases h : 2 ∣ m
    · rw [if_pos (by omega), if_pos h, coeff_dis5C, show (5 * (m + 1) + 3) / 2 = 5 * (m / 2) + 4 by omega]
    · rw [if_neg (by omega), if_neg h]

lemma E2C_injective : Function.Injective E2C := by
  intro f g h
  ext n
  have := congrArg (coeff (2 * n)) h
  rwa [coeff_E2C, coeff_E2C, if_pos (dvd_mul_right 2 n), if_pos (dvd_mul_right 2 n),
    Nat.mul_div_cancel_left _ (by norm_num)] at this

lemma sζ_ne : sζ ≠ 0 := by
  intro h
  have h3 : ω5 ^ 3 = -ω5 ^ 2 := by rw [sζ] at h; linear_combination h
  have h1 : ω5 = -1 := by
    have := mul_left_cancel₀ (pow_ne_zero 2 ω5_ne) (show ω5 ^ 2 * ω5 = ω5 ^ 2 * (-1) by linear_combination h3)
    exact this
  have : ω5 ^ 2 = 1 := by rw [h1]; norm_num
  exact ω5_prim.pow_ne_one_of_pos_of_lt (by norm_num) (by norm_num) this

/-- the components `R_k` of `R(ζ;q) = Σ q^k R_k(q⁵)`. -/
noncomputable def Rk (k : ℕ) : PowerSeries ℂ := dis5C k (Dser ω5 ω5⁻¹)

theorem eqs23 :
    ψ (Jab 5 2) * dis5C 3 (E2C (Dser ω5 ω5⁻¹)) + C sζ * ψ (Jab 5 1) * dis5C 2 (E2C (Dser ω5 ω5⁻¹))
        = C sζ * ψ (eQ * Jab 10 3 * betaQ)
      ∧ ψ (Jab 5 2) * dis5C 4 (E2C (Dser ω5 ω5⁻¹)) + C sζ * ψ (Jab 5 1) * dis5C 3 (E2C (Dser ω5 ω5⁻¹))
        = C tζ * ψ (eQ * Jab 10 1 * alphaQ) := by
  set S := E2C (Dser ω5 ω5⁻¹)
  set B := ψ (Jab 5 1)
  have hF : Fz ω5 = E5C (ψ (Jab 5 2)) + X * C sζ * E5C B := by
    rw [Fz, F_dissect ω5_pow5 (ω5_prim.ne_one (by norm_num))]; rfl
  have hshift : ∀ r, 1 ≤ r → r < 5 → ∀ G : PowerSeries ℂ,
      dis5C r (X * C sζ * E5C B * G) = C sζ * B * dis5C (r - 1) G := by
    intro r h1 h5 G
    rw [show X * C sζ * E5C B * G = E5C (C sζ * B) * (X ^ 1 * G) by rw [map_mul, E5C_C]; ring,
      dis5C_E5C_mul r h5, dis5C_X_pow_mul h1]
  constructor
  · rw [← eq_class3, hF, add_mul, dis5C_add, dis5C_E5C_mul 3 (by norm_num), hshift 3 (by norm_num) (by norm_num)]
  · rw [← eq_class4, hF, add_mul, dis5C_add, dis5C_E5C_mul 4 (by norm_num), hshift 4 (by norm_num) (by norm_num)]

/-- **`R₄ = 0`** (Dyson's mod-5 conjecture in generating-function form). -/
theorem rank_R4 : Rk 4 = 0 := by
  have h := S3_zero
  rw [dis5C_three_E2C] at h
  apply E2C_injective
  rw [map_zero]
  exact (mul_eq_zero.mp h).resolve_left X_ne_zero

/-- **`R₁ = J₅²/J_{5,1}`.** -/
theorem rank_R1 : Rk 1 = ψ (eQ ^ 2 * Ring.inverse (Jab 5 1)) := by
  have hu1 := isUnit_Jab 5 1 (by norm_num) (by norm_num)
  have h := eqs23.1
  rw [S3_zero, mul_zero, zero_add, dis5C_two_E2C, mul_assoc] at h
  have h' := mul_left_cancel₀ ((isUnit_iff_ne_zero.mpr sζ_ne).map C).ne_zero h
  apply E2C_injective
  apply (hu1.map ψ).mul_left_cancel
  rw [Rk, h', ← ψ_E2, ← map_mul]
  congr 1
  have hE2u : IsUnit (E2 (Jab 5 1)) := hu1.map E2
  have p2 := prod_id2
  have hb := betaQ_eq
  rw [map_mul, map_pow, map_inverse E2 hu1]
  apply hE2u.mul_left_cancel
  have hc := Ring.mul_inverse_cancel _ hE2u
  linear_combination p2 * betaQ - E2 eQ ^ 2 * Jab 5 1 * hc - E2 eQ ^ 2 * hb

/-- **`R₂ = (ζ + ζ⁴) J₅²/J_{5,2}`.** -/
theorem rank_R2 : Rk 2 = C tζ * ψ (eQ ^ 2 * Ring.inverse (Jab 5 2)) := by
  have hu2 := isUnit_Jab 5 2 (by norm_num) (by norm_num)
  have h := eqs23.2
  rw [S3_zero, mul_zero, add_zero, dis5C_four_E2C] at h
  apply E2C_injective
  apply (hu2.map ψ).mul_left_cancel
  rw [Rk, h, RingHom.map_mul E2C, E2C_C, ← ψ_E2, mul_left_comm, ← map_mul]
  congr 2
  have hE2u : IsUnit (E2 (Jab 5 2)) := hu2.map E2
  have p1 := prod_id1
  have ha := J5_relations.2
  rw [map_mul, map_pow, map_inverse E2 hu2]
  apply hE2u.mul_left_cancel
  have hc := Ring.mul_inverse_cancel _ hE2u
  linear_combination p1 * alphaQ - E2 eQ ^ 2 * Jab 5 2 * hc - E2 eQ ^ 2 * ha

end CrankProof
