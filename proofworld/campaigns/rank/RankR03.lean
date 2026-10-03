/-
# Ramanujan's Lost Notebook identity for `R(ζ₅;q)`: the components `R₀` and `R₃`

`R₀ = J₅²J_{5,2}/J_{5,1}² + (ζ+ζ⁴−2)·φ`,  `q·R₃ = (ζ²+ζ³−ζ−ζ⁴)·ψ + (ζ²+ζ³+1)·q·J₅²J_{5,1}/J_{5,2}²`,
with Ramanujan's `φ = Φ₁ − 1`, `ψ = Φ₂ − 1` (`RankSpec.Phi`). Together with `RankRamanujan5` this is the
full 5-dissection of `R(ζ;q)` from p. 20 of the Lost Notebook (Garvan, (4.1)–(4.4)).
-/
import RamanujanTau.RankP2

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5 eQ alphaQ betaQ dis5 ω5 ω5_prim ω5_pow5)
open RankProof (Gorb Phi eP eM coeff_Hblk_box eP_two eM_two orbit_mock)
local notation "ψ" => MockTheta5.JTP.ψC

section Vanish0

lemma hb_zero_of_class {a r m : ℕ} (ha : a % 5 = 0) (hm : m % 5 = 1 ∨ m % 5 = 3) : RankProof.hb a r m = 0 := by
  have hax : (a : ZMod 5) = 0 := by
    rw [ZMod.natCast_eq_zero_iff]; omega
  have hmz : (m : ZMod 5) = 1 ∨ (m : ZMod 5) = 3 := by
    rcases hm with h | h
    · left; rw [← Nat.mod_add_div m 5, h]; push_cast; rw [show (5 : ZMod 5) = 0 from by decide]; ring
    · right; rw [← Nat.mod_add_div m 5, h]; push_cast; rw [show (5 : ZMod 5) = 0 from by decide]; ring
  unfold RankProof.hb
  rw [if_neg, if_neg]
  · ring
  · rintro ⟨hr, he⟩
    have h2 := congrArg (Int.cast : ℤ → ZMod 5) (eM_two a r hr)
    rw [he] at h2
    push_cast [RankProof.eM_le a r hr] at h2
    rw [hax] at h2
    generalize (r : ZMod 5) = x at h2
    rcases hmz with h | h <;> rw [h] at h2 <;> revert x <;> decide
  · intro he
    have h2 := congrArg (Int.cast : ℤ → ZMod 5) (eP_two a r)
    rw [he] at h2
    push_cast at h2
    rw [hax] at h2
    generalize (r : ZMod 5) = x at h2
    rcases hmz with h | h <;> rw [h] at h2 <;> revert x <;> decide

end Vanish0

section Orbits

lemma coeff_Hblk_zero0 {a m : ℕ} (ha : a % 5 = 0) (hm : m % 5 = 1 ∨ m % 5 = 3) : coeff m (Hblk a) = 0 := by
  rw [coeff_Hblk_box a m (m + 1) (by omega)]
  exact sum_eq_zero fun r _ => hb_zero_of_class ha hm

/-- classes 1 and 3 of `F(z)R(z;q)`, split by the orbits of `a mod 5`. -/
theorem class_orbits {z : ℂ} (hz : z ≠ 0) (w1 w2 : ℂ) (h1 : ∀ a, (a % 5 = 1 ∨ a % 5 = 4) → wz z a = w1)
    (h2 : ∀ a, (a % 5 = 2 ∨ a % 5 = 3) → wz z a = w2) {c : ℕ} (hc : c = 1 ∨ c = 3) :
    dis5C c (Fz z * Dser z z⁻¹) = C w1 * ψ (dis5 c (Gorb 2)) + C w2 * ψ (dis5 c (Gorb 1)) := by
  ext n
  rw [coeff_dis5C, hr18_E hz, map_add, coeff_C_mul, coeff_C_mul, PowerSeries.coeff_map, PowerSeries.coeff_map,
    MockTheta5.JTP.coeff_dis5, MockTheta5.JTP.coeff_dis5, Gorb, Gorb, coeff_mk, coeff_mk]
  simp only [map_sum, mul_sum, ← sum_add_distrib]
  refine sum_congr rfl fun a _ => ?_
  have hm : (5 * n + c) % 5 = 1 ∨ (5 * n + c) % 5 = 3 := by omega
  rw [PowerSeries.coeff_map]
  have hr := Nat.mod_lt a (show 0 < 5 by norm_num)
  by_cases ha0 : a % 5 = 0
  · rw [coeff_Hblk_zero0 ha0 hm]
    simp [show ¬ (a % 5 = 5 - 2 * 2 ∨ a % 5 = 2 * 2) by omega, show ¬ (a % 5 = 5 - 2 * 1 ∨ a % 5 = 2 * 1) by omega]
  by_cases ha1 : a % 5 = 1 ∨ a % 5 = 4
  · rw [h1 a ha1, if_pos (by omega), if_neg (by omega)]; simp
  · have ha2 : a % 5 = 2 ∨ a % 5 = 3 := by omega
    rw [h2 a ha2, if_neg (by omega), if_pos (by omega)]; simp

lemma wz_ω5_14 (a : ℕ) (h : a % 5 = 1 ∨ a % 5 = 4) : wz ω5 a = tζ := by
  rcases h with h | h
  · rw [wz_ω5_of_mod 1 h (by norm_num)]; simp [tζ]
  · rw [wz_ω5_of_mod 4 h (by norm_num)]; simp [tζ]; ring

lemma wz_ω5_23 (a : ℕ) (h : a % 5 = 2 ∨ a % 5 = 3) : wz ω5 a = sζ := by
  rcases h with h | h
  · rw [wz_ω5_of_mod 2 h (by norm_num)]; simp [sζ]
  · rw [wz_ω5_of_mod 3 h (by norm_num)]; simp [sζ]; ring

lemma wz_one_ne (a : ℕ) (h : a % 5 ≠ 0) : wz 1 a = 2 := wz_one (by rintro rfl; simp at h)

end Orbits

section Final

lemma dis5_sq_classes : dis5 1 (qfacInf ^ 2) = -(2 * eQ ^ 2 * alphaQ) ∧ dis5 3 (qfacInf ^ 2) = 2 * eQ ^ 2 * betaQ := by
  have hn : qfacInf ^ 2 = E5 (eQ ^ 2 * alphaQ ^ 2) + X * E5 (-(2 * eQ ^ 2 * alphaQ))
      + X ^ 2 * E5 (eQ ^ 2 * (1 - 2 * (alphaQ * betaQ))) + X ^ 3 * E5 (2 * eQ ^ 2 * betaQ)
      + X ^ 4 * E5 (eQ ^ 2 * betaQ ^ 2) := by
    rw [MockTheta5.JTP.qfacInf_dissection]
    simp only [map_add, map_sub, map_mul, map_neg, map_pow, map_one, map_ofNat]
    ring
  rw [hn, MockTheta5.JTP.dis5_normal 1 (by norm_num), MockTheta5.JTP.dis5_normal 3 (by norm_num)]
  simp

lemma Fz_dissect : Fz ω5 = E5C (ψ (Jab 5 2)) + X * C sζ * E5C (ψ (Jab 5 1)) := by
  rw [Fz, F_dissect ω5_pow5 (ω5_prim.ne_one (by norm_num))]; rfl

lemma FD_class {r : ℕ} (h1 : 1 ≤ r) (h5 : r < 5) :
    dis5C r (Fz ω5 * Dser ω5 ω5⁻¹) = ψ (Jab 5 2) * Rk r + C sζ * ψ (Jab 5 1) * Rk (r - 1) := by
  rw [Fz_dissect, add_mul, dis5C_add, dis5C_E5C_mul r h5,
    show X * C sζ * E5C (ψ (Jab 5 1)) * Dser ω5 ω5⁻¹ = E5C (C sζ * ψ (Jab 5 1)) * (X ^ 1 * Dser ω5 ω5⁻¹) by
      rw [map_mul, E5C_C]; ring,
    dis5C_E5C_mul r h5, dis5C_X_pow_mul h1, Rk, Rk]

lemma class_one (c : ℕ) (hc : c = 1 ∨ c = 3) :
    dis5C c (Fz 1 * Dser 1 1⁻¹) = C 2 * ψ (dis5 c (Gorb 2)) + C 2 * ψ (dis5 c (Gorb 1)) :=
  class_orbits one_ne_zero 2 2 (fun a h => wz_one_ne a (by omega)) (fun a h => wz_one_ne a (by omega)) hc

/-- **`R₀ = J₅²J_{5,2}/J_{5,1}² + (ζ+ζ⁴−2)φ`** (Ramanujan, Lost Notebook p. 20; Garvan (4.3)). -/
theorem rank_R0 : Rk 0 = ψ (eQ ^ 2 * Jab 5 2 * Ring.inverse (Jab 5 1) ^ 2) + C (tζ - 2) * ψ (Phi 1 - 1) := by
  set A := ψ (Jab 5 2)
  set B := ψ (Jab 5 1)
  set G1 := ψ (dis5 1 (Gorb 1))
  set G2 := ψ (dis5 1 (Gorb 2))
  have hu1 := isUnit_Jab 5 1 (by norm_num) (by norm_num)
  have e1 : A * Rk 1 + C sζ * B * Rk 0 = C tζ * G2 + C sζ * G1 := by
    rw [← class_orbits ω5_ne tζ sζ wz_ω5_14 wz_ω5_23 (Or.inl rfl), FD_class (r := 1) le_rfl (by norm_num)]
  have e2 : ψ (-(2 * eQ ^ 2 * alphaQ)) = C 2 * G2 + C 2 * G1 := by
    rw [← class_one 1 (Or.inl rfl), FD_one, dis5C_ψ, dis5_sq_classes.1]
  have e3 : G1 = B * (1 - ψ (Phi 1)) := by
    have := congrArg ψ (orbit_mock (k := 1) (Or.inl rfl))
    simp only [show (2 * 1 - 1 : ℕ) = 1 from rfl, show (1 - 1 : ℕ) = 0 from rfl, pow_zero, map_one, one_mul,
      map_mul, map_sub] at this
    exact this
  have e4 := rank_R1
  have e5 : A = ψ alphaQ * B := by rw [← map_mul]; exact congrArg ψ J5_relations.2
  have e6 : B * ψ (Ring.inverse (Jab 5 1)) = 1 := by rw [← map_mul, Ring.mul_inverse_cancel _ hu1, map_one]
  have hst : C sζ + C tζ = (-1 : PowerSeries ℂ) := by rw [← map_add, s_add_t, map_neg, map_one]
  have h2 : (C 2 : PowerSeries ℂ) = 2 := by rw [map_ofNat]
  have hB : C sζ * B ≠ 0 :=
    mul_ne_zero ((isUnit_iff_ne_zero.mpr sζ_ne).map C).ne_zero ((hu1.map ψ).ne_zero)
  apply mul_left_cancel₀ hB
  simp only [map_mul, map_pow, map_neg, map_sub, map_one, map_ofNat] at e2 e4 ⊢
  have h20 : (2 : PowerSeries ℂ) ≠ 0 := by
    rw [← h2]; intro h; exact two_ne_zero (PowerSeries.C_injective (h.trans (map_zero C).symm))
  have hG2 : G2 = -(ψ eQ ^ 2 * ψ alphaQ) - G1 := by
    have : (2 : PowerSeries ℂ) * (G2 + ψ eQ ^ 2 * ψ alphaQ + G1) = 0 := by linear_combination -e2
    have h := (mul_eq_zero.mp this).resolve_left h20
    linear_combination h
  have hst2 : C sζ * C tζ = (-1 : PowerSeries ℂ) := by
    rw [← map_mul, ← map_one C, ← map_neg]; congr 1
    have := MockTheta5.JTP.phi_of_prim ω5_prim
    rw [sζ, tζ]; linear_combination (ω5 + ω5 ^ 2) * ω5_pow5 + this
  have key : C sζ * B * Rk 0 = C tζ * G2 + C sζ * G1 - A * Rk 1 := by linear_combination e1
  rw [key, hG2, e3, e4, e5]
  have e5' : ψ (Jab 5 2) = ψ alphaQ * B := e5
  linear_combination (-(ψ alphaQ * ψ eQ ^ 2) - C sζ * ψ eQ ^ 2 * ψ alphaQ * (B * ψ (Ring.inverse (Jab 5 1)) + 1)) * e6
    - ψ eQ ^ 2 * ψ alphaQ * hst + B * (1 - ψ (Phi 1)) * hst2 + B * (ψ (Phi 1) - 1) * hst
    - ψ eQ ^ 2 * B * C sζ * ψ (Ring.inverse (Jab 5 1)) ^ 2 * e5'

/-- **`q·R₃ = (ζ²+ζ³−ζ−ζ⁴)ψ + (ζ²+ζ³+1)·q·J₅²J_{5,1}/J_{5,2}²`** (Ramanujan; Garvan (4.4)). -/
theorem rank_R3 : X * Rk 3 = C (sζ - tζ) * ψ (Phi 2 - 1)
    + C (sζ + 1) * X * ψ (eQ ^ 2 * Jab 5 1 * Ring.inverse (Jab 5 2) ^ 2) := by
  set A := ψ (Jab 5 2)
  set B := ψ (Jab 5 1)
  set H1 := ψ (dis5 3 (Gorb 1))
  set H2 := ψ (dis5 3 (Gorb 2))
  have hu2 := isUnit_Jab 5 2 (by norm_num) (by norm_num)
  have e1 : A * Rk 3 + C sζ * B * Rk 2 = C tζ * H2 + C sζ * H1 := by
    rw [← class_orbits ω5_ne tζ sζ wz_ω5_14 wz_ω5_23 (Or.inr rfl), FD_class (r := 3) (by norm_num) (by norm_num)]
  have e2 : ψ (2 * eQ ^ 2 * betaQ) = C 2 * H2 + C 2 * H1 := by
    rw [← class_one 3 (Or.inr rfl), FD_one, dis5C_ψ, dis5_sq_classes.2]
  have e3 : X * H2 = A * (1 - ψ (Phi 2)) := by
    have := congrArg ψ (orbit_mock (k := 2) (Or.inr rfl))
    simp only [show (2 * 2 - 1 : ℕ) = 3 from rfl, show (2 - 1 : ℕ) = 1 from rfl, pow_one, map_mul, map_sub,
      map_one, PowerSeries.map_X] at this
    exact this
  have e4 := rank_R2
  have e5 : B = ψ betaQ * A := by rw [← map_mul]; exact congrArg ψ betaQ_eq
  have e6 : A * ψ (Ring.inverse (Jab 5 2)) = 1 := by rw [← map_mul, Ring.mul_inverse_cancel _ hu2, map_one]
  have hst : C sζ + C tζ = (-1 : PowerSeries ℂ) := by rw [← map_add, s_add_t, map_neg, map_one]
  have hst2 : C sζ * C tζ = (-1 : PowerSeries ℂ) := by
    rw [← map_mul, ← map_one C, ← map_neg]; congr 1
    have := MockTheta5.JTP.phi_of_prim ω5_prim
    rw [sζ, tζ]; linear_combination (ω5 + ω5 ^ 2) * ω5_pow5 + this
  have h2 : (C 2 : PowerSeries ℂ) = 2 := by rw [map_ofNat]
  have h20 : (2 : PowerSeries ℂ) ≠ 0 := by
    rw [← h2]; intro h; exact two_ne_zero (PowerSeries.C_injective (h.trans (map_zero C).symm))
  simp only [map_mul, map_pow, map_ofNat, map_sub, map_one, map_add] at e2 e4 ⊢
  have hH1 : H1 = ψ eQ ^ 2 * ψ betaQ - H2 := by
    have : (2 : PowerSeries ℂ) * (H1 + H2 - ψ eQ ^ 2 * ψ betaQ) = 0 := by linear_combination -e2
    have h := (mul_eq_zero.mp this).resolve_left h20
    linear_combination h
  apply mul_left_cancel₀ (show A ≠ 0 from (hu2.map ψ).ne_zero)
  have key : A * Rk 3 = C tζ * H2 + C sζ * H1 - C sζ * B * Rk 2 := by linear_combination e1
  have e5' : ψ (Jab 5 1) = ψ betaQ * A := e5
  set I := ψ (Ring.inverse (Jab 5 2))
  set Q := ψ eQ ^ 2
  set β' := ψ betaQ
  linear_combination X * key + X * C sζ * hH1 - X * C sζ * B * e4 + (C tζ - C sζ) * e3
    + (-X * C sζ * Q * β' * (A * I + 1) - X * Q * β' * (A * I + 1) - X * C tζ * C sζ * Q * β') * e6
    + (-X * C sζ * Q * A * I ^ 2 - X * Q * A * I ^ 2 - X * C tζ * C sζ * Q * I) * e5 - X * Q * β' * hst2

end Final

end CrankProof

namespace CrankProof
open PowerSeries
open MockTheta5.JTP (eQ)
local notation "ψ" => MockTheta5.JTP.ψC

/-- **Ramanujan's 5-dissection of the rank generating function** (Lost Notebook p. 20; Garvan (4.1)–(4.4)).
With `R(ζ₅;q) = Σ_{k<5} q^k R_k(q⁵)`, `s = ζ²+ζ³`, `t = ζ+ζ⁴`, `φ = Φ₁ − 1`, `ψ = Φ₂ − 1`:
`R₀ = J₅²J_{5,2}/J_{5,1}² + (t−2)φ`, `R₁ = J₅²/J_{5,1}`, `R₂ = t·J₅²/J_{5,2}`,
`q R₃ = (s−t)ψ + (s+1)·q·J₅²J_{5,1}/J_{5,2}²`, `R₄ = 0`. -/
theorem lost_notebook_rank_mod5 :
    Rk 0 = ψ (eQ ^ 2 * Jab 5 2 * Ring.inverse (Jab 5 1) ^ 2) + C (tζ - 2) * ψ (RankProof.Phi 1 - 1)
    ∧ Rk 1 = ψ (eQ ^ 2 * Ring.inverse (Jab 5 1))
    ∧ Rk 2 = C tζ * ψ (eQ ^ 2 * Ring.inverse (Jab 5 2))
    ∧ X * Rk 3 = C (sζ - tζ) * ψ (RankProof.Phi 2 - 1)
        + C (sζ + 1) * X * ψ (eQ ^ 2 * Jab 5 1 * Ring.inverse (Jab 5 2) ^ 2)
    ∧ Rk 4 = 0 :=
  ⟨rank_R0, rank_R1, rank_R2, rank_R3, rank_R4⟩

end CrankProof
