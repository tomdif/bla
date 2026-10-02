/-
# Dyson's rank conjecture mod 5 (Atkin–Swinnerton-Dyer), via Garvan's new approach

`N(k, 5, 5n+4) = p(5n+4)/5` for every residue `k`.
From (2.18)/(2.19) at `ζ = ζ₅` we get three linear equations for the 5-dissection components
`R₂, R₃, R₄` of `R(ζ;q²)`; with the product identities of `RankDissect` they force `R₃ = 0`,
i.e. `[q^{5n+4}] R(ζ;q) = 0`, and cyclotomy finishes.
-/
import RamanujanTau.RankDissect

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5 eQ alphaQ betaQ dis5 ω5 ω5_prim ω5_pow5)
local notation "ψ" => MockTheta5.JTP.ψC

/-! ## 5-dissection over `ℂ` -/

section DisC

noncomputable def dis5C (r : ℕ) (f : PowerSeries ℂ) : PowerSeries ℂ := mk fun n => coeff (5 * n + r) f

@[simp] lemma coeff_dis5C (r n : ℕ) (f : PowerSeries ℂ) : coeff n (dis5C r f) = coeff (5 * n + r) f := by
  rw [dis5C, coeff_mk]

lemma dis5C_add (r : ℕ) (f g : PowerSeries ℂ) : dis5C r (f + g) = dis5C r f + dis5C r g := by ext n; simp

lemma dis5C_ψ (r : ℕ) (f : PowerSeries ℤ) : dis5C r (ψ f) = ψ (dis5 r f) := by
  ext n; simp [PowerSeries.coeff_map]

lemma dis5C_X_pow_mul {j r : ℕ} (hjr : j ≤ r) (g : PowerSeries ℂ) : dis5C r (X ^ j * g) = dis5C (r - j) g := by
  ext n; rw [coeff_dis5C, coeff_dis5C, coeff_X_pow_mul', if_pos (by omega), show 5 * n + r - j = 5 * n + (r - j) by omega]

lemma coeff_E5C_mul (n r : ℕ) (hr : r < 5) (F G : PowerSeries ℂ) :
    coeff (5 * n + r) (E5C F * G) = coeff n (F * dis5C r G) := by
  rw [coeff_mul, coeff_mul]
  -- pair (i, j) with i = 5a: coefficient of E5C F is nonzero only on multiples of 5
  rw [← sum_subset (s₁ := (antidiagonal n).map ⟨fun p => (5 * p.1, 5 * p.2 + r), fun p q h => by
      simp only [Prod.mk.injEq] at h; ext <;> omega⟩) ?_ ?_]
  · rw [sum_map]
    refine sum_congr rfl fun p _ => ?_
    simp only [Function.Embedding.coeFn_mk, coeff_E5C, coeff_dis5C]
    rw [if_pos (dvd_mul_right 5 _), Nat.mul_div_cancel_left _ (by norm_num)]
  · intro p hp
    simp only [mem_map, mem_antidiagonal, Function.Embedding.coeFn_mk] at hp ⊢
    obtain ⟨q, hq, rfl⟩ := hp
    omega
  · intro p hp hp'
    simp only [mem_antidiagonal] at hp
    rw [coeff_E5C]
    split_ifs with h
    · exfalso; apply hp'
      obtain ⟨c, hc⟩ := h
      simp only [mem_map, mem_antidiagonal, Function.Embedding.coeFn_mk]
      refine ⟨(c, (p.2 - r) / 5), ?_, ?_⟩ <;> [skip; ext] <;> simp only <;> omega
    · rw [zero_mul]

lemma dis5C_E5C_mul (r : ℕ) (hr : r < 5) (F G : PowerSeries ℂ) : dis5C r (E5C F * G) = F * dis5C r G := by
  ext n; rw [coeff_dis5C, coeff_E5C_mul n r hr]

lemma E5C_C (c : ℂ) : E5C (C c) = C c := by
  ext n; rw [coeff_E5C, coeff_C, coeff_C]; split_ifs <;> first | rfl | omega

lemma E5C_X : E5C X = X ^ 5 := PowerSeries.expand_X 5 (by norm_num)

lemma E2C_E5C (f : PowerSeries ℂ) : E2C (E5C f) = E5C (E2C f) := by
  ext n
  rw [coeff_E2C, coeff_E5C, coeff_E5C, coeff_E2C]
  split_ifs <;> first | rfl | omega | (rw [show n / 2 / 5 = n / 5 / 2 by omega])

/-- class 4 of `f(q²)` is class 2 of `f`, at `q²`. -/
lemma dis5C_four_E2C (f : PowerSeries ℂ) : dis5C 4 (E2C f) = E2C (dis5C 2 f) := by
  ext n
  rw [coeff_dis5C, coeff_E2C, coeff_E2C]
  by_cases h : 2 ∣ n
  · rw [if_pos (by omega), if_pos h, coeff_dis5C, show (5 * n + 4) / 2 = 5 * (n / 2) + 2 by omega]
  · rw [if_neg (by omega), if_neg h]

end DisC

/-! ## residue bookkeeping -/

section Residue

lemma natCast_zmod5 {e c : ℕ} (h : e % 5 = c) : (e : ZMod 5) = c := by
  rw [← Nat.mod_add_div e 5, h]; push_cast
  rw [show (5 : ZMod 5) = 0 from by decide]; ring

lemma dvd5_of_zmod {a : ℕ} (h : (a : ZMod 5) = 0) : 5 ∣ a := (ZMod.natCast_eq_zero_iff a 5).mp h

lemma two_T (a : ℕ) : 2 * (a * (a + 1) / 2) = a * (a + 1) := Nat.mul_div_cancel' (Nat.even_mul_succ_self a).two_dvd

lemma two_P (t : ℕ) : 2 * (t * (3 * t + 1) / 2) = t * (3 * t + 1) := by
  apply Nat.mul_div_cancel'
  rw [show t * (3 * t + 1) = t * (t + 1) + 2 * t ^ 2 by ring]
  exact dvd_add (Nat.even_mul_succ_self _).two_dvd (dvd_mul_right _ _)

/-- (2.18) exponents in class 2 force `5 ∣ a`. -/
lemma hr1_classP {a r m : ℕ} (he : a * (a + 1) / 2 + (r ^ 2 + a * r + (2 * r ^ 2 + 2 * a * r + r)) = m)
    (hm : m % 5 = 2) : 5 ∣ a := by
  have h2 : 2 * m = a * (a + 1) + 2 * (r ^ 2 + a * r + (2 * r ^ 2 + 2 * a * r + r)) := by
    rw [← he, mul_add, two_T]
  have hz := congrArg (fun x : ℕ => (x : ZMod 5)) h2
  simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, natCast_zmod5 hm] at hz
  apply dvd5_of_zmod
  generalize (a : ZMod 5) = x at hz
  generalize (r : ZMod 5) = y at hz
  revert x y; decide

lemma hr1_classM {a s m : ℕ}
    (he : a * (a + 1) / 2 + ((s + 1) ^ 2 + a * (s + 1) + (2 * s ^ 2 + 3 * s + 1 + 2 * a * s + a)) = m)
    (hm : m % 5 = 2) : 5 ∣ a := by
  have h2 : 2 * m = a * (a + 1) + 2 * ((s + 1) ^ 2 + a * (s + 1) + (2 * s ^ 2 + 3 * s + 1 + 2 * a * s + a)) := by
    rw [← he, mul_add, two_T]
  have hz := congrArg (fun x : ℕ => (x : ZMod 5)) h2
  simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, natCast_zmod5 hm] at hz
  apply dvd5_of_zmod
  generalize (a : ZMod 5) = x at hz
  generalize (s : ZMod 5) = y at hz
  revert x y; decide

/-- (2.19) exponents: class `c ∈ {3,4}` forces `a mod 5`. -/
lemma hr2_class {a t m c : ℕ} (hc : c = 3 ∨ c = 4)
    (he : RankProof.eRt a t = m ∨ RankProof.eRt a t + (2 * a + 2 * t + 1) = m) (hm : m % 5 = c) :
    (c = 3 → a % 5 = 2 ∨ a % 5 = 3) ∧ (c = 4 → a % 5 = 1 ∨ a % 5 = 4) := by
  have h2e : 2 * RankProof.eRt a t = a * (a + 1) + 6 * a * t + t * (3 * t + 1) := by
    unfold RankProof.eRt; rw [mul_add, mul_add, two_T, two_P]; ring
  have key : ∀ x y : ZMod 5, ∀ k : ZMod 5,
      (x * (x + 1) + 6 * x * y + y * (3 * y + 1) = 2 * k ∨ x * (x + 1) + 6 * x * y + y * (3 * y + 1)
        + 2 * (2 * x + 2 * y + 1) = 2 * k) →
      (k = 3 → x = 2 ∨ x = 3) ∧ (k = 4 → x = 1 ∨ x = 4) := by decide
  have hx : ∀ v : ℕ, v < 5 → ((a : ZMod 5) = v ↔ a % 5 = v) := fun v hv => by
    constructor
    · intro h
      have := (ZMod.natCast_eq_natCast_iff' a v 5).mp h
      rwa [Nat.mod_eq_of_lt hv] at this
    · intro h; rw [natCast_zmod5 h]
  have hk := key (a : ZMod 5) (t : ZMod 5) (c : ZMod 5) (by
    rcases he with he | he
    · left
      have := congrArg (fun x : ℕ => (x : ZMod 5)) (show 2 * m = _ from he ▸ h2e)
      simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, natCast_zmod5 hm] at this
      exact this.symm
    · right
      have h2 : 2 * m = a * (a + 1) + 6 * a * t + t * (3 * t + 1) + 2 * (2 * a + 2 * t + 1) := by
        rw [← he, mul_add, h2e]
      have := congrArg (fun x : ℕ => (x : ZMod 5)) h2
      simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, natCast_zmod5 hm] at this
      exact this.symm)
  rcases hc with rfl | rfl
  · refine ⟨fun _ => ?_, fun h => absurd h (by norm_num)⟩
    rcases hk.1 (by norm_num) with h | h
    · left; exact (hx 2 (by norm_num)).mp (by exact_mod_cast h)
    · right; exact (hx 3 (by norm_num)).mp (by exact_mod_cast h)
  · refine ⟨fun h => absurd h (by norm_num), fun _ => ?_⟩
    rcases hk.2 (by norm_num) with h | h
    · left; exact (hx 1 (by norm_num)).mp (by exact_mod_cast h)
    · right; exact (hx 4 (by norm_num)).mp (by exact_mod_cast h)

lemma ω5_pow_mod (a : ℕ) : ω5 ^ a = ω5 ^ (a % 5) := by
  conv_lhs => rw [← Nat.mod_add_div a 5, pow_add, pow_mul, ω5_pow5, one_pow, mul_one]

lemma wz_ω5_of_mod {a : ℕ} (k : ℕ) (hk : a % 5 = k) (hk0 : k ≠ 0) :
    wz ω5 a = ω5 ^ k + ω5 ^ ((5 - k) % 5) := by
  have ha : a ≠ 0 := by rintro rfl; simp at hk; omega
  rw [wz, if_neg ha, ω5_inv, ← pow_mul, ω5_pow_mod a, ω5_pow_mod (4 * a), hk]
  congr 2; omega

lemma wz_ω5_dvd {a : ℕ} (h : 5 ∣ a) : wz ω5 a = wz 1 a := by
  unfold wz
  split_ifs with ha
  · rfl
  · obtain ⟨c, rfl⟩ := h
    simp only [inv_pow, pow_mul, ω5_pow5, one_pow, inv_one]

lemma wz_one {a : ℕ} (ha : a ≠ 0) : wz 1 a = 2 := by rw [wz, if_neg ha]; norm_num

end Residue

/-! ## vanishing classes of the Hecke–Rogers blocks -/

section Vanish

/-- `E · (b-th block of (2.18))`: `(−1)^b q^{b(b+1)/2} Σ_r q^{r²+br} α*_r`. -/
noncomputable def Hblk (a : ℕ) : PowerSeries ℤ := C ((-1 : ℤ) ^ a) * X ^ (a * (a + 1) / 2) * RankProof.RHSz a

lemma qfacInf_h1Z (a : ℕ) : qfacInf * h1Z a = Hblk a := by
  rw [h1Z, Hblk, RankProof.bailey_k]
  have := Ring.mul_inverse_cancel _ isUnit_qfacInf
  linear_combination (C ((-1 : ℤ) ^ a) * X ^ (a * (a + 1) / 2) * RankProof.RHSz a) * this

lemma coeff_Xpow (k c : ℕ) : coeff c ((X : PowerSeries ℤ) ^ k) = if c = k then 1 else 0 := coeff_X_pow c k

theorem coeff_Hblk_zero {a m : ℕ} (hm : m % 5 = 2) (ha : ¬ 5 ∣ a) : coeff m (Hblk a) = 0 := by
  rw [Hblk, mul_assoc, coeff_C_mul, coeff_X_pow_mul']
  split_ifs with hT
  swap; · simp
  rw [RankProof.RHSz, coeff_mk, map_sum]
  refine mul_eq_zero_of_right _ (sum_eq_zero fun r _ => ?_)
  rcases r with _ | s
  · simp only [RankProof.αz, RankProof.Bz, sub_zero, ← pow_add, coeff_Xpow]
    rw [if_neg]; intro h
    exact ha (hr1_classP (r := 0) (by simp at h ⊢; omega) hm)
  · simp only [RankProof.αz, RankProof.Bz, mul_sub, ← pow_add, map_sub, coeff_Xpow]
    rw [if_neg, if_neg, sub_zero]
    · intro h; exact ha (hr1_classM (s := s) (by omega) hm)
    · intro h; exact ha (hr1_classP (r := s + 1) (by omega) hm)

theorem coeff_Gsum_zero {a m N c : ℕ} (hc : c = 3 ∨ c = 4) (hm : m % 5 = c)
    (ha : ¬ ((c = 3 → a % 5 = 2 ∨ a % 5 = 3) ∧ (c = 4 → a % 5 = 1 ∨ a % 5 = 4))) :
    coeff m (RankProof.Gsum a N) = 0 := by
  rw [RankProof.Gsum, map_sum]
  refine sum_eq_zero fun t _ => ?_
  rw [mul_sub, mul_one, ← pow_add, map_sub, coeff_Xpow, coeff_Xpow, if_neg, if_neg, sub_zero]
  · exact fun h => ha (hr2_class hc (Or.inr h.symm) hm)
  · exact fun h => ha (hr2_class hc (Or.inl h.symm) hm)

end Vanish

/-! ## the three equations -/

section Equations

/-- `(q)_∞ (zq)_∞ (q/z)_∞`. -/
noncomputable def Fz (z : ℂ) : PowerSeries ℂ := pochInf 1 1 * pochInf z 1 * pochInf z⁻¹ 1

lemma Dser_one : Dser 1 1⁻¹ = ψ MockTheta5.JTP.partitionGF := by
  ext n
  rw [← rank_durfee one_ne_zero n, PowerSeries.coeff_map, MockTheta5.JTP.coeff_partitionGF_eq_card]
  simp

lemma F_one : Fz 1 = ψ (qfacInf ^ 3) := by
  rw [Fz, inv_one, pochInf_one_eq, map_pow]; ring

lemma FD_one : Fz 1 * Dser 1 1⁻¹ = ψ (qfacInf ^ 2) := by
  rw [F_one, Dser_one, ← map_mul, MockTheta5.JTP.partitionGF]
  congr 1
  rw [pow_succ, mul_assoc, Ring.mul_inverse_cancel _ isUnit_qfacInf, mul_one]

lemma E2_eq_Ea (f : PowerSeries ℤ) : E2 f = RankProof.Ea 2 (by norm_num) f := rfl

lemma Jab21_mul : Jab 2 1 * E2 qfacInf = qfacInf ^ 2 := by
  have h1 := RankProof.Pinf_split 1 1 le_rfl le_rfl
  have h2 : E2 qfacInf = RankProof.Pinf 2 2 := by
    rw [E2_eq_Ea, ← RankProof.Pinf_one_one, RankProof.Ea_Pinf 2 1 1 _ le_rfl le_rfl]
  rw [RankProof.Pinf_one_one] at h1
  rw [Jab, h2, h1]; norm_num; ring

lemma FS_one : Fz 1 * E2C (Dser 1 1⁻¹) = ψ (Jab 2 1 * qfacInf) := by
  rw [F_one, Dser_one, ← ψ_E2, ← map_mul, MockTheta5.JTP.partitionGF, MockTheta5.JTP.E2_inverse_qfacInf]
  congr 1
  have hu : IsUnit (E2 qfacInf) := isUnit_qfacInf.map E2
  have := Jab21_mul
  calc qfacInf ^ 3 * Ring.inverse (E2 qfacInf)
      = (Jab 2 1 * E2 qfacInf) * qfacInf * Ring.inverse (E2 qfacInf) := by rw [this]; ring
    _ = Jab 2 1 * qfacInf * (E2 qfacInf * Ring.inverse (E2 qfacInf)) := by ring
    _ = _ := by rw [Ring.mul_inverse_cancel _ hu, mul_one]

/-- (2.18) multiplied through by `(q)_∞`. -/
lemma hr18_E {z : ℂ} (hz : z ≠ 0) (m : ℕ) :
    coeff m (Fz z * Dser z z⁻¹) = ∑ a ∈ range (m + 1), wz z a * coeff m (ψ (Hblk a)) := by
  have hH1 : ∀ k, coeff k (pochInf z 1 * pochInf z⁻¹ 1 * Dser z z⁻¹)
      = ∑ a ∈ range (k + 1), wz z a * coeff k (ψ (h1Z a)) := fun k => by
    rw [hecke_rogers_18 hz k]
    exact sum_congr rfl fun a _ => by rw [ψ_h1Z]
  have hT := trunc_of_coeff _ _ h1Z_dvd hH1 m
  rw [show Fz z * Dser z z⁻¹ = ψ qfacInf * (pochInf z 1 * pochInf z⁻¹ 1 * Dser z z⁻¹) by
    rw [Fz, pochInf_one_eq]; ring, coeffC_congr hT, mul_sum, map_sum]
  refine sum_congr rfl fun a _ => ?_
  rw [show ψ qfacInf * (C (wz z a) * ψ (h1Z a)) = C (wz z a) * ψ (qfacInf * h1Z a) by rw [map_mul]; ring,
    coeff_C_mul, qfacInf_h1Z]

lemma dis5_two_sq : dis5 2 (qfacInf ^ 2) = -eQ ^ 2 := by
  have hn : qfacInf ^ 2 = E5 (eQ ^ 2 * alphaQ ^ 2) + X * E5 (-(2 * eQ ^ 2 * alphaQ))
      + X ^ 2 * E5 (eQ ^ 2 * (1 - 2 * (alphaQ * betaQ))) + X ^ 3 * E5 (2 * eQ ^ 2 * betaQ)
      + X ^ 4 * E5 (eQ ^ 2 * betaQ ^ 2) := by
    rw [MockTheta5.JTP.qfacInf_dissection]
    simp only [map_add, map_sub, map_mul, map_neg, map_pow, map_one, map_ofNat]
    ring
  rw [hn, MockTheta5.JTP.dis5_normal 2 (by norm_num), MockTheta5.JTP.alphaQ_mul_betaQ]
  simp only [show (2 : ℕ) ≠ 0 by norm_num, show (2 : ℕ) ≠ 1 by norm_num, if_false, if_true]
  ring

/-- **equation 1** (class 2 of (2.18)): `U_{5,2}(F(ζ)R(ζ;q)) = −E(q⁵)²`. -/
theorem eq_class2 : dis5C 2 (Fz ω5 * Dser ω5 ω5⁻¹) = ψ (-eQ ^ 2) := by
  ext n
  rw [coeff_dis5C, ← dis5_two_sq, ← dis5C_ψ, coeff_dis5C, ← FD_one, hr18_E ω5_ne,
    hr18_E one_ne_zero]
  refine sum_congr rfl fun a _ => ?_
  by_cases ha : 5 ∣ a
  · rw [wz_ω5_dvd ha]
  · rw [PowerSeries.coeff_map, coeff_Hblk_zero (by omega) ha]; simp

/-- (2.19) at `ζ` vs at `1`, on the classes 3, 4. -/
lemma eq_class34 {c : ℕ} (hc : c = 3 ∨ c = 4) (k : ℂ)
    (hk : ∀ a : ℕ, ((c = 3 → a % 5 = 2 ∨ a % 5 = 3) ∧ (c = 4 → a % 5 = 1 ∨ a % 5 = 4)) → wz ω5 a = k * wz 1 a) :
    dis5C c (Fz ω5 * E2C (Dser ω5 ω5⁻¹)) = C k * dis5C c (Fz 1 * E2C (Dser 1 1⁻¹)) := by
  ext n
  rw [coeff_C_mul, coeff_dis5C, coeff_dis5C, Fz, Fz, hecke_rogers_19 ω5_ne, hecke_rogers_19 one_ne_zero,
    mul_sum]
  refine sum_congr rfl fun a _ => ?_
  by_cases ha : (c = 3 → a % 5 = 2 ∨ a % 5 = 3) ∧ (c = 4 → a % 5 = 1 ∨ a % 5 = 4)
  · rw [hk a ha]; ring
  · rw [PowerSeries.coeff_map, coeff_C_mul, coeff_Gsum_zero hc (by omega) ha]; simp

/-- `s = ζ² + ζ³`, `t = ζ + ζ⁴`. -/
noncomputable def sζ : ℂ := ω5 ^ 2 + ω5 ^ 3
noncomputable def tζ : ℂ := ω5 + ω5 ^ 4

lemma s_add_t : sζ + tζ = -1 := by
  have := MockTheta5.JTP.phi_of_prim ω5_prim
  rw [sζ, tζ]; linear_combination this

lemma s_add_one_ne : sζ + 1 ≠ 0 := by
  intro h
  have hphi := MockTheta5.JTP.phi_of_prim ω5_prim
  have h1 : ω5 ^ 4 = -ω5 := by rw [sζ] at h; linear_combination hphi - h
  have h3 : ω5 ^ 3 = -1 := by
    have := mul_left_cancel₀ ω5_ne (show ω5 * ω5 ^ 3 = ω5 * (-1) by linear_combination h1)
    exact this
  have h5 := ω5_pow5
  have h2 : ω5 ^ 2 = -1 := by linear_combination (ω5 ^ 2) * h3 - h5
  have : ω5 = 1 := by linear_combination ω5 * h2 - h3
  exact ω5_prim.ne_one (by norm_num) this

/-- **equations 2, 3** (classes 3, 4 of (2.19)). -/
theorem eq_class3 : dis5C 3 (Fz ω5 * E2C (Dser ω5 ω5⁻¹)) = C sζ * ψ (eQ * Jab 10 3 * betaQ) := by
  rw [eq_class34 (Or.inl rfl) (sζ / 2) ?_, FS_one, dis5C_ψ, (theta4E_classes).1]
  · simp only [map_mul, map_ofNat]
    rw [show C (sζ / 2) = C sζ * C (1 / 2 : ℂ) by rw [← map_mul]; ring_nf]
    rw [show (2 : PowerSeries ℂ) = C 2 from (map_ofNat C 2).symm]
    have : C (1 / 2 : ℂ) * C 2 = (1 : PowerSeries ℂ) := by rw [← map_mul]; norm_num
    linear_combination (C sζ * ψ eQ * ψ (Jab 10 3) * ψ betaQ) * this
  · intro a ha
    have h2 := ha.1 rfl
    have ha0 : a ≠ 0 := by rintro rfl; omega
    rw [wz_one ha0]
    rcases h2 with h | h
    · rw [wz_ω5_of_mod 2 h (by norm_num)]; simp [sζ]
    · rw [wz_ω5_of_mod 3 h (by norm_num)]; simp [sζ]; ring

theorem eq_class4 : dis5C 4 (Fz ω5 * E2C (Dser ω5 ω5⁻¹)) = C tζ * ψ (eQ * Jab 10 1 * alphaQ) := by
  rw [eq_class34 (Or.inr rfl) (tζ / 2) ?_, FS_one, dis5C_ψ, (theta4E_classes).2]
  · simp only [map_mul, map_ofNat]
    rw [show C (tζ / 2) = C tζ * C (1 / 2 : ℂ) by rw [← map_mul]; ring_nf]
    rw [show (2 : PowerSeries ℂ) = C 2 from (map_ofNat C 2).symm]
    have : C (1 / 2 : ℂ) * C 2 = (1 : PowerSeries ℂ) := by rw [← map_mul]; norm_num
    linear_combination (C tζ * ψ eQ * ψ (Jab 10 1) * ψ alphaQ) * this
  · intro a ha
    have h2 := ha.2 rfl
    have ha0 : a ≠ 0 := by rintro rfl; omega
    rw [wz_one ha0]
    rcases h2 with h | h
    · rw [wz_ω5_of_mod 1 h (by norm_num)]; simp [tζ]
    · rw [wz_ω5_of_mod 4 h (by norm_num)]; simp [tζ]; ring

end Equations

/-! ## the product identities (Garvan (3.19)) -/

section Products
open RankProof (Pinf Pinf_split Ea_Pinf)

lemma E2_Pinf (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) : E2 (Pinf b a) = Pinf (2 * b) (2 * a) := by
  rw [E2_eq_Ea, Ea_Pinf 2 b a _ hb ha]

lemma eQ_Pinf : eQ = Pinf 5 5 := by
  rw [show eQ = E5 qfacInf from rfl, ← RankProof.Pinf_one_one,
    show E5 (Pinf 1 1) = RankProof.Ea 5 (by norm_num) (Pinf 1 1) from rfl, Ea_Pinf 5 1 1 _ le_rfl le_rfl]

lemma E2eQ_Pinf : E2 eQ = Pinf 10 10 := by rw [eQ_Pinf, E2_Pinf 5 5 (by norm_num) (by norm_num)]

/-- `J_{5,2}(q²) · J₅ · J_{10,1} = J₁₀² · J_{5,1}`. -/
theorem prod_id1 : E2 (Jab 5 2) * eQ * Jab 10 1 = E2 eQ ^ 2 * Jab 5 1 := by
  rw [E2eQ_Pinf, eQ_Pinf, Jab, Jab, Jab, map_mul, map_mul, E2_Pinf 2 5 (by norm_num) (by norm_num),
    E2_Pinf (5 - 2) 5 (by norm_num) (by norm_num), E2_Pinf 5 5 (by norm_num) (by norm_num),
    Pinf_split 1 5 (by norm_num) (by norm_num), Pinf_split (5 - 1) 5 (by norm_num) (by norm_num)]
  norm_num; ring

/-- `J_{5,1}(q²) · J₅ · J_{10,3} = J₁₀² · J_{5,2}`. -/
theorem prod_id2 : E2 (Jab 5 1) * eQ * Jab 10 3 = E2 eQ ^ 2 * Jab 5 2 := by
  rw [E2eQ_Pinf, eQ_Pinf, Jab, Jab, Jab, map_mul, map_mul, E2_Pinf 1 5 (by norm_num) (by norm_num),
    E2_Pinf (5 - 1) 5 (by norm_num) (by norm_num), E2_Pinf 5 5 (by norm_num) (by norm_num),
    Pinf_split 2 5 (by norm_num) (by norm_num), Pinf_split (5 - 2) 5 (by norm_num) (by norm_num)]
  norm_num; ring

lemma coeff_zero_Pinf (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) : constantCoeff (Pinf b a) = 1 := by
  have := (X_pow_dvd_iff.mp (RankProof.X_pow_dvd_Pinf_sub b a hb ha 0)) 0 (by omega)
  rw [map_sub, sub_eq_zero, RankProof.Pfin] at this
  rw [← coeff_zero_eq_constantCoeff_apply, this]; simp

lemma constantCoeff_Jab (a b : ℕ) (hb : 1 ≤ b) (hab : b < a) : constantCoeff (Jab a b) = 1 := by
  rw [Jab, map_mul, map_mul, coeff_zero_Pinf b a hb (by omega), coeff_zero_Pinf (a - b) a (by omega) (by omega),
    coeff_zero_Pinf a a (by omega) (by omega)]; norm_num

lemma constantCoeff_ψ (f : PowerSeries ℤ) : constantCoeff (ψ f) = ((constantCoeff f : ℤ) : ℂ) := by
  rw [← coeff_zero_eq_constantCoeff_apply, PowerSeries.coeff_map, coeff_zero_eq_constantCoeff_apply]; rfl

lemma constantCoeff_E2C (f : PowerSeries ℂ) : constantCoeff (E2C f) = constantCoeff f := by
  rw [← coeff_zero_eq_constantCoeff_apply, coeff_E2C, if_pos (dvd_zero 2), coeff_zero_eq_constantCoeff_apply]

end Products

/-! ## the solve and the theorem -/

section Solve

/-- **`U_{5,3} R(ζ;q²) = 0`.** -/
theorem S3_zero : dis5C 3 (E2C (Dser ω5 ω5⁻¹)) = 0 := by
  set S := E2C (Dser ω5 ω5⁻¹) with hS
  set A := ψ (Jab 5 2)
  set B := ψ (Jab 5 1)
  have hF : Fz ω5 = E5C A + X * C sζ * E5C B := by
    rw [Fz, F_dissect ω5_pow5 (ω5_prim.ne_one (by norm_num))]; rfl
  have hshift : ∀ r, 1 ≤ r → r < 5 → ∀ G : PowerSeries ℂ, dis5C r (X * C sζ * E5C B * G) = C sζ * B * dis5C (r - 1) G := by
    intro r h1 h5 G
    rw [show X * C sζ * E5C B * G = E5C (C sζ * B) * (X ^ 1 * G) by rw [map_mul, E5C_C]; ring,
      dis5C_E5C_mul r h5, dis5C_X_pow_mul h1]
  -- equation 2 and 3
  have e2 : A * dis5C 3 S + C sζ * B * dis5C 2 S = C sζ * ψ (eQ * Jab 10 3 * betaQ) := by
    rw [← eq_class3, hF, add_mul, dis5C_add, dis5C_E5C_mul 3 (by norm_num), hshift 3 (by norm_num) (by norm_num)]
  have e3 : A * dis5C 4 S + C sζ * B * dis5C 3 S = C tζ * ψ (eQ * Jab 10 1 * alphaQ) := by
    rw [← eq_class4, hF, add_mul, dis5C_add, dis5C_E5C_mul 4 (by norm_num), hshift 4 (by norm_num) (by norm_num)]
  -- equation 1, at q²
  have e1 : E2C A * dis5C 4 S + C sζ * E2C B * dis5C 2 S = -ψ (E2 eQ ^ 2) := by
    have h := congrArg E2C eq_class2
    rw [← dis5C_four_E2C, map_mul, hF] at h
    rw [show -ψ (E2 eQ ^ 2) = E2C (ψ (-eQ ^ 2)) by rw [← ψ_E2]; simp [map_neg, map_pow], ← h]
    simp only [map_add, map_mul, E2C_E5C, E2C_X, E2C_C, add_mul, dis5C_add, dis5C_E5C_mul 4 (by norm_num : 4 < 5)]
    rw [show X ^ 2 * C sζ * E5C (E2C B) * S = E5C (C sζ * E2C B) * (X ^ 2 * S) by rw [map_mul, E5C_C]; ring,
      dis5C_E5C_mul 4 (by norm_num), dis5C_X_pow_mul (by norm_num : 2 ≤ 4)]
  -- product identities
  have i1 : E2C A * ψ eQ * ψ (Jab 10 1) = ψ (E2 eQ ^ 2) * B := by
    rw [← ψ_E2, ← map_mul, ← map_mul, ← map_mul, prod_id1]
  have i2 : E2C B * ψ eQ * ψ (Jab 10 3) = ψ (E2 eQ ^ 2) * A := by
    rw [← ψ_E2, ← map_mul, ← map_mul, ← map_mul, prod_id2]
  have rα : A = ψ alphaQ * B := by rw [← map_mul]; exact congrArg ψ J5_relations.2
  have rβ : B = ψ betaQ * A := by rw [← map_mul]; exact congrArg ψ betaQ_eq
  have hst : C sζ + C tζ = (-1 : PowerSeries ℂ) := by rw [← map_add, s_add_t, map_neg, map_one]
  set Z := ψ (E2 eQ ^ 2)
  set c := C sζ * E2C A * B ^ 2 + A ^ 2 * E2C B
  have key : c * dis5C 3 S = 0 := by
    simp only [map_mul] at e2 e3
    linear_combination -(A * B) * e1 + E2C A * B * e3 + A * E2C B * e2 + C tζ * ψ alphaQ * B * i1
      - C tζ * Z * B * rα + C sζ * A * ψ betaQ * i2 - C sζ * Z * A * rβ + Z * A * B * hst
  have hc : c ≠ 0 := by
    intro h0
    have := congrArg constantCoeff h0
    simp only [c, A, B, map_add, map_mul, map_pow, constantCoeff_C, constantCoeff_E2C, constantCoeff_ψ,
      constantCoeff_Jab 5 2 (by norm_num) (by norm_num), constantCoeff_Jab 5 1 (by norm_num) (by norm_num),
      map_zero] at this
    norm_num at this
    exact s_add_one_ne this
  exact (mul_eq_zero.mp key).resolve_left hc

/-- `Σ_{λ ⊢ 5n+4} ζ^{rank λ} = 0`. -/
theorem rank_sum_root_zero (n : ℕ) : ∑ l : (5 * n + 4).Partition, ω5 ^ rank l = 0 := by
  have h := congrArg (coeff (2 * n + 1)) S3_zero
  rw [coeff_dis5C, coeff_E2C, if_pos (by omega), show (5 * (2 * n + 1) + 3) / 2 = 5 * n + 4 by omega,
    map_zero] at h
  rw [rank_durfee ω5_ne, h]

/-- the number of partitions of `n` with rank `≡ k (mod 5)`. -/
noncomputable def rankCount (n k : ℕ) : ℕ := (univ.filter fun l : n.Partition => (rank l % 5).toNat = k).card

lemma rank_sum_grouped (n : ℕ) :
    ∑ l : n.Partition, ω5 ^ rank l = ∑ k ∈ range 5, (rankCount n k : ℂ) * ω5 ^ k := by
  simp_rw [ω5_zpow_crank]
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range 5) (g := fun l : n.Partition => (rank l % 5).toNat)
    (fun l _ => by simp only [Finset.mem_range]; omega)]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_congr rfl (g := fun _ => ω5 ^ k) (fun l hl => by rw [(Finset.mem_filter.mp hl).2]),
    Finset.sum_const, nsmul_eq_mul, rankCount]

lemma rankCount_sum (n : ℕ) : ∑ k ∈ range 5, rankCount n k = Fintype.card n.Partition := by
  rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun l : n.Partition => (rank l % 5).toNat)
    (t := range 5) (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (rank l % 5).toNat < 5 by omega)))]
  rfl

/-- **Dyson's rank conjecture mod 5** (Atkin–Swinnerton-Dyer 1954): the rank splits the partitions of `5n+4`
into five equal classes, `N(i, 5, 5n+4) = p(5n+4)/5`. -/
theorem rank_equidistribution_mod5 (n : ℕ) {i : ℕ} (hi : i < 5) :
    5 * (univ.filter fun l : (5 * n + 4).Partition => rank l % 5 = i).card
      = Fintype.card (5 * n + 4).Partition := by
  have h := rank_sum_root_zero n
  rw [rank_sum_grouped] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, mul_one, pow_one] at h
  obtain ⟨e0, e1, e2, e3⟩ := cyc_equal _ _ _ _ _ h
  have hs := rankCount_sum (5 * n + 4)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  have hfilt : (univ.filter fun l : (5 * n + 4).Partition => rank l % 5 = i) =
      univ.filter fun l => (rank l % 5).toNat = i :=
    Finset.filter_congr fun l _ => by omega
  rw [hfilt, ← rankCount]
  interval_cases i <;> omega

end Solve

end CrankProof
