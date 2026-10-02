/-
# 5-dissections of theta series (for the ζ₅ endgame of Dyson's rank)

`thL a b u = Σ_{m∈ℤ} (−u)^m q^{a·C(m,2)+bm}` as a lattice sum; for `u⁵ = 1` it splits by `m mod 5`
(representatives `r ∈ {−2,…,2}`) into `Σ_r (−u)^r q^{a·C(r,2)+br} · thL(5a, a(r+2)+b, 1)(q⁵)`.
With `jtp_ab` each piece is a product `J_{5a, a(r+2)+b}`.
-/
import RamanujanTau.RankTheta

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5 eQ alphaQ betaQ dis5 triE ω5 ω5_prim ω5_pow5)
open scoped PowerSeries.WithPiTopology
local notation "ψ" => MockTheta5.JTP.ψC

/-! ## exponents -/

section Expo

/-- `a·C(m,2) + bm` (nonnegative when `b ≤ a`). -/
def expo (a b : ℕ) (m : ℤ) : ℕ := ((a : ℤ) * (triE m : ℤ) + b * m).toNat

lemma two_expo_raw (a b : ℕ) (m : ℤ) :
    2 * ((a : ℤ) * (triE m : ℤ) + b * m) = a * (m * (m - 1)) + 2 * b * m := by
  rw [mul_add, ← mul_assoc, mul_comm 2 (a : ℤ), mul_assoc, two_triE]; ring

lemma expo_nonneg {a b : ℕ} (hba : b ≤ a) (m : ℤ) : 0 ≤ (a : ℤ) * (triE m : ℤ) + b * m := by
  have h := two_expo_raw a b m
  have hb' : (b : ℤ) ≤ a := by exact_mod_cast hba
  have ha0 : (0 : ℤ) ≤ a := by positivity
  have hb0 : (0 : ℤ) ≤ b := by positivity
  rcases le_or_gt 0 m with hm | hm
  · have : 0 ≤ (a : ℤ) * (m * (m - 1)) := by
      rcases le_or_gt 1 m with h1 | h1
      · exact mul_nonneg ha0 (by nlinarith)
      · have : m = 0 := by omega
        subst this; simp
    nlinarith
  · have h1 : (0 : ℤ) ≤ -m := by omega
    have h2 : (0 : ℤ) ≤ a * (1 - m) - 2 * b := by nlinarith
    nlinarith [mul_nonneg h1 h2]

lemma two_expo {a b : ℕ} (hba : b ≤ a) (m : ℤ) : 2 * (expo a b m : ℤ) = a * (m * (m - 1)) + 2 * b * m := by
  rw [expo, Int.toNat_of_nonneg (expo_nonneg hba m), two_expo_raw]

lemma expo_bound {a b : ℕ} (ha : 1 ≤ a) (hba : b ≤ a) {m : ℤ} {k : ℕ} (h : expo a b m ≤ k) :
    -((k : ℤ) + 1) ≤ m ∧ m ≤ k + 1 := by
  have h2 := two_expo hba m
  have hk : (expo a b m : ℤ) ≤ k := by exact_mod_cast h
  have ha' : (1 : ℤ) ≤ a := by exact_mod_cast ha
  have hb' : (b : ℤ) ≤ a := by exact_mod_cast hba
  have hb0 : (0 : ℤ) ≤ b := by positivity
  constructor
  · by_contra hc
    push_neg at hc
    set n := -m with hn
    have hn2 : (k : ℤ) + 2 ≤ n := by omega
    have e : a * (m * (m - 1)) + 2 * b * m = a * (n * (n + 1)) - 2 * b * n := by rw [hn]; ring
    have h3 : (a : ℤ) * (n * (n - 1)) ≤ a * (n * (n + 1)) - 2 * b * n := by nlinarith
    have h4 : n * (n - 1) ≤ (a : ℤ) * (n * (n - 1)) := le_mul_of_one_le_left (by nlinarith) ha'
    nlinarith
  · by_contra hc
    push_neg at hc
    have hmm : 0 ≤ m * (m - 1) := by nlinarith
    have h4 : m * (m - 1) ≤ (a : ℤ) * (m * (m - 1)) := le_mul_of_one_le_left hmm ha'
    nlinarith

lemma proper_expo {a b : ℕ} (ha : 1 ≤ a) (hba : b ≤ a) : Proper (expo a b) := fun k =>
  (Set.finite_Icc (-((k : ℤ) + 1)) (k + 1)).subset fun _ hm => expo_bound ha hba hm

lemma expo_natCast (a b d : ℕ) : expo a b (d : ℤ) = a * d.choose 2 + b * d := by
  have h := two_triE (d : ℤ)
  have hc := two_choose_two d
  have : (triE (d : ℤ) : ℤ) = (d.choose 2 : ℤ) := by linarith
  rw [expo, this]; exact_mod_cast Int.toNat_natCast _

lemma expo_negSucc {a b : ℕ} (hba : b ≤ a) (d : ℕ) :
    expo a b (-((d : ℤ) + 1)) = a * (d + 1).choose 2 + (a - b) * (d + 1) := by
  have h := two_expo hba (-((d : ℤ) + 1))
  have hc := two_choose_two (d + 1)
  have : ((a * (d + 1).choose 2 + (a - b) * (d + 1) : ℕ) : ℤ) = expo a b (-((d : ℤ) + 1)) := by
    push_cast [Nat.cast_sub hba] at hc ⊢; nlinarith
  exact_mod_cast this.symm

end Expo

/-! ## theta series as lattice sums -/

section ThetaLat

/-- `Σ_{m∈ℤ} (−u)^m q^{a·C(m,2)+bm}`. -/
noncomputable def thL (a b : ℕ) (u : ℂ) : PowerSeries ℂ := lat (fun m : ℤ => (-u) ^ m) (expo a b)

/-- `thetaS` (the `ℤ⟦q⟧` theta series of `jtp_ab`) is the lattice sum at `u = 1`. -/
theorem thetaS_eq_thL {a b : ℕ} (hb : 1 ≤ b) (hab : b < a) : ψ (RankProof.thetaS a b) = thL a b 1 := by
  ext k
  rw [thL, coeff_lat _ (proper_expo (by omega) hab.le), tsum_ite_eq_box fun m h => expo_bound (by omega) hab.le (le_of_eq h),
    show -((k : ℤ) + 1) = -(((k + 1 : ℕ)) : ℤ) by push_cast; ring, show (k : ℤ) + 1 = ((k + 1 : ℕ) : ℤ) by push_cast; ring,
    Icc_split _ (k + 1), PowerSeries.coeff_map, RankProof.thetaS, coeff_mk, RankProof.thetaTr, map_add, map_sum,
    map_sum, sum_range_succ _ (k + 1)]
  simp only [coeff_C_mul_X_pow]
  have hlast : (if expo a b ((k + 1 : ℕ) : ℤ) = k then (-1 : ℂ) ^ ((k + 1 : ℕ) : ℤ) else 0) = 0 := by
    rw [if_neg]; rw [expo_natCast]
    have : k + 1 ≤ b * (k + 1) := Nat.le_mul_of_pos_left _ (by omega)
    omega
  rw [hlast, add_zero]
  simp only [map_add, map_sum, apply_ite (Int.castRingHom ℂ), map_pow, map_neg, map_one, map_zero]
  congr 1
  · refine sum_congr rfl fun d _ => ?_
    rw [expo_natCast, zpow_natCast]
    by_cases h : a * d.choose 2 + b * d = k
    · rw [if_pos h.symm, if_pos h]
    · rw [if_neg (Ne.symm h), if_neg h]
  · refine sum_congr rfl fun d _ => ?_
    rw [expo_negSucc hab.le]
    by_cases h : a * (d + 1).choose 2 + (a - b) * (d + 1) = k
    · rw [if_pos h.symm, if_pos h, zpow_neg, show ((d : ℤ) + 1) = ((d + 1 : ℕ) : ℤ) by push_cast; ring,
        zpow_natCast, ← inv_pow, inv_neg, inv_one]
    · rw [if_neg (Ne.symm h), if_neg h]

end ThetaLat

/-! ## the 5-dissection -/

section Dissect

noncomputable def E5C : PowerSeries ℂ →+* PowerSeries ℂ := (PowerSeries.expand 5 (by norm_num)).toRingHom

lemma coeff_E5C (n : ℕ) (f : PowerSeries ℂ) : coeff n (E5C f) = if 5 ∣ n then coeff (n / 5) f else 0 := by
  have h : E5C f = PowerSeries.expand 5 (by norm_num) f := rfl
  rw [h, PowerSeries.coeff_expand]

lemma ψ_E5 (f : PowerSeries ℤ) : ψ (E5 f) = E5C (ψ f) := by
  ext n
  have h : E5 f = PowerSeries.expand 5 (by norm_num) f := rfl
  rw [coeff_E5C, PowerSeries.coeff_map, h, PowerSeries.coeff_expand]
  split_ifs <;> simp [PowerSeries.coeff_map]

lemma proper_five {ι : Type*} {e : ι → ℕ} (he : Proper e) : Proper fun i => 5 * e i := fun k =>
  (he k).subset fun i hi => by simp only [Set.mem_setOf_eq] at hi ⊢; omega

lemma lat_five {ι : Type*} (w : ι → ℂ) {e : ι → ℕ} (he : Proper e) :
    lat w (fun i => 5 * e i) = E5C (lat w e) := by
  ext k
  rw [coeff_lat _ (proper_five he), coeff_E5C]
  split_ifs with hk
  · rw [coeff_lat _ he]
    refine tsum_congr fun i => ?_
    obtain ⟨j, rfl⟩ := hk
    by_cases h : e i = j
    · rw [if_pos (by omega), if_pos (by omega)]
    · rw [if_neg (by omega), if_neg (by omega)]
  · simp only [show ∀ i, ¬ (5 * e i = k) from fun i h => hk ⟨e i, h.symm⟩, if_false, tsum_zero]

/-- `(r, k) ↦ 5k + r − 2`, `r ∈ {0,…,4}`. -/
def σ5 : Fin 5 × ℤ ≃ ℤ where
  toFun p := 5 * p.2 + (p.1 : ℤ) - 2
  invFun m := (⟨((m + 2) % 5).toNat, by omega⟩, (m + 2) / 5)
  left_inv p := by
    obtain ⟨⟨i, hi⟩, k⟩ := p
    simp only [Prod.mk.injEq, Fin.mk.injEq]
    constructor <;> omega
  right_inv m := by simp only; omega

lemma expo_split {a b : ℕ} (hba : b ≤ a) (i : ℕ) (hi : i < 5) (k : ℤ) :
    expo a b (5 * k + i - 2) = expo a b ((i : ℤ) - 2) + 5 * expo (5 * a) (a * i + b) k := by
  have h1 := two_expo hba (5 * k + i - 2)
  have h2 := two_expo hba ((i : ℤ) - 2)
  have h3 := two_expo (show a * i + b ≤ 5 * a by nlinarith) k
  have : 2 * (expo a b (5 * k + i - 2) : ℤ) = 2 * ((expo a b ((i : ℤ) - 2) : ℤ) + 5 * expo (5 * a) (a * i + b) k) := by
    push_cast at h3
    linear_combination h1 - h2 - 5 * h3
  omega

/-- **the 5-dissection of a theta series** (`u⁵ = 1`). -/
theorem thL_dissect {a b : ℕ} (ha : 1 ≤ a) (hba : b ≤ a) {u : ℂ} (hu : u ^ 5 = 1) :
    thL a b u = ∑ i : Fin 5, C ((-u) ^ ((i : ℤ) - 2)) * X ^ expo a b ((i : ℤ) - 2)
      * E5C (thL (5 * a) (a * i + b) 1) := by
  have hu0 : -u ≠ 0 := by
    intro h; rw [neg_eq_zero] at h; rw [h] at hu; norm_num at hu
  rw [thL, ← lat_equiv σ5, lat_fubini _ (proper_equiv (proper_expo ha hba) σ5), tsum_fintype]
  refine sum_congr rfl fun i _ => ?_
  have hi := i.isLt
  have hp : Proper (expo (5 * a) (a * i + b)) := proper_expo (by omega) (by nlinarith)
  have hw : (fun k : ℤ => (((fun m : ℤ => (-u) ^ m) ∘ σ5) (i, k)))
      = fun k => (-u) ^ ((i : ℤ) - 2) * (-1 : ℂ) ^ k := by
    funext k
    simp only [Function.comp_apply, σ5, Equiv.coe_fn_mk]
    rw [show 5 * k + (i : ℤ) - 2 = 5 * k + ((i : ℤ) - 2) by ring, zpow_add₀ hu0, zpow_mul,
      show (-u) ^ (5 : ℤ) = -1 by rw [zpow_ofNat, neg_pow, hu]; norm_num]
    ring
  have he : (fun k : ℤ => ((expo a b ∘ σ5) (i, k))) = fun k => expo a b ((i : ℤ) - 2) + 5 * expo (5 * a) (a * i + b) k := by
    funext k
    simp only [Function.comp_apply, σ5, Equiv.coe_fn_mk]
    exact expo_split hba i hi k
  rw [hw, he, lat_shift _ _ _ (proper_five hp), lat_five _ hp]
  rfl

end Dissect

/-! ## the two concrete dissections -/

section Concrete

/-- `J_{a,b} = (q^b;q^a)(q^{a−b};q^a)(q^a;q^a)`. -/
noncomputable def Jab (a b : ℕ) : PowerSeries ℤ := RankProof.Pinf b a * RankProof.Pinf (a - b) a * RankProof.Pinf a a

lemma Jab_symm (a b : ℕ) (hab : b ≤ a) : Jab a (a - b) = Jab a b := by
  rw [Jab, Jab, Nat.sub_sub_self hab]; ring

lemma thL_eq_Jab {a b : ℕ} (hb : 1 ≤ b) (hab : b < a) : thL a b 1 = ψ (Jab a b) := by
  rw [← thetaS_eq_thL hb hab, Jab, RankProof.jtp_ab a b hb hab]

lemma triE_one_sub (m : ℤ) : triE (1 - m) = triE m := by
  have h1 := two_triE (1 - m)
  have h2 := two_triE m
  have : (triE (1 - m) : ℤ) = triE m := by nlinarith
  exact_mod_cast this

lemma thL_zero (a : ℕ) : thL a 0 1 = 0 := by
  refine lat_eq_zero_of_invol (Equiv.subLeft 1) _ _ fun m => ⟨?_, ?_⟩
  · simp only [Equiv.subLeft_apply]
    have h1 : (-1 : ℂ) ^ (1 - m) * (-1) ^ m = -1 := by
      rw [← zpow_add₀ (by norm_num), sub_add_cancel, zpow_one]
    have h2 : (-1 : ℂ) ^ m * (-1) ^ m = 1 := by
      rw [← zpow_add₀ (by norm_num), ← two_mul, zpow_mul]; norm_num
    linear_combination (-((-1 : ℂ) ^ (1 - m))) * h2 + (-1 : ℂ) ^ m * h1
  · simp only [Equiv.subLeft_apply, expo, triE_one_sub, Nat.cast_zero, zero_mul, add_zero]

lemma expo_one_zero : expo 1 0 = triE := by
  funext m; simp [expo]

/-- **5-dissection of `(wq;q)_∞(q/w;q)_∞(q;q)_∞`** for `w⁵ = 1`, `w ≠ 1` (Garvan (3.3)). -/
theorem F_dissect {w : ℂ} (hw : w ^ 5 = 1) (hw1 : w ≠ 1) :
    pochInf 1 1 * pochInf w 1 * pochInf w⁻¹ 1
      = E5C (ψ (Jab 5 2)) + X * C (w ^ 2 + w ^ 3) * E5C (ψ (Jab 5 1)) := by
  have hw0 : w ≠ 0 := by rintro rfl; norm_num at hw
  have hnw : -w ≠ 0 := neg_ne_zero.mpr hw0
  set u : ℂˣ := Units.mk0 (-w) hnw
  have hJ := jtp_eval u
  rw [theta_lat u] at hJ
  have hu : (u : ℂ) = -w := rfl
  have hui : -((u⁻¹ : ℂˣ) : ℂ) = w⁻¹ := by rw [Units.val_inv_eq_inv_val, hu, inv_neg, neg_neg]
  rw [hui, hu, neg_neg, show (fun m : ℤ => (-w) ^ m) = fun m : ℤ => (-w) ^ m from rfl,
    ← expo_one_zero, show lat (fun m : ℤ => (-w) ^ m) (expo 1 0) = thL 1 0 w from rfl,
    thL_dissect le_rfl (by norm_num) hw, Fin.sum_univ_five] at hJ
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two, one_mul, add_zero] at hJ
  rw [thL_zero, map_zero, mul_zero, zero_add] at hJ
  have h3 : ((3 : Fin 5) : ℕ) = 3 := rfl
  have h4 : ((4 : Fin 5) : ℕ) = 4 := rfl
  rw [h3, h4, thL_eq_Jab (by norm_num) (by norm_num), thL_eq_Jab (by norm_num) (by norm_num),
    thL_eq_Jab (by norm_num) (by norm_num), thL_eq_Jab (by norm_num) (by norm_num),
    show Jab 5 3 = Jab 5 2 from Jab_symm 5 2 (by norm_num), show Jab 5 4 = Jab 5 1 from Jab_symm 5 1 (by norm_num)] at hJ
  have e1 : expo 1 0 ((1 : ℕ) - 2 : ℤ) = 1 := by decide
  have e2 : expo 1 0 ((2 : ℕ) - 2 : ℤ) = 0 := by decide
  have e3 : expo 1 0 ((3 : ℕ) - 2 : ℤ) = 0 := by decide
  have e4 : expo 1 0 ((4 : ℕ) - 2 : ℤ) = 1 := by decide
  push_cast at hJ e1 e2 e3 e4
  rw [e1, e2, e3, e4] at hJ
  have hwinv : w⁻¹ = w ^ 4 := inv_eq_of_mul_eq_one_right (by rw [← pow_succ', hw])
  have hc : (1 : ℂ) - w ≠ 0 := sub_ne_zero.mpr (Ne.symm hw1)
  have key : C (1 - w) * (pochInf 1 1 * pochInf w 1 * pochInf w⁻¹ 1)
      = C (1 - w) * (E5C (ψ (Jab 5 2)) + X * C (w ^ 2 + w ^ 3) * E5C (ψ (Jab 5 1))) := by
    rw [show C (1 - w) * (pochInf 1 1 * pochInf w 1 * pochInf w⁻¹ 1)
        = (1 + C (-w)) * pochInf 1 1 * pochInf w 1 * pochInf w⁻¹ 1 by rw [map_neg, map_sub, map_one]; ring, hJ]
    simp only [show (1 : ℤ) - 2 = -1 by norm_num, show (2 : ℤ) - 2 = 0 by norm_num,
      show (3 : ℤ) - 2 = 1 by norm_num, show (4 : ℤ) - 2 = 2 by norm_num, zpow_neg_one, zpow_zero, zpow_one,
      pow_zero, pow_one, map_one, one_mul, mul_one]
    rw [show ((2 : ℤ)) = ((2 : ℕ) : ℤ) by rfl, zpow_natCast, inv_neg, hwinv]
    have : C (-w ^ 4) + C ((-w) ^ 2) = C (1 - w) * C (w ^ 2 + w ^ 3) := by
      rw [← map_add, ← map_mul]; congr 1
      ring
    simp only [map_sub, map_add, map_one, map_neg, map_mul, map_pow] at this ⊢
    linear_combination (X * E5C (ψ (Jab 5 1))) * this
  have hu' : IsUnit (C (1 - w) : PowerSeries ℂ) := (isUnit_iff_ne_zero.mpr hc).map C
  exact hu'.mul_left_cancel key

lemma isUnit_Jab (a b : ℕ) (hb : 1 ≤ b) (hab : b < a) : IsUnit (Jab a b) :=
  ((RankProof.isUnit_Pinf b a hb (by omega)).mul (RankProof.isUnit_Pinf (a - b) a (by omega) (by omega))).mul
    (RankProof.isUnit_Pinf a a (by omega) (by omega))

/-- **5-dissection of `θ₄ = Σ(−1)ⁿq^{n²} = J_{2,1}`** (Garvan (3.5)). -/
theorem theta4_dissect :
    Jab 2 1 = E5 (Jab 10 5) - 2 * X * E5 (Jab 10 3) + 2 * X ^ 4 * E5 (Jab 10 1) := by
  apply MockTheta5.JTP.ψC_injective
  rw [← thL_eq_Jab le_rfl (by norm_num), thL_dissect (by norm_num) (by norm_num) (by norm_num : (1 : ℂ) ^ 5 = 1),
    Fin.sum_univ_five]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two, mul_zero, zero_add]
  have h3 : ((3 : Fin 5) : ℕ) = 3 := rfl
  have h4 : ((4 : Fin 5) : ℕ) = 4 := rfl
  have e0 : expo 2 1 ((0 : ℕ) - 2 : ℤ) = 4 := by decide
  have e1 : expo 2 1 ((1 : ℕ) - 2 : ℤ) = 1 := by decide
  have e2 : expo 2 1 ((2 : ℕ) - 2 : ℤ) = 0 := by decide
  have e3 : expo 2 1 ((3 : ℕ) - 2 : ℤ) = 1 := by decide
  have e4 : expo 2 1 ((4 : ℕ) - 2 : ℤ) = 4 := by decide
  rw [h3, h4]
  push_cast at e0 e1 e2 e3 e4 ⊢
  rw [e0, e1, e2, e3, e4, thL_eq_Jab (by norm_num) (by norm_num), thL_eq_Jab (by norm_num) (by norm_num),
    thL_eq_Jab (by norm_num) (by norm_num), thL_eq_Jab (by norm_num) (by norm_num), thL_eq_Jab (by norm_num) (by norm_num),
    show Jab 10 7 = Jab 10 3 from Jab_symm 10 3 (by norm_num), show Jab 10 9 = Jab 10 1 from Jab_symm 10 1 (by norm_num)]
  simp only [map_add, map_sub, map_mul, map_pow, map_ofNat, PowerSeries.map_X, ψ_E5]
  norm_num
  ring

/-- `E(q)E(q⁵) = F(ω)F(ω²)`, read off in 5-dissected form. -/
theorem EE5_eq : qfacInf * E5 qfacInf
    = E5 (Jab 5 2 ^ 2) - X * E5 (Jab 5 2 * Jab 5 1) - X ^ 2 * E5 (Jab 5 1 ^ 2) := by
  apply MockTheta5.JTP.ψC_injective
  have hw := MockTheta5.JTP.ω5_pow5
  have hphi := MockTheta5.JTP.phi_of_prim ω5_prim
  have hw2 : (ω5 ^ 2) ^ 5 = 1 := by rw [← pow_mul, mul_comm, pow_mul, hw, one_pow]
  have h1 := F_dissect hw (ω5_prim.ne_one (by norm_num))
  have h2 := F_dissect hw2 (ω5_prim.pow_ne_one_of_pos_of_lt (by norm_num) (by norm_num))
  have hroots := prod_pochInf_roots
  have hi1 : ω5⁻¹ = ω5 ^ 4 := ω5_inv
  have hi2 : (ω5 ^ 2)⁻¹ = ω5 ^ 3 := inv_eq_of_mul_eq_one_right (by rw [← pow_add, hw])
  rw [hi1] at h1
  rw [hi2] at h2
  have hprod : (pochInf 1 1 * pochInf ω5 1 * pochInf (ω5 ^ 4) 1) * (pochInf 1 1 * pochInf (ω5 ^ 2) 1 * pochInf (ω5 ^ 3) 1)
      = ψ (qfacInf * E5 qfacInf) := by
    rw [map_mul, ← pochInf_one_eq, ← hroots]; ring
  rw [← hprod, h1, h2]
  simp only [map_sub, map_mul, map_pow, PowerSeries.map_X, ψ_E5]
  have hs : (C (ω5 ^ 2 + ω5 ^ 3) : PowerSeries ℂ) + C ((ω5 ^ 2) ^ 2 + (ω5 ^ 2) ^ 3) = -1 := by
    rw [← map_add, ← map_one C, ← map_neg]; congr 1; linear_combination hphi + ω5 * hw
  have ht : (C (ω5 ^ 2 + ω5 ^ 3) : PowerSeries ℂ) * C ((ω5 ^ 2) ^ 2 + (ω5 ^ 2) ^ 3) = -1 := by
    rw [← map_mul, ← map_one C, ← map_neg]; congr 1
    linear_combination hphi + (ω5 + ω5 ^ 2 + ω5 ^ 3 + ω5 ^ 4) * hw
  linear_combination (X * E5C (ψ (Jab 5 2)) * E5C (ψ (Jab 5 1))) * hs
    + (X ^ 2 * E5C (ψ (Jab 5 1)) ^ 2) * ht

open MockTheta5.JTP (dis5_normal dis5_E5_mul dis5_one_qfacInf isUnit_eQ alphaQ_mul_betaQ qfacInf_dissection E5_X) in
/-- `J_{5,2} J_{5,1} = E(q)E(q⁵)` and Ramanujan's `α = J_{5,2}/J_{5,1}` (in the `q⁵`-variable). -/
theorem J5_relations : Jab 5 2 * Jab 5 1 = qfacInf * eQ ∧ Jab 5 2 = alphaQ * Jab 5 1 := by
  have hn : qfacInf * E5 qfacInf = E5 (Jab 5 2 ^ 2) + X * E5 (-(Jab 5 2 * Jab 5 1)) + X ^ 2 * E5 (-(Jab 5 1 ^ 2))
      + X ^ 3 * E5 0 + X ^ 4 * E5 0 := by
    rw [EE5_eq]; simp only [map_neg, map_zero]; ring
  have h1 := congrArg (dis5 1) hn
  have h0 := congrArg (dis5 0) hn
  rw [mul_comm qfacInf, dis5_E5_mul 1 (by norm_num), dis5_normal 1 (by norm_num), dis5_one_qfacInf] at h1
  rw [mul_comm qfacInf, dis5_E5_mul 0 (by norm_num), dis5_normal 0 (by norm_num)] at h0
  rw [if_neg one_ne_zero, if_pos rfl] at h1
  rw [if_pos rfl] at h0
  have hprod : Jab 5 2 * Jab 5 1 = qfacInf * eQ := by
    rw [show eQ = E5 qfacInf from rfl]; linear_combination h1
  refine ⟨hprod, ?_⟩
  have hd0 : MockTheta5.JTP.dis5 0 qfacInf = alphaQ * eQ := by
    rw [alphaQ, mul_assoc, Ring.inverse_mul_cancel _ isUnit_eQ, mul_one]
  rw [hd0] at h0
  have hu := isUnit_Jab 5 2 (by norm_num) (by norm_num)
  apply hu.mul_left_cancel
  linear_combination -h0 - alphaQ * hprod

lemma betaQ_eq : Jab 5 1 = betaQ * Jab 5 2 := by
  rw [J5_relations.2, ← mul_assoc, mul_comm betaQ, MockTheta5.JTP.alphaQ_mul_betaQ, one_mul]

open MockTheta5.JTP (dis5_normal qfacInf_dissection E5_X) in
/-- classes 3 and 4 of `θ₄·E = E³/E(q²)` (Garvan (3.7), (3.8)). -/
theorem theta4E_classes :
    dis5 3 (Jab 2 1 * qfacInf) = 2 * eQ * Jab 10 3 * betaQ ∧ dis5 4 (Jab 2 1 * qfacInf) = 2 * eQ * Jab 10 1 * alphaQ := by
  have hn : Jab 2 1 * qfacInf = E5 (eQ * (Jab 10 5 * alphaQ - 2 * X * Jab 10 1))
      + X * E5 (eQ * (-Jab 10 5 - 2 * Jab 10 3 * alphaQ - 2 * X * Jab 10 1 * betaQ))
      + X ^ 2 * E5 (eQ * (-(Jab 10 5 * betaQ) + 2 * Jab 10 3))
      + X ^ 3 * E5 (eQ * (2 * Jab 10 3 * betaQ)) + X ^ 4 * E5 (eQ * (2 * Jab 10 1 * alphaQ)) := by
    rw [theta4_dissect, qfacInf_dissection, show eQ = E5 qfacInf from rfl]
    simp only [map_add, map_sub, map_mul, map_neg, map_ofNat, E5_X]
    ring
  rw [hn, dis5_normal 3 (by norm_num), dis5_normal 4 (by norm_num)]
  simp only [show (3 : ℕ) ≠ 0 by norm_num, show (3 : ℕ) ≠ 1 by norm_num, show (3 : ℕ) ≠ 2 by norm_num,
    show (4 : ℕ) ≠ 0 by norm_num, show (4 : ℕ) ≠ 1 by norm_num, show (4 : ℕ) ≠ 2 by norm_num,
    show (4 : ℕ) ≠ 3 by norm_num, if_false, if_true]
  constructor <;> ring

end Concrete

end CrankProof
