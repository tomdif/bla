/-
# Ramanujan's mod-7 identity  `Σ p(7n+5) qⁿ = 7·(q⁷;q⁷)³/(q;q)⁴ + 49q·(q⁷;q⁷)⁷/(q;q)⁸`
-/
import RamanujanTau.CrankWinquistMod11
import RamanujanTau.MockTheta5PartitionCongruence7
import RamanujanTau.Ramanujan7Norm

set_option autoImplicit false

namespace MockTheta5.JTP
open PowerSeries MockTheta5.Bailey

/-! ## S1: the 7-dissection operator -/

noncomputable def dis7 (r : ℕ) (f : PowerSeries ℤ) : PowerSeries ℤ := mk fun n => coeff (7 * n + r) f

@[simp] lemma coeff_dis7 (r n : ℕ) (f : PowerSeries ℤ) : coeff n (dis7 r f) = coeff (7 * n + r) f := by
  rw [dis7, coeff_mk]

lemma dis7_add (r : ℕ) (f g : PowerSeries ℤ) : dis7 r (f + g) = dis7 r f + dis7 r g := by ext n; simp

lemma coeff_E7 (n : ℕ) (f : PowerSeries ℤ) : coeff n (E7 f) = if 7 ∣ n then coeff (n / 7) f else 0 := by
  have h : E7 f = PowerSeries.expand 7 (by norm_num) f := rfl
  rw [h, PowerSeries.coeff_expand]

lemma coeff_X_pow_mul_E7 (j r n : ℕ) (hj : j < 7) (hr : r < 7) (F : PowerSeries ℤ) :
    coeff (7 * n + r) (X ^ j * E7 F) = if j = r then coeff n F else 0 := by
  rw [coeff_X_pow_mul', coeff_E7]
  by_cases hjr : j = r
  · subst hjr
    rw [if_pos (by omega), if_pos (by omega), if_pos rfl, show (7 * n + j - j) / 7 = n by omega]
  · rw [if_neg hjr]
    by_cases h1 : j ≤ 7 * n + r
    · rw [if_pos h1]
      by_cases h2 : 7 ∣ 7 * n + r - j
      · exfalso; omega
      · rw [if_neg h2]
    · rw [if_neg h1]

lemma dis7_X_pow_mul_E7 (j r : ℕ) (hj : j < 7) (hr : r < 7) (F : PowerSeries ℤ) :
    dis7 r (X ^ j * E7 F) = if j = r then F else 0 := by
  ext n
  rw [coeff_dis7, coeff_X_pow_mul_E7 j r n hj hr]
  split_ifs <;> simp

/-- the dissection of a normal form `Σ_{j<7} q^j F_j(q⁷)`. -/
lemma dis7_normal (r : ℕ) (hr : r < 7) (F0 F1 F2 F3 F4 F5 F6 : PowerSeries ℤ) :
    dis7 r (X ^ 0 * E7 F0 + X ^ 1 * E7 F1 + X ^ 2 * E7 F2 + X ^ 3 * E7 F3 + X ^ 4 * E7 F4 + X ^ 5 * E7 F5
      + X ^ 6 * E7 F6)
      = if r = 0 then F0 else if r = 1 then F1 else if r = 2 then F2 else if r = 3 then F3
        else if r = 4 then F4 else if r = 5 then F5 else F6 := by
  simp only [dis7_add, dis7_X_pow_mul_E7 0 r (by norm_num) hr, dis7_X_pow_mul_E7 1 r (by norm_num) hr,
    dis7_X_pow_mul_E7 2 r (by norm_num) hr, dis7_X_pow_mul_E7 3 r (by norm_num) hr,
    dis7_X_pow_mul_E7 4 r (by norm_num) hr, dis7_X_pow_mul_E7 5 r (by norm_num) hr,
    dis7_X_pow_mul_E7 6 r (by norm_num) hr]
  interval_cases r <;> simp

lemma eq_dissection7 (f : PowerSeries ℤ) :
    f = X ^ 0 * E7 (dis7 0 f) + X ^ 1 * E7 (dis7 1 f) + X ^ 2 * E7 (dis7 2 f) + X ^ 3 * E7 (dis7 3 f)
      + X ^ 4 * E7 (dis7 4 f) + X ^ 5 * E7 (dis7 5 f) + X ^ 6 * E7 (dis7 6 f) := by
  ext k
  obtain ⟨n, r, hr, rfl⟩ : ∃ n r, r < 7 ∧ k = 7 * n + r := ⟨k / 7, k % 7, Nat.mod_lt _ (by norm_num), by omega⟩
  simp only [map_add, coeff_X_pow_mul_E7 0 r n (by norm_num) hr, coeff_X_pow_mul_E7 1 r n (by norm_num) hr,
    coeff_X_pow_mul_E7 2 r n (by norm_num) hr, coeff_X_pow_mul_E7 3 r n (by norm_num) hr,
    coeff_X_pow_mul_E7 4 r n (by norm_num) hr, coeff_X_pow_mul_E7 5 r n (by norm_num) hr,
    coeff_X_pow_mul_E7 6 r n (by norm_num) hr, coeff_dis7]
  interval_cases r <;> simp

lemma dis7_E7_mul (r : ℕ) (hr : r < 7) (F G : PowerSeries ℤ) : dis7 r (E7 F * G) = F * dis7 r G := by
  conv_lhs => rw [eq_dissection7 G]
  rw [show E7 F * (X ^ 0 * E7 (dis7 0 G) + X ^ 1 * E7 (dis7 1 G) + X ^ 2 * E7 (dis7 2 G) + X ^ 3 * E7 (dis7 3 G)
        + X ^ 4 * E7 (dis7 4 G) + X ^ 5 * E7 (dis7 5 G) + X ^ 6 * E7 (dis7 6 G))
      = X ^ 0 * E7 (F * dis7 0 G) + X ^ 1 * E7 (F * dis7 1 G) + X ^ 2 * E7 (F * dis7 2 G)
        + X ^ 3 * E7 (F * dis7 3 G) + X ^ 4 * E7 (F * dis7 4 G) + X ^ 5 * E7 (F * dis7 5 G)
        + X ^ 6 * E7 (F * dis7 6 G) by simp only [map_mul]; ring,
    dis7_normal r hr]
  interval_cases r <;> simp

/-! ## S2: empty classes, and the class-2 component `−E(q⁴⁹)` -/

lemma pent_mod7_ne3 : ∀ x : ZMod 7, x * (3 * x - 1) ≠ 2 * 3 := by decide
lemma pent_mod7_ne4 : ∀ x : ZMod 7, x * (3 * x - 1) ≠ 2 * 4 := by decide
lemma pent_mod7_ne6 : ∀ x : ZMod 7, x * (3 * x - 1) ≠ 2 * 6 := by decide
lemma pent_mod7_two : ∀ x : ZMod 7, x * (3 * x - 1) = 2 * 2 → x = 6 := by decide
lemma tri_mod7_ne2 : ∀ x : ZMod 7, x * (x + 1) ≠ 2 * 2 := by decide
lemma tri_mod7_ne4 : ∀ x : ZMod 7, x * (x + 1) ≠ 2 * 4 := by decide
lemma tri_mod7_ne5 : ∀ x : ZMod 7, x * (x + 1) ≠ 2 * 5 := by decide

lemma pent_cast7 {s : ℤ} {n r : ℕ} (h : pentE s = 7 * n + r) :
    (s : ZMod 7) * (3 * (s : ZMod 7) - 1) = 2 * (r : ZMod 7) := by
  have h2 := two_mul_pentE s
  rw [h] at h2
  have := congrArg (fun z : ℤ => (z : ZMod 7)) h2
  push_cast at this
  rw [← this, show (7 : ZMod 7) = 0 from rfl]; ring

lemma dis7_qfacInf_empty (r : ℕ) (hr : r = 3 ∨ r = 4 ∨ r = 6) : dis7 r qfacInf = 0 := by
  ext n
  rw [coeff_dis7, map_zero, coeff_qfacInf_box (7 * n + r) (7 * n + r) le_rfl]
  refine Finset.sum_eq_zero fun s _ => if_neg fun h => ?_
  have hc := pent_cast7 h
  rcases hr with rfl | rfl | rfl
  · exact pent_mod7_ne3 _ (by simpa using hc)
  · exact pent_mod7_ne4 _ (by simpa using hc)
  · exact pent_mod7_ne6 _ (by simpa using hc)

lemma dis7_qfacInf_cube_empty (r : ℕ) (hr : r = 2 ∨ r = 4 ∨ r = 5) : dis7 r (qfacInf ^ 3) = 0 := by
  ext n
  rw [coeff_dis7, map_zero, jacobi_cube_identity]
  by_contra h0
  obtain ⟨m, hm, _⟩ := coeff_jacobiCubeSum_value h0
  have hcast : ((m : ZMod 7) * ((m : ZMod 7) + 1)) = 2 * (r : ZMod 7) := by
    have := congrArg (fun z : ℤ => (z : ZMod 7)) hm
    push_cast at this
    rw [this, show (7 : ZMod 7) = 0 from rfl]; ring
  rcases hr with rfl | rfl | rfl
  · exact tri_mod7_ne2 _ (by simpa using hcast)
  · exact tri_mod7_ne4 _ (by simpa using hcast)
  · exact tri_mod7_ne5 _ (by simpa using hcast)

lemma sgn_seven_mul (s : ℤ) : sgn (7 * s) = sgn s := by
  unfold sgn
  simp [Int.even_mul, show ¬ Even (7 : ℤ) by decide]

lemma pentE_seven_mul_sub_one (m : ℤ) : pentE (7 * m - 1) = 49 * pentE m + 2 := by
  have h1 := two_mul_pentE (7 * m - 1)
  have h2 := two_mul_pentE m
  have : ((pentE (7 * m - 1) : ℕ) : ℤ) = 49 * (pentE m : ℤ) + 2 := by nlinarith
  exact_mod_cast this

lemma coeff_E7_qfacInf_box (n : ℕ) :
    coeff n (E7 qfacInf) = ∑ m ∈ Finset.Icc (-((n : ℤ) + 1)) ((n : ℤ) + 1),
      if 7 * pentE m = n then sgn m else 0 := by
  rw [coeff_E7]
  split_ifs with h7
  · rw [coeff_qfacInf_box (n / 7) n (Nat.div_le_self _ _)]
    refine Finset.sum_congr rfl fun m _ => ?_
    have : pentE m = n / 7 ↔ 7 * pentE m = n := by omega
    simp only [this]
  · symm; exact Finset.sum_eq_zero fun m _ => if_neg fun h => h7 ⟨pentE m, h.symm⟩

/-- **the class-2 component of Euler's product**: `Σ_n [q^{7n+2}]E · Qⁿ = −E(Q⁷)`. -/
lemma dis7_two_qfacInf : dis7 2 qfacInf = -E7 qfacInf := by
  ext n
  rw [coeff_dis7, map_neg, coeff_qfacInf_box (7 * n + 2) (7 * n + 2) le_rfl, coeff_E7_qfacInf_box,
    ← Finset.sum_neg_distrib]
  symm
  refine Finset.sum_bij_ne_zero (fun m _ _ => 7 * m - 1) ?_ ?_ ?_ ?_
  · intro m _ hne
    have hc : 7 * pentE m = n := by by_contra hc; exact hne (by rw [if_neg hc, neg_zero])
    have hp := two_mul_pentE (7 * m - 1)
    rw [pentE_seven_mul_sub_one] at hp
    have hc' : 7 * (pentE m : ℤ) = n := by exact_mod_cast hc
    have hb := pent_bound (t := 7 * m - 1) (k := 7 * (n : ℤ) + 2) (by push_cast at hp; nlinarith)
    simp only [Finset.mem_Icc]; push_cast; constructor <;> linarith [hb.1, hb.2]
  · intro a _ _ b _ _ hab; linarith
  · intro s hs hne
    have hc : pentE s = 7 * n + 2 := by by_contra hc; exact hne (if_neg hc)
    have hmod : (s : ZMod 7) = 6 := pent_mod7_two _ (by simpa using pent_cast7 hc)
    obtain ⟨m, hm⟩ : ∃ m : ℤ, s = 7 * m - 1 := by
      have : (7 : ℤ) ∣ s + 1 := by
        have h0 : ((s + 1 : ℤ) : ZMod 7) = 0 := by push_cast; rw [hmod]; rfl
        have := (ZMod.intCast_zmod_eq_zero_iff_dvd (s + 1) 7).mp h0
        exact_mod_cast this
      obtain ⟨c, hc'⟩ := this
      exact ⟨c, by linarith⟩
    subst hm
    have hpm : 7 * pentE m = n := by
      have := pentE_seven_mul_sub_one m
      omega
    refine ⟨m, ?_, ?_, rfl⟩
    · have h2' := two_mul_pentE m
      have hpm' : 7 * (pentE m : ℤ) = n := by exact_mod_cast hpm
      have hP : (0 : ℤ) ≤ (pentE m : ℤ) := Nat.cast_nonneg _
      have hb := pent_bound (t := m) (k := (n : ℤ)) (by rw [← h2']; linarith)
      simp only [Finset.mem_Icc]; constructor <;> linarith [hb.1, hb.2]
    · rw [if_pos hpm]; simp [sgn]; split_ifs <;> norm_num
  · intro m _ hne
    have hc : 7 * pentE m = n := by by_contra hc; exact hne (by rw [if_neg hc, neg_zero])
    rw [if_pos hc, if_pos (show pentE (7 * m - 1) = 7 * n + 2 by rw [pentE_seven_mul_sub_one]; omega)]
    show -sgn m = sgn (7 * m - 1)
    rw [sub_eq_add_neg, sgn_add, sgn_seven_mul, show sgn (-1 : ℤ) = -1 by decide]
    ring

/-! ## S3: the dissection `E = E(q⁴⁹)(x − qy − q² + q⁵z)` and the cube relations -/

noncomputable def eQ7 : PowerSeries ℤ := E7 qfacInf
noncomputable def xQ : PowerSeries ℤ := dis7 0 qfacInf * Ring.inverse eQ7
noncomputable def yQ : PowerSeries ℤ := -(dis7 1 qfacInf * Ring.inverse eQ7)
noncomputable def zQ : PowerSeries ℤ := dis7 5 qfacInf * Ring.inverse eQ7

lemma isUnit_eQ7 : IsUnit eQ7 := isUnit_qfacInf.map E7

/-- the dissected factor `x − qy − q² + q⁵z`. -/
noncomputable def L7 (a b c : PowerSeries ℤ) : PowerSeries ℤ := E7 a - X * E7 b - X ^ 2 + X ^ 5 * E7 c

theorem qfacInf_dissection7 : qfacInf = E7 eQ7 * L7 xQ yQ zQ := by
  have hu : eQ7 * Ring.inverse eQ7 = 1 := Ring.mul_inverse_cancel _ isUnit_eQ7
  conv_lhs => rw [eq_dissection7 qfacInf]
  rw [dis7_two_qfacInf, dis7_qfacInf_empty 3 (by norm_num), dis7_qfacInf_empty 4 (by norm_num),
    dis7_qfacInf_empty 6 (by norm_num)]
  have h0 : E7 (dis7 0 qfacInf) = E7 eQ7 * E7 xQ := by
    rw [← map_mul, xQ, ← mul_assoc, mul_comm eQ7, mul_assoc, hu, mul_one]
  have h1 : E7 (dis7 1 qfacInf) = -(E7 eQ7 * E7 yQ) := by
    rw [← map_mul, yQ, mul_neg, ← mul_assoc, mul_comm eQ7, mul_assoc, hu, mul_one, map_neg, neg_neg]
  have h5 : E7 (dis7 5 qfacInf) = E7 eQ7 * E7 zQ := by
    rw [← map_mul, zQ, ← mul_assoc, mul_comm eQ7, mul_assoc, hu, mul_one]
  rw [h0, h1, h5, L7]
  simp only [map_neg, map_zero, mul_zero, add_zero]
  unfold eQ7
  ring

/-- the cube of the dissected factor, in normal form (computed by sympy). -/
lemma cube_normal7 (a b c : PowerSeries ℤ) :
    L7 a b c ^ 3
      = X ^ 0 * E7 (a ^ 3 + X * (3 * b ^ 2 * c - 6 * a * c)) + X ^ 1 * E7 (X ^ 2 * c ^ 3 + 6 * X * b * c - 3 * a ^ 2 * b)
        + X ^ 2 * E7 (-3 * a ^ 2 + 3 * a * b ^ 2 + 3 * X * c) + X ^ 3 * E7 (3 * X * a * c ^ 2 + 6 * a * b - b ^ 3)
        + X ^ 4 * E7 (3 * a - 3 * b ^ 2 - 3 * X * b * c ^ 2) + X ^ 5 * E7 (3 * a ^ 2 * c - 3 * b - 3 * X * c ^ 2)
        + X ^ 6 * E7 (-6 * a * b * c - 1) := by
  simp only [L7, map_sub, map_add, map_mul, map_pow, map_neg, map_one, map_ofNat, E7_X]
  ring

lemma cube_class (r : ℕ) (hr : r < 7) (c : PowerSeries ℤ)
    (h : dis7 r (L7 xQ yQ zQ ^ 3) = c) (h0 : dis7 r (qfacInf ^ 3) = 0) : c = 0 := by
  rw [qfacInf_dissection7, mul_pow, ← map_pow, dis7_E7_mul r hr, h] at h0
  exact (mul_eq_zero.mp h0).resolve_left (pow_ne_zero _ isUnit_eQ7.ne_zero)

lemma three_mul_eq_zero {f : PowerSeries ℤ} (h : 3 * f = 0) : f = 0 :=
  (mul_eq_zero.mp h).resolve_left (by
    intro h3; have := congrArg (coeff 0) h3
    rw [show (3 : PowerSeries ℤ) = PowerSeries.C (3 : ℤ) from (map_ofNat PowerSeries.C 3).symm, coeff_C,
      if_pos rfl, map_zero] at this
    norm_num at this)

/-- the three cube relations (classes 2, 4, 5 of Jacobi's `E³` are empty). -/
theorem rel2 : xQ ^ 2 - xQ * yQ ^ 2 - X * zQ = 0 := by
  have h := cube_class 2 (by norm_num) _ (by rw [cube_normal7, dis7_normal 2 (by norm_num)])
    (dis7_qfacInf_cube_empty 2 (by norm_num))
  norm_num at h
  exact three_mul_eq_zero (by linear_combination -h)

theorem rel4 : xQ - yQ ^ 2 - X * yQ * zQ ^ 2 = 0 := by
  have h := cube_class 4 (by norm_num) _ (by rw [cube_normal7, dis7_normal 4 (by norm_num)])
    (dis7_qfacInf_cube_empty 4 (by norm_num))
  norm_num at h
  exact three_mul_eq_zero (by linear_combination h)

theorem rel5 : xQ ^ 2 * zQ - yQ - X * zQ ^ 2 = 0 := by
  have h := cube_class 5 (by norm_num) _ (by rw [cube_normal7, dis7_normal 5 (by norm_num)])
    (dis7_qfacInf_cube_empty 5 (by norm_num))
  norm_num at h
  exact three_mul_eq_zero (by linear_combination h)

lemma coeff_zero_inv_eQ7 : coeff 0 (Ring.inverse eQ7) = 1 := by
  have hm : eQ7 * Ring.inverse eQ7 = 1 := Ring.mul_inverse_cancel _ isUnit_eQ7
  have := congrArg (coeff 0) hm
  rw [PowerSeries.coeff_mul, Finset.antidiagonal_zero, Finset.sum_singleton, coeff_one, if_pos rfl,
    eQ7, coeff_E7, if_pos (dvd_zero 7)] at this
  simpa [coeff_zero_qfacInf] using this

lemma coeff_zero_xQ : coeff 0 xQ = 1 := by
  rw [xQ, PowerSeries.coeff_mul, Finset.antidiagonal_zero, Finset.sum_singleton, coeff_dis7, coeff_zero_inv_eQ7]
  simp [coeff_zero_qfacInf]

lemma coeff_zero_yQ : coeff 0 yQ = 1 := by
  rw [yQ, map_neg, PowerSeries.coeff_mul, Finset.antidiagonal_zero, Finset.sum_singleton, coeff_dis7,
    coeff_zero_inv_eQ7, show 7 * 0 + 1 = 1 from rfl, coeff_qfacInf_box 1 1 le_rfl]
  decide

lemma isUnit_xQ : IsUnit xQ := by
  rw [PowerSeries.isUnit_iff_constantCoeff, ← PowerSeries.coeff_zero_eq_constantCoeff_apply, coeff_zero_xQ]
  exact isUnit_one

/-- **`xyz = 1`**: `x·(xyz − 1) = −R₄ + y·R₅`, and `x` is a unit. -/
theorem xyz_eq_one : xQ * yQ * zQ = 1 := by
  have h : xQ * (xQ * yQ * zQ - 1) = 0 := by linear_combination -rel4 + yQ * rel5
  exact sub_eq_zero.mp ((mul_eq_zero.mp h).resolve_left isUnit_xQ.ne_zero)

/-! ## S4: the norm over the seventh roots of unity -/

lemma rescale_ψC_E7 (c : ℂ) (hc : c ^ 7 = 1) (F : PowerSeries ℤ) : rescale c (ψC (E7 F)) = ψC (E7 F) := by
  ext n
  rw [coeff_rescale, PowerSeries.coeff_map, coeff_E7]
  split_ifs with h
  · obtain ⟨m, rfl⟩ := h; rw [pow_mul, hc, one_pow, one_mul]
  · simp

lemma prod_rescale_factor7 (n : ℕ) :
    ∏ j ∈ Finset.range 7, (1 - PowerSeries.C ((CrankProof.ω7 ^ j) ^ n) * X ^ n : PowerSeries ℂ)
      = if 7 ∣ n then (1 - X ^ n) ^ 7 else 1 - X ^ (7 * n) := by
  split_ifs with h
  · have hj : ∀ j, (CrankProof.ω7 ^ j) ^ n = 1 := by
      intro j; obtain ⟨m, rfl⟩ := h
      rw [← pow_mul, show j * (7 * m) = 7 * (j * m) by ring, pow_mul, CrankProof.ω7_pow7, one_pow]
    simp only [hj, map_one, one_mul, Finset.prod_const, Finset.card_range]
  · have hcop : n.Coprime 7 := Nat.coprime_comm.mp ((Nat.Prime.coprime_iff_not_dvd (by norm_num)).mpr h)
    have hζ := CrankProof.ω7_prim.pow_of_coprime n hcop
    have hrw : ∀ j, PowerSeries.C ((CrankProof.ω7 ^ j) ^ n) = PowerSeries.C ((CrankProof.ω7 ^ n) ^ j) := by
      intro j; rw [← pow_mul, mul_comm, pow_mul]
    simp only [hrw]
    rw [CrankProof.prod_one_sub_root_gen hζ (by norm_num), ← pow_mul, Nat.mul_comm]

noncomputable def Gfac7 (n : ℕ) : PowerSeries ℂ := if 7 ∣ n then (1 - X ^ n) ^ 7 else 1 - X ^ (7 * n)

lemma rescale_ψC_qfac_prod7 (N : ℕ) :
    ∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC (qfac N)) = ∏ k ∈ Finset.range N, Gfac7 (k + 1) := by
  have h1 : ∀ j, rescale (CrankProof.ω7 ^ j) (ψC (qfac N))
      = ∏ k ∈ Finset.range N, (1 - PowerSeries.C ((CrankProof.ω7 ^ j) ^ (k + 1)) * X ^ (k + 1)) := by
    intro j
    rw [qfac, map_prod, map_prod]
    refine Finset.prod_congr rfl fun k _ => ?_
    rw [map_sub, map_one, map_pow, PowerSeries.map_X, map_sub, map_one, map_pow, rescale_X, mul_pow,
      ← map_pow]
  simp only [h1]
  rw [Finset.prod_comm]
  refine Finset.prod_congr rfl fun k _ => ?_
  rw [prod_rescale_factor7, Gfac7]

lemma block_identity7 : ∀ M : ℕ,
    (∏ k ∈ Finset.range (7 * M), Gfac7 (k + 1)) * ∏ i ∈ Finset.range M, (1 - (X : PowerSeries ℂ) ^ (49 * (i + 1)))
      = (∏ k ∈ Finset.range (7 * M), (1 - (X : PowerSeries ℂ) ^ (7 * (k + 1))))
        * (∏ i ∈ Finset.range M, (1 - (X : PowerSeries ℂ) ^ (7 * (i + 1)))) ^ 7 := by
  intro M
  induction M with
  | zero => simp
  | succ M ih =>
      rw [show 7 * (M + 1) = 7 * M + 7 by ring, Finset.prod_range_add (fun k => Gfac7 (k + 1)),
        Finset.prod_range_add (fun k => 1 - (X : PowerSeries ℂ) ^ (7 * (k + 1))),
        Finset.prod_range_succ (fun i => 1 - (X : PowerSeries ℂ) ^ (49 * (i + 1))) M,
        Finset.prod_range_succ (fun i => 1 - (X : PowerSeries ℂ) ^ (7 * (i + 1))) M]
      simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul]
      have g1 : Gfac7 (7 * M + 0 + 1) = 1 - X ^ (7 * (7 * M + 0 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g2 : Gfac7 (7 * M + 1 + 1) = 1 - X ^ (7 * (7 * M + 1 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g3 : Gfac7 (7 * M + 2 + 1) = 1 - X ^ (7 * (7 * M + 2 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g4 : Gfac7 (7 * M + 3 + 1) = 1 - X ^ (7 * (7 * M + 3 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g5 : Gfac7 (7 * M + 4 + 1) = 1 - X ^ (7 * (7 * M + 4 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g6 : Gfac7 (7 * M + 5 + 1) = 1 - X ^ (7 * (7 * M + 5 + 1)) := by rw [Gfac7, if_neg (by omega)]
      have g7 : Gfac7 (7 * M + 6 + 1) = (1 - X ^ (7 * M + 6 + 1)) ^ 7 := by rw [Gfac7, if_pos (by omega)]
      rw [g1, g2, g3, g4, g5, g6, g7]
      have e1 : (X : PowerSeries ℂ) ^ (49 * (M + 1)) = X ^ (7 * (7 * M + 6 + 1)) := by ring
      have e2 : (X : PowerSeries ℂ) ^ (7 * (M + 1)) = X ^ (7 * M + 6 + 1) := by ring
      rw [e1, e2]
      linear_combination (1 - X ^ (7 * (7 * M + 0 + 1))) * (1 - X ^ (7 * (7 * M + 1 + 1)))
        * (1 - X ^ (7 * (7 * M + 2 + 1))) * (1 - X ^ (7 * (7 * M + 3 + 1))) * (1 - X ^ (7 * (7 * M + 4 + 1)))
        * (1 - X ^ (7 * (7 * M + 5 + 1))) * (1 - X ^ (7 * (7 * M + 6 + 1))) * (1 - X ^ (7 * M + 6 + 1)) ^ 7 * ih

lemma X_pow_dvd_E7 {K : ℕ} {f : PowerSeries ℤ} (h : X ^ K ∣ f) : (X : PowerSeries ℤ) ^ K ∣ E7 f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨X ^ (6 * K) * E7 d, by rw [map_mul, map_pow, E7_X, ← pow_mul]; ring⟩

/-- **the product over the seven rescalings**: `∏_ζ E(ζq) · E(q⁴⁹) = E(q⁷)⁸`. -/
theorem prod_rescale_qfacInf7 :
    (∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC qfacInf)) * ψC (E7 (E7 qfacInf)) = ψC (E7 qfacInf) ^ 8 := by
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  set K := k + 1
  have hfin : (∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC (qfac (7 * K)))) * ψC (E7 (E7 (qfac K)))
      = ψC (E7 (qfac (7 * K))) * ψC (E7 (qfac K)) ^ 7 := by
    rw [rescale_ψC_qfac_prod7]
    have hE7E7 : ψC (E7 (E7 (qfac K))) = ∏ i ∈ Finset.range K, (1 - (X : PowerSeries ℂ) ^ (49 * (i + 1))) := by
      rw [E7_qfac, map_prod, map_prod]
      refine Finset.prod_congr rfl fun i _ => ?_
      simp only [map_sub, map_one, map_pow, E7_X, PowerSeries.map_X]
      rw [← pow_mul]; ring_nf
    have hE7 : ∀ N, ψC (E7 (qfac N)) = ∏ i ∈ Finset.range N, (1 - (X : PowerSeries ℂ) ^ (7 * (i + 1))) := by
      intro N; rw [E7_qfac, map_prod]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [map_sub, map_one, map_pow, PowerSeries.map_X]; ring_nf
    rw [hE7E7, hE7, hE7, block_identity7]
  have hL : (X : PowerSeries ℂ) ^ K ∣ (∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC qfacInf)) * ψC (E7 (E7 qfacInf))
      - (∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC (qfac (7 * K)))) * ψC (E7 (E7 (qfac K))) := by
    refine dvd_sub_mul' (CrankProof.dvd_sub_prod_gen _ _ _ fun j => ?_) ?_
    · rw [← map_sub]; exact X_pow_dvd_rescale _ (by rw [← map_sub]; exact X_pow_dvd_ψC (X_pow_dvd_qfacInf_sub K _ (by omega)))
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub, ← map_sub]; exact X_pow_dvd_E7 (X_pow_dvd_E7 (X_pow_dvd_qfacInf_sub K K le_rfl)))
  have hR : (X : PowerSeries ℂ) ^ K ∣ ψC (E7 qfacInf) ^ 8 - ψC (E7 (qfac (7 * K))) * ψC (E7 (qfac K)) ^ 7 := by
    rw [show ψC (E7 qfacInf) ^ 8 = ψC (E7 qfacInf) * ψC (E7 qfacInf) ^ 7 by ring]
    refine dvd_sub_mul' ?_ (dvd_sub_pow' 7 ?_)
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub]; exact X_pow_dvd_E7 (X_pow_dvd_qfacInf_sub K _ (by omega)))
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub]; exact X_pow_dvd_E7 (X_pow_dvd_qfacInf_sub K K le_rfl))
  have htot : (X : PowerSeries ℂ) ^ K ∣ (∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC qfacInf)) * ψC (E7 (E7 qfacInf))
      - ψC (E7 qfacInf) ^ 8 := by
    have := dvd_sub hL hR
    rwa [hfin, show ∀ a b c : PowerSeries ℂ, a - c - (b - c) = a - b by intros; ring] at this
  obtain ⟨d, hd⟩ := htot
  rw [hd, coeff_X_pow_mul', if_neg (by omega)]

/-- the norm `N(x, y, z, Q)` of the dissected factor (computed by sympy). -/
noncomputable def Nser : PowerSeries ℤ := X^5*zQ^7 - 7*X^4*yQ*zQ^5 - 7*X^3*xQ^2*yQ*zQ^4 + 7*X^3*xQ*zQ^3 + 14*X^3*yQ^2*zQ^3 + 14*X^2*xQ^3*zQ^2 + 7*X^2*xQ^2*yQ^2*zQ^2 + 7*X^2*xQ*yQ^4*zQ^2 - 14*X^2*xQ*yQ*zQ - 7*X^2*yQ^3*zQ - X^2 + 7*X*xQ^5*zQ + 7*X*xQ^4*yQ^2*zQ - 7*X*xQ^3*yQ - 14*X*xQ^2*yQ^3 - 7*X*xQ*yQ^5 - X*yQ^7 + xQ^7

/-- the dissected factor under the seven rescalings: the **norm** `N`. -/
theorem prod_rescale_dissection7 :
    ∏ j ∈ Finset.range 7, rescale (CrankProof.ω7 ^ j) (ψC qfacInf) = ψC (E7 eQ7) ^ 7 * ψC (E7 Nser) := by
  set w : PowerSeries ℂ := PowerSeries.C CrankProof.ω7
  set e := ψC (E7 eQ7)
  set A := ψC (E7 xQ)
  set B := ψC (E7 yQ)
  set Cc := ψC (E7 zQ)
  have hA : ∀ j, rescale (CrankProof.ω7 ^ j) (ψC qfacInf)
      = e * (A - w ^ j * X * B - (w ^ j * X) ^ 2 + (w ^ j * X) ^ 5 * Cc) := by
    intro j
    have hc : (CrankProof.ω7 ^ j) ^ 7 = 1 := by rw [← pow_mul, mul_comm, pow_mul, CrankProof.ω7_pow7, one_pow]
    conv_lhs => rw [qfacInf_dissection7]
    simp only [L7, map_mul, map_sub, map_add, map_pow, PowerSeries.map_X, rescale_X, rescale_ψC_E7 _ hc]
    simp only [w, A, B, Cc, e]
  have hw : w ^ 6 + w ^ 5 + w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0 := by
    have h := CrankProof.ω7_prim.geom_sum_eq_zero (by norm_num)
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one, zero_add] at h
    simp only [w, ← map_pow, ← map_add, ← map_one PowerSeries.C]
    rw [show CrankProof.ω7 ^ 6 + CrankProof.ω7 ^ 5 + CrankProof.ω7 ^ 4 + CrankProof.ω7 ^ 3 + CrankProof.ω7 ^ 2
        + CrankProof.ω7 + 1 = 0 by linear_combination h, map_zero]
  rw [Finset.prod_congr rfl fun j _ => hA j, Finset.prod_mul_distrib, Finset.prod_const, Finset.card_range]
  simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul]
  rw [norm7_poly w A B Cc X hw]
  simp only [Nser, map_sub, map_add, map_mul, map_pow, map_neg, map_ofNat, E7_X, PowerSeries.map_X]
  ring

lemma E7_injective : Function.Injective E7 := by
  intro f g h
  ext n
  have := congrArg (coeff (7 * n)) h
  rwa [coeff_E7, coeff_E7, if_pos (dvd_mul_right 7 n), if_pos (dvd_mul_right 7 n),
    Nat.mul_div_cancel_left _ (by norm_num)] at this

/-- **the norm identity**: `E(Q⁷)⁸·N = E(Q)⁸`. -/
theorem norm_identity7 : eQ7 ^ 8 * Nser = qfacInf ^ 8 := by
  have h := prod_rescale_qfacInf7
  rw [prod_rescale_dissection7, show E7 (E7 qfacInf) = E7 eQ7 from rfl] at h
  apply E7_injective
  apply ψC_injective
  rw [map_mul, map_mul, map_pow, map_pow, map_pow, map_pow, ← h, eQ7]
  ring

/-! ## S5: inversion, and Ramanujan's identity -/

/-- `F = x³y + Qy³z − Q²xz³ − 8Q`, which will be `(E(Q)/E(Q⁷))⁴`. -/
noncomputable def Fser : PowerSeries ℤ := xQ ^ 3 * yQ + X * yQ ^ 3 * zQ - X ^ 2 * xQ * zQ ^ 3 - 8 * X

/-- the inversion polynomial `∏_{j≠0}` of the dissected factor, in 7-dissection normal form (sympy). -/
noncomputable def Qpoly7 : PowerSeries ℤ :=
  X ^ 0 * E7 (10*X^3*zQ^3 + 14*X^2*yQ*zQ + 3*X*xQ^2*yQ - 2*X*yQ^5 + xQ^6) + X ^ 1 * E7 (-3*X^3*xQ*zQ^4 + X^2*zQ + 4*X*xQ^2 + 5*X*yQ^4 + xQ^5*yQ)
    + X ^ 2 * E7 (X^4*zQ^6 - 3*X^2*xQ*zQ^2 + 14*X*xQ*yQ - 10*X*yQ^3 + 2*xQ^5) + X ^ 3 * E7 (5*X^3*zQ^4 + X*xQ + X*yQ^5*zQ + 4*X*yQ^2 + 3*xQ^4*yQ)
    + X ^ 4 * E7 (-X^3*xQ*zQ^5 + 4*X^2*zQ^2 - 3*X*yQ^4*zQ - X*yQ + 5*xQ^4) + X ^ 5 * E7 (7 * (Fser + 7 * X))
    + X ^ 6 * E7 (2*X^3*zQ^5 - 14*X*xQ*zQ - 3*X*yQ^2*zQ + 10*xQ^3 + yQ^6)

lemma dis7_five_Qpoly7 : dis7 5 Qpoly7 = 7 * (Fser + 7 * X) := by
  rw [Qpoly7, dis7_normal 5 (by norm_num)]; simp

lemma rel1 : xQ * yQ * zQ - 1 = 0 := sub_eq_zero.mpr xyz_eq_one

/-- `(x − qy − q² + q⁵z)·𝒬₇ = N` (using the cube relations). -/
lemma factor_mul_Qpoly7 : L7 xQ yQ zQ * Qpoly7 = E7 Nser := by
  have h2 := congrArg E7 rel2
  have h4 := congrArg E7 rel4
  have h5 := congrArg E7 rel5
  have h1 := congrArg E7 rel1
  simp only [map_sub, map_add, map_mul, map_pow, map_ofNat, map_one, map_zero, E7_X] at h2 h4 h5 h1
  simp only [L7, Qpoly7, Fser, Nser, map_sub, map_add, map_mul, map_pow, map_ofNat, map_neg, map_one, E7_X]
  set A := E7 xQ
  set B := E7 yQ
  set Cc := E7 zQ
  linear_combination (X^19*Cc^3 + 3*X^16*Cc^2 + 21*X^14*A*Cc^2 + 7*X^14*B^2*Cc^2 + 11*X^13*Cc - 7*X^12*B*Cc + 3*X^11*A*Cc + 3*X^11*B^2*Cc - X^10*A*B*Cc - X^10*B^3*Cc + 8*X^10 + 13*X^9*B - 3*X^8*A - 6*X^8*B^2 + 7*X^7*A^3*Cc - 19*X^7*A*B - 5*X^7*B^3 + 6*X^6*A^2 - X^6*A*B^2 - X^6*B^4 + 3*X^4*A^3 + X^2*A^4) * h2 + (X^24*Cc^4 - 5*X^21*Cc^3 - X^19*A*Cc^3 + 6*X^18*Cc^2 - X^17*B*Cc^2 - 3*X^16*A*Cc^2 + 13*X^15*Cc - 7*X^14*A^2*Cc^2 + 12*X^14*B*Cc - 7*X^13*A*Cc - 8*X^12 - 6*X^11*B - 11*X^10*A - 5*X^10*B^2 + 3*X^9*B^3 - 3*X^8*A^2 - X^8*B^4 - X^6*A^3) * h4
    + (-X^25*Cc^4 - 3*X^22*Cc^3 - 6*X^19*Cc^2 + 3*X^16*Cc - 33*X^14*A*Cc - 8*X^13 + X^12*A^2*Cc + 11*X^12*B + 10*X^11*A + 8*X^10*A*B + 6*X^9*A^2 + 3*X^8*A^2*B - 12*X^7*A^3 + X^6*A^3*B + X^5*A^4) * h5 + (-8*X^14) * h1

/-- `F² = N`. -/
lemma Fser_sq : Fser ^ 2 = Nser := by
  simp only [Fser, Nser]
  linear_combination (2*X^3*yQ*zQ^4 + 21*X^2*xQ*zQ^2 + 7*X^2*yQ^2*zQ^2 + 3*X*xQ^3*zQ - 2*X*xQ^2*yQ^2*zQ - 22*X*xQ*yQ - 8*X*yQ^3 - xQ^5) * rel2 + (-10*X^3*zQ^3 - 5*X^2*xQ^2*zQ^2 + 17*X^2*yQ*zQ + 2*X*xQ^4*zQ - X*yQ^5) * rel4 + (X^4*zQ^5 - 40*X^2*xQ*zQ - 13*X*xQ^3) * rel5 + (-65*X^2) * rel1

lemma constantCoeff_eQ7 : constantCoeff eQ7 = 1 := by
  rw [← PowerSeries.coeff_zero_eq_constantCoeff_apply, eQ7, coeff_E7, if_pos (dvd_zero 7)]
  exact coeff_zero_qfacInf

/-- **`F = (E(Q)/E(Q⁷))⁴`**: the square root of the norm identity, fixed by the constant term. -/
theorem eQ7_pow4_mul_Fser : eQ7 ^ 4 * Fser = qfacInf ^ 4 := by
  have h8 : (eQ7 ^ 4 * Fser) ^ 2 = (qfacInf ^ 4) ^ 2 := by
    rw [mul_pow, Fser_sq, ← pow_mul, ← pow_mul, ← norm_identity7]
  have hfac : (eQ7 ^ 4 * Fser - qfacInf ^ 4) * (eQ7 ^ 4 * Fser + qfacInf ^ 4) = 0 := by
    linear_combination h8
  have hne : eQ7 ^ 4 * Fser + qfacInf ^ 4 ≠ 0 := by
    intro h0
    have := congrArg constantCoeff h0
    have hx : constantCoeff xQ = 1 := by rw [← PowerSeries.coeff_zero_eq_constantCoeff_apply, coeff_zero_xQ]
    have hy : constantCoeff yQ = 1 := by rw [← PowerSeries.coeff_zero_eq_constantCoeff_apply, coeff_zero_yQ]
    have hq : constantCoeff qfacInf = 1 := by
      rw [← PowerSeries.coeff_zero_eq_constantCoeff_apply]; exact coeff_zero_qfacInf
    simp only [Fser, map_add, map_sub, map_mul, map_pow, map_ofNat, constantCoeff_X, constantCoeff_eQ7, hx, hy,
      hq, map_zero] at this
    norm_num at this
  exact sub_eq_zero.mp ((mul_eq_zero.mp hfac).resolve_right hne)

lemma isUnit_eQ7_mul_Nser : IsUnit (eQ7 * Nser) := by
  have h : eQ7 ^ 7 * (eQ7 * Nser) = qfacInf ^ 8 := by rw [← norm_identity7]; ring
  have hu : IsUnit (eQ7 ^ 7 * (eQ7 * Nser)) := by rw [h]; exact isUnit_qfacInf.pow 8
  exact isUnit_of_mul_isUnit_right hu

lemma E7_inverse {u : PowerSeries ℤ} (hu : IsUnit u) : Ring.inverse (E7 u) = E7 (Ring.inverse u) :=
  (CrankProof.map_inverse E7 hu).symm

lemma partitionGF_eq_Qpoly7 : partitionGF = E7 (Ring.inverse (eQ7 * Nser)) * Qpoly7 := by
  have hE : qfacInf * Qpoly7 = E7 (eQ7 * Nser) := by
    conv_lhs => rw [qfacInf_dissection7]
    rw [mul_assoc, factor_mul_Qpoly7, ← map_mul]
  have hu := isUnit_eQ7_mul_Nser
  rw [partitionGF, ← E7_inverse hu]
  calc Ring.inverse qfacInf = Ring.inverse qfacInf * (E7 (eQ7 * Nser) * Ring.inverse (E7 (eQ7 * Nser))) := by
        rw [Ring.mul_inverse_cancel _ (hu.map E7), mul_one]
    _ = Ring.inverse qfacInf * (qfacInf * Qpoly7) * Ring.inverse (E7 (eQ7 * Nser)) := by rw [hE]; ring
    _ = Ring.inverse (E7 (eQ7 * Nser)) * Qpoly7 := by
        rw [← mul_assoc, Ring.inverse_mul_cancel _ isUnit_qfacInf, one_mul, mul_comm]

/-- **Ramanujan's mod-7 identity, generating-function form**:
`(Σ p(7n+5) qⁿ)·(q;q)⁸ = 7·(q⁷;q⁷)³·(q;q)⁴ + 49q·(q⁷;q⁷)⁷`. -/
theorem dis7_five_partitionGF_mul :
    dis7 5 partitionGF * qfacInf ^ 8 = 7 * eQ7 ^ 3 * qfacInf ^ 4 + 49 * X * eQ7 ^ 7 := by
  rw [partitionGF_eq_Qpoly7, dis7_E7_mul 5 (by norm_num), dis7_five_Qpoly7, ← norm_identity7,
    ← eQ7_pow4_mul_Fser]
  have hu := Ring.inverse_mul_cancel _ isUnit_eQ7_mul_Nser
  rw [show Ring.inverse (eQ7 * Nser) * (7 * (Fser + 7 * X)) * (eQ7 ^ 8 * Nser)
      = 7 * eQ7 ^ 7 * (Fser + 7 * X) * (Ring.inverse (eQ7 * Nser) * (eQ7 * Nser)) by ring, hu, mul_one]
  ring

/-- **Ramanujan's identity for 7**: `Σ_{n≥0} p(7n+5) qⁿ · (q;q)_∞⁸ = 7(q⁷;q⁷)³(q;q)⁴ + 49q(q⁷;q⁷)⁷`. -/
theorem ramanujan_identity_mod7 :
    (PowerSeries.mk fun n => (Fintype.card (Nat.Partition (7 * n + 5)) : ℤ)) * qfacInf ^ 8
      = 7 * (E7 qfacInf) ^ 3 * qfacInf ^ 4 + 49 * X * (E7 qfacInf) ^ 7 := by
  have h : (PowerSeries.mk fun n => (Fintype.card (Nat.Partition (7 * n + 5)) : ℤ)) = dis7 5 partitionGF := by
    ext n; rw [coeff_mk, coeff_dis7, coeff_partitionGF_eq_card]
  rw [h, dis7_five_partitionGF_mul]; rfl

/-! ## S6: Ramanujan's congruence `p(49n+47) ≡ 0 (mod 49)` -/

lemma tri_mod7_six : ∀ x : ZMod 7, x * (x + 1) = 2 * 6 → 2 * x + 1 = 0 := by decide

/-- every coefficient of Jacobi's `E³` in the class `7m+6` is divisible by 7 (`2k+1 ≡ 0`). -/
lemma seven_dvd_dis7_six_cube (m : ℕ) : (7 : ℤ) ∣ coeff m (dis7 6 (qfacInf ^ 3)) := by
  rw [coeff_dis7, jacobi_cube_identity]
  by_cases h0 : coeff (7 * m + 6) jacobiCubeSum = 0
  · rw [h0]; exact dvd_zero _
  obtain ⟨k, hk, hv⟩ := coeff_jacobiCubeSum_value h0
  rw [hv]
  have hcast : ((k : ZMod 7) * ((k : ZMod 7) + 1)) = 2 * 6 := by
    have := congrArg (fun z : ℤ => (z : ZMod 7)) hk
    push_cast at this
    rw [this, show (7 : ZMod 7) = 0 from rfl]; ring
  have h2 := tri_mod7_six _ hcast
  have h7 : (7 : ℤ) ∣ 2 * (k : ℤ) + 1 := by
    have := (ZMod.intCast_zmod_eq_zero_iff_dvd (2 * (k : ℤ) + 1) 7).mp (by push_cast; exact h2)
    exact_mod_cast this
  exact Dvd.dvd.mul_left h7 _

/-- mod 7 (Frobenius): `E(q⁷)³/E⁴ ≡ E(q⁷)²·E³`. -/
lemma Ψ7_key : Ψ7 (eQ7 ^ 3 * Ring.inverse (qfacInf ^ 4)) = Ψ7 (E7 (qfacInf ^ 2) * qfacInf ^ 3) := by
  have hu := isUnit_qfacInf.map Ψ7
  rw [eQ7, map_mul, map_mul, CrankProof.map_inverse Ψ7 (isUnit_qfacInf.pow 4)]
  simp only [map_pow]
  rw [← frobenius_qfacInf7]
  set g := Ψ7 qfacInf
  have hg4 : g ^ 4 * Ring.inverse (g ^ 4) = 1 := Ring.mul_inverse_cancel _ (hu.pow 4)
  calc (g ^ 7) ^ 3 * Ring.inverse (g ^ 4) = (g ^ 7) ^ 2 * g ^ 3 * (g ^ 4 * Ring.inverse (g ^ 4)) := by ring
    _ = (g ^ 7) ^ 2 * g ^ 3 := by rw [hg4, mul_one]

lemma seven_dvd_coeff_key (n : ℕ) : (7 : ℤ) ∣ coeff (7 * n + 6) (eQ7 ^ 3 * Ring.inverse (qfacInf ^ 4)) := by
  have h1 : (7 : ℤ) ∣ coeff (7 * n + 6) (E7 (qfacInf ^ 2) * qfacInf ^ 3) := by
    rw [← coeff_dis7, dis7_E7_mul 6 (by norm_num), PowerSeries.coeff_mul]
    exact Finset.dvd_sum fun p _ => Dvd.dvd.mul_left (seven_dvd_dis7_six_cube p.2) _
  have h2 := congrArg (coeff (7 * n + 6)) Ψ7_key
  rw [Ψ7, PowerSeries.coeff_map, PowerSeries.coeff_map] at h2
  have h3 := (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ 7).mp h2
  have h4 := dvd_sub h1 h3
  rwa [sub_sub_cancel] at h4

lemma coeff_nat_mul (k n : ℕ) (f : PowerSeries ℤ) : coeff n ((k : PowerSeries ℤ) * f) = k * coeff n f := by
  rw [show (k : PowerSeries ℤ) = PowerSeries.C (k : ℤ) from (map_natCast PowerSeries.C k).symm,
    PowerSeries.coeff_C_mul]

/-- **Ramanujan's congruence modulo 49**: `49 ∣ [q^{49n+47}] (1/(q;q)∞)`. -/
theorem fortynine_dvd_coeff_partitionGF (n : ℕ) : (49 : ℤ) ∣ coeff (49 * n + 47) partitionGF := by
  have hu8 := Ring.mul_inverse_cancel _ (isUnit_qfacInf.pow 8)
  have hu4 := Ring.mul_inverse_cancel _ (isUnit_qfacInf.pow 4)
  have hinv : qfacInf ^ 4 * Ring.inverse (qfacInf ^ 8) = Ring.inverse (qfacInf ^ 4) := by
    rw [show qfacInf ^ 8 = qfacInf ^ 4 * qfacInf ^ 4 by ring, Ring.mul_inverse_rev, ← mul_assoc, hu4, one_mul]
  have hS : dis7 5 partitionGF = ((7 : ℕ) : PowerSeries ℤ) * (eQ7 ^ 3 * Ring.inverse (qfacInf ^ 4))
      + ((49 : ℕ) : PowerSeries ℤ) * (X * eQ7 ^ 7 * Ring.inverse (qfacInf ^ 8)) := by
    calc dis7 5 partitionGF = dis7 5 partitionGF * qfacInf ^ 8 * Ring.inverse (qfacInf ^ 8) := by
          rw [mul_assoc, hu8, mul_one]
      _ = (7 * eQ7 ^ 3 * qfacInf ^ 4 + 49 * X * eQ7 ^ 7) * Ring.inverse (qfacInf ^ 8) := by
          rw [dis7_five_partitionGF_mul]
      _ = _ := by rw [← hinv]; push_cast; ring
  have h0 : coeff (49 * n + 47) partitionGF = coeff (7 * n + 6) (dis7 5 partitionGF) := by
    rw [coeff_dis7]; ring_nf
  rw [h0, hS, map_add, coeff_nat_mul, coeff_nat_mul]
  push_cast
  exact dvd_add (by rw [show (49 : ℤ) = 7 * 7 by norm_num]; exact mul_dvd_mul_left 7 (seven_dvd_coeff_key n))
    (dvd_mul_right _ _)

/-- **Ramanujan's congruence** `49 ∣ p(49n+47)` for the partition count. -/
theorem fortynine_dvd_partition_card (n : ℕ) : 49 ∣ Fintype.card (Nat.Partition (49 * n + 47)) := by
  have h := fortynine_dvd_coeff_partitionGF n
  rw [coeff_partitionGF_eq_card] at h
  exact_mod_cast h

end MockTheta5.JTP
