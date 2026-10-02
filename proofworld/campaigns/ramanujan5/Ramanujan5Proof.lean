/-
# Ramanujan's "most beautiful identity"  `Σ p(5n+4) qⁿ = 5·(q⁵;q⁵)⁵/(q;q)⁶`

**`ramanujan_most_beautiful_identity`**: `(Σₙ #(Nat.Partition (5n+4)) qⁿ)·(q;q)∞⁶ = 5·(q⁵;q⁵)∞⁵`.
Axioms `propext, Classical.choice, Quot.sound` only (no `sorry`, no `native_decide`).

Ramanujan's own argument, made formal:
1. **5-dissection of Euler's product** (`qfacInf_dissection`): from the pentagonal theorem,
   `E = E(q²⁵)(A − q − q²B)` with `A = α(q⁵)`, `B = β(q⁵)`. The classes `≡ 3, 4 (mod 5)` are empty, and the
   class `≡ 1` is exactly `−q E(q²⁵)` (`dis5_one_qfacInf`, via the bijection `s = 1 − 5m`).
2. **`AB = 1`** (`alphaQ_mul_betaQ`): cube the dissection and compare with Jacobi's `E³`, which has no
   exponent `≡ 2 (mod 5)`. The class-2 part is `3A(1 − AB)`.
3. **The norm** (`norm_identity`): `∏_{ω⁵=1} E(ωq) = E(q⁵)⁶/E(q²⁵)`, proved in `ℂ⟦X⟧` with
   `PowerSeries.rescale` (finite blocks of 5 plus an `X`-adic limit). With the dissection this gives
   `A⁵ − 11q⁵ − q¹⁰B⁵ = E(q⁵)⁶/E(q²⁵)⁶`.
4. **Inversion** (`factor_mul_Qpoly`): `(A − q − q²B)·𝒬 = A⁵ − 11q⁵ − q¹⁰B⁵`. The only `q^{≡4}` term of
   `𝒬` is `5q⁴`, so the class-4 part of `1/E` is `5q⁴E(q²⁵)⁵/E(q⁵)⁶`.

**Corollary** (`twentyfive_dvd_partition_card`): Ramanujan's congruence `25 ∣ p(25n+24)`.
The identity gives `p(25m+24) = 5·[qᵐ](E⁵·[q^{5j+4}](1/E⁶))`. Mod 5, Frobenius gives `1/E⁶ ≡ (1/E)(1/E(q⁵))`,
whose class-4 part is built from the `p(5j+4) ≡ 0 (mod 5)`.
-/
import RamanujanTau.PartitionCongruenceMod11
import Mathlib.RingTheory.RootsOfUnity.Complex

set_option autoImplicit false

namespace MockTheta5.JTP
open PowerSeries MockTheta5.Bailey

/-! ## Stage R1: the 5-dissection operator -/

/-- `dis5 r f = Σ_n [q^{5n+r}] f · Qⁿ`, the `r`-th 5-dissection component (as a series in `Q = q⁵`). -/
noncomputable def dis5 (r : ℕ) (f : PowerSeries ℤ) : PowerSeries ℤ := mk fun n => coeff (5 * n + r) f

@[simp] lemma coeff_dis5 (r n : ℕ) (f : PowerSeries ℤ) : coeff n (dis5 r f) = coeff (5 * n + r) f := by
  rw [dis5, coeff_mk]

lemma dis5_add (r : ℕ) (f g : PowerSeries ℤ) : dis5 r (f + g) = dis5 r f + dis5 r g := by
  ext n; simp

lemma dis5_neg (r : ℕ) (f : PowerSeries ℤ) : dis5 r (-f) = -dis5 r f := by
  ext n; simp

lemma dis5_sub (r : ℕ) (f g : PowerSeries ℤ) : dis5 r (f - g) = dis5 r f - dis5 r g := by
  ext n; simp

lemma coeff_E5 (n : ℕ) (f : PowerSeries ℤ) : coeff n (E5 f) = if 5 ∣ n then coeff (n / 5) f else 0 := by
  have h : E5 f = PowerSeries.expand 5 (by norm_num) f := rfl
  rw [h, PowerSeries.coeff_expand]

/-- the monomial rule: `[q^{5n+r}] (q^j · F(q⁵)) = [j = r]·[Qⁿ]F` for `j, r < 5`. -/
lemma coeff_X_pow_mul_E5 (j r n : ℕ) (hj : j < 5) (hr : r < 5) (F : PowerSeries ℤ) :
    coeff (5 * n + r) (X ^ j * E5 F) = if j = r then coeff n F else 0 := by
  rw [coeff_X_pow_mul', coeff_E5]
  by_cases hjr : j = r
  · subst hjr
    rw [if_pos (by omega), if_pos (by omega), if_pos rfl, show (5 * n + j - j) / 5 = n by omega]
  · rw [if_neg hjr]
    by_cases h1 : j ≤ 5 * n + r
    · rw [if_pos h1]
      by_cases h2 : 5 ∣ 5 * n + r - j
      · exfalso; omega
      · rw [if_neg h2]
    · rw [if_neg h1]

lemma dis5_X_pow_mul_E5 (j r : ℕ) (hj : j < 5) (hr : r < 5) (F : PowerSeries ℤ) :
    dis5 r (X ^ j * E5 F) = if j = r then F else 0 := by
  ext n
  rw [coeff_dis5, coeff_X_pow_mul_E5 j r n hj hr]
  split_ifs <;> simp

lemma dis5_E5 (r : ℕ) (hr : r < 5) (F : PowerSeries ℤ) : dis5 r (E5 F) = if r = 0 then F else 0 := by
  have := dis5_X_pow_mul_E5 0 r (by norm_num) hr F
  rw [pow_zero, one_mul] at this
  rw [this]; split_ifs <;> first | rfl | omega

/-- **the dissection of a normal form** `Σ_{j<5} q^j F_j(q⁵)`. -/
lemma dis5_normal (r : ℕ) (hr : r < 5) (F0 F1 F2 F3 F4 : PowerSeries ℤ) :
    dis5 r (E5 F0 + X * E5 F1 + X ^ 2 * E5 F2 + X ^ 3 * E5 F3 + X ^ 4 * E5 F4)
      = if r = 0 then F0 else if r = 1 then F1 else if r = 2 then F2 else if r = 3 then F3 else F4 := by
  rw [show (X : PowerSeries ℤ) * E5 F1 = X ^ 1 * E5 F1 by rw [pow_one]]
  simp only [dis5_add, dis5_E5 r hr, dis5_X_pow_mul_E5 1 r (by norm_num) hr,
    dis5_X_pow_mul_E5 2 r (by norm_num) hr, dis5_X_pow_mul_E5 3 r (by norm_num) hr,
    dis5_X_pow_mul_E5 4 r (by norm_num) hr]
  interval_cases r <;> simp

/-- every series is the sum of its five dissection components. -/
lemma eq_dissection (f : PowerSeries ℤ) :
    f = E5 (dis5 0 f) + X * E5 (dis5 1 f) + X ^ 2 * E5 (dis5 2 f) + X ^ 3 * E5 (dis5 3 f)
      + X ^ 4 * E5 (dis5 4 f) := by
  ext k
  obtain ⟨n, r, hr, rfl⟩ : ∃ n r, r < 5 ∧ k = 5 * n + r := ⟨k / 5, k % 5, Nat.mod_lt _ (by norm_num), by omega⟩
  rw [show (X : PowerSeries ℤ) * E5 (dis5 1 f) = X ^ 1 * E5 (dis5 1 f) by rw [pow_one]]
  have h0 := coeff_X_pow_mul_E5 0 r n (by norm_num) hr (dis5 0 f)
  rw [pow_zero, one_mul] at h0
  simp only [map_add, h0, coeff_X_pow_mul_E5 1 r n (by norm_num) hr, coeff_X_pow_mul_E5 2 r n (by norm_num) hr,
    coeff_X_pow_mul_E5 3 r n (by norm_num) hr, coeff_X_pow_mul_E5 4 r n (by norm_num) hr, coeff_dis5]
  interval_cases r <;> simp

/-- multiplication by a `q⁵`-series commutes with dissection. -/
lemma dis5_E5_mul (r : ℕ) (hr : r < 5) (F G : PowerSeries ℤ) : dis5 r (E5 F * G) = F * dis5 r G := by
  conv_lhs => rw [eq_dissection G]
  rw [show E5 F * (E5 (dis5 0 G) + X * E5 (dis5 1 G) + X ^ 2 * E5 (dis5 2 G) + X ^ 3 * E5 (dis5 3 G)
        + X ^ 4 * E5 (dis5 4 G))
      = E5 (F * dis5 0 G) + X * E5 (F * dis5 1 G) + X ^ 2 * E5 (F * dis5 2 G)
        + X ^ 3 * E5 (F * dis5 3 G) + X ^ 4 * E5 (F * dis5 4 G) by simp only [map_mul]; ring,
    dis5_normal r hr]
  interval_cases r <;> simp

/-! ## Stage R1b: empty dissection classes of `E` and `E³` -/

lemma pent_mod5_ne3 : ∀ x : ZMod 5, x * (3 * x - 1) ≠ 1 := by decide
lemma pent_mod5_ne4 : ∀ x : ZMod 5, x * (3 * x - 1) ≠ 3 := by decide
lemma tri_mod5_ne2 : ∀ x : ZMod 5, x * (x + 1) ≠ 4 := by decide

lemma no_pent_mod5 {s : ℤ} {n : ℕ} (r : ℕ) (hr : r = 3 ∨ r = 4) (h : pentE s = 5 * n + r) : False := by
  have h2 := two_mul_pentE s
  rw [h] at h2
  have hcast : ((s : ZMod 5) * (3 * (s : ZMod 5) - 1)) = (((2 * ((5 * n + r : ℕ) : ℤ)) : ℤ) : ZMod 5) := by
    rw [h2]; push_cast; ring
  push_cast at hcast
  rw [show (5 : ZMod 5) = 0 from rfl] at hcast
  rcases hr with rfl | rfl
  · exact pent_mod5_ne3 _ (by rw [hcast]; simp only [zero_mul, zero_add]; decide)
  · exact pent_mod5_ne4 _ (by rw [hcast]; simp only [zero_mul, zero_add]; decide)

lemma dis5_qfacInf_empty (r : ℕ) (hr : r = 3 ∨ r = 4) : dis5 r qfacInf = 0 := by
  ext n
  rw [coeff_dis5, map_zero, coeff_qfacInf_box (5 * n + r) (5 * n + r) le_rfl]
  exact Finset.sum_eq_zero fun s _ => if_neg fun h => no_pent_mod5 r hr h

lemma dis5_two_qfacInf_cube : dis5 2 (qfacInf ^ 3) = 0 := by
  ext n
  rw [coeff_dis5, map_zero, jacobi_cube_identity]
  by_contra h0
  obtain ⟨m, hm, _⟩ := coeff_jacobiCubeSum_value h0
  have hcast : ((m : ZMod 5) * ((m : ZMod 5) + 1)) = ((2 * ((5 * n + 2 : ℕ) : ℤ) : ℤ) : ZMod 5) := by
    have := congrArg (fun z : ℤ => (z : ZMod 5)) hm
    push_cast at this ⊢; exact this
  push_cast at hcast
  rw [show (5 : ZMod 5) = 0 from rfl] at hcast
  exact tri_mod5_ne2 _ (by rw [hcast]; simp only [zero_mul, zero_add]; decide)


/-! ## Stage R2: the class-1 component `−E(q²⁵)` and the factorization `E = E(q²⁵)(A − q − q²B)` -/

lemma sgn_five_mul (s : ℤ) : sgn (5 * s) = sgn s := by
  unfold sgn
  simp [Int.even_mul, show ¬ Even (5 : ℤ) by decide]

lemma pentE_one_sub_five_mul (m : ℤ) : pentE (1 - 5 * m) = 25 * pentE m + 1 := by
  have h1 := two_mul_pentE (1 - 5 * m)
  have h2 := two_mul_pentE m
  have : ((pentE (1 - 5 * m) : ℕ) : ℤ) = 25 * (pentE m : ℤ) + 1 := by nlinarith
  exact_mod_cast this

lemma pent_mod5_one : ∀ x : ZMod 5, x * (3 * x - 1) = 2 → x = 1 := by decide

lemma coeff_E5_qfacInf_box (n : ℕ) :
    coeff n (E5 qfacInf) = ∑ m ∈ Finset.Icc (-((n : ℤ) + 1)) ((n : ℤ) + 1),
      if 5 * pentE m = n then sgn m else 0 := by
  rw [coeff_E5]
  split_ifs with h5
  · rw [coeff_qfacInf_box (n / 5) n (Nat.div_le_self _ _)]
    refine Finset.sum_congr rfl fun m _ => ?_
    have : pentE m = n / 5 ↔ 5 * pentE m = n := by omega
    simp only [this]
  · symm; exact Finset.sum_eq_zero fun m _ => if_neg fun h => h5 ⟨pentE m, h.symm⟩

/-- **the class-1 component of Euler's product**: `Σ_n [q^{5n+1}]E · Qⁿ = −E(Q⁵)`. -/
lemma dis5_one_qfacInf : dis5 1 qfacInf = -E5 qfacInf := by
  ext n
  rw [coeff_dis5, map_neg, coeff_qfacInf_box (5 * n + 1) (5 * n + 1) le_rfl, coeff_E5_qfacInf_box,
    ← Finset.sum_neg_distrib]
  symm
  refine Finset.sum_bij_ne_zero (fun m _ _ => 1 - 5 * m) ?_ ?_ ?_ ?_
  · intro m _ hne
    have hc : 5 * pentE m = n := by by_contra hc; exact hne (by rw [if_neg hc, neg_zero])
    have hp := two_mul_pentE (1 - 5 * m)
    rw [pentE_one_sub_five_mul] at hp
    have hc' : 5 * (pentE m : ℤ) = n := by exact_mod_cast hc
    have hb := pent_bound (t := 1 - 5 * m) (k := 5 * (n : ℤ)) (by push_cast at hp; linarith)
    simp only [Finset.mem_Icc]; push_cast; constructor <;> linarith [hb.1, hb.2]
  · intro a _ _ b _ _ hab; linarith
  · intro s hs hne
    have hc : pentE s = 5 * n + 1 := by by_contra hc; exact hne (if_neg hc)
    have h2 := two_mul_pentE s
    rw [hc] at h2
    have hmod : (s : ZMod 5) = 1 := by
      apply pent_mod5_one
      have := congrArg (fun z : ℤ => (z : ZMod 5)) h2
      push_cast at this
      rw [← this, show (5 : ZMod 5) = 0 from rfl]; ring
    obtain ⟨m, hm⟩ : ∃ m : ℤ, s = 1 - 5 * m := by
      have : (5 : ℤ) ∣ s - 1 := by
        have h0 : ((s - 1 : ℤ) : ZMod 5) = 0 := by push_cast; rw [hmod]; ring
        have := (ZMod.intCast_zmod_eq_zero_iff_dvd (s - 1) 5).mp h0
        exact_mod_cast this
      obtain ⟨c, hc'⟩ := this
      exact ⟨-c, by linarith⟩
    subst hm
    have hpm : 5 * pentE m = n := by
      have := pentE_one_sub_five_mul m
      omega
    refine ⟨m, ?_, ?_, rfl⟩
    · have h2' := two_mul_pentE m
      have hpm' : 5 * (pentE m : ℤ) = n := by exact_mod_cast hpm
      have hP : (0 : ℤ) ≤ (pentE m : ℤ) := Nat.cast_nonneg _
      have hb := pent_bound (t := m) (k := (n : ℤ)) (by rw [← h2']; linarith)
      simp only [Finset.mem_Icc]; constructor <;> linarith [hb.1, hb.2]
    · rw [if_pos hpm]; simp [sgn]; split_ifs <;> norm_num
  · intro m _ hne
    have hc : 5 * pentE m = n := by by_contra hc; exact hne (by rw [if_neg hc, neg_zero])
    rw [if_pos hc, if_pos (show pentE (1 - 5 * m) = 5 * n + 1 by rw [pentE_one_sub_five_mul]; omega)]
    show -sgn m = sgn (1 - 5 * m)
    rw [sub_eq_add_neg, sgn_add, show -(5 * m) = 5 * (-m) by ring, sgn_five_mul, sgn_neg,
      show sgn (1 : ℤ) = -1 by decide]
    ring

/-- `e = E(Q⁵)`, and the two remaining components `c₀, c₂` of Euler's product. -/
noncomputable def eQ : PowerSeries ℤ := E5 qfacInf
noncomputable def alphaQ : PowerSeries ℤ := dis5 0 qfacInf * Ring.inverse eQ
noncomputable def betaQ : PowerSeries ℤ := -(dis5 2 qfacInf * Ring.inverse eQ)

lemma isUnit_eQ : IsUnit eQ := isUnit_qfacInf.map E5

/-- **the 5-dissection of Euler's product**: `E = E(q²⁵)·(A − q − q²B)` with `A = α(q⁵)`, `B = β(q⁵)`. -/
theorem qfacInf_dissection :
    qfacInf = E5 eQ * (E5 alphaQ - X - X ^ 2 * E5 betaQ) := by
  have hu : eQ * Ring.inverse eQ = 1 := Ring.mul_inverse_cancel _ isUnit_eQ
  conv_lhs => rw [eq_dissection qfacInf]
  rw [dis5_one_qfacInf, dis5_qfacInf_empty 3 (Or.inl rfl), dis5_qfacInf_empty 4 (Or.inr rfl)]
  have h0 : E5 (dis5 0 qfacInf) = E5 eQ * E5 alphaQ := by
    rw [← map_mul, alphaQ, ← mul_assoc, mul_comm eQ, mul_assoc, hu, mul_one]
  have h2 : E5 (dis5 2 qfacInf) = -(E5 eQ * E5 betaQ) := by
    rw [← map_mul, betaQ, mul_neg, ← mul_assoc, mul_comm eQ, mul_assoc, hu, mul_one, map_neg, neg_neg]
  rw [h0, h2]
  simp only [map_neg, map_zero, mul_zero, add_zero]
  unfold eQ
  ring

/-- the cube of the dissected factor, in normal form (computed by sympy). -/
lemma cube_normal (a b : PowerSeries ℤ) :
    (E5 a - X - X ^ 2 * E5 b) ^ 3
      = E5 (a ^ 3 - 3 * X * b ^ 2) + X * E5 (-(X * b ^ 3) - 3 * a ^ 2) + X ^ 2 * E5 (3 * a - 3 * a ^ 2 * b)
        + X ^ 3 * E5 (6 * a * b - 1) + X ^ 4 * E5 (3 * a * b ^ 2 - 3 * b) := by
  simp only [map_sub, map_mul, map_pow, map_neg, map_one, map_ofNat, E5_X]
  ring

/-- **`AB = 1`**: forced by Jacobi's cube having no exponent `≡ 2 (mod 5)`. -/
theorem alphaQ_mul_betaQ : alphaQ * betaQ = 1 := by
  have h := dis5_two_qfacInf_cube
  rw [qfacInf_dissection, mul_pow, ← map_pow, dis5_E5_mul 2 (by norm_num), cube_normal,
    dis5_normal 2 (by norm_num)] at h
  simp only [show (2 : ℕ) ≠ 0 by norm_num, show (2 : ℕ) ≠ 1 by norm_num, if_false, if_true] at h
  have he : eQ ^ 3 ≠ 0 := pow_ne_zero _ isUnit_eQ.ne_zero
  have h3 : 3 * alphaQ * (1 - alphaQ * betaQ) = 0 := by
    have := (mul_eq_zero.mp h).resolve_left he
    rw [← this]; ring
  have hα : IsUnit alphaQ := by
    rw [PowerSeries.isUnit_iff_constantCoeff]
    have hc : constantCoeff alphaQ = 1 := by
      rw [alphaQ, map_mul, ← PowerSeries.coeff_zero_eq_constantCoeff_apply, coeff_dis5,
        show 5 * 0 + 0 = 0 by rfl, ← PowerSeries.coeff_zero_eq_constantCoeff_apply,
        show coeff 0 (Ring.inverse eQ) = 1 by
          have hm : eQ * Ring.inverse eQ = 1 := Ring.mul_inverse_cancel _ isUnit_eQ
          have := congrArg (coeff 0) hm
          rw [PowerSeries.coeff_mul, Finset.antidiagonal_zero, Finset.sum_singleton, coeff_one, if_pos rfl,
            eQ, coeff_E5, if_pos (dvd_zero 5)] at this
          simpa [coeff_zero_qfacInf] using this]
      simp [coeff_zero_qfacInf]
    rw [hc]; exact isUnit_one
  have h3' : (3 * alphaQ) * (1 - alphaQ * betaQ) = 0 := h3
  rcases mul_eq_zero.mp h3' with h0 | h0
  · exfalso
    have := congrArg constantCoeff h0
    rw [map_mul, map_zero] at this
    have hc := (PowerSeries.isUnit_iff_constantCoeff.mp hα)
    rcases mul_eq_zero.mp this with h | h
    · rw [show constantCoeff (3 : PowerSeries ℤ) = 3 from map_ofNat _ 3] at h; norm_num at h
    · exact hc.ne_zero h
  · exact (sub_eq_zero.mp h0).symm


/-! ## Stage R3: the norm over the fifth roots of unity -/

lemma norm_poly {R : Type*} [CommRing R] (w A B t : R) (hw : w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0)
    (hAB : A * B = 1) :
    (A - t - t ^ 2 * B) * (A - w * t - w ^ 2 * t ^ 2 * B) * (A - w ^ 2 * t - w ^ 4 * t ^ 2 * B)
      * (A - w ^ 3 * t - w * t ^ 2 * B) * (A - w ^ 4 * t - w ^ 3 * t ^ 2 * B)
      = A ^ 5 - 11 * t ^ 5 - t ^ 10 * B ^ 5 := by
  linear_combination (-A^4*B*t^2 - A^4*t + A^3*B^2*t^4*w^3 + A^3*B^2*t^4*w + A^3*B*t^3*w^4 + A^3*B*t^3*w^2 + 2*A^3*B*t^3*w + A^3*t^2*w^3 + A^3*t^2*w - A^2*B^3*t^6*w^5 - A^2*B^3*t^6*w^3 - 2*A^2*B^2*t^5*w^6 - A^2*B^2*t^5*w^4 - A^2*B^2*t^5*w^3 - A^2*B^2*t^5*w^2 - 5*A^2*B^2*t^5*w + 5*A^2*B^2*t^5 - A^2*B*t^4*w^7 + A^2*B*t^4*w^6 - 2*A^2*B*t^4*w^5 - 2*A^2*B*t^4*w^4 - A^2*B*t^4*w^3 - A^2*B*t^4*w^2 - A^2*t^3*w^5 - A^2*t^3*w^3 + A*B^4*t^8*w^6 + A*B^3*t^7*w^8 + A*B^3*t^7*w^6 + 2*A*B^3*t^7*w^5 + A*B^2*t^6*w^9 - A*B^2*t^6*w^8 + 2*A*B^2*t^6*w^7 + 2*A*B^2*t^6*w^6 + A*B^2*t^6*w^5 + A*B^2*t^6*w^4 + A*B*t^5*w^8 + A*B*t^5*w^7 - 2*A*B*t^5*w^6 + 4*A*B*t^5*w^5 + A*B*t^5*w^4 - 5*A*B*t^5*w + 5*A*B*t^5 + A*t^4*w^6 - B^5*t^10*w^6 + B^5*t^10*w^5 - B^5*t^10*w + B^5*t^10 - B^4*t^9*w^8 - B^3*t^8*w^9 - B^3*t^8*w^7 - B^2*t^7*w^9 - B^2*t^7*w^7 - B*t^6*w^8 - t^5*w^6 + t^5*w^5 - t^5*w + t^5) * hw + (-5 * A * B * t ^ 5 - 10 * t ^ 5) * hAB

lemma prod_one_sub_root {R : Type*} [CommRing R] (w y : R) (hw : w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0) :
    (1 - y) * (1 - w * y) * (1 - w ^ 2 * y) * (1 - w ^ 3 * y) * (1 - w ^ 4 * y) = 1 - y ^ 5 := by
  linear_combination (-w^6*y^5 + w^6*y^4 + w^5*y^5 - w^5*y^3 - w^3*y^3 + w^3*y^2 - w*y^5 + w*y^2 + y^5 - y) * hw

lemma pow5_of_phi {R : Type*} [CommRing R] (w : R) (hw : w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0) : w ^ 5 = 1 := by
  linear_combination (w - 1) * hw

noncomputable def ω5 : ℂ := Complex.exp (2 * ↑Real.pi * Complex.I / ((5 : ℕ) : ℂ))

lemma ω5_prim : IsPrimitiveRoot ω5 5 := Complex.isPrimitiveRoot_exp 5 (by norm_num)

lemma phi_of_prim {ζ : ℂ} (h : IsPrimitiveRoot ζ 5) : ζ ^ 4 + ζ ^ 3 + ζ ^ 2 + ζ + 1 = 0 := by
  have := h.geom_sum_eq_zero (by norm_num : 1 < 5)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one, zero_add] at this
  linear_combination this

lemma ω5_pow5 : ω5 ^ 5 = 1 := ω5_prim.pow_eq_one

/-- integer series into complex series. -/
noncomputable abbrev ψC : PowerSeries ℤ →+* PowerSeries ℂ := PowerSeries.map (Int.castRingHom ℂ)

lemma ψC_injective : Function.Injective ψC :=
  PowerSeries.map_injective _ (Int.cast_injective (α := ℂ))

lemma E5_injective : Function.Injective E5 := by
  intro f g h
  ext n
  have := congrArg (coeff (5 * n)) h
  rwa [coeff_E5, coeff_E5, if_pos (dvd_mul_right 5 n), if_pos (dvd_mul_right 5 n),
    Nat.mul_div_cancel_left _ (by norm_num)] at this

lemma rescale_ψC_E5 (c : ℂ) (hc : c ^ 5 = 1) (F : PowerSeries ℤ) : rescale c (ψC (E5 F)) = ψC (E5 F) := by
  ext n
  rw [coeff_rescale, PowerSeries.coeff_map, coeff_E5]
  split_ifs with h
  · obtain ⟨m, rfl⟩ := h; rw [pow_mul, hc, one_pow, one_mul]
  · simp

/-- the factor product over the five rescalings, one Euler factor at a time. -/
lemma prod_rescale_factor (n : ℕ) :
    ∏ j ∈ Finset.range 5, (1 - PowerSeries.C ((ω5 ^ j) ^ n) * X ^ n : PowerSeries ℂ)
      = if 5 ∣ n then (1 - X ^ n) ^ 5 else 1 - X ^ (5 * n) := by
  split_ifs with h
  · have hj : ∀ j, (ω5 ^ j) ^ n = 1 := by
      intro j; obtain ⟨m, rfl⟩ := h
      rw [← pow_mul, show j * (5 * m) = 5 * (j * m) by ring, pow_mul, ω5_pow5, one_pow]
    simp only [hj, map_one, one_mul, Finset.prod_const, Finset.card_range]
  · have hcop : n.Coprime 5 := (Nat.coprime_comm.mp ((Nat.Prime.coprime_iff_not_dvd Nat.prime_five).mpr h))
    have hζ := ω5_prim.pow_of_coprime n hcop
    have hphi : (PowerSeries.C (ω5 ^ n) : PowerSeries ℂ) ^ 4 + PowerSeries.C (ω5 ^ n) ^ 3
        + PowerSeries.C (ω5 ^ n) ^ 2 + PowerSeries.C (ω5 ^ n) + 1 = 0 := by
      rw [← map_pow, ← map_pow, ← map_pow, ← map_add, ← map_add, ← map_add, ← map_one PowerSeries.C,
        ← map_add, phi_of_prim hζ, map_zero]
    have hrw : ∀ j, PowerSeries.C ((ω5 ^ j) ^ n) = (PowerSeries.C (ω5 ^ n) : PowerSeries ℂ) ^ j := by
      intro j; rw [← map_pow, ← pow_mul, mul_comm, pow_mul]
    simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul, hrw, pow_zero, pow_one]
    rw [prod_one_sub_root _ _ hphi, ← pow_mul, mul_comm]

/-- `G n = ∏_j (1 − (ωʲq)ⁿ)`. -/
noncomputable def Gfac (n : ℕ) : PowerSeries ℂ := if 5 ∣ n then (1 - X ^ n) ^ 5 else 1 - X ^ (5 * n)

lemma rescale_ψC_qfac_prod (N : ℕ) :
    ∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC (qfac N)) = ∏ k ∈ Finset.range N, Gfac (k + 1) := by
  have h1 : ∀ j, rescale (ω5 ^ j) (ψC (qfac N))
      = ∏ k ∈ Finset.range N, (1 - PowerSeries.C ((ω5 ^ j) ^ (k + 1)) * X ^ (k + 1)) := by
    intro j
    rw [qfac, map_prod, map_prod]
    refine Finset.prod_congr rfl fun k _ => ?_
    rw [map_sub, map_one, map_pow, PowerSeries.map_X, map_sub, map_one, map_pow, rescale_X, mul_pow,
      ← map_pow]
  simp only [h1]
  rw [Finset.prod_comm]
  refine Finset.prod_congr rfl fun k _ => ?_
  rw [prod_rescale_factor, Gfac]

lemma block_identity : ∀ M : ℕ,
    (∏ k ∈ Finset.range (5 * M), Gfac (k + 1)) * ∏ i ∈ Finset.range M, (1 - (X : PowerSeries ℂ) ^ (25 * (i + 1)))
      = (∏ k ∈ Finset.range (5 * M), (1 - (X : PowerSeries ℂ) ^ (5 * (k + 1))))
        * (∏ i ∈ Finset.range M, (1 - (X : PowerSeries ℂ) ^ (5 * (i + 1)))) ^ 5 := by
  intro M
  induction M with
  | zero => simp
  | succ M ih =>
      rw [show 5 * (M + 1) = 5 * M + 5 by ring, Finset.prod_range_add (fun k => Gfac (k + 1)),
        Finset.prod_range_add (fun k => 1 - (X : PowerSeries ℂ) ^ (5 * (k + 1))),
        Finset.prod_range_succ (fun i => 1 - (X : PowerSeries ℂ) ^ (25 * (i + 1))) M,
        Finset.prod_range_succ (fun i => 1 - (X : PowerSeries ℂ) ^ (5 * (i + 1))) M]
      simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul]
      have g1 : Gfac (5 * M + 0 + 1) = 1 - X ^ (5 * (5 * M + 0 + 1)) := by rw [Gfac, if_neg (by omega)]
      have g2 : Gfac (5 * M + 1 + 1) = 1 - X ^ (5 * (5 * M + 1 + 1)) := by rw [Gfac, if_neg (by omega)]
      have g3 : Gfac (5 * M + 2 + 1) = 1 - X ^ (5 * (5 * M + 2 + 1)) := by rw [Gfac, if_neg (by omega)]
      have g4 : Gfac (5 * M + 3 + 1) = 1 - X ^ (5 * (5 * M + 3 + 1)) := by rw [Gfac, if_neg (by omega)]
      have g5 : Gfac (5 * M + 4 + 1) = (1 - X ^ (5 * M + 4 + 1)) ^ 5 := by rw [Gfac, if_pos (by omega)]
      rw [g1, g2, g3, g4, g5]
      have e1 : (X : PowerSeries ℂ) ^ (25 * (M + 1)) = X ^ (5 * (5 * M + 4 + 1)) := by ring
      have e2 : (X : PowerSeries ℂ) ^ (5 * (M + 1)) = X ^ (5 * M + 4 + 1) := by ring
      rw [e1, e2]
      linear_combination (1 - X ^ (5 * (5 * M + 0 + 1))) * (1 - X ^ (5 * (5 * M + 1 + 1)))
        * (1 - X ^ (5 * (5 * M + 2 + 1))) * (1 - X ^ (5 * (5 * M + 3 + 1)))
        * (1 - X ^ (5 * (5 * M + 4 + 1))) * (1 - X ^ (5 * M + 4 + 1)) ^ 5 * ih

/-! ### passing to the limit: congruences modulo `X^K` -/

lemma dvd_sub_mul' {R : Type*} [CommRing R] {d a a' b b' : R} (ha : d ∣ a - a') (hb : d ∣ b - b') :
    d ∣ a * b - a' * b' := by
  have : a * b - a' * b' = (a - a') * b + a' * (b - b') := by ring
  rw [this]; exact dvd_add (dvd_mul_of_dvd_left ha _) (dvd_mul_of_dvd_right hb _)

lemma dvd_sub_pow' {R : Type*} [CommRing R] {d a a' : R} (n : ℕ) (ha : d ∣ a - a') : d ∣ a ^ n - a' ^ n :=
  dvd_trans ha (sub_dvd_pow_sub_pow a a' n)

lemma X_pow_dvd_qfacInf_sub (K N : ℕ) (h : K ≤ N) : (X : PowerSeries ℤ) ^ K ∣ qfacInf - qfac N := by
  rw [PowerSeries.X_pow_dvd_iff]; intro k hk
  rw [map_sub, coeff_qfacInf (show k + 1 ≤ N by omega), sub_self]

lemma X_pow_dvd_ψC {K : ℕ} {f : PowerSeries ℤ} (h : X ^ K ∣ f) : (X : PowerSeries ℂ) ^ K ∣ ψC f := by
  obtain ⟨d, rfl⟩ := h; exact ⟨ψC d, by rw [map_mul, map_pow, PowerSeries.map_X]⟩

lemma X_pow_dvd_rescale {K : ℕ} (c : ℂ) {f : PowerSeries ℂ} (h : X ^ K ∣ f) : (X : PowerSeries ℂ) ^ K ∣ rescale c f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨PowerSeries.C c ^ K * rescale c d, by rw [map_mul, map_pow, rescale_X]; ring⟩

lemma X_pow_dvd_E5 {K : ℕ} {f : PowerSeries ℤ} (h : X ^ K ∣ f) : (X : PowerSeries ℤ) ^ K ∣ E5 f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨X ^ (4 * K) * E5 d, by rw [map_mul, map_pow, E5_X, ← pow_mul]; ring⟩

lemma dvd_sub_prod_range5 {K : ℕ} (F G : ℕ → PowerSeries ℂ) (h : ∀ j, (X : PowerSeries ℂ) ^ K ∣ F j - G j) :
    (X : PowerSeries ℂ) ^ K ∣ ∏ j ∈ Finset.range 5, F j - ∏ j ∈ Finset.range 5, G j := by
  simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul]
  exact dvd_sub_mul' (dvd_sub_mul' (dvd_sub_mul' (dvd_sub_mul' (h 0) (h 1)) (h 2)) (h 3)) (h 4)

/-- **the product over the five rescalings**: `∏_ω E(ωq) · E(q²⁵) = E(q⁵)⁶`. -/
theorem prod_rescale_qfacInf :
    (∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC qfacInf)) * ψC (E5 (E5 qfacInf)) = ψC (E5 qfacInf) ^ 6 := by
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  set K := k + 1
  have hfin : (∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC (qfac (5 * K)))) * ψC (E5 (E5 (qfac K)))
      = ψC (E5 (qfac (5 * K))) * ψC (E5 (qfac K)) ^ 5 := by
    rw [rescale_ψC_qfac_prod]
    have hE5E5 : ψC (E5 (E5 (qfac K))) = ∏ i ∈ Finset.range K, (1 - (X : PowerSeries ℂ) ^ (25 * (i + 1))) := by
      rw [E5_qfac, map_prod, map_prod]
      refine Finset.prod_congr rfl fun i _ => ?_
      simp only [map_sub, map_one, map_pow, E5_X, PowerSeries.map_X]
      rw [← pow_mul]; ring_nf
    have hE5 : ∀ N, ψC (E5 (qfac N)) = ∏ i ∈ Finset.range N, (1 - (X : PowerSeries ℂ) ^ (5 * (i + 1))) := by
      intro N; rw [E5_qfac, map_prod]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [map_sub, map_one, map_pow, PowerSeries.map_X]; ring_nf
    rw [hE5E5, hE5, hE5, block_identity]
  have hL : (X : PowerSeries ℂ) ^ K ∣ (∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC qfacInf)) * ψC (E5 (E5 qfacInf))
      - (∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC (qfac (5 * K)))) * ψC (E5 (E5 (qfac K))) := by
    refine dvd_sub_mul' (dvd_sub_prod_range5 _ _ fun j => ?_) ?_
    · rw [← map_sub]; exact X_pow_dvd_rescale _ (by rw [← map_sub]; exact X_pow_dvd_ψC (X_pow_dvd_qfacInf_sub K _ (by omega)))
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub, ← map_sub]; exact X_pow_dvd_E5 (X_pow_dvd_E5 (X_pow_dvd_qfacInf_sub K K le_rfl)))
  have hR : (X : PowerSeries ℂ) ^ K ∣ ψC (E5 qfacInf) ^ 6 - ψC (E5 (qfac (5 * K))) * ψC (E5 (qfac K)) ^ 5 := by
    rw [show ψC (E5 qfacInf) ^ 6 = ψC (E5 qfacInf) * ψC (E5 qfacInf) ^ 5 by ring]
    refine dvd_sub_mul' ?_ (dvd_sub_pow' 5 ?_)
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub]; exact X_pow_dvd_E5 (X_pow_dvd_qfacInf_sub K _ (by omega)))
    · rw [← map_sub]; exact X_pow_dvd_ψC (by rw [← map_sub]; exact X_pow_dvd_E5 (X_pow_dvd_qfacInf_sub K K le_rfl))
  have htot : (X : PowerSeries ℂ) ^ K ∣ (∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC qfacInf)) * ψC (E5 (E5 qfacInf))
      - ψC (E5 qfacInf) ^ 6 := by
    have := dvd_sub hL hR
    rwa [hfin, show ∀ a b c : PowerSeries ℂ, a - c - (b - c) = a - b by intros; ring] at this
  obtain ⟨d, hd⟩ := htot
  rw [hd, coeff_X_pow_mul', if_neg (by omega)]

lemma norm_assembled {R : Type*} [CommRing R] (e A B t w : R) (hw : w ^ 4 + w ^ 3 + w ^ 2 + w + 1 = 0)
    (hAB : A * B = 1) :
    e * (A - w ^ 0 * t - (w ^ 0 * t) ^ 2 * B) * (e * (A - w ^ 1 * t - (w ^ 1 * t) ^ 2 * B))
      * (e * (A - w ^ 2 * t - (w ^ 2 * t) ^ 2 * B)) * (e * (A - w ^ 3 * t - (w ^ 3 * t) ^ 2 * B))
      * (e * (A - w ^ 4 * t - (w ^ 4 * t) ^ 2 * B)) = e ^ 5 * (A ^ 5 - 11 * t ^ 5 - (t ^ 5) ^ 2 * B ^ 5) := by
  have h5 := pow5_of_phi w hw
  have e6 : w ^ 6 = w := by rw [show w ^ 6 = w ^ 5 * w by ring, h5, one_mul]
  have e8 : w ^ 8 = w ^ 3 := by rw [show w ^ 8 = w ^ 5 * w ^ 3 by ring, h5, one_mul]
  calc e * (A - w ^ 0 * t - (w ^ 0 * t) ^ 2 * B) * (e * (A - w ^ 1 * t - (w ^ 1 * t) ^ 2 * B))
        * (e * (A - w ^ 2 * t - (w ^ 2 * t) ^ 2 * B)) * (e * (A - w ^ 3 * t - (w ^ 3 * t) ^ 2 * B))
        * (e * (A - w ^ 4 * t - (w ^ 4 * t) ^ 2 * B))
      = e ^ 5 * ((A - t - t ^ 2 * B) * (A - w * t - w ^ 2 * t ^ 2 * B) * (A - w ^ 2 * t - w ^ 4 * t ^ 2 * B)
          * (A - w ^ 3 * t - w ^ 6 * t ^ 2 * B) * (A - w ^ 4 * t - w ^ 8 * t ^ 2 * B)) := by ring
    _ = e ^ 5 * (A ^ 5 - 11 * t ^ 5 - (t ^ 5) ^ 2 * B ^ 5) := by
        rw [e6, e8, norm_poly w A B t hw hAB]; ring

/-- the dissected factor under the five rescalings: the **norm** `A⁵ − 11q⁵ − q¹⁰B⁵`. -/
theorem prod_rescale_dissection :
    ∏ j ∈ Finset.range 5, rescale (ω5 ^ j) (ψC qfacInf)
      = ψC (E5 eQ) ^ 5 * ψC (E5 (alphaQ ^ 5 - 11 * X - X ^ 2 * betaQ ^ 5)) := by
  have hA : ∀ j, rescale (ω5 ^ j) (ψC qfacInf)
      = ψC (E5 eQ) * (ψC (E5 alphaQ) - (PowerSeries.C ω5) ^ j * X
          - ((PowerSeries.C ω5) ^ j * X) ^ 2 * ψC (E5 betaQ)) := by
    intro j
    have hc : (ω5 ^ j) ^ 5 = 1 := by rw [← pow_mul, mul_comm, pow_mul, ω5_pow5, one_pow]
    conv_lhs => rw [qfacInf_dissection]
    simp only [map_mul, map_sub, map_pow, PowerSeries.map_X, rescale_X, rescale_ψC_E5 _ hc]
  have hw : (PowerSeries.C ω5 : PowerSeries ℂ) ^ 4 + PowerSeries.C ω5 ^ 3 + PowerSeries.C ω5 ^ 2
      + PowerSeries.C ω5 + 1 = 0 := by
    rw [← map_pow, ← map_pow, ← map_pow, ← map_add, ← map_add, ← map_add, ← map_one PowerSeries.C,
      ← map_add, phi_of_prim ω5_prim, map_zero]
  have hAB : ψC (E5 alphaQ) * ψC (E5 betaQ) = 1 := by
    rw [← map_mul, ← map_mul, alphaQ_mul_betaQ, map_one, map_one]
  simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul, hA]
  rw [norm_assembled _ _ _ _ _ hw hAB]
  congr 1
  simp only [map_sub, map_mul, map_pow, map_ofNat, E5_X, PowerSeries.map_X]

/-- **the norm identity**: `E(Q⁵)⁶·(α⁵ − 11Q − Q²β⁵) = E(Q)⁶`. -/
theorem norm_identity : eQ ^ 6 * (alphaQ ^ 5 - 11 * X - X ^ 2 * betaQ ^ 5) = qfacInf ^ 6 := by
  have h := prod_rescale_qfacInf
  rw [prod_rescale_dissection, show E5 (E5 qfacInf) = E5 eQ from rfl] at h
  apply E5_injective
  apply ψC_injective
  rw [map_mul, map_mul, map_pow, map_pow, map_pow, map_pow, ← h]
  ring


/-! ## Stage R4: inversion and extraction — Ramanujan's identity -/

/-- the inversion polynomial `𝒬 = A⁴ + qA³ + 2q²A² + 3q³A + 5q⁴ − 3q⁵B + 2q⁶B² − q⁷B³ + q⁸B⁴`,
written in 5-dissection normal form. -/
noncomputable def Qpoly : PowerSeries ℤ :=
  E5 (alphaQ ^ 4 - 3 * X * betaQ) + X * E5 (alphaQ ^ 3 + 2 * X * betaQ ^ 2)
    + X ^ 2 * E5 (2 * alphaQ ^ 2 - X * betaQ ^ 3) + X ^ 3 * E5 (3 * alphaQ + X * betaQ ^ 4) + X ^ 4 * E5 5

lemma dis5_four_Qpoly : dis5 4 Qpoly = 5 := by
  rw [Qpoly, dis5_normal 4 (by norm_num)]; simp

/-- `(A − q − q²B)·𝒬 = A⁵ − 11q⁵ − q¹⁰B⁵` (using `AB = 1`). -/
lemma factor_mul_Qpoly :
    (E5 alphaQ - X - X ^ 2 * E5 betaQ) * Qpoly = E5 (alphaQ ^ 5 - 11 * X - X ^ 2 * betaQ ^ 5) := by
  have hAB : E5 alphaQ * E5 betaQ = 1 := by rw [← map_mul, alphaQ_mul_betaQ, map_one]
  simp only [Qpoly, map_sub, map_add, map_mul, map_pow, map_ofNat, E5_X]
  set A := E5 alphaQ
  set B := E5 betaQ
  linear_combination (-A ^ 3 * X ^ 2 - A ^ 2 * X ^ 3 - 2 * A * X ^ 4 + B ^ 3 * X ^ 8 - B ^ 2 * X ^ 7
    + 2 * B * X ^ 6 - 6 * X ^ 5) * hAB

/-- `N = α⁵ − 11Q − Q²β⁵` and the unit `eQ·N`. -/
noncomputable def normQ : PowerSeries ℤ := alphaQ ^ 5 - 11 * X - X ^ 2 * betaQ ^ 5

lemma isUnit_eQ_mul_normQ : IsUnit (eQ * normQ) := by
  have h : eQ ^ 5 * (eQ * normQ) = qfacInf ^ 6 := by rw [← norm_identity, normQ]; ring
  have hu : IsUnit (eQ ^ 5 * (eQ * normQ)) := by rw [h]; exact isUnit_qfacInf.pow 6
  exact isUnit_of_mul_isUnit_right hu

lemma E5_inverse {u : PowerSeries ℤ} (hu : IsUnit u) : Ring.inverse (E5 u) = E5 (Ring.inverse u) := by
  have h1 : E5 u * E5 (Ring.inverse u) = 1 := by rw [← map_mul, Ring.mul_inverse_cancel u hu, map_one]
  calc Ring.inverse (E5 u) = Ring.inverse (E5 u) * (E5 u * E5 (Ring.inverse u)) := by rw [h1, mul_one]
    _ = (Ring.inverse (E5 u) * E5 u) * E5 (Ring.inverse u) := by ring
    _ = E5 (Ring.inverse u) := by rw [Ring.inverse_mul_cancel _ (hu.map E5), one_mul]

/-- `1/E = 𝒬 · E5((eQ·N)⁻¹)`. -/
lemma partitionGF_eq_Qpoly : partitionGF = E5 (Ring.inverse (eQ * normQ)) * Qpoly := by
  have hE : qfacInf * Qpoly = E5 (eQ * normQ) := by
    conv_lhs => rw [qfacInf_dissection]
    rw [mul_assoc, factor_mul_Qpoly, ← map_mul, normQ]
  have hu := isUnit_eQ_mul_normQ
  rw [partitionGF, ← E5_inverse hu]
  calc Ring.inverse qfacInf = Ring.inverse qfacInf * (E5 (eQ * normQ) * Ring.inverse (E5 (eQ * normQ))) := by
        rw [Ring.mul_inverse_cancel _ (hu.map E5), mul_one]
    _ = Ring.inverse qfacInf * (qfacInf * Qpoly) * Ring.inverse (E5 (eQ * normQ)) := by rw [hE]; ring
    _ = Ring.inverse (E5 (eQ * normQ)) * Qpoly := by
        rw [← mul_assoc, Ring.inverse_mul_cancel _ isUnit_qfacInf, one_mul, mul_comm]

/-- **Ramanujan's identity, generating-function form**: `(Σ p(5n+4) qⁿ)·(q;q)⁶ = 5·(q⁵;q⁵)⁵`
(here `dis5 4 partitionGF = Σ_n [q^{5n+4}](1/(q;q)∞) qⁿ`). -/
theorem dis5_four_partitionGF_mul : dis5 4 partitionGF * qfacInf ^ 6 = 5 * eQ ^ 5 := by
  rw [partitionGF_eq_Qpoly, dis5_E5_mul 4 (by norm_num), dis5_four_Qpoly, ← norm_identity]
  have hu := Ring.inverse_mul_cancel _ isUnit_eQ_mul_normQ
  rw [show Ring.inverse (eQ * normQ) * 5 * (eQ ^ 6 * (alphaQ ^ 5 - 11 * X - X ^ 2 * betaQ ^ 5))
      = 5 * eQ ^ 5 * (Ring.inverse (eQ * normQ) * (eQ * normQ)) by rw [normQ]; ring, hu, mul_one]

/-- **Ramanujan's "most beautiful identity"**: `Σ_{n≥0} p(5n+4) qⁿ · (q;q)_∞⁶ = 5·(q⁵;q⁵)_∞⁵`, with `p(n)`
the number of partitions of `n` (Mathlib's `Nat.Partition`). -/
theorem ramanujan_most_beautiful_identity :
    (PowerSeries.mk fun n => (Fintype.card (Nat.Partition (5 * n + 4)) : ℤ)) * qfacInf ^ 6
      = 5 * (E5 qfacInf) ^ 5 := by
  have h : (PowerSeries.mk fun n => (Fintype.card (Nat.Partition (5 * n + 4)) : ℤ)) = dis5 4 partitionGF := by
    ext n; rw [coeff_mk, coeff_dis5, coeff_partitionGF_eq_card]
  rw [h, dis5_four_partitionGF_mul]; rfl

/-- the same identity with the inverse written out: `Σ p(5n+4) qⁿ = 5·(q⁵;q⁵)⁵ / (q;q)⁶`. -/
theorem ramanujan_most_beautiful_identity' :
    (PowerSeries.mk fun n => (Fintype.card (Nat.Partition (5 * n + 4)) : ℤ))
      = 5 * (E5 qfacInf) ^ 5 * Ring.inverse (qfacInf ^ 6) := by
  rw [← ramanujan_most_beautiful_identity, mul_assoc,
    Ring.mul_inverse_cancel _ (isUnit_qfacInf.pow 6), mul_one]


/-! ## Stage R5: Ramanujan's congruence `p(25n+24) ≡ 0 (mod 25)` -/

lemma coeff_Ψ5_cast (n : ℕ) (f : PowerSeries ℤ) : coeff n (Ψ5 f) = ((coeff n f : ℤ) : ZMod 5) := by
  rw [Ψ5, PowerSeries.coeff_map]; rfl

lemma Ψ5_inverse {u : PowerSeries ℤ} (hu : IsUnit u) : Ψ5 (Ring.inverse u) = Ring.inverse (Ψ5 u) := by
  have h1 : Ψ5 u * Ψ5 (Ring.inverse u) = 1 := by rw [← map_mul, Ring.mul_inverse_cancel u hu, map_one]
  calc Ψ5 (Ring.inverse u) = (Ring.inverse (Ψ5 u) * Ψ5 u) * Ψ5 (Ring.inverse u) := by
        rw [Ring.inverse_mul_cancel _ (hu.map Ψ5), one_mul]
    _ = Ring.inverse (Ψ5 u) * (Ψ5 u * Ψ5 (Ring.inverse u)) := by ring
    _ = Ring.inverse (Ψ5 u) := by rw [h1, mul_one]

/-- mod 5 (Frobenius): `1/E⁶ ≡ (1/E)·(1/E(q⁵))`. -/
lemma Ψ5_inv_qfac6 : Ψ5 (Ring.inverse (qfacInf ^ 6)) = Ψ5 (partitionGF * E5 partitionGF) := by
  have hE := isUnit_qfacInf
  rw [Ψ5_inverse (hE.pow 6), map_mul, partitionGF, ← E5_inverse hE, Ψ5_inverse hE, Ψ5_inverse (hE.map E5),
    ← frobenius_qfacInf, map_pow, show (Ψ5 qfacInf) ^ 6 = (Ψ5 qfacInf) ^ 5 * Ψ5 qfacInf by ring,
    Ring.mul_inverse_rev' (Commute.all _ _)]

lemma five_dvd_coeff_inv_qfac6 (k : ℕ) : (5 : ℤ) ∣ coeff (5 * k + 4) (Ring.inverse (qfacInf ^ 6)) := by
  have h1 : (5 : ℤ) ∣ coeff (5 * k + 4) (partitionGF * E5 partitionGF) := by
    have hd : coeff (5 * k + 4) (E5 partitionGF * partitionGF) = coeff k (partitionGF * dis5 4 partitionGF) := by
      rw [← coeff_dis5, dis5_E5_mul 4 (by norm_num)]
    rw [mul_comm partitionGF (E5 partitionGF), hd, PowerSeries.coeff_mul]
    exact Finset.dvd_sum fun p _ => by
      rw [coeff_dis5]; exact Dvd.dvd.mul_left (five_dvd_coeff_partitionGF p.2) _
  have h2 := congrArg (coeff (5 * k + 4)) Ψ5_inv_qfac6
  rw [coeff_Ψ5_cast, coeff_Ψ5_cast] at h2
  have h3 := (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ 5).mp h2
  have h4 := dvd_sub h1 h3
  rwa [sub_sub_cancel] at h4

lemma coeff_five_mul (n : ℕ) (f : PowerSeries ℤ) : coeff n (5 * f) = 5 * coeff n f := by
  rw [show (5 : PowerSeries ℤ) = PowerSeries.C (5 : ℤ) from (map_ofNat PowerSeries.C 5).symm,
    PowerSeries.coeff_C_mul]

/-- **Ramanujan's congruence modulo 25**: `25 ∣ [q^{25m+24}] (1/(q;q)∞)`. -/
theorem twentyfive_dvd_coeff_partitionGF (m : ℕ) : (25 : ℤ) ∣ coeff (25 * m + 24) partitionGF := by
  have hdis : dis5 4 partitionGF = 5 * (E5 (qfacInf ^ 5) * Ring.inverse (qfacInf ^ 6)) := by
    have h := dis5_four_partitionGF_mul
    calc dis5 4 partitionGF = dis5 4 partitionGF * qfacInf ^ 6 * Ring.inverse (qfacInf ^ 6) := by
          rw [mul_assoc, Ring.mul_inverse_cancel _ (isUnit_qfacInf.pow 6), mul_one]
      _ = 5 * (E5 (qfacInf ^ 5) * Ring.inverse (qfacInf ^ 6)) := by rw [h, eQ, map_pow]; ring
  have h0 : coeff (25 * m + 24) partitionGF = coeff (5 * m + 4) (dis5 4 partitionGF) := by
    rw [coeff_dis5]; ring_nf
  rw [h0, hdis, coeff_five_mul, ← coeff_dis5, dis5_E5_mul 4 (by norm_num), PowerSeries.coeff_mul]
  rw [show (25 : ℤ) = 5 * 5 by norm_num]
  refine mul_dvd_mul_left 5 (Finset.dvd_sum fun p _ => ?_)
  rw [coeff_dis5]
  exact Dvd.dvd.mul_left (five_dvd_coeff_inv_qfac6 p.2) _

/-- **Ramanujan's congruence** `25 ∣ p(25n+24)` for the partition count. -/
theorem twentyfive_dvd_partition_card (m : ℕ) : 25 ∣ Fintype.card (Nat.Partition (25 * m + 24)) := by
  have h := twentyfive_dvd_coeff_partitionGF m
  rw [coeff_partitionGF_eq_card] at h
  exact_mod_cast h

end MockTheta5.JTP
