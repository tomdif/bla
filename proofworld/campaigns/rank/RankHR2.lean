/-
# Dyson's rank: Garvan's (2.19), the `R(z;q²)` identity

`(q)_∞(zq)_∞(q/z)_∞ · R(z;q²) = Σ_{a∈ℤ} z^a (−1)^a Σ_{n≥|a|} q^{n(3n+1)/2−a²}(1−q^{2n+1})`.
Route: `(zq;q)_∞ = (zq;q²)_∞ · (zq²;q²)_∞`; the odd products give `Σ_m (−1)^m z^m q^{m²}/(q²;q²)_∞`
(odd Euler + Durfee rectangle), the even ones with `R(z;q²)` are (2.18) at `q²`, and the
convolution of the two is the reduced lattice identity (`RankHR2Series.lattice_trunc`).
-/
import RamanujanTau.RankHR1
import RamanujanTau.RankHR2Series

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf rectTerm rectPartial rectInf durfee_rect_base psiSum qfac2Inf)
local notation "ψ" => MockTheta5.JTP.ψC

/-! ## `q ↦ q²` over `ℂ` -/

section Expand

noncomputable def E2C : PowerSeries ℂ →+* PowerSeries ℂ := (PowerSeries.expand 2 (by norm_num)).toRingHom

lemma coeff_E2C (n : ℕ) (f : PowerSeries ℂ) : coeff n (E2C f) = if 2 ∣ n then coeff (n / 2) f else 0 := by
  have h : E2C f = PowerSeries.expand 2 (by norm_num) f := rfl
  rw [h, PowerSeries.coeff_expand]

lemma E2C_X : E2C X = X ^ 2 := PowerSeries.expand_X 2 (by norm_num)

lemma E2C_C (c : ℂ) : E2C (C c) = C c := by
  ext n; rw [coeff_E2C, coeff_C, coeff_C]
  split_ifs <;> first | rfl | omega

lemma ψ_E2 (f : PowerSeries ℤ) : ψ (E2 f) = E2C (ψ f) := by
  ext n
  have h : E2 f = PowerSeries.expand 2 (by norm_num) f := rfl
  rw [coeff_E2C, PowerSeries.coeff_map, h, PowerSeries.coeff_expand]
  split_ifs <;> simp [PowerSeries.coeff_map]

lemma E2C_dvd {m : ℕ} {f : PowerSeries ℂ} (h : (X : PowerSeries ℂ) ^ m ∣ f) : (X : PowerSeries ℂ) ^ (2 * m) ∣ E2C f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨E2C d, by rw [map_mul, map_pow, E2C_X, ← pow_mul, mul_comm 2 m]⟩

end Expand

/-! ## the odd products `(cq;q²)_∞` -/

section Odd

/-- `∏_{i<M} (1 − c q^{2i+1})`. -/
noncomputable def oddP (c : ℂ) (M : ℕ) : PowerSeries ℂ := ∏ i ∈ range M, (1 - C c * X ^ (2 * i + 1))

lemma poch_split2 (c : ℂ) : ∀ M, poch c 1 (2 * M) = oddP c M * E2C (poch c 1 M)
  | 0 => by simp [poch, oddP]
  | M + 1 => by
    rw [show 2 * (M + 1) = 2 * M + 1 + 1 by ring, poch_succ, poch_succ, poch_split2 c M, poch_succ c 1 M,
      oddP, oddP, prod_range_succ, map_mul, map_sub, map_one, map_mul, E2C_C, map_pow, E2C_X, ← pow_mul]
    ring

lemma isUnit_E2C_pochInf (c : ℂ) : IsUnit (E2C (pochInf c 1)) := (isUnit_pochInf c le_rfl).map E2C

/-- `(cq;q²)_∞ := (cq;q)_∞ / (cq²;q²)_∞`. -/
noncomputable def OddInf (c : ℂ) : PowerSeries ℂ := pochInf c 1 * Ring.inverse (E2C (pochInf c 1))

lemma pochInf_eq_odd (c : ℂ) : pochInf c 1 = OddInf c * E2C (pochInf c 1) := by
  rw [OddInf, mul_assoc, Ring.inverse_mul_cancel _ (isUnit_E2C_pochInf c), mul_one]

lemma X_pow_dvd_OddInf_sub (c : ℂ) (M : ℕ) : (X : PowerSeries ℂ) ^ (2 * M) ∣ OddInf c - oddP c M := by
  have h1 : (X : PowerSeries ℂ) ^ (2 * M) ∣ pochInf c 1 - poch c 1 (2 * M) := X_pow_dvd_pochInf_sub c le_rfl _
  have h2 : (X : PowerSeries ℂ) ^ (2 * M) ∣ E2C (poch c 1 M) - E2C (pochInf c 1) := by
    rw [← map_sub]; exact E2C_dvd (dvd_sub_comm.mp (X_pow_dvd_pochInf_sub c le_rfl M))
  have key : OddInf c - oddP c M
      = (pochInf c 1 - poch c 1 (2 * M) + oddP c M * (E2C (poch c 1 M) - E2C (pochInf c 1)))
        * Ring.inverse (E2C (pochInf c 1)) := by
    rw [poch_split2, OddInf]
    have := Ring.mul_inverse_cancel _ (isUnit_E2C_pochInf c)
    linear_combination (oddP c M) * this
  rw [key]
  exact dvd_mul_of_dvd_left (dvd_add h1 (dvd_mul_of_dvd_right h2 _)) _

/-- the odd product, expanded by the `q`-binomial theorem at base `q²`. -/
lemma oddP_qbinom (c : ℂ) (M : ℕ) :
    oddP c M = ∑ k ∈ range (M + 1), C ((-c) ^ k) * X ^ (k ^ 2) * ψ (E2 (gaussBinom M k)) := by
  have h := congrArg (fun p => Polynomial.eval₂ (MockTheta5.JTP.ψC.comp E2) (-(C c * X)) p) (qbinom M)
  simp only [qprod, qbRHS, Polynomial.eval₂_finset_prod, Polynomial.eval₂_finset_sum, Polynomial.eval₂_add,
    Polynomial.eval₂_one, Polynomial.eval₂_mul, Polynomial.eval₂_C, Polynomial.eval₂_X,
    Polynomial.eval₂_pow, qcoeff, map_mul, map_pow, RingHom.coe_comp, Function.comp_apply, E2_X,
    PowerSeries.map_X, qq] at h
  rw [oddP]
  convert h using 2 with i _ k _
  · ring
  · rw [neg_pow (C c * X), mul_pow, ← pow_mul, show k ^ 2 = 2 * k.choose 2 + k by
      have h := two_choose_two k
      have : ((k ^ 2 : ℕ) : ℤ) = ((2 * k.choose 2 + k : ℕ) : ℤ) := by push_cast; linarith
      exact_mod_cast this, pow_add, pow_mul]
    simp only [map_mul, map_pow, map_neg, map_one]; ring

/-- **odd Euler**, truncated: `(cq;q²)_∞ ≡ Σ_{i≤N} (−c)^i q^{i²}/(q²;q²)_i (mod q^{N+1})`. -/
theorem odd_euler_dvd (c : ℂ) (N : ℕ) :
    (X : PowerSeries ℂ) ^ (N + 1) ∣ OddInf c
      - ∑ i ∈ range (N + 1), C ((-c) ^ i) * X ^ (i ^ 2) * ψ (E2 (Ring.inverse (qfac i))) := by
  set M := 2 * N + 1
  have h1 : (X : PowerSeries ℂ) ^ (N + 1) ∣ OddInf c - oddP c M :=
    (pow_dvd_pow X (by omega)).trans (X_pow_dvd_OddInf_sub c M)
  have h2 : (X : PowerSeries ℂ) ^ (N + 1) ∣ oddP c M
      - ∑ i ∈ range (N + 1), C ((-c) ^ i) * X ^ (i ^ 2) * ψ (E2 (Ring.inverse (qfac i))) := by
    rw [oddP_qbinom, show M + 1 = (N + 1) + (N + 1) by omega, sum_range_add, add_sub_right_comm,
      ← sum_sub_distrib]
    refine dvd_add (dvd_sum fun k hk => ?_) (dvd_sum fun k _ => ?_)
    · have hk' := mem_range.mp hk
      rw [← mul_sub, ← map_sub, ← map_sub]
      refine dvd_mul_of_dvd_right (CrankProof.ψ_dvd ((pow_dvd_pow X ?_).trans
        (RankProof.E2_dvd (gauss_dvd M k (by omega))))) _
      omega
    · rw [show (N + 1 + k) ^ 2 = (N + 1) + ((N + 1 + k) ^ 2 - (N + 1)) by
        have : N + 1 + k ≤ (N + 1 + k) ^ 2 := Nat.le_self_pow two_ne_zero _
        omega, pow_add]
      exact ⟨C ((-c) ^ (N + 1 + k)) * X ^ ((N + 1 + k) ^ 2 - (N + 1)) * ψ (E2 (gaussBinom M (N + 1 + k))), by ring⟩
  have := dvd_add h1 h2
  rwa [sub_add_sub_cancel] at this

end Odd

/-! ## `(zq;q²)_∞(q/z;q²)_∞ = Σ_a w_a(z) (−1)^a q^{a²} / (q²;q²)_∞` -/

section OddJTP

/-- `(−1)^a q^{a²}/(q²;q²)_∞`. -/
noncomputable def Aterm (a : ℕ) : PowerSeries ℤ := C ((-1 : ℤ) ^ a) * X ^ (a ^ 2) * E2 (Ring.inverse qfacInf)

noncomputable def Ocoef (n i j : ℕ) : ℂ :=
  coeff n (X ^ (i ^ 2 + j ^ 2) * ψ (E2 (Ring.inverse (qfac i) * Ring.inverse (qfac j))))

lemma Ocoef_symm (n i j : ℕ) : Ocoef n i j = Ocoef n j i := by
  unfold Ocoef; rw [add_comm (i ^ 2), mul_comm (Ring.inverse (qfac i))]

lemma odd_coeff_expand (z : ℂ) (n : ℕ) :
    coeff n (OddInf z * OddInf z⁻¹)
      = ∑ i ∈ range (n + 1), ∑ j ∈ range (n + 1), z ^ i * z⁻¹ ^ j * ((-1) ^ (i + j) * Ocoef n i j) := by
  rw [coeffC_congr (odd_euler_dvd z⁻¹ n), mul_comm, coeffC_congr (odd_euler_dvd z n), mul_sum]
  simp only [sum_mul, map_sum]
  refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
  have e : C ((-z⁻¹) ^ j) * X ^ (j ^ 2) * ψ (E2 (Ring.inverse (qfac j)))
      * (C ((-z) ^ i) * X ^ (i ^ 2) * ψ (E2 (Ring.inverse (qfac i))))
      = C (z ^ i * z⁻¹ ^ j * (-1) ^ (i + j)) * (X ^ (i ^ 2 + j ^ 2)
        * ψ (E2 (Ring.inverse (qfac i) * Ring.inverse (qfac j)))) := by
    rw [neg_pow z, neg_pow z⁻¹, pow_add (-1 : ℂ), pow_add (X : PowerSeries ℂ), map_mul E2, RingHom.map_mul ψ, RingHom.map_mul C,
      RingHom.map_mul C, RingHom.map_mul C, RingHom.map_mul C, RingHom.map_mul C]
    ring
  rw [e, coeff_C_mul, Ocoef, mul_assoc]

/-- one `a`-block: the Durfee rectangle at base `q²`. -/
lemma odd_block (n a : ℕ) :
    ∑ j ∈ range (n + 1 - a), (-1 : ℂ) ^ (j + a + j) * Ocoef n (j + a) j = coeff n (ψ (Aterm a)) := by
  have hsign : ∀ j, (-1 : ℂ) ^ (j + a + j) = (-1) ^ a := fun j => by
    rw [show j + a + j = a + 2 * j by ring, pow_add, pow_mul]; norm_num
  have hO : ∀ j, Ocoef n (j + a) j = coeff n (X ^ (a ^ 2) * ψ (E2 (rectTerm a j))) := fun j => by
    rw [Ocoef, rectTerm, add_comm a j]
    simp only [map_mul, map_pow, E2_X, PowerSeries.map_X]
    rw [show (j + a) ^ 2 + j ^ 2 = a ^ 2 + 2 * (j ^ 2 + a * j) by ring, pow_add, pow_mul]
    ring_nf
  have hdvd : (X : PowerSeries ℂ) ^ (n + 1) ∣ ψ (E2 (Ring.inverse qfacInf))
      - ψ (E2 (rectPartial a (n + 1))) := by
    rw [← map_sub, ← map_sub, ← durfee_rect_base a]
    exact CrankProof.ψ_dvd ((pow_dvd_pow X (by omega)).trans (RankProof.E2_dvd (RankProof.rect_dvd a n)))
  have hext : ∑ j ∈ range (n + 1 - a), Ocoef n (j + a) j = ∑ j ∈ range (n + 1), Ocoef n (j + a) j := by
    refine sum_subset (range_subset_range.mpr (by omega)) fun j _ hj => ?_
    simp only [mem_range, not_lt] at hj
    have : j + a ≤ (j + a) ^ 2 := Nat.le_self_pow two_ne_zero _
    exact coeffC_Xpow_zero (by omega) _
  simp only [hsign, ← mul_sum]
  rw [hext, sum_congr rfl fun j _ => hO j]
  conv_rhs => rw [Aterm, RingHom.map_mul ψ, RingHom.map_mul ψ, PowerSeries.map_C, RingHom.map_pow ψ,
    PowerSeries.map_X, mul_assoc, coeff_C_mul, coeffC_congr hdvd, rectPartial, map_sum, map_sum, mul_sum, map_sum]
  simp

/-- **the odd Jacobi product, coefficientwise.** -/
theorem odd_jtp_coeff {z : ℂ} (hz : z ≠ 0) (n : ℕ) :
    coeff n (OddInf z * OddInf z⁻¹) = ∑ a ∈ range (n + 1), wz z a * coeff n (ψ (Aterm a)) := by
  have key := sum_regroup hz (fun i j => (-1 : ℂ) ^ (i + j) * Ocoef n i j)
    (fun i j => by beta_reduce; rw [add_comm i j, Ocoef_symm]) (n + 1)
  beta_reduce at key
  rw [odd_coeff_expand, key]
  exact sum_congr rfl fun a _ => by rw [odd_block]

lemma Aterm_dvd (a : ℕ) : (X : PowerSeries ℂ) ^ a ∣ ψ (Aterm a) := by
  refine CrankProof.ψ_dvd ((pow_dvd_pow X (Nat.le_self_pow two_ne_zero a)).trans ?_)
  rw [Aterm, mul_comm (C _), mul_assoc]; exact dvd_mul_right _ _

/-- generic: a coefficientwise `wz`-expansion with orders `≥ a` lifts to a truncation. -/
lemma trunc_of_coeff {z : ℂ} (F : PowerSeries ℂ) (f : ℕ → PowerSeries ℂ) (hf : ∀ a, (X : PowerSeries ℂ) ^ a ∣ f a)
    (h : ∀ m, coeff m F = ∑ a ∈ range (m + 1), wz z a * coeff m (f a)) (N : ℕ) :
    (X : PowerSeries ℂ) ^ (N + 1) ∣ F - ∑ a ∈ range (N + 1), C (wz z a) * f a := by
  rw [X_pow_dvd_iff]
  intro m hm
  rw [map_sub, sub_eq_zero, h, map_sum]
  simp only [coeff_C_mul]
  refine sum_subset (range_subset_range.mpr (by omega)) fun a _ ha => ?_
  simp only [mem_range, not_lt] at ha
  obtain ⟨d, hd⟩ := hf a
  rw [hd, coeffC_Xpow_zero (by omega), mul_zero]

theorem odd_jtp_trunc {z : ℂ} (hz : z ≠ 0) (N : ℕ) :
    (X : PowerSeries ℂ) ^ (N + 1) ∣ OddInf z * OddInf z⁻¹ - ∑ a ∈ range (N + 1), C (wz z a) * ψ (Aterm a) :=
  trunc_of_coeff _ _ Aterm_dvd (odd_jtp_coeff hz) N

end OddJTP

/-! ## multiplying two `w_a(z)`-expansions -/

section ZConv

lemma Icc_split {M : Type*} [AddCommMonoid M] (G : ℤ → M) : ∀ N : ℕ,
    ∑ m ∈ Icc (-(N : ℤ)) N, G m = ∑ m ∈ range (N + 1), G m + ∑ m ∈ range N, G (-((m : ℤ) + 1))
  | 0 => by simp
  | N + 1 => by
    have hs : Icc (-((N + 1 : ℕ) : ℤ)) ((N + 1 : ℕ) : ℤ)
        = insert ((N : ℤ) + 1) (insert (-((N : ℤ) + 1)) (Icc (-(N : ℤ)) N)) := by
      ext x; simp only [mem_Icc, mem_insert]; push_cast; omega
    rw [hs, sum_insert (by simp only [mem_insert, mem_Icc]; omega), sum_insert (by simp only [mem_Icc]; omega),
      Icc_split G N, sum_range_succ (fun m : ℕ => G (m : ℤ)) (N + 1),
      sum_range_succ (fun m : ℕ => G (-((m : ℤ) + 1))) N]
    push_cast
    abel

lemma wz_int {z : ℂ} (hz : z ≠ 0) (F : ℕ → ℂ) (N : ℕ) :
    ∑ a ∈ range (N + 1), wz z a * F a = ∑ m ∈ Icc (-(N : ℤ)) N, z ^ m * F m.natAbs := by
  rw [Icc_split, sum_range_succ', sum_range_succ' (fun m : ℕ => z ^ (m : ℤ) * F (m : ℤ).natAbs)]
  simp only [wz, if_pos rfl, one_mul, Nat.cast_zero, zpow_zero, Int.natAbs_zero, add_eq_zero, one_ne_zero,
    and_false, if_false]
  have h1 : ∀ m : ℕ, z ^ (-((m : ℤ) + 1)) = z⁻¹ ^ (m + 1) := fun m => by
    rw [zpow_neg, inv_pow, ← zpow_natCast]; push_cast; rfl
  simp only [h1, Int.natAbs_neg, Int.natAbs_natCast, show ∀ m : ℕ, ((m : ℤ) + 1).natAbs = m + 1 from
    fun m => by omega, zpow_natCast, add_mul, sum_add_distrib]
  push_cast
  ring_nf

/-- convolution over `ℤ`, for a kernel supported on `|m| + |b| ≤ n`. -/
lemma conv_regroup (z : ℂ) (n : ℕ) (F : ℤ → ℤ → ℂ)
    (hF : ∀ m b, n < m.natAbs + b.natAbs → F m b = 0) :
    ∑ m ∈ Icc (-(n : ℤ)) n, ∑ b ∈ Icc (-(n : ℤ)) n, z ^ (m + b) * F m b
      = ∑ c ∈ Icc (-(n : ℤ)) n, z ^ c * ∑ b ∈ Icc (-(n : ℤ)) n, F (c - b) b := by
  rw [sum_comm]
  have step : ∀ b ∈ Icc (-(n : ℤ)) n, ∑ m ∈ Icc (-(n : ℤ)) n, z ^ (m + b) * F m b
      = ∑ c ∈ Icc (-(2 * n : ℤ)) (2 * n), z ^ c * F (c - b) b := by
    intro b hb
    have hb' := mem_Icc.mp hb
    rw [← sum_subset (s₁ := Icc (b - n) (b + n)) (s₂ := Icc (-(2 * n : ℤ)) (2 * n)) (fun c hc => by simp only [mem_Icc] at hc ⊢; omega)
      (fun c _ hc => by
        simp only [mem_Icc, not_and_or, not_le] at hc
        rw [hF _ _ (by omega), mul_zero])]
    refine sum_nbij' (fun m => m + b) (fun c => c - b) (fun m hm => ?_) (fun c hc => ?_) (fun m _ => by ring)
      (fun c _ => by ring) (fun m _ => by rw [add_sub_cancel_right])
    · simp only [mem_Icc] at hm ⊢; omega
    · simp only [mem_Icc] at hc ⊢; omega
  rw [sum_congr rfl step, sum_comm]
  simp only [← mul_sum]
  symm
  refine sum_subset (fun c hc => by simp only [mem_Icc] at hc ⊢; omega) fun c _ hc => ?_
  simp only [mem_Icc, not_and_or, not_le] at hc
  rw [sum_eq_zero fun b hb => hF _ _ (by simp only [mem_Icc] at hb; omega), mul_zero]

/-- **product of two `w_a(z)`-expansions** with orders `≥ a`. -/
theorem wz_prod {z : ℂ} (hz : z ≠ 0) (f g : ℕ → PowerSeries ℂ) (hf : ∀ m, (X : PowerSeries ℂ) ^ m ∣ f m)
    (hg : ∀ m, (X : PowerSeries ℂ) ^ m ∣ g m) (n : ℕ) :
    coeff n ((∑ m ∈ range (n + 1), C (wz z m) * f m) * (∑ b ∈ range (n + 1), C (wz z b) * g b))
      = ∑ c ∈ range (n + 1), wz z c
        * coeff n (∑ b ∈ Icc (-(n : ℤ)) n, f ((c : ℤ) - b).natAbs * g b.natAbs) := by
  let F' : ℕ → ℕ → ℂ := fun m b => coeff n (f m * g b)
  let F : ℤ → ℤ → ℂ := fun m b => F' m.natAbs b.natAbs
  have hF : ∀ m b, n < m.natAbs + b.natAbs → F m b = 0 := fun m b h => by
    obtain ⟨d, hd⟩ := hf m.natAbs
    obtain ⟨e, he⟩ := hg b.natAbs
    show coeff n (f m.natAbs * g b.natAbs) = 0
    rw [hd, he, show X ^ m.natAbs * d * (X ^ b.natAbs * e) = X ^ (m.natAbs + b.natAbs) * (d * e) by ring]
    exact coeffC_Xpow_zero h _
  have hS : ∀ c : ℤ, ∑ b ∈ Icc (-(n : ℤ)) n, F (-c - b) b = ∑ b ∈ Icc (-(n : ℤ)) n, F (c - b) b := by
    intro c
    refine sum_nbij' (fun b => -b) (fun b => -b) (fun b hb => ?_) (fun b hb => ?_) (fun b _ => neg_neg b)
      (fun b _ => neg_neg b) (fun b _ => ?_)
    · simp only [mem_Icc] at hb ⊢; omega
    · simp only [mem_Icc] at hb ⊢; omega
    · show F' _ _ = F' _ _
      rw [show c - -b = -(-c - b) by ring, Int.natAbs_neg, Int.natAbs_neg]
  have hL : coeff n ((∑ m ∈ range (n + 1), C (wz z m) * f m) * (∑ b ∈ range (n + 1), C (wz z b) * g b))
      = ∑ m ∈ Icc (-(n : ℤ)) n, ∑ b ∈ Icc (-(n : ℤ)) n, z ^ (m + b) * F m b := by
    rw [sum_mul]
    simp only [mul_sum]
    have e1 : ∀ m b, coeff n (C (wz z m) * f m * (C (wz z b) * g b)) = wz z m * (wz z b * F' m b) := fun m b => by
      rw [show C (wz z m) * f m * (C (wz z b) * g b) = C (wz z m * wz z b) * (f m * g b) by
        rw [map_mul]; ring, coeff_C_mul, mul_assoc]
    simp only [map_sum, e1]
    simp only [← mul_sum]
    calc ∑ m ∈ range (n + 1), wz z m * ∑ b ∈ range (n + 1), wz z b * F' m b
        = ∑ m ∈ range (n + 1), wz z m * ∑ b ∈ Icc (-(n : ℤ)) n, z ^ b * F' m b.natAbs :=
          sum_congr rfl fun m _ => by rw [wz_int hz (F' m)]
      _ = ∑ m ∈ Icc (-(n : ℤ)) n, z ^ m * ∑ b ∈ Icc (-(n : ℤ)) n, z ^ b * F' m.natAbs b.natAbs :=
          wz_int hz (fun m => ∑ b ∈ Icc (-(n : ℤ)) n, z ^ b * F' m b.natAbs) n
      _ = _ := by
          refine sum_congr rfl fun m _ => ?_
          rw [mul_sum]
          refine sum_congr rfl fun b _ => ?_
          rw [zpow_add₀ hz]; ring
  have hR : ∑ c ∈ range (n + 1), wz z c
        * coeff n (∑ b ∈ Icc (-(n : ℤ)) n, f ((c : ℤ) - b).natAbs * g b.natAbs)
      = ∑ c ∈ Icc (-(n : ℤ)) n, z ^ c * ∑ b ∈ Icc (-(n : ℤ)) n, F (c - b) b := by
    rw [wz_int hz (fun c : ℕ => coeff n (∑ b ∈ Icc (-(n : ℤ)) n, f ((c : ℤ) - b).natAbs * g b.natAbs))]
    refine sum_congr rfl fun c _ => ?_
    congr 1
    rw [map_sum]
    have h2 : ∑ b ∈ Icc (-(n : ℤ)) n, coeff n (f (((c.natAbs : ℕ) : ℤ) - b).natAbs * g b.natAbs)
        = ∑ b ∈ Icc (-(n : ℤ)) n, F ((c.natAbs : ℤ) - b) b := rfl
    rw [h2]
    rcases Int.natAbs_eq c with h | h
    · rw [← h]
    · rw [show ((c.natAbs : ℕ) : ℤ) = -c by omega, hS]
  rw [hL, hR, conv_regroup z n F hF]

end ZConv

/-! ## the `z`-free blocks: Bailey + lattice -/

section Blocks

/-- the `b`-th block of (2.18): `(−1)^b q^{b(b+1)/2} Σ_m q^{m²+bm} β*_m(b)`. -/
noncomputable def h1Z (b : ℕ) : PowerSeries ℤ := C ((-1 : ℤ) ^ b) * X ^ (b * (b + 1) / 2) * RankProof.LHSz b

lemma E2_Cz (c : ℤ) : E2 (C c) = C c := by
  ext n
  have h : E2 (C c) = PowerSeries.expand 2 (by norm_num) (C c) := rfl
  rw [h, PowerSeries.coeff_expand, coeff_C, coeff_C]
  split_ifs <;> first | rfl | omega

lemma neg_one_pow_parity {a b : ℕ} (h : a % 2 = b % 2) : (-1 : ℤ) ^ a = (-1) ^ b := by
  rw [neg_one_pow_eq_pow_mod_two, h, ← neg_one_pow_eq_pow_mod_two]

lemma psi_unit_identity : qfacInf * E2 (Ring.inverse qfacInf) ^ 2 * psiSum = 1 := by
  have hodd := MockTheta5.JTP.qfac2Inf_mul_oddPochInf
  have hu2 : IsUnit qfac2Inf := isUnit_of_mul_isUnit_left (hodd ▸ isUnit_qfacInf)
  have huo : IsUnit MockTheta5.JTP.oddPochInf := isUnit_of_mul_isUnit_right (hodd ▸ isUnit_qfacInf)
  rw [← MockTheta5.JTP.psi_eq_series, MockTheta5.JTP.psi, MockTheta5.JTP.E2_inverse_qfacInf,
    ← MockTheta5.JTP.qfac2Inf]
  have h1 := Ring.mul_inverse_cancel _ hu2
  have h2 := Ring.mul_inverse_cancel _ huo
  rw [← hodd]
  linear_combination (qfac2Inf * Ring.inverse qfac2Inf) ^ 2 * h2 + (qfac2Inf * Ring.inverse qfac2Inf + 1) * h1

/-- **one `c`-block of (2.19)**: `(q)_∞ Σ_b A_{|c−b|} E₂(h1_{|b|}) ≡ (−1)^c G_c (mod q^{N+1})`. -/
theorem block_c (c N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣
    ∑ b ∈ Icc (-(N : ℤ)) N, qfacInf * Aterm ((c : ℤ) - b).natAbs * E2 (h1Z b.natAbs)
      - C ((-1 : ℤ) ^ c) * RankProof.Gsum c N := by
  have hterm : ∀ b : ℤ, qfacInf * Aterm ((c : ℤ) - b).natAbs * E2 (h1Z b.natAbs)
      = C ((-1 : ℤ) ^ c) * (qfacInf * E2 (Ring.inverse qfacInf) ^ 2)
        * (X ^ RankProof.eLb c b * E2 (RankProof.RHSz b.natAbs)) := fun b => by
    have htri : 2 * (b.natAbs * (b.natAbs + 1) / 2) = b.natAbs * (b.natAbs + 1) :=
      Nat.mul_div_cancel' (Nat.even_mul_succ_self _).two_dvd
    have hsign : (-1 : ℤ) ^ (((c : ℤ) - b).natAbs + b.natAbs) = (-1) ^ c := by
      apply neg_one_pow_parity; omega
    rw [Aterm, h1Z, RankProof.bailey_k, RankProof.eLb]
    simp only [map_mul, map_pow, E2_Cz, E2_X]
    have hsign' : (C (-1 : ℤ) : PowerSeries ℤ) ^ (((c : ℤ) - b).natAbs + b.natAbs) = C (-1) ^ c := by
      rw [← map_pow, ← map_pow, hsign]
    rw [← pow_mul, htri, ← hsign', pow_add, pow_add]
    ring
  simp only [hterm, ← mul_sum]
  rw [show (∑ b ∈ Icc (-(N : ℤ)) N, X ^ RankProof.eLb c b * E2 (RankProof.RHSz b.natAbs)) = RankProof.Lsum c N
    from rfl]
  have hL := RankProof.lattice_trunc c N
  obtain ⟨d, hd⟩ := hL
  rw [show C ((-1 : ℤ) ^ c) * (qfacInf * E2 (Ring.inverse qfacInf) ^ 2) * RankProof.Lsum c N
      - C ((-1 : ℤ) ^ c) * RankProof.Gsum c N
      = C ((-1 : ℤ) ^ c) * (qfacInf * E2 (Ring.inverse qfacInf) ^ 2)
        * (RankProof.Lsum c N - psiSum * RankProof.Gsum c N)
        + C ((-1 : ℤ) ^ c) * RankProof.Gsum c N * (qfacInf * E2 (Ring.inverse qfacInf) ^ 2 * psiSum - 1) by ring,
    psi_unit_identity, sub_self, mul_zero, add_zero, hd]
  exact dvd_mul_of_dvd_right (dvd_mul_right _ _) _

end Blocks

/-! ## assembly -/

section Assembly

lemma ψ_h1Z (a : ℕ) : ψ (h1Z a) = C ((-1) ^ a) * X ^ (a * (a + 1) / 2) * ψ (RankProof.LHSz a) := by
  rw [h1Z, RingHom.map_mul ψ, RingHom.map_mul ψ, PowerSeries.map_C, RingHom.map_pow ψ, PowerSeries.map_X]
  simp

lemma h1Z_dvd (a : ℕ) : (X : PowerSeries ℂ) ^ a ∣ ψ (h1Z a) := by
  rw [ψ_h1Z, mul_comm (C _), mul_assoc]
  exact dvd_mul_of_dvd_left (pow_dvd_pow X (MockTheta5.JTP.tri_ge a)) _

/-- **Garvan's (2.19)**: `(q)_∞(zq)_∞(q/z)_∞ R(z;q²) = Σ_a w_a(z) (−1)^a G_a(q)`, where
`G_a = Σ_{t≥0} q^{a(a+1)/2+3at+t(3t+1)/2}(1 − q^{2a+2t+1}) = Σ_{n≥a} q^{n(3n+1)/2−a²}(1−q^{2n+1})`
(truncated at `t ≤ n`, exact at `qⁿ`). -/
theorem hecke_rogers_19 {z : ℂ} (hz : z ≠ 0) (n : ℕ) :
    coeff n (pochInf 1 1 * pochInf z 1 * pochInf z⁻¹ 1 * E2C (Dser z z⁻¹))
      = ∑ a ∈ range (n + 1), wz z a * coeff n (ψ (C ((-1 : ℤ) ^ a) * RankProof.Gsum a n)) := by
  set PPD := pochInf z 1 * pochInf z⁻¹ 1 * Dser z z⁻¹ with hPPD
  have hH1 : ∀ m, coeff m PPD = ∑ a ∈ range (m + 1), wz z a * coeff m (ψ (h1Z a)) := fun m => by
    rw [hPPD, hecke_rogers_18 hz m]
    exact sum_congr rfl fun a _ => by rw [ψ_h1Z]
  have hT1 := trunc_of_coeff PPD _ h1Z_dvd hH1 n
  have hT1' : (X : PowerSeries ℂ) ^ (n + 1) ∣ E2C PPD
      - ∑ a ∈ range (n + 1), C (wz z a) * ψ (E2 (h1Z a)) := by
    have h := (pow_dvd_pow X (by omega : n + 1 ≤ 2 * (n + 1))).trans (E2C_dvd hT1)
    rw [map_sub, map_sum] at h
    simp only [map_mul, E2C_C] at h
    simpa only [ψ_E2] using h
  have hOdd := odd_jtp_trunc hz n
  have hser : pochInf 1 1 * pochInf z 1 * pochInf z⁻¹ 1 * E2C (Dser z z⁻¹)
      = (OddInf z * OddInf z⁻¹ * ψ qfacInf) * E2C PPD := by
    rw [hPPD, map_mul, map_mul, pochInf_one_eq]
    conv_lhs => rw [pochInf_eq_odd z, pochInf_eq_odd z⁻¹]
    ring
  set SA := ∑ a ∈ range (n + 1), C (wz z a) * ψ (Aterm a)
  set SB := ∑ a ∈ range (n + 1), C (wz z a) * ψ (E2 (h1Z a))
  have hSA : ψ qfacInf * SA = ∑ m ∈ range (n + 1), C (wz z m) * ψ (qfacInf * Aterm m) := by
    rw [mul_sum]
    exact sum_congr rfl fun m _ => by rw [RingHom.map_mul ψ]; ring
  have hf : ∀ m, (X : PowerSeries ℂ) ^ m ∣ ψ (qfacInf * Aterm m) := fun m => by
    rw [RingHom.map_mul ψ]; exact dvd_mul_of_dvd_right (Aterm_dvd m) _
  have hg : ∀ m, (X : PowerSeries ℂ) ^ m ∣ ψ (E2 (h1Z m)) := fun m => by
    rw [ψ_E2]; exact (pow_dvd_pow X (by omega)).trans (E2C_dvd (h1Z_dvd m))
  rw [hser, coeffC_congr hT1', show OddInf z * OddInf z⁻¹ * ψ qfacInf * SB
      = SB * ψ qfacInf * (OddInf z * OddInf z⁻¹) by ring, coeffC_congr hOdd,
    show SB * ψ qfacInf * SA = (ψ qfacInf * SA) * SB by ring, hSA, wz_prod hz _ _ hf hg n]
  refine sum_congr rfl fun c _ => ?_
  congr 1
  have hb := ψ_dvd (block_c c n)
  rw [map_sub] at hb
  rw [← coeff_eq_of_X_pow_dvd hb (by omega)]
  simp only [map_sum, RingHom.map_mul ψ]

end Assembly

end CrankProof
