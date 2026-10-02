/-
# Theta products `J_{a,b}` and the Jacobi triple product for them

`P b a = ∏_{i≥0} (1 − q^{b+ai})`, `J_{a,b} = P b a · P (a−b) a · P a a`, and
`J_{a,b} = Σ_{m∈ℤ} (−1)^m q^{a·m(m−1)/2 + bm}` (`0 < b < a`), proved by Euler-expanding the two
Pochhammers in base `q^a` and summing the `m = i − j` diagonals with the Durfee rectangle identity.
-/
import RamanujanTau.RankHR2

set_option autoImplicit false

namespace RankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf rectTerm rectPartial rectInf durfee_rect_base)

/-! ## step-`a` Pochhammers -/

section Poch

/-- `q ↦ q^a`. -/
noncomputable def Ea (a : ℕ) (ha : a ≠ 0) : PowerSeries ℤ →+* PowerSeries ℤ := (PowerSeries.expand a ha).toRingHom

lemma Ea_X (a : ℕ) (ha : a ≠ 0) : Ea a ha X = X ^ a := PowerSeries.expand_X a ha

lemma coeff_Ea (a : ℕ) (ha : a ≠ 0) (n : ℕ) (f : PowerSeries ℤ) :
    coeff n (Ea a ha f) = if a ∣ n then coeff (n / a) f else 0 := by
  have h : Ea a ha f = PowerSeries.expand a ha f := rfl
  rw [h, PowerSeries.coeff_expand]

lemma Ea_dvd (a : ℕ) (ha : a ≠ 0) {m : ℕ} {f : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ m ∣ f) :
    (X : PowerSeries ℤ) ^ (a * m) ∣ Ea a ha f := by
  obtain ⟨d, rfl⟩ := h
  exact ⟨Ea a ha d, by rw [map_mul, map_pow, Ea_X, ← pow_mul]⟩

noncomputable def Pfin (b a M : ℕ) : PowerSeries ℤ := ∏ i ∈ range M, (1 - X ^ (b + a * i))

lemma Pfin_succ (b a M : ℕ) : Pfin b a (M + 1) = Pfin b a M * (1 - X ^ (b + a * M)) := by
  rw [Pfin, prod_range_succ]; rfl

lemma X_pow_dvd_Pfin_sub (b a N : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) :
    ∀ M, N ≤ M → (X : PowerSeries ℤ) ^ (N + 1) ∣ Pfin b a M - Pfin b a N := by
  intro M hM
  induction M, hM using Nat.le_induction with
  | base => simp
  | succ M hNM ih =>
      rw [Pfin_succ, show Pfin b a M * (1 - X ^ (b + a * M)) - Pfin b a N
          = (Pfin b a M - Pfin b a N) - Pfin b a M * X ^ (b + a * M) by ring]
      exact dvd_sub ih (Dvd.dvd.mul_left (pow_dvd_pow X (by nlinarith)) _)

/-- `∏_{i≥0}(1 − q^{b+ai})`, by coefficient stabilization. -/
noncomputable def Pinf (b a : ℕ) : PowerSeries ℤ := mk fun k => coeff k (Pfin b a (k + 1))

lemma X_pow_dvd_Pinf_sub (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) (N : ℕ) :
    (X : PowerSeries ℤ) ^ (N + 1) ∣ Pinf b a - Pfin b a N := by
  rw [X_pow_dvd_iff]
  intro k hk
  have hst : ∀ M, k ≤ M → coeff k (Pfin b a M) = coeff k (Pfin b a k) := fun M hM => by
    have := (X_pow_dvd_iff.mp (X_pow_dvd_Pfin_sub b a k hb ha M hM)) k (by omega)
    rwa [map_sub, sub_eq_zero] at this
  rw [map_sub, Pinf, coeff_mk, sub_eq_zero, hst (k + 1) (by omega), hst N (by omega)]

lemma isUnit_Pinf (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) : IsUnit (Pinf b a) := by
  rw [isUnit_iff_constantCoeff, ← coeff_zero_eq_constantCoeff_apply]
  have := (X_pow_dvd_iff.mp (X_pow_dvd_Pinf_sub b a hb ha 0)) 0 (by omega)
  rw [map_sub, sub_eq_zero, Pfin] at this
  rw [this]; simp

/-- equality of two stabilized products from equality of their truncations. -/
lemma Pinf_ext {f : PowerSeries ℤ} {b a : ℕ} (hb : 1 ≤ b) (ha : 1 ≤ a)
    (h : ∀ N, (X : PowerSeries ℤ) ^ (N + 1) ∣ f - Pfin b a N) : f = Pinf b a := by
  ext k
  have h1 := (X_pow_dvd_iff.mp (h k)) k (by omega)
  have h2 := (X_pow_dvd_iff.mp (X_pow_dvd_Pinf_sub b a hb ha k)) k (by omega)
  rw [map_sub, sub_eq_zero] at h1 h2
  rw [h1, h2]

lemma Pfin_split (b a : ℕ) : ∀ M, Pfin b a (2 * M) = Pfin b (2 * a) M * Pfin (b + a) (2 * a) M
  | 0 => by simp [Pfin]
  | M + 1 => by
    rw [show 2 * (M + 1) = 2 * M + 1 + 1 by ring, Pfin_succ, Pfin_succ, Pfin_split b a M, Pfin_succ, Pfin_succ]
    rw [show b + a * (2 * M + 1) = b + a + 2 * a * M by ring, show b + a * (2 * M) = b + 2 * a * M by ring]
    ring

/-- `(q^b;q^a)_∞ = (q^b;q^{2a})_∞ (q^{b+a};q^{2a})_∞`. -/
theorem Pinf_split (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) : Pinf b a = Pinf b (2 * a) * Pinf (b + a) (2 * a) := by
  symm
  refine Pinf_ext hb ha fun N => ?_
  have h1 := X_pow_dvd_Pinf_sub b (2 * a) hb (by omega) N
  have h2 := X_pow_dvd_Pinf_sub (b + a) (2 * a) (by omega) (by omega) N
  have h3 := X_pow_dvd_Pfin_sub b a N hb ha (2 * N) (by omega)
  rw [show Pinf b (2 * a) * Pinf (b + a) (2 * a) - Pfin b a N
      = (Pinf b (2 * a) - Pfin b (2 * a) N) * Pinf (b + a) (2 * a)
        + Pfin b (2 * a) N * (Pinf (b + a) (2 * a) - Pfin (b + a) (2 * a) N)
        + (Pfin b a (2 * N) - Pfin b a N) by rw [Pfin_split]; ring]
  exact dvd_add (dvd_add (dvd_mul_of_dvd_left h1 _) (dvd_mul_of_dvd_right h2 _)) h3

lemma Ea_Pfin (c b a M : ℕ) (hc : c ≠ 0) : Ea c hc (Pfin b a M) = Pfin (c * b) (c * a) M := by
  rw [Pfin, Pfin, map_prod]
  refine prod_congr rfl fun i _ => ?_
  rw [map_sub, map_one, map_pow, Ea_X, ← pow_mul]; congr 2; ring

/-- `(q^b;q^a)_∞ ↦ (q^{cb};q^{ca})_∞` under `q ↦ q^c`. -/
theorem Ea_Pinf (c b a : ℕ) (hc : c ≠ 0) (hb : 1 ≤ b) (ha : 1 ≤ a) :
    Ea c hc (Pinf b a) = Pinf (c * b) (c * a) := by
  refine Pinf_ext (Nat.one_le_iff_ne_zero.mpr (by positivity)) (Nat.one_le_iff_ne_zero.mpr (by positivity))
    fun N => ?_
  rw [← Ea_Pfin c b a N hc, ← map_sub]
  exact (pow_dvd_pow X (by nlinarith [Nat.one_le_iff_ne_zero.mpr hc])).trans (Ea_dvd c hc (X_pow_dvd_Pinf_sub b a hb ha N))

lemma Pfin_one_one (M : ℕ) : Pfin 1 1 M = qfac M := by
  rw [Pfin, qfac]; refine prod_congr rfl fun i _ => ?_; congr 2; ring

theorem Pinf_one_one : Pinf 1 1 = qfacInf := by
  symm
  refine Pinf_ext le_rfl le_rfl fun N => ?_
  rw [Pfin_one_one]
  have h1 := MockTheta5.JTP.X_pow_dvd_qfacInf_sub (N + 1) (N + 1) le_rfl
  have h2 : (X : PowerSeries ℤ) ^ (N + 1) ∣ qfac (N + 1) - qfac N := by
    rw [← Pfin_one_one, ← Pfin_one_one]; exact X_pow_dvd_Pfin_sub 1 1 N le_rfl le_rfl (N + 1) (by omega)
  have := dvd_add h1 h2
  rwa [sub_add_sub_cancel] at this

end Poch

/-! ## Euler's expansion of `(q^b;q^a)_∞` -/

section Euler

lemma Pfin_qbinom (b a : ℕ) (ha : a ≠ 0) (M : ℕ) :
    Pfin b a M = ∑ k ∈ range (M + 1), C ((-1 : ℤ) ^ k) * X ^ (a * k.choose 2 + b * k)
      * Ea a ha (gaussBinom M k) := by
  have h := congrArg (fun p => Polynomial.eval₂ (Ea a ha) (-(X ^ b)) p) (qbinom M)
  simp only [qprod, qbRHS, Polynomial.eval₂_finset_prod, Polynomial.eval₂_finset_sum, Polynomial.eval₂_add,
    Polynomial.eval₂_one, Polynomial.eval₂_mul, Polynomial.eval₂_C, Polynomial.eval₂_X,
    Polynomial.eval₂_pow, qcoeff, map_mul, map_pow, Ea_X, qq] at h
  rw [Pfin]
  convert h using 2 with i _ k _
  · rw [← pow_mul, pow_add]; ring
  · rw [neg_pow (X ^ b), ← pow_mul, pow_add]
    simp only [one_pow, mul_one, map_pow, map_neg, map_one]
    ring

/-- **Euler**, truncated: `(q^b;q^a)_∞ ≡ Σ_{i≤N} (−1)^i q^{a·C(i,2)+bi}/(q^a;q^a)_i (mod q^{N+1})`. -/
theorem Pinf_euler_dvd (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) (N : ℕ) :
    (X : PowerSeries ℤ) ^ (N + 1) ∣ Pinf b a - ∑ i ∈ range (N + 1),
      C ((-1 : ℤ) ^ i) * X ^ (a * i.choose 2 + b * i) * Ea a (by omega) (Ring.inverse (qfac i)) := by
  have ha0 : a ≠ 0 := by omega
  set M := 2 * N + 1
  have h1 : (X : PowerSeries ℤ) ^ (N + 1) ∣ Pinf b a - Pfin b a M :=
    (pow_dvd_pow X (by omega)).trans (X_pow_dvd_Pinf_sub b a hb ha M)
  have h2 : (X : PowerSeries ℤ) ^ (N + 1) ∣ Pfin b a M - ∑ i ∈ range (N + 1),
      C ((-1 : ℤ) ^ i) * X ^ (a * i.choose 2 + b * i) * Ea a ha0 (Ring.inverse (qfac i)) := by
    rw [Pfin_qbinom b a ha0, show M + 1 = (N + 1) + (N + 1) by omega, sum_range_add, add_sub_right_comm,
      ← sum_sub_distrib]
    refine dvd_add (dvd_sum fun k hk => ?_) (dvd_sum fun k _ => ?_)
    · have hk' := mem_range.mp hk
      rw [← mul_sub, ← map_sub]
      have hMk : N + 1 ≤ M - k + 1 := by omega
      exact dvd_mul_of_dvd_right ((pow_dvd_pow X (by nlinarith)).trans
        (Ea_dvd a ha0 (CrankProof.gauss_dvd M k (by omega)))) _
    · rw [show a * (N + 1 + k).choose 2 + b * (N + 1 + k)
          = (N + 1) + (a * (N + 1 + k).choose 2 + b * (N + 1 + k) - (N + 1)) by
        have : N + 1 ≤ b * (N + 1 + k) := by nlinarith
        omega, pow_add]
      exact ⟨C ((-1 : ℤ) ^ (N + 1 + k)) * X ^ (a * (N + 1 + k).choose 2 + b * (N + 1 + k) - (N + 1))
        * Ea a ha0 (gaussBinom M (N + 1 + k)), by ring⟩
  have := dvd_add h1 h2
  rwa [sub_add_sub_cancel] at this

end Euler

/-! ## the Jacobi triple product for `J_{a,b}` -/

section JTP

lemma sum_square_splitZ (g : ℕ → ℕ → ℤ) : ∀ N : ℕ,
    ∑ i ∈ range N, ∑ j ∈ range N, g i j
      = ∑ a ∈ range N, ∑ j ∈ range (N - a), g (j + a) j + ∑ b ∈ range N, ∑ i ∈ range (N - b - 1), g i (i + b + 1) := by
  intro N
  induction N with
  | zero => simp
  | succ N ih =>
      have hrow : ∑ j ∈ range (N + 1), g N j = ∑ a ∈ range (N + 1), g N (N - a) := by
        rw [← sum_range_reflect]; exact sum_congr rfl fun a ha => by
          rw [show N + 1 - 1 - a = N - a by omega]
      have hcol : ∑ i ∈ range N, g i N = ∑ b ∈ range N, g (N - b - 1) N := by
        rw [← sum_range_reflect]; exact sum_congr rfl fun b hb => by
          rw [show N - 1 - b = N - b - 1 by omega]
      have e1 : ∑ i ∈ range (N + 1), ∑ j ∈ range (N + 1), g i j
          = ∑ i ∈ range N, ∑ j ∈ range N, g i j + ∑ i ∈ range N, g i N + ∑ j ∈ range (N + 1), g N j := by
        rw [sum_range_succ (fun i => ∑ j ∈ range (N + 1), g i j)]
        simp only [sum_range_succ (fun j => g _ j) N, sum_add_distrib]
      have e2 : ∑ a ∈ range (N + 1), ∑ j ∈ range (N + 1 - a), g (j + a) j
          = ∑ a ∈ range N, ∑ j ∈ range (N - a), g (j + a) j + ∑ a ∈ range (N + 1), g N (N - a) := by
        rw [sum_range_succ (fun a => ∑ j ∈ range (N + 1 - a), g (j + a) j),
          sum_range_succ (fun a => g N (N - a)), show N + 1 - N = 1 by omega, sum_range_one, zero_add,
          show N - N = 0 by omega, ← add_assoc, ← sum_add_distrib]
        congr 1
        refine sum_congr rfl fun a ha => ?_
        have ha' := mem_range.mp ha
        rw [show N + 1 - a = N - a + 1 by omega, sum_range_succ, show N - a + a = N by omega]
      have e3 : ∑ b ∈ range (N + 1), ∑ i ∈ range (N + 1 - b - 1), g i (i + b + 1)
          = ∑ b ∈ range N, ∑ i ∈ range (N - b - 1), g i (i + b + 1) + ∑ b ∈ range N, g (N - b - 1) N := by
        rw [sum_range_succ (fun b => ∑ i ∈ range (N + 1 - b - 1), g i (i + b + 1)),
          show N + 1 - N - 1 = 0 by omega, range_zero, sum_empty, add_zero, ← sum_add_distrib]
        refine sum_congr rfl fun b hb => ?_
        have hb' := mem_range.mp hb
        rw [show N + 1 - b - 1 = N - b - 1 + 1 by omega, sum_range_succ, show N - b - 1 + b + 1 = N by omega]
      rw [e1, e2, e3, ih, ← hrow, ← hcol]
      ring


lemma coeffZ_congr {n : ℕ} {f g g' : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ (n + 1) ∣ g - g') :
    coeff n (f * g) = coeff n (f * g') := coeff_mul_congr_right' h

lemma coeffZ_Xpow_zero {n d : ℕ} (h : n < d) (f : PowerSeries ℤ) : coeff n (X ^ d * f) = 0 := by
  rw [coeff_X_pow_mul', if_neg (by omega)]

/-- the Durfee rectangle at base `q^a`, truncated. -/
lemma Ea_rect_dvd (a : ℕ) (ha : 1 ≤ a) (d n : ℕ) :
    (X : PowerSeries ℤ) ^ (n + 1) ∣ Ea a (by omega) (Ring.inverse qfacInf) - Ea a (by omega) (rectPartial d (n + 1)) := by
  rw [← map_sub, ← durfee_rect_base d]
  exact (pow_dvd_pow X (by nlinarith)).trans (Ea_dvd a (by omega) (rect_dvd d n))

lemma Ea_rectTerm (a : ℕ) (ha : a ≠ 0) (d j : ℕ) : Ea a ha (rectTerm d j)
    = X ^ (a * (j ^ 2 + d * j)) * (Ea a ha (Ring.inverse (qfac j)) * Ea a ha (Ring.inverse (qfac (d + j)))) := by
  rw [rectTerm, RingHom.map_mul (Ea a ha), RingHom.map_mul (Ea a ha), RingHom.map_pow (Ea a ha), Ea_X, ← pow_mul]
  ring

variable (a b : ℕ)

/-- the theta series `Σ_{m∈ℤ} (−1)^m q^{a·C(m,2)+bm}`, truncated (`m = d ≥ 0` and `m = −(d+1)`). -/
noncomputable def thA (d : ℕ) : PowerSeries ℤ := C ((-1 : ℤ) ^ d) * X ^ (a * d.choose 2 + b * d)
noncomputable def thB (d : ℕ) : PowerSeries ℤ :=
  C ((-1 : ℤ) ^ (d + 1)) * X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1))

noncomputable def thetaTr (n : ℕ) : PowerSeries ℤ :=
  ∑ d ∈ range (n + 1), C ((-1 : ℤ) ^ d) * X ^ (a * d.choose 2 + b * d)
    + ∑ d ∈ range (n + 1), C ((-1 : ℤ) ^ (d + 1)) * X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1))

lemma thetaTr_eq (n : ℕ) : thetaTr a b n = ∑ d ∈ range (n + 1), thA a b d + ∑ d ∈ range (n + 1), thB a b d := rfl

theorem jtp_coeff (hb : 1 ≤ b) (hab : b < a) (n : ℕ) :
    coeff n (Pinf b a * Pinf (a - b) a) = coeff n (Ea a (by omega) (Ring.inverse qfacInf) * thetaTr a b n) := by
  have ha0 : a ≠ 0 := by omega
  set IE := Ea a ha0 (Ring.inverse qfacInf)
  let g : ℕ → ℕ → ℤ := fun i j => (-1) ^ (i + j) * coeff n (X ^ (a * i.choose 2 + b * i + (a * j.choose 2
    + (a - b) * j)) * Ea a ha0 (Ring.inverse (qfac i) * Ring.inverse (qfac j)))
  have hexp : coeff n (Pinf b a * Pinf (a - b) a) = ∑ i ∈ range (n + 1), ∑ j ∈ range (n + 1), g i j := by
    rw [coeffZ_congr (Pinf_euler_dvd (a - b) a (by omega) (by omega) n), mul_comm,
      coeffZ_congr (Pinf_euler_dvd b a hb (by omega) n), mul_sum]
    simp only [sum_mul, map_sum]
    refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
    show _ = (-1) ^ (i + j) * _
    rw [show C ((-1 : ℤ) ^ j) * X ^ (a * j.choose 2 + (a - b) * j) * Ea a ha0 (Ring.inverse (qfac j))
        * (C ((-1 : ℤ) ^ i) * X ^ (a * i.choose 2 + b * i) * Ea a ha0 (Ring.inverse (qfac i)))
        = C ((-1 : ℤ) ^ (i + j)) * (X ^ (a * i.choose 2 + b * i + (a * j.choose 2 + (a - b) * j))
          * Ea a ha0 (Ring.inverse (qfac i) * Ring.inverse (qfac j))) by
      rw [pow_add (-1 : ℤ), map_mul, pow_add X, map_mul]; ring, coeff_C_mul]
  -- lower blocks
  have hlow : ∀ d ∈ range (n + 1), ∑ j ∈ range (n + 1 - d), g (j + d) j
      = (-1) ^ d * coeff n (X ^ (a * d.choose 2 + b * d) * IE) := by
    intro d _
    have hg : ∀ j, g (j + d) j = (-1) ^ d * coeff n (X ^ (a * d.choose 2 + b * d) * Ea a ha0 (rectTerm d j)) := by
      intro j
      show (-1) ^ (j + d + j) * _ = _
      have hE : a * (j + d).choose 2 + b * (j + d) + (a * j.choose 2 + (a - b) * j)
          = a * d.choose 2 + b * d + a * (j ^ 2 + d * j) := by
        have h1 := CrankProof.two_choose_two (j + d)
        have h2 := CrankProof.two_choose_two j
        have h3 := CrankProof.two_choose_two d
        zify [show b ≤ a by omega]
        push_cast at h1
        nlinarith
      rw [show j + d + j = d + 2 * j by ring, pow_add, pow_mul, neg_one_sq, one_pow, mul_one, Ea_rectTerm, hE,
        RingHom.map_mul (Ea a ha0), pow_add, add_comm d j]
      ring_nf
    rw [sum_congr rfl fun j _ => hg j, ← mul_sum]
    congr 1
    rw [show ∑ j ∈ range (n + 1 - d), coeff n (X ^ (a * d.choose 2 + b * d) * Ea a ha0 (rectTerm d j))
        = ∑ j ∈ range (n + 1), coeff n (X ^ (a * d.choose 2 + b * d) * Ea a ha0 (rectTerm d j)) by
      refine sum_subset (range_subset_range.mpr (by omega)) fun j _ hj => ?_
      simp only [mem_range, not_lt] at hj
      rw [Ea_rectTerm, ← mul_assoc, ← pow_add]
      have h1 : j ≤ a * (j ^ 2 + d * j) := le_trans (Nat.le_self_pow two_ne_zero j)
        (le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_left _ (by omega)))
      have h2 : d ≤ b * d := Nat.le_mul_of_pos_left _ (by omega)
      exact coeffZ_Xpow_zero (by omega) _,
      ← map_sum, ← mul_sum, ← map_sum, ← rectPartial, coeffZ_congr (Ea_rect_dvd a (by omega) d n)]
  -- upper blocks
  have hup : ∀ d ∈ range (n + 1), ∑ i ∈ range (n + 1 - d - 1), g i (i + d + 1)
      = (-1) ^ (d + 1) * coeff n (X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1)) * IE) := by
    intro d _
    have hg : ∀ i, g i (i + d + 1) = (-1) ^ (d + 1)
        * coeff n (X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1)) * Ea a ha0 (rectTerm (d + 1) i)) := by
      intro i
      show (-1) ^ (i + (i + d + 1)) * _ = _
      have hE : a * i.choose 2 + b * i + (a * (i + d + 1).choose 2 + (a - b) * (i + d + 1))
          = a * (d + 1).choose 2 + (a - b) * (d + 1) + a * (i ^ 2 + (d + 1) * i) := by
        have h1 := CrankProof.two_choose_two (i + d + 1)
        have h2 := CrankProof.two_choose_two i
        have h3 := CrankProof.two_choose_two (d + 1)
        zify [show b ≤ a by omega]
        push_cast at h1 h3
        nlinarith
      rw [show i + (i + d + 1) = (d + 1) + 2 * i by ring, pow_add, pow_mul, neg_one_sq, one_pow, mul_one,
        Ea_rectTerm, hE, RingHom.map_mul (Ea a ha0), pow_add, show d + 1 + i = i + d + 1 by ring]
      ring_nf
    rw [sum_congr rfl fun i _ => hg i, ← mul_sum]
    congr 1
    rw [show ∑ i ∈ range (n + 1 - d - 1), coeff n (X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1))
          * Ea a ha0 (rectTerm (d + 1) i))
        = ∑ i ∈ range (n + 1), coeff n (X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1))
          * Ea a ha0 (rectTerm (d + 1) i)) by
      refine sum_subset (range_subset_range.mpr (by omega)) fun i _ hi => ?_
      simp only [mem_range, not_lt] at hi
      rw [Ea_rectTerm, ← mul_assoc, ← pow_add]
      have : d + 1 ≤ (a - b) * (d + 1) := Nat.le_mul_of_pos_left _ (by omega)
      have h1 : i ≤ a * (i ^ 2 + (d + 1) * i) := le_trans (Nat.le_self_pow two_ne_zero i)
        (le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_left _ (by omega)))
      exact coeffZ_Xpow_zero (by omega) _,
      ← map_sum, ← mul_sum, ← map_sum, ← rectPartial, coeffZ_congr (Ea_rect_dvd a (by omega) (d + 1) n)]
  rw [hexp, sum_square_splitZ, sum_congr rfl hlow, sum_congr rfl hup, thetaTr, mul_add, map_add, mul_sum, mul_sum,
    map_sum, map_sum]
  congr 1
  · refine sum_congr rfl fun d _ => ?_
    rw [show IE * (C ((-1 : ℤ) ^ d) * X ^ (a * d.choose 2 + b * d)) = C ((-1 : ℤ) ^ d)
      * (X ^ (a * d.choose 2 + b * d) * IE) by ring, coeff_C_mul]
  · refine sum_congr rfl fun d _ => ?_
    rw [show IE * (C ((-1 : ℤ) ^ (d + 1)) * X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1))) = C ((-1 : ℤ) ^ (d + 1))
      * (X ^ (a * (d + 1).choose 2 + (a - b) * (d + 1)) * IE) by ring, coeff_C_mul]

lemma thetaTr_dvd (hb : 1 ≤ b) (hab : b < a) {m n : ℕ} (hmn : m ≤ n) :
    (X : PowerSeries ℤ) ^ (m + 1) ∣ thetaTr a b n - thetaTr a b m := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  rw [thetaTr_eq, thetaTr_eq, show m + k + 1 = (m + 1) + k by ring, sum_range_add (thA a b), sum_range_add (thB a b),
    show ∑ x ∈ range (m + 1), thA a b x + ∑ x ∈ range k, thA a b (m + 1 + x)
      + (∑ x ∈ range (m + 1), thB a b x + ∑ x ∈ range k, thB a b (m + 1 + x))
      - (∑ d ∈ range (m + 1), thA a b d + ∑ d ∈ range (m + 1), thB a b d)
      = ∑ x ∈ range k, thA a b (m + 1 + x) + ∑ x ∈ range k, thB a b (m + 1 + x) by ring]
  refine dvd_add (dvd_sum fun x _ => ?_) (dvd_sum fun x _ => ?_)
  · have : m + 1 + x ≤ b * (m + 1 + x) := Nat.le_mul_of_pos_left _ (by omega)
    exact dvd_mul_of_dvd_right (pow_dvd_pow X (by omega)) _
  · have : m + 1 + x + 1 ≤ (a - b) * (m + 1 + x + 1) := Nat.le_mul_of_pos_left _ (by omega)
    exact dvd_mul_of_dvd_right (pow_dvd_pow X (by omega)) _

/-- the theta series `Σ_{m∈ℤ} (−1)^m q^{a·C(m,2)+bm}`. -/
noncomputable def thetaS : PowerSeries ℤ := mk fun n => coeff n (thetaTr a b n)

lemma thetaS_dvd (hb : 1 ≤ b) (hab : b < a) (n : ℕ) :
    (X : PowerSeries ℤ) ^ (n + 1) ∣ thetaS a b - thetaTr a b n := by
  rw [X_pow_dvd_iff]
  intro m hm
  have := (X_pow_dvd_iff.mp (thetaTr_dvd a b hb hab (show m ≤ n by omega))) m (by omega)
  rw [map_sub, sub_eq_zero] at this
  rw [map_sub, thetaS, coeff_mk, this, sub_self]

lemma Pinf_aa (a : ℕ) (ha : a ≠ 0) : Pinf a a = Ea a ha qfacInf := by
  rw [← Pinf_one_one, Ea_Pinf a 1 1 ha le_rfl le_rfl, mul_one]

/-- **Jacobi's triple product for `J_{a,b}`**:
`(q^b;q^a)_∞ (q^{a−b};q^a)_∞ (q^a;q^a)_∞ = Σ_{m∈ℤ} (−1)^m q^{a·m(m−1)/2+bm}`. -/
theorem jtp_ab (hb : 1 ≤ b) (hab : b < a) : Pinf b a * Pinf (a - b) a * Pinf a a = thetaS a b := by
  have ha0 : a ≠ 0 := by omega
  have hu : Ea a ha0 (Ring.inverse qfacInf) * Pinf a a = 1 := by
    rw [Pinf_aa a ha0, ← map_mul, Ring.inverse_mul_cancel _ isUnit_qfacInf, map_one]
  have key : ∀ n, (X : PowerSeries ℤ) ^ (n + 1) ∣ Pinf b a * Pinf (a - b) a
      - Ea a ha0 (Ring.inverse qfacInf) * thetaTr a b n := by
    intro n
    rw [X_pow_dvd_iff]
    intro m hm
    rw [map_sub, sub_eq_zero, jtp_coeff a b hb hab m]
    exact (coeffZ_congr (thetaTr_dvd a b hb hab (show m ≤ n by omega))).symm
  ext n
  have h1 : coeff n (Pinf b a * Pinf (a - b) a * Pinf a a)
      = coeff n (Pinf a a * (Ea a ha0 (Ring.inverse qfacInf) * thetaTr a b n)) := by
    rw [mul_comm, coeffZ_congr (key n)]
  rw [h1, ← mul_assoc, mul_comm (Pinf a a), hu, one_mul, coeff_eq_of_dvd (thetaS_dvd a b hb hab n) le_rfl]

end JTP

end RankProof
