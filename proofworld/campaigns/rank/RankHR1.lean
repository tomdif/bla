/-
# Dyson's rank: Garvan's Hecke–Rogers identity (2.18) from the Durfee form

`(zq;q)_∞(q/z;q)_∞(q;q)_∞ · R(z;q) = Σ_{a∈ℤ} z^a (−1)^a q^{a(a+1)/2} PT_{|a|}(q)`.
Step H1: Euler's second identity `(cq^s;q)_∞ = Σ_i (−c)^i q^{C(i,2)+si}/(q;q)_i` (truncated form).
-/
import RamanujanTau.CrankAndrewsGarvan
import RamanujanTau.RankBailey
import RamanujanTau.RankGF
import RamanujanTau.MockTheta5EulerCauchy

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset

section Euler2
open MockTheta5.Bailey
local notation "ψ" => MockTheta5.JTP.ψC

lemma gauss_dvd (M k : ℕ) (hk : k ≤ M) :
    (X : PowerSeries ℤ) ^ (M - k + 1) ∣ gaussBinom M k - Ring.inverse (qfac k) := by
  rw [PowerSeries.X_pow_dvd_iff]
  intro m hm
  rw [map_sub, gaussBinom_stable M k hk hm, sub_self]

lemma ψ_dvd {K : ℕ} {f : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ K ∣ f) : (X : PowerSeries ℂ) ^ K ∣ ψ f :=
  MockTheta5.JTP.X_pow_dvd_ψC h

/-- the finite product, expanded by the `q`-binomial theorem. -/
lemma poch_qbinom (c : ℂ) (s M : ℕ) :
    poch c s M = ∑ k ∈ range (M + 1), C ((-c) ^ k) * X ^ (k.choose 2 + s * k) * ψ (gaussBinom M k) := by
  have h := congrArg (fun p => Polynomial.eval₂ ψ (-(C c * X ^ s)) p) (qbinom M)
  simp only [qprod, qbRHS, Polynomial.eval₂_finset_prod, Polynomial.eval₂_finset_sum, Polynomial.eval₂_add,
    Polynomial.eval₂_one, Polynomial.eval₂_mul, Polynomial.eval₂_C, Polynomial.eval₂_X, Polynomial.eval₂_X_pow,
    Polynomial.eval₂_pow, qcoeff, map_mul, map_pow, PowerSeries.map_X, qq] at h
  rw [poch]
  convert h using 2 with i _ k _
  · rw [pow_add]; ring
  · rw [neg_pow (C c * X ^ s), mul_pow, ← pow_mul, pow_add, mul_comm s k]
    simp only [map_mul, map_pow, map_neg, map_one, one_pow, mul_one]; ring

/-- **Euler's second identity**, truncated: `X^{N+1} ∣ (cq^s;q)_∞ − Σ_{i≤N} (−c)^i q^{C(i,2)+si}/(q;q)_i`. -/
theorem euler2_dvd (c : ℂ) {s : ℕ} (hs : 1 ≤ s) (N : ℕ) :
    (X : PowerSeries ℂ) ^ (N + 1) ∣ pochInf c s
      - ∑ i ∈ range (N + 1), C ((-c) ^ i) * X ^ (i.choose 2 + s * i) * ψ (Ring.inverse (qfac i)) := by
  set M := 2 * N + 1
  have h1 : (X : PowerSeries ℂ) ^ (N + 1) ∣ pochInf c s - poch c s M :=
    (pow_dvd_pow X (by omega)).trans (X_pow_dvd_pochInf_sub c hs M)
  have h2 : (X : PowerSeries ℂ) ^ (N + 1) ∣ poch c s M
      - ∑ i ∈ range (N + 1), C ((-c) ^ i) * X ^ (i.choose 2 + s * i) * ψ (Ring.inverse (qfac i)) := by
    rw [poch_qbinom, show M + 1 = (N + 1) + (N + 1) by omega, sum_range_add, add_sub_right_comm,
      ← sum_sub_distrib]
    refine dvd_add (dvd_sum fun k hk => ?_) (dvd_sum fun k _ => ?_)
    · have hk' := mem_range.mp hk
      rw [← mul_sub, ← map_sub]
      exact dvd_mul_of_dvd_right (ψ_dvd ((pow_dvd_pow X (by omega)).trans (gauss_dvd M k (by omega)))) _
    · rw [show (N + 1 + k).choose 2 + s * (N + 1 + k) = (N + 1) + ((N + 1 + k).choose 2 + s * (N + 1 + k) - (N + 1))
        by have : N + 1 ≤ s * (N + 1 + k) := by nlinarith
           omega, pow_add]
      exact ⟨C ((-c) ^ (N + 1 + k)) * X ^ ((N + 1 + k).choose 2 + s * (N + 1 + k) - (N + 1))
        * ψ (gaussBinom M (N + 1 + k)), by ring⟩
  have := dvd_add h1 h2
  rwa [sub_add_sub_cancel] at this

end Euler2

/-! ## H2: regrouping a double sum by `i − j` -/

/-- the square `[0,N)²` = lower triangle (incl. diagonal, by `a = i − j`) + strict upper triangle. -/
lemma sum_square_split (g : ℕ → ℕ → ℂ) : ∀ N : ℕ,
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

/-- weight `w₀ = 1`, `w_a = z^a + z^{−a}`. -/
noncomputable def wz (z : ℂ) (a : ℕ) : ℂ := if a = 0 then 1 else z ^ a + z⁻¹ ^ a

/-- regrouping a symmetric double sum weighted by `z^i z^{−j}` by `a = |i − j|`. -/
lemma sum_regroup {z : ℂ} (hz : z ≠ 0) (f : ℕ → ℕ → ℂ) (hf : ∀ i j, f i j = f j i) (N : ℕ) :
    ∑ i ∈ range N, ∑ j ∈ range N, z ^ i * z⁻¹ ^ j * f i j
      = ∑ a ∈ range N, wz z a * ∑ j ∈ range (N - a), f (j + a) j := by
  have hzz : ∀ m : ℕ, z ^ m * z⁻¹ ^ m = 1 := fun m => by rw [← mul_pow, mul_inv_cancel₀ hz, one_pow]
  rw [sum_square_split]
  have hA : ∀ a j, z ^ (j + a) * z⁻¹ ^ j * f (j + a) j = z ^ a * f (j + a) j := fun a j => by
    rw [pow_add]; linear_combination (z ^ a * f (j + a) j) * hzz j
  have hB : ∀ b i, z ^ i * z⁻¹ ^ (i + b + 1) * f i (i + b + 1) = z⁻¹ ^ (b + 1) * f (i + (b + 1)) i := fun b i => by
    rw [show i + b + 1 = i + (b + 1) by ring, pow_add z⁻¹, ← mul_assoc, hzz, one_mul, hf]
  simp only [hA, hB, ← mul_sum]
  have hw : ∀ a, wz z a = z ^ a + (if a = 0 then 0 else z⁻¹ ^ a) := by
    intro a; unfold wz; split_ifs with h
    · subst h; simp
    · rfl
  simp only [hw, add_mul, sum_add_distrib]
  congr 1
  rcases N with _ | M
  · simp
  rw [sum_range_succ (fun x => z⁻¹ ^ (x + 1) * ∑ i ∈ range (M + 1 - x - 1), f (i + (x + 1)) i),
    show M + 1 - M - 1 = 0 by omega, range_zero, sum_empty, mul_zero, add_zero,
    sum_range_succ' (fun a => (if a = 0 then (0 : ℂ) else z⁻¹ ^ a) * ∑ j ∈ range (M + 1 - a), f (j + a) j),
    if_pos rfl, zero_mul, add_zero]
  refine sum_congr rfl fun x _ => ?_
  rw [if_neg (by omega), show M + 1 - x - 1 = M + 1 - (x + 1) by omega]

/-! ## H3: exponent bookkeeping and truncations -/

lemma two_choose_two (m : ℕ) : 2 * (m.choose 2 : ℤ) = m * ((m : ℤ) - 1) := by
  induction m with
  | zero => simp
  | succ m ih => rw [Nat.choose_succ_succ, Nat.choose_one_right]; push_cast; linear_combination ih

lemma two_tri (a : ℕ) : 2 * ((a * (a + 1) / 2 : ℕ) : ℤ) = a * ((a : ℤ) + 1) := by
  have h : 2 ∣ a * (a + 1) := (Nat.even_mul_succ_self a).two_dvd
  have : 2 * (a * (a + 1) / 2) = a * (a + 1) := Nat.mul_div_cancel' h
  exact_mod_cast this

lemma exp_HR1 (a j k : ℕ) :
    k ^ 2 + (j + a).choose 2 + j.choose 2 + (1 + k) * ((j + a) + j)
      = a * (a + 1) / 2 + ((j + k) ^ 2 + a * (j + k)) + j := by
  have h1 := two_choose_two (j + a)
  have h2 := two_choose_two j
  have h3 := two_tri a
  generalize (j + a).choose 2 = c1 at h1 ⊢
  generalize j.choose 2 = c2 at h2 ⊢
  generalize a * (a + 1) / 2 = t at h3 ⊢
  push_cast at h1 h2 h3
  have : (2 : ℤ) * ((k ^ 2 + c1 + c2 + (1 + k) * ((j + a) + j) : ℕ) : ℤ)
      = 2 * ((t + ((j + k) ^ 2 + a * (j + k)) + j : ℕ) : ℤ) := by
    push_cast; linear_combination h1 + h2 - h3
  exact_mod_cast (mul_left_cancel₀ (by norm_num) this)

section HR1
open MockTheta5.Bailey
local notation "ψ" => MockTheta5.JTP.ψC

/-- the summand `T_k(i,j)`: the `qⁿ`-coefficient of `q^{k²+C(i,2)+C(j,2)+(1+k)(i+j)}/((q;q)_i(q;q)_j)`. -/
noncomputable def Tcoef (n k i j : ℕ) : ℂ :=
  coeff n (X ^ (k ^ 2 + i.choose 2 + j.choose 2 + (1 + k) * (i + j))
    * ψ (Ring.inverse (qfac i) * Ring.inverse (qfac j)))

lemma Tcoef_symm (n k i j : ℕ) : Tcoef n k i j = Tcoef n k j i := by
  unfold Tcoef
  rw [show k ^ 2 + i.choose 2 + j.choose 2 + (1 + k) * (i + j)
      = k ^ 2 + j.choose 2 + i.choose 2 + (1 + k) * (j + i) by ring, mul_comm (Ring.inverse (qfac i))]

lemma Tcoef_zero {n k i j : ℕ} (h : n < i + j) : Tcoef n k i j = 0 := by
  unfold Tcoef
  have : i + j ≤ (1 + k) * (i + j) := Nat.le_mul_of_pos_left _ (by omega)
  exact coeffC_Xpow_zero (by omega) _

/-- **HA**: the Durfee series truncated at degree `n`. -/
lemma Dser_dvd (x y : ℂ) (n : ℕ) :
    (X : PowerSeries ℂ) ^ (n + 1) ∣ Dser x y - ∑ k ∈ range (n + 1),
      C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k) := by
  rw [X_pow_dvd_iff]; intro i hi
  rw [map_sub, Dser, coeff_mk, sub_eq_zero, map_sum, map_sum]
  refine sum_subset (range_subset_range.mpr (by omega)) fun k _ hk => ?_
  simp only [mem_range, not_lt] at hk
  rw [show C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k)
      = X ^ (k ^ 2) * (C ((x * y) ^ k) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k)) by ring]
  exact coeffC_Xpow_zero (by nlinarith) _

lemma LHSz_dvd (a c : ℕ) :
    (X : PowerSeries ℤ) ^ (c + 1) ∣ RankProof.LHSz a
      - ∑ m ∈ range (c + 1), X ^ (m ^ 2 + a * m) * RankProof.SLz a m := by
  rw [X_pow_dvd_iff]; intro i hi
  rw [map_sub, RankProof.LHSz, coeff_mk, sub_eq_zero, map_sum, map_sum]
  refine sum_subset (range_subset_range.mpr (by omega)) fun m _ hm => ?_
  simp only [mem_range, not_lt] at hm
  exact RankProof.coeff_Xpow_mul_eq_zero (by nlinarith) _

/-- **HB**: absorbing the finite Pochhammers into the infinite ones. -/
lemma hb {z : ℂ} (hz : z ≠ 0) (k : ℕ) :
    pochInf z 1 * pochInf z⁻¹ 1 * (C ((z * z⁻¹) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch z 1 k)
      * Ring.inverse (poch z⁻¹ 1 k)) = X ^ (k ^ 2) * pochInf z (1 + k) * pochInf z⁻¹ (1 + k) := by
  rw [pochInf_split z le_rfl k, pochInf_split z⁻¹ le_rfl k, mul_inv_cancel₀ hz, one_pow, map_one, one_mul]
  have h1 := Ring.mul_inverse_cancel _ (isUnit_poch z le_rfl k)
  have h2 := Ring.mul_inverse_cancel _ (isUnit_poch z⁻¹ le_rfl k)
  linear_combination X ^ (k ^ 2) * pochInf z (1 + k) * pochInf z⁻¹ (1 + k)
    * (poch z⁻¹ 1 k * Ring.inverse (poch z⁻¹ 1 k) * h1 + h2)

/-- **HC**: Euler-expanding both infinite products. -/
lemma hc (z : ℂ) (n k : ℕ) :
    coeff n (X ^ (k ^ 2) * pochInf z (1 + k) * pochInf z⁻¹ (1 + k))
      = ∑ i ∈ range (n + 1), ∑ j ∈ range (n + 1), z ^ i * z⁻¹ ^ j * ((-1) ^ (i + j) * Tcoef n k i j) := by
  rw [coeffC_congr (euler2_dvd z⁻¹ (by omega : 1 ≤ 1 + k) n), mul_right_comm,
    coeffC_congr (euler2_dvd z (by omega : 1 ≤ 1 + k) n), mul_sum, mul_sum]
  simp only [sum_mul, map_sum]
  refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
  have e : X ^ (k ^ 2) * (C ((-z⁻¹) ^ j) * X ^ (j.choose 2 + (1 + k) * j) * ψ (Ring.inverse (qfac j)))
      * (C ((-z) ^ i) * X ^ (i.choose 2 + (1 + k) * i) * ψ (Ring.inverse (qfac i)))
      = C (z ^ i * z⁻¹ ^ j * (-1) ^ (i + j)) * (X ^ (k ^ 2 + i.choose 2 + j.choose 2 + (1 + k) * (i + j))
        * ψ (Ring.inverse (qfac i) * Ring.inverse (qfac j))) := by
    rw [show k ^ 2 + i.choose 2 + j.choose 2 + (1 + k) * (i + j)
        = k ^ 2 + (i.choose 2 + (1 + k) * i) + (j.choose 2 + (1 + k) * j) by ring, pow_add, pow_add,
      neg_pow z, neg_pow z⁻¹, pow_add (-1 : ℂ), RingHom.map_mul ψ, RingHom.map_mul C, RingHom.map_mul C,
      RingHom.map_mul C, RingHom.map_mul C, RingHom.map_mul C]
    ring
  rw [e, coeff_C_mul, Tcoef, mul_assoc]

noncomputable def Ucoef (n a m l : ℕ) : ℂ := coeff n (X ^ (a * (a + 1) / 2 + (m ^ 2 + a * m) + l)
    * ψ (Ring.inverse (qfac l) * Ring.inverse (qfac (l + a))))

/-- **HD**: one `a`-block is the Bailey sum `(−1)^a q^{a(a+1)/2} Σ_m q^{m²+am} β*_m`. -/
lemma hd (n a : ℕ) :
    ∑ j ∈ range (n + 1 - a), (-1 : ℂ) ^ (j + a + j) * ∑ k ∈ range (n + 1), Tcoef n k (j + a) j
      = coeff n (C ((-1) ^ a) * X ^ (a * (a + 1) / 2) * ψ (RankProof.LHSz a)) := by
  let U : ℕ → ℕ → ℂ := Ucoef n a
  have hUdef : ∀ m l, U m l = coeff n (X ^ (a * (a + 1) / 2 + (m ^ 2 + a * m) + l)
    * ψ (Ring.inverse (qfac l) * Ring.inverse (qfac (l + a)))) := fun _ _ => rfl
  have hT : ∀ k j, Tcoef n k (j + a) j = U (j + k) j := fun k j => by
    rw [hUdef, Tcoef, exp_HR1, mul_comm (Ring.inverse (qfac (j + a)))]
  have hU0 : ∀ m l, n < m → U m l = 0 := fun m l hm => by
    have : m ≤ m ^ 2 := by nlinarith
    rw [hUdef]; exact coeffC_Xpow_zero (by omega) _
  have hsign : ∀ j, (-1 : ℂ) ^ (j + a + j) = (-1) ^ a := fun j => by
    rw [show j + a + j = a + 2 * j by ring, pow_add, pow_mul]; norm_num
  have hdvd := ψ_dvd (LHSz_dvd a n)
  rw [map_sub] at hdvd
  rw [mul_assoc, coeff_C_mul, coeffC_congr hdvd]
  simp only [hsign, ← mul_sum]
  congr 1
  calc ∑ j ∈ range (n + 1 - a), ∑ k ∈ range (n + 1), Tcoef n k (j + a) j
      = ∑ j ∈ range (n + 1), ∑ k ∈ range (n + 1), Tcoef n k (j + a) j := by
        refine sum_subset (range_subset_range.mpr (by omega)) fun j hj hj' => ?_
        simp only [mem_range, not_lt] at hj hj'
        exact sum_eq_zero fun k _ => Tcoef_zero (by omega)
    _ = ∑ j ∈ range (n + 1), ∑ k ∈ range (n + 1 - j), U (j + k) j := by
        refine sum_congr rfl fun j _ => ?_
        simp only [hT]
        symm
        refine sum_subset (range_subset_range.mpr (by omega)) fun k _ hk => ?_
        simp only [mem_range, not_lt] at hk
        exact hU0 _ _ (by omega)
    _ = ∑ m ∈ range (n + 1), ∑ l ∈ range (m + 1), U m l := by
        rw [← sum_range_diag_flip (n + 1) (fun l k => U (l + k) l)]
        refine sum_congr rfl fun m _ => sum_congr rfl fun l hl => ?_
        rw [show l + (m - l) = m by have := mem_range.mp hl; omega]
    _ = _ := by
        rw [map_sum, mul_sum, map_sum]
        refine sum_congr rfl fun m _ => ?_
        rw [RankProof.SLz, mul_sum, map_sum, mul_sum, map_sum]
        refine sum_congr rfl fun l _ => ?_
        rw [hUdef]
        congr 1
        simp only [map_mul, map_pow, PowerSeries.map_X, pow_add]
        ring

/-- **(2.18)**, Garvan's Hecke–Rogers form of the rank generating function:
`(zq)_∞(q/z)_∞ R(z;q) = Σ_{a≥0} w_a(z) (−1)^a q^{a(a+1)/2} Σ_m q^{m²+am} β*_m(a)`. -/
theorem hecke_rogers_18 {z : ℂ} (hz : z ≠ 0) (n : ℕ) :
    coeff n (pochInf z 1 * pochInf z⁻¹ 1 * Dser z z⁻¹)
      = ∑ a ∈ range (n + 1), wz z a * coeff n (C ((-1) ^ a) * X ^ (a * (a + 1) / 2) * ψ (RankProof.LHSz a)) := by
  rw [coeffC_congr (Dser_dvd z z⁻¹ n), mul_sum, map_sum]
  simp only [hb hz, hc]
  have key := sum_regroup hz (fun i j => (-1 : ℂ) ^ (i + j) * ∑ k ∈ range (n + 1), Tcoef n k i j)
    (fun i j => by beta_reduce; rw [add_comm i j]; exact congrArg _ (sum_congr rfl fun k _ => Tcoef_symm n k i j))
    (n + 1)
  beta_reduce at key
  have e : ∑ k ∈ range (n + 1), ∑ i ∈ range (n + 1), ∑ j ∈ range (n + 1),
      z ^ i * z⁻¹ ^ j * ((-1) ^ (i + j) * Tcoef n k i j)
      = ∑ i ∈ range (n + 1), ∑ j ∈ range (n + 1),
        z ^ i * z⁻¹ ^ j * ((-1) ^ (i + j) * ∑ k ∈ range (n + 1), Tcoef n k i j) := by
    rw [sum_comm]
    refine sum_congr rfl fun i _ => ?_
    rw [sum_comm]
    refine sum_congr rfl fun j _ => ?_
    rw [mul_sum, mul_sum]
  rw [e, key]
  exact sum_congr rfl fun a _ => by rw [hd]

end HR1

end CrankProof
