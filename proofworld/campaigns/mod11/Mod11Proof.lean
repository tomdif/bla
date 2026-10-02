/-
# Ramanujan's partition congruence `p(11n+6) ≡ 0 (mod 11)` — unconditional, kernel-checked

**`eleven_dvd_partition_card : 11 ∣ #(Nat.Partition (11n+6))`.**  Axioms: `propext, Classical.choice,
Quot.sound` only (no `sorry`, no `native_decide`).  Built by the proofworld campaign (gated by
`proofworld.gate`), following Winquist's elementary route but entirely in one variable:

1. **Third derivative of the triangular JTP** (`L3_triTheta_cube`, `six_coeff_qfac9`):
   with `R_N(q) = Σ_{n₁+n₂+n₃=N} q^{Σ nᵢ(nᵢ−1)/2}` (the `z^N`-slice of `triTheta³`),
   `6·(q;q)⁹ = Σ_N w3(N)·R_N`, `w3(N) = d³/dz³ z^N |_{z=−1}`.
2. **Quasi-periodicity / reflection** (`RZ_shift`, `RZ_refl`, closed forms `RZ_three_mul…`):
   `R_{N+3} = q^N R_N`, `R_{3−N} = R_N`, so every slice is a `q`-shift of `R₀` or `R₁`.
3. **Winquist's specialization `a = q^{1/3}`** (`coeff_E_RS0`, `coeff_E_RS1`): regrouping the triple sum by
   `3·Σ nᵢ(nᵢ−1)/2 + Σ nᵢ = Σ nᵢ(3nᵢ−1)/2` turns it into Euler's pentagonal series cubed, so
   `(q;q)·R₀ = [3-dissection₀ of (q;q)³]`, `(q;q)·R₁ = −[3-dissection₁ of (q;q)³]` (Jacobi's cube identity).
4. **The mod-11 heart** (`eleven_dvd_coeff_E_RS`): each term of `6·(q;q)¹⁰ = Σ_N w3(N)·(q;q)R_N` at
   `q^a`, `a ≡ 6 (mod 11)`, carries a Jacobi weight `±(2n+1)` with `(2n+1)² + y² = 24a+10 ≡ 0 (mod 11)`;
   as `−1` is a non-residue mod 11, `11 ∣ 2n+1`.
5. **Frobenius** (port of the mod-7 file): `1/(q;q) ≡ (q;q)¹⁰/(q¹¹;q¹¹) (mod 11)`, and the partition-count
   bridge `coeff_partitionGF_eq_card`.
-/
import RamanujanTau.MockTheta5JacobiCubeProof
import RamanujanTau.MockTheta5QuintIdentity
import RamanujanTau.MockTheta5Qfac4
import RamanujanTau.MockTheta5EulerPentagonal
import Mathlib.Data.Int.Interval
import RamanujanTau.MockTheta5PartitionCount

namespace MockTheta5.JTP
open PowerSeries LaurentPolynomial

/-- `d³/dz³ zᵏ` at `z = −1`: `k(k−1)(k−2)(−1)^{k−3} = −k(k−1)(k−2)·sgn k`. -/
def w3 (κ : ℤ) : ℤ := -(κ * (κ - 1) * (κ - 2)) * sgn κ

lemma w3_third_difference (κ : ℤ) :
    w3 κ + 3 * w3 (κ + 1) + 3 * w3 (κ + 2) + w3 (κ + 3) = 6 * sgn κ := by
  have h1 : sgn (κ + 1) = -sgn κ := sgn_succ κ
  have h2 : sgn (κ + 2) = sgn κ := by
    rw [show κ + 2 = (κ + 1) + 1 by ring, sgn_succ, h1, neg_neg]
  have h3 : sgn (κ + 3) = -sgn κ := by
    rw [show κ + 3 = (κ + 2) + 1 by ring, sgn_succ, h2]
  simp only [w3, h1, h2, h3]
  ring

/-- the linear third-derivative-at-`(−1)` functional on `LaurentPolynomial ℤ`. -/
noncomputable def D3 : LaurentPolynomial ℤ →+ ℤ where
  toFun p := p.sum (fun κ c => c * w3 κ)
  map_zero' := Finsupp.sum_zero_index
  map_add' _ _ := Finsupp.sum_add_index' (fun i => zero_mul (w3 i)) (fun i a b => add_mul a b (w3 i))

lemma D3_CT (b κ : ℤ) : D3 (LaurentPolynomial.C b * T κ) = b * w3 κ := by
  rw [← LaurentPolynomial.single_eq_C_mul_T]
  simp only [D3, AddMonoidHom.coe_mk, ZeroHom.coe_mk, Finsupp.sum_single_index, zero_mul]

lemma D3_T (κ : ℤ) : D3 (T κ) = w3 κ := by
  rw [show (T κ : LaurentPolynomial ℤ) = LaurentPolynomial.C (1 : ℤ) * T κ by rw [map_one, one_mul],
      D3_CT, one_mul]

lemma T_mul_CT (j b κ : ℤ) :
    (T j : LaurentPolynomial ℤ) * (LaurentPolynomial.C b * T κ) = LaurentPolynomial.C b * T (κ + j) := by
  rw [← mul_assoc, mul_comm (T j) (LaurentPolynomial.C b), mul_assoc, ← T_add, add_comm j κ]

lemma one_add_T1_cube :
    (1 + T 1 : LaurentPolynomial ℤ) ^ 3 = 1 + 3 * T 1 + 3 * T 2 + T 3 := by
  have h2 : (T 2 : LaurentPolynomial ℤ) = T 1 * T 1 := by rw [← T_add]; norm_num
  have h3 : (T 3 : LaurentPolynomial ℤ) = T 1 * T 1 * T 1 := by rw [← T_add, ← T_add]; norm_num
  rw [h2, h3]; ring

lemma D3_one_add_T1_cube_CT (b κ : ℤ) :
    D3 ((1 + T 1) ^ 3 * (LaurentPolynomial.C b * T κ)) = 6 * evm1 (LaurentPolynomial.C b * T κ) := by
  rw [one_add_T1_cube,
      show ((1 : LaurentPolynomial ℤ) + 3 * T 1 + 3 * T 2 + T 3) * (LaurentPolynomial.C b * T κ)
          = LaurentPolynomial.C b * T κ + (T 1 * (LaurentPolynomial.C b * T κ)
              + T 1 * (LaurentPolynomial.C b * T κ) + T 1 * (LaurentPolynomial.C b * T κ))
            + (T 2 * (LaurentPolynomial.C b * T κ) + T 2 * (LaurentPolynomial.C b * T κ)
              + T 2 * (LaurentPolynomial.C b * T κ)) + T 3 * (LaurentPolynomial.C b * T κ) by ring,
      T_mul_CT, T_mul_CT, T_mul_CT]
  simp only [map_add, D3_CT, evm1_CT]
  have := w3_third_difference κ
  linear_combination b * this

lemma D3_one_add_T1_cube_mul (c : LaurentPolynomial ℤ) : D3 ((1 + T 1) ^ 3 * c) = 6 * evm1 c := by
  have h : D3.comp (AddMonoidHom.mulLeft ((1 + T 1) ^ 3 : LaurentPolynomial ℤ))
      = (6 : ℤ) • (evm1 : LaurentPolynomial ℤ →+* ℤ).toAddMonoidHom := by
    apply Finsupp.addHom_ext
    intro κ b
    rw [show (Finsupp.single κ b : LaurentPolynomial ℤ) = LaurentPolynomial.C b * T κ from
          LaurentPolynomial.single_eq_C_mul_T b κ]
    simp only [RingHom.toAddMonoidHom_eq_coe, smul_eq_mul]
    exact D3_one_add_T1_cube_CT b κ
  have := DFunLike.congr_fun h c
  simpa using this

/-- the third-derivative functional on power series (coefficient-wise). -/
noncomputable def L3 (f : PowerSeries (LaurentPolynomial ℤ)) : PowerSeries ℤ := mk fun n => D3 (coeff n f)

@[simp] lemma coeff_L3 (f : PowerSeries (LaurentPolynomial ℤ)) (n : ℕ) :
    coeff n (L3 f) = D3 (coeff n f) := by rw [L3, coeff_mk]

/-- **the cube vanishing rule**: `L3((1+z)³·G) = 6·G|_{z=−1}`. -/
lemma L3_one_add_z_cube_mul (G : PowerSeries (LaurentPolynomial ℤ)) :
    L3 ((1 + PowerSeries.C (T 1)) ^ 3 * G) = PowerSeries.C (6 : ℤ) * PowerSeries.map evm1 G := by
  rw [show (1 + PowerSeries.C (T 1) : PowerSeries (LaurentPolynomial ℤ)) ^ 3
        = PowerSeries.C ((1 + T 1) ^ 3) from by rw [map_pow, map_add, map_one]]
  ext n
  rw [coeff_L3, PowerSeries.coeff_C_mul, PowerSeries.coeff_C_mul, PowerSeries.coeff_map,
    D3_one_add_T1_cube_mul]

/-- **product side**: `L3 (triTheta³) = 6·(q;q)⁹`. -/
theorem L3_triTheta_cube : L3 (triTheta ^ 3) = PowerSeries.C (6 : ℤ) * qfacInf ^ 9 := by
  rw [← bilateral_triangular_JTP, triProdQInf_eq_split,
    show (qfacInfL * ((1 + PowerSeries.C (T 1)) * triProdQ1Inf)
            * (PowerSeries.map invertHom triProdQ1Inf)) ^ 3
          = (1 + PowerSeries.C (T 1)) ^ 3
            * (qfacInfL * triProdQ1Inf * (PowerSeries.map invertHom triProdQ1Inf)) ^ 3 by ring,
    L3_one_add_z_cube_mul, map_pow, map_mul, map_mul, map_evm1_qfacInfL, map_evm1_triProdQ1Inf,
    map_evm1_map_invert, map_evm1_triProdQ1Inf]
  ring


/-! ## Stage 2: the series side, as finite box sums -/

/-- a box sum over `[-k, k+1]` is the paired sum `Σ_{m≤k} (f(m+1) + f(−m))`. -/
lemma sum_Icc_neg_succ {M : Type*} [AddCommMonoid M] (f : ℤ → M) : ∀ k : ℕ,
    ∑ n ∈ Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1), f n
      = ∑ m ∈ Finset.range (k + 1), (f ((m : ℤ) + 1) + f (-(m : ℤ))) := by
  intro k
  induction k with
  | zero =>
      rw [show Finset.Icc (-((0 : ℕ) : ℤ)) (((0 : ℕ) : ℤ) + 1) = {0, 1} by decide]
      simp [add_comm]
  | succ k ih =>
      have hset : Finset.Icc (-((k + 1 : ℕ) : ℤ)) (((k + 1 : ℕ) : ℤ) + 1)
          = insert (-((k + 1 : ℕ) : ℤ)) (insert (((k + 1 : ℕ) : ℤ) + 1)
              (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1))) := by
        ext x; simp only [Finset.mem_Icc, Finset.mem_insert]; push_cast; omega
      rw [hset, Finset.sum_insert (by simp <;> omega), Finset.sum_insert (by simp <;> omega), ih]
      conv_rhs => rw [Finset.sum_range_succ]
      push_cast
      abel

lemma dT_nonneg (n : ℤ) : 0 ≤ n * (n - 1) := by
  rcases le_or_gt n 0 with h | h <;> nlinarith

lemma two_mul_triExp (n : ℤ) : 2 * (((n * (n - 1) / 2).toNat : ℕ) : ℤ) = n * (n - 1) := by
  have he : Even (n * (n - 1)) := by
    have := Int.even_mul_pred_self n; simpa [mul_comm] using this
  obtain ⟨r, hr⟩ := he
  have h0 := dT_nonneg n
  rw [hr]; omega

lemma triExp_pos (m : ℕ) : ((((m : ℤ) + 1) * ((m : ℤ) + 1 - 1) / 2).toNat) = m * (m + 1) / 2 := by
  have h : ((m : ℤ) + 1) * ((m : ℤ) + 1 - 1) = ((m * (m + 1) : ℕ) : ℤ) := by push_cast; ring
  rw [h]; omega

lemma triExp_neg (m : ℕ) : (((-(m : ℤ)) * (-(m : ℤ) - 1) / 2).toNat) = m * (m + 1) / 2 := by
  have h : (-(m : ℤ)) * (-(m : ℤ) - 1) = ((m * (m + 1) : ℕ) : ℤ) := by push_cast; ring
  rw [h]; omega

/-- the triangular exponent `n(n−1)/2` as a natural number. -/
def triE (n : ℤ) : ℕ := (n * (n - 1) / 2).toNat

/-- **coefficients of `triTheta` as a box sum** (any box `[-k,k+1]` with `i ≤ k`). -/
lemma coeff_triTheta_box (i k : ℕ) (hik : i ≤ k) :
    coeff i triTheta = ∑ n ∈ Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1), if triE n = i then T n else 0 := by
  rw [coeff_triTheta (show i + 1 ≤ k + 1 by omega), triFinite, map_sum, sum_Icc_neg_succ]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [coeff_triTerm_val, triE, triE, triExp_pos, triExp_neg]
  by_cases h : i = m * (m + 1) / 2
  · rw [if_pos h, if_pos h.symm, if_pos h.symm]
  · rw [if_neg h, if_neg (Ne.symm h), if_neg (Ne.symm h), add_zero]

/-- **cube of a series with finitely supported low coefficients.** -/
lemma coeff_cube_of_box {R : Type*} [CommRing R] (f : PowerSeries R) (g : ℤ → ℕ) (c : ℤ → R)
    (S : Finset ℤ) (k : ℕ) (hf : ∀ i ≤ k, coeff i f = ∑ n ∈ S, if g n = i then c n else 0) :
    coeff k (f ^ 3) = ∑ t ∈ S ×ˢ S ×ˢ S,
      if g t.1 + g t.2.1 + g t.2.2 = k then c t.1 * c t.2.1 * c t.2.2 else 0 := by
  set F : PowerSeries R := ∑ n ∈ S, PowerSeries.C (c n) * X ^ (g n) with hFdef
  have hF : ∀ i, coeff i F = ∑ n ∈ S, if g n = i then c n else 0 := by
    intro i
    rw [hFdef, map_sum]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [PowerSeries.coeff_C_mul_X_pow]
    by_cases h : g n = i
    · rw [if_pos h.symm, if_pos h]
    · rw [if_neg (Ne.symm h), if_neg h]
  have hdiff : (X : PowerSeries R) ^ (k + 1) ∣ f - F := by
    rw [PowerSeries.X_pow_dvd_iff]
    intro i hi
    rw [map_sub, hf i (by omega), hF, sub_self]
  have h3 : (X : PowerSeries R) ^ (k + 1) ∣ f ^ 3 - F ^ 3 := dvd_trans hdiff (sub_dvd_pow_sub_pow _ _ 3)
  have hk : coeff k (f ^ 3) = coeff k (F ^ 3) := by
    obtain ⟨d, hd⟩ := h3
    have := congrArg (coeff k) hd
    rw [map_sub, coeff_X_pow_mul', if_neg (by omega)] at this
    exact sub_eq_zero.mp this
  have hF3 : F ^ 3 = ∑ a ∈ S, ∑ b ∈ S, ∑ d ∈ S,
      PowerSeries.C (c a * c b * c d) * X ^ (g a + g b + g d) := by
    rw [pow_three, hFdef]
    simp only [Finset.sum_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ =>
      Finset.sum_congr rfl fun d _ => ?_
    rw [map_mul, map_mul, pow_add, pow_add]
    ring
  rw [hk, hF3, Finset.sum_product (s := S) (t := S ×ˢ S), map_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.sum_product (s := S) (t := S), map_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [map_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  rw [PowerSeries.coeff_C_mul_X_pow]
  by_cases h : g a + g b + g d = k
  · rw [if_pos h.symm, if_pos h]
  · rw [if_neg (Ne.symm h), if_neg h]

/-- the doubled triangular exponent sum of a triple, and its index sum. -/
def dsum (t : ℤ × ℤ × ℤ) : ℤ := t.1 * (t.1 - 1) + t.2.1 * (t.2.1 - 1) + t.2.2 * (t.2.2 - 1)
def tsum (t : ℤ × ℤ × ℤ) : ℤ := t.1 + t.2.1 + t.2.2

lemma triE_sum_eq_iff (t : ℤ × ℤ × ℤ) (k : ℕ) :
    triE t.1 + triE t.2.1 + triE t.2.2 = k ↔ dsum t = 2 * k := by
  have h1 := two_mul_triExp t.1
  have h2 := two_mul_triExp t.2.1
  have h3 := two_mul_triExp t.2.2
  simp only [triE] at *
  unfold dsum
  constructor
  · intro h; rw [← h1, ← h2, ← h3]; exact_mod_cast (by omega : _)
  · intro h; omega

/-- **`D3` of the `q^k` coefficient of `triTheta³`**, as a triple box sum. -/
lemma D3_coeff_triTheta_cube (k : ℕ) :
    D3 (coeff k (triTheta ^ 3)) = ∑ t ∈ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ
        (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)),
      if dsum t = 2 * k then w3 (tsum t) else 0 := by
  rw [coeff_cube_of_box triTheta triE T _ k (fun i hi => coeff_triTheta_box i k hi), map_sum]
  refine Finset.sum_congr rfl fun t _ => ?_
  by_cases h : triE t.1 + triE t.2.1 + triE t.2.2 = k
  · rw [if_pos h, if_pos ((triE_sum_eq_iff t k).mp h), ← T_add, ← T_add, D3_T]; rfl
  · rw [if_neg h, if_neg (fun h' => h ((triE_sum_eq_iff t k).mpr h')), map_zero]


/-! ## Stage 3: the slice counts `RZ N K` (z-degree `N`, doubled q-degree `2K`) -/

def box3 (B : ℕ) : Finset (ℤ × ℤ × ℤ) :=
  Finset.Icc (-(B : ℤ)) B ×ˢ Finset.Icc (-(B : ℤ)) B ×ˢ Finset.Icc (-(B : ℤ)) B

/-- `RZ N K = #{t ∈ ℤ³ : Σ tᵢ(tᵢ−1) = 2K, Σ tᵢ = N}` = `[q^K] zProj_N (triTheta³)`. -/
def RZ (N K : ℤ) : ℤ := ∑ t ∈ box3 (K.toNat + 1), if dsum t = 2 * K ∧ tsum t = N then 1 else 0

lemma comp_bound {n K : ℤ} (h : n * (n - 1) ≤ 2 * K) : -(K + 1) ≤ n ∧ n ≤ K + 1 := by
  have h0 := dT_nonneg n
  constructor <;> nlinarith [sq_nonneg (n + K + 1), sq_nonneg (n - K - 1)]

lemma mem_box3_of_dsum {t : ℤ × ℤ × ℤ} {K : ℤ} (h : dsum t = 2 * K) : t ∈ box3 (K.toNat + 1) := by
  unfold dsum at h
  have h1 := dT_nonneg t.1
  have h2 := dT_nonneg t.2.1
  have h3 := dT_nonneg t.2.2
  have b1 := comp_bound (n := t.1) (K := K) (by linarith)
  have b2 := comp_bound (n := t.2.1) (K := K) (by linarith)
  have b3 := comp_bound (n := t.2.2) (K := K) (by linarith)
  have hK : 0 ≤ K := by linarith
  simp only [box3, Finset.mem_product, Finset.mem_Icc]
  push_cast
  refine ⟨⟨by omega, by omega⟩, ⟨by omega, by omega⟩, ⟨by omega, by omega⟩⟩

lemma RZ_eq_of_box (N K : ℤ) (B : ℕ) (hB : K.toNat + 1 ≤ B) :
    RZ N K = ∑ t ∈ box3 B, if dsum t = 2 * K ∧ tsum t = N then 1 else 0 := by
  unfold RZ
  apply Finset.sum_subset
  · intro t ht
    simp only [box3, Finset.mem_product, Finset.mem_Icc] at ht ⊢
    push_cast at ht ⊢
    omega
  · intro t _ hnot
    rw [if_neg]
    rintro ⟨hd, _⟩
    exact hnot (mem_box3_of_dsum hd)

lemma RZ_neg (N K : ℤ) (hK : K < 0) : RZ N K = 0 := by
  unfold RZ
  refine Finset.sum_eq_zero fun t _ => if_neg ?_
  rintro ⟨hd, _⟩
  unfold dsum at hd
  have := dT_nonneg t.1; have := dT_nonneg t.2.1; have := dT_nonneg t.2.2
  linarith

def shiftT (c : ℤ) (t : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ := (t.1 + c, t.2.1 + c, t.2.2 + c)

lemma dsum_shift1 (t : ℤ × ℤ × ℤ) : dsum (shiftT 1 t) = dsum t + 2 * tsum t := by
  simp only [dsum, tsum, shiftT]; ring

lemma tsum_shift (c : ℤ) (t : ℤ × ℤ × ℤ) : tsum (shiftT c t) = tsum t + 3 * c := by
  simp only [tsum, shiftT]; ring

/-- **quasi-periodicity**: `R_{N+3}(K) = R_N(K − N)`  (i.e. `R_{N+3} = q^N R_N`). -/
lemma RZ_shift (N K : ℤ) : RZ (N + 3) K = RZ N (K - N) := by
  symm
  unfold RZ
  refine Finset.sum_bij_ne_zero (fun t _ _ => shiftT 1 t) ?_ ?_ ?_ ?_
  · intro t _ hne
    have hc : dsum t = 2 * (K - N) ∧ tsum t = N := by by_contra hc; exact hne (if_neg hc)
    apply mem_box3_of_dsum
    rw [dsum_shift1, hc.1, hc.2]; ring
  · intro a _ _ b _ _ hab
    simp only [shiftT, Prod.mk.injEq] at hab
    obtain ⟨h1, h2, h3⟩ := hab
    exact Prod.ext (by linarith) (Prod.ext (by linarith) (by linarith))
  · intro b _ hne
    have hc : dsum b = 2 * K ∧ tsum b = N + 3 := by by_contra hc; exact hne (if_neg hc)
    refine ⟨shiftT (-1) b, ?_, ?_, ?_⟩
    · apply mem_box3_of_dsum
      have hs : shiftT 1 (shiftT (-1) b) = b := by simp [shiftT]
      have := dsum_shift1 (shiftT (-1) b)
      rw [hs, tsum_shift, hc.2] at this
      rw [hc.1] at this; linarith
    · have hs : shiftT 1 (shiftT (-1) b) = b := by simp [shiftT]
      have h1 := dsum_shift1 (shiftT (-1) b)
      rw [hs, tsum_shift, hc.2] at h1
      rw [if_pos ⟨by rw [hc.1] at h1; linarith, by rw [tsum_shift, hc.2]; ring⟩]
      exact one_ne_zero
    · simp [shiftT]
  · intro t _ hne
    have hc : dsum t = 2 * (K - N) ∧ tsum t = N := by by_contra hc; exact hne (if_neg hc)
    rw [if_pos hc, if_pos ⟨by rw [dsum_shift1, hc.1, hc.2]; ring, by rw [tsum_shift, hc.2]; ring⟩]

def reflT (t : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ := (1 - t.1, 1 - t.2.1, 1 - t.2.2)

lemma dsum_refl (t : ℤ × ℤ × ℤ) : dsum (reflT t) = dsum t := by simp only [dsum, reflT]; ring
lemma tsum_refl (t : ℤ × ℤ × ℤ) : tsum (reflT t) = 3 - tsum t := by simp only [tsum, reflT]; ring
lemma reflT_reflT (t : ℤ × ℤ × ℤ) : reflT (reflT t) = t := by simp [reflT]

/-- **reflection**: `R_{3−N} = R_N`. -/
lemma RZ_refl (N K : ℤ) : RZ (3 - N) K = RZ N K := by
  unfold RZ
  refine Finset.sum_bij_ne_zero (fun t _ _ => reflT t) ?_ ?_ ?_ ?_
  · intro t _ hne
    have hc : dsum t = 2 * K ∧ tsum t = 3 - N := by by_contra hc; exact hne (if_neg hc)
    show reflT t ∈ box3 (K.toNat + 1)
    exact mem_box3_of_dsum (by rw [dsum_refl, hc.1])
  · intro a _ _ b _ _ hab
    have := congrArg reflT hab
    simpa only [reflT_reflT] using this
  · intro b _ hne
    have hc : dsum b = 2 * K ∧ tsum b = N := by by_contra hc; exact hne (if_neg hc)
    refine ⟨reflT b, mem_box3_of_dsum (by rw [dsum_refl, hc.1]), ?_, reflT_reflT b⟩
    rw [if_pos ⟨by rw [dsum_refl, hc.1], by rw [tsum_refl, hc.2]⟩]
    exact one_ne_zero
  · intro t _ hne
    have hc : dsum t = 2 * K ∧ tsum t = 3 - N := by by_contra hc; exact hne (if_neg hc)
    show _ = (if dsum (reflT t) = 2 * K ∧ tsum (reflT t) = N then 1 else 0)
    rw [if_pos hc, if_pos ⟨by rw [dsum_refl, hc.1], by rw [tsum_refl, hc.2]; ring⟩]

/-- **the `E⁹` formula**: `6·[q^k](q;q)⁹ = Σ_N w3(N)·R_N(k)`. -/
theorem six_coeff_qfac9 (k : ℕ) :
    6 * coeff k (qfacInf ^ 9) = ∑ N ∈ Finset.Icc (-(3 * ((k : ℤ) + 1))) (3 * ((k : ℤ) + 1)),
      w3 N * RZ N k := by
  have h := congrArg (coeff k) L3_triTheta_cube
  rw [coeff_L3, PowerSeries.coeff_C_mul, D3_coeff_triTheta_cube] at h
  rw [← h]
  have hbox : ∀ N ∈ Finset.Icc (-(3 * ((k : ℤ) + 1))) (3 * ((k : ℤ) + 1)),
      RZ N k = ∑ t ∈ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ
        (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)),
        if dsum t = 2 * k ∧ tsum t = N then 1 else 0 := by
    intro N _
    rw [RZ_eq_of_box N k (k + 1) (by simp)]
    symm
    apply Finset.sum_subset
    · intro t ht
      simp only [box3, Finset.mem_product, Finset.mem_Icc] at ht ⊢
      push_cast at ht ⊢; omega
    · intro t _ hnot
      rw [if_neg]
      rintro ⟨hd, _⟩
      apply hnot
      have hm := mem_box3_of_dsum hd
      unfold dsum at hd
      have h1 := dT_nonneg t.1; have h2 := dT_nonneg t.2.1; have h3 := dT_nonneg t.2.2
      have b1 := comp_bound (n := t.1) (K := (k : ℤ) - 1 + 1) (by linarith)
      have b2 := comp_bound (n := t.2.1) (K := (k : ℤ) - 1 + 1) (by linarith)
      have b3 := comp_bound (n := t.2.2) (K := (k : ℤ) - 1 + 1) (by linarith)
      have l1 : -(k : ℤ) ≤ t.1 := by nlinarith [sq_nonneg (t.1 + k)]
      have l2 : -(k : ℤ) ≤ t.2.1 := by nlinarith [sq_nonneg (t.2.1 + k)]
      have l3 : -(k : ℤ) ≤ t.2.2 := by nlinarith [sq_nonneg (t.2.2 + k)]
      simp only [Finset.mem_product, Finset.mem_Icc]
      refine ⟨⟨l1, by linarith⟩, ⟨l2, by linarith⟩, ⟨l3, by linarith⟩⟩
  symm
  calc ∑ N ∈ Finset.Icc (-(3 * ((k : ℤ) + 1))) (3 * ((k : ℤ) + 1)), w3 N * RZ N k
      = ∑ N ∈ Finset.Icc (-(3 * ((k : ℤ) + 1))) (3 * ((k : ℤ) + 1)),
          ∑ t ∈ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ
            (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)),
            (if dsum t = 2 * k ∧ tsum t = N then w3 N else 0) := by
        refine Finset.sum_congr rfl fun N hN => ?_
        rw [hbox N hN, Finset.mul_sum]
        refine Finset.sum_congr rfl fun t _ => ?_
        split_ifs <;> simp
    _ = ∑ t ∈ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ
            (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)) ×ˢ (Finset.Icc (-(k : ℤ)) ((k : ℤ) + 1)),
          ∑ N ∈ Finset.Icc (-(3 * ((k : ℤ) + 1))) (3 * ((k : ℤ) + 1)),
            (if dsum t = 2 * k ∧ tsum t = N then w3 N else 0) := Finset.sum_comm
    _ = _ := by
        refine Finset.sum_congr rfl fun t ht => ?_
        by_cases hd : dsum t = 2 * k
        · simp only [hd, true_and, if_true]
          rw [Finset.sum_ite_eq, if_pos]
          simp only [Finset.mem_product, Finset.mem_Icc] at ht
          simp only [Finset.mem_Icc, tsum]
          constructor <;> linarith [ht.1.1, ht.1.2, ht.2.1.1, ht.2.1.2, ht.2.2.1, ht.2.2.2]
        · simp only [hd, false_and, if_false, Finset.sum_const_zero]


/-! ## Stage 4a: closed forms of the slices via quasi-periodicity -/

lemma half_ss1 (s : ℤ) : 2 * (s * (s - 1) / 2) = s * (s - 1) :=
  Int.two_mul_ediv_two_of_even (by simpa [mul_comm] using Int.even_mul_pred_self s)

lemma half_s3m (s : ℤ) : 2 * (s * (3 * s - 1) / 2) = s * (3 * s - 1) := by
  apply Int.two_mul_ediv_two_of_even
  have h : s * (3 * s - 1) = s * (s - 1) + 2 * (s * s) := by ring
  rw [h]
  exact (by simpa [mul_comm] using Int.even_mul_pred_self s : Even (s * (s - 1))).add (even_two_mul _)

lemma half_s3p (s : ℤ) : 2 * (s * (3 * s + 1) / 2) = s * (3 * s + 1) := by
  apply Int.two_mul_ediv_two_of_even
  have h : s * (3 * s + 1) = s * (s - 1) + 2 * (s * s + s) := by ring
  rw [h]
  exact (by simpa [mul_comm] using Int.even_mul_pred_self s : Even (s * (s - 1))).add (even_two_mul _)

/-- `R_{3s}(K) = R_0(K − 3·s(s−1)/2)`. -/
lemma RZ_three_mul : ∀ s K : ℤ, RZ (3 * s) K = RZ 0 (K - 3 * (s * (s - 1) / 2)) := by
  intro s
  induction s using Int.induction_on with
  | zero => intro K; simp
  | succ i ih =>
      intro K
      rw [show 3 * ((i : ℤ) + 1) = 3 * (i : ℤ) + 3 by ring, RZ_shift, ih]
      congr 1
      have h1 := half_ss1 (i : ℤ)
      have h2 := half_ss1 ((i : ℤ) + 1)
      have h3 : ((i : ℤ) + 1) * ((i : ℤ) + 1 - 1) - (i : ℤ) * ((i : ℤ) - 1) = 2 * (i : ℤ) := by ring
      linarith
  | pred i ih =>
      intro K
      have h := RZ_shift (3 * (-(i : ℤ) - 1)) (K + 3 * (-(i : ℤ) - 1))
      rw [show 3 * (-(i : ℤ) - 1) + 3 = 3 * -(i : ℤ) by ring,
        show K + 3 * (-(i : ℤ) - 1) - 3 * (-(i : ℤ) - 1) = K by ring] at h
      rw [← h, ih]
      congr 1
      have h1 := half_ss1 (-(i : ℤ))
      have h2 := half_ss1 (-(i : ℤ) - 1)
      have h3 : (-(i : ℤ)) * (-(i : ℤ) - 1) - (-(i : ℤ) - 1) * (-(i : ℤ) - 1 - 1)
          = -2 * ((i : ℤ) + 1) := by ring
      linarith

/-- `R_{3s+1}(K) = R_1(K − s(3s−1)/2)`. -/
lemma RZ_three_mul_add_one : ∀ s K : ℤ, RZ (3 * s + 1) K = RZ 1 (K - s * (3 * s - 1) / 2) := by
  intro s
  induction s using Int.induction_on with
  | zero => intro K; simp
  | succ i ih =>
      intro K
      rw [show 3 * ((i : ℤ) + 1) + 1 = (3 * (i : ℤ) + 1) + 3 by ring, RZ_shift, ih]
      congr 1
      have h1 := half_s3m (i : ℤ)
      have h2 := half_s3m ((i : ℤ) + 1)
      have h3 : ((i : ℤ) + 1) * (3 * ((i : ℤ) + 1) - 1) - (i : ℤ) * (3 * (i : ℤ) - 1)
          = 2 * (3 * (i : ℤ) + 1) := by ring
      linarith
  | pred i ih =>
      intro K
      have h := RZ_shift (3 * (-(i : ℤ) - 1) + 1) (K + (3 * (-(i : ℤ) - 1) + 1))
      rw [show 3 * (-(i : ℤ) - 1) + 1 + 3 = 3 * -(i : ℤ) + 1 by ring,
        show K + (3 * (-(i : ℤ) - 1) + 1) - (3 * (-(i : ℤ) - 1) + 1) = K by ring] at h
      rw [← h, ih]
      congr 1
      have h1 := half_s3m (-(i : ℤ))
      have h2 := half_s3m (-(i : ℤ) - 1)
      have h3 : (-(i : ℤ)) * (3 * (-(i : ℤ)) - 1) - (-(i : ℤ) - 1) * (3 * (-(i : ℤ) - 1) - 1)
          = 2 * (3 * (-(i : ℤ) - 1) + 1) := by ring
      linarith

/-- `R_{3s+2}(K) = R_1(K − s(3s+1)/2)` (using the reflection `R_2 = R_1`). -/
lemma RZ_three_mul_add_two : ∀ s K : ℤ, RZ (3 * s + 2) K = RZ 1 (K - s * (3 * s + 1) / 2) := by
  intro s
  induction s using Int.induction_on with
  | zero =>
      intro K
      have := RZ_refl 1 K
      rw [show (3 : ℤ) - 1 = 2 by norm_num] at this
      simpa using this
  | succ i ih =>
      intro K
      rw [show 3 * ((i : ℤ) + 1) + 2 = (3 * (i : ℤ) + 2) + 3 by ring, RZ_shift, ih]
      congr 1
      have h1 := half_s3p (i : ℤ)
      have h2 := half_s3p ((i : ℤ) + 1)
      have h3 : ((i : ℤ) + 1) * (3 * ((i : ℤ) + 1) + 1) - (i : ℤ) * (3 * (i : ℤ) + 1)
          = 2 * (3 * (i : ℤ) + 2) := by ring
      linarith
  | pred i ih =>
      intro K
      have h := RZ_shift (3 * (-(i : ℤ) - 1) + 2) (K + (3 * (-(i : ℤ) - 1) + 2))
      rw [show 3 * (-(i : ℤ) - 1) + 2 + 3 = 3 * -(i : ℤ) + 2 by ring,
        show K + (3 * (-(i : ℤ) - 1) + 2) - (3 * (-(i : ℤ) - 1) + 2) = K by ring] at h
      rw [← h, ih]
      congr 1
      have h1 := half_s3p (-(i : ℤ))
      have h2 := half_s3p (-(i : ℤ) - 1)
      have h3 : (-(i : ℤ)) * (3 * (-(i : ℤ)) + 1) - (-(i : ℤ) - 1) * (3 * (-(i : ℤ) - 1) + 1)
          = 2 * (3 * (-(i : ℤ) - 1) + 2) := by ring
      linarith


/-! ## Stage 4b: Euler's pentagonal coefficients as a box sum; sign bookkeeping -/

def pentE (s : ℤ) : ℕ := (s * (3 * s - 1) / 2).toNat

lemma pent_nonneg (s : ℤ) : 0 ≤ s * (3 * s - 1) := by
  rcases le_or_gt s 0 with h | h <;> nlinarith

lemma two_mul_pentE (s : ℤ) : 2 * ((pentE s : ℕ) : ℤ) = s * (3 * s - 1) := by
  have h := half_s3m s
  have h0 := pent_nonneg s
  unfold pentE
  generalize s * (3 * s - 1) = X at h h0 ⊢
  omega

lemma pentE_pos (m : ℕ) : pentE ((m : ℤ) + 1) = (m + 1) * (3 * m + 2) / 2 := by
  have h : ((m : ℤ) + 1) * (3 * ((m : ℤ) + 1) - 1) = (((m + 1) * (3 * m + 2) : ℕ) : ℤ) := by
    push_cast; ring
  unfold pentE; rw [h]; omega

lemma pentE_neg (m : ℕ) : pentE (-((m : ℤ) + 1)) = (m + 1) * (3 * m + 4) / 2 := by
  have h : (-((m : ℤ) + 1)) * (3 * (-((m : ℤ) + 1)) - 1) = (((m + 1) * (3 * m + 4) : ℕ) : ℤ) := by
    push_cast; ring
  unfold pentE; rw [h]; omega

/-- **Euler's coefficients**: `[q^a](q;q)_∞ = Σ_{s, s(3s−1)/2 = a} (−1)^s` (any box `|s| ≤ k+1`, `a ≤ k`). -/
lemma coeff_qfacInf_box (a k : ℕ) (h : a ≤ k) :
    coeff a qfacInf = ∑ s ∈ Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1),
      if pentE s = a then sgn s else 0 := by
  rw [euler_pentagonal, pentSeries, PowerSeries.coeff_map, coeff_pentTheta (show a + 1 ≤ k + 1 by omega),
    pentFiniteP]
  simp only [map_add, map_sum]
  rw [show Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1)
        = Finset.Icc (-(((k + 1 : ℕ)) : ℤ)) ((k + 1 : ℕ) : ℤ) by push_cast; rfl,
    sum_Icc_neg_nat_nat]
  congr 1
  · rw [PowerSeries.coeff_one, show pentE 0 = 0 by decide]
    by_cases ha : a = 0
    · subst ha; simp [sgn]
    · rw [if_neg ha, if_neg (Ne.symm ha), map_zero]
  · refine Finset.sum_congr rfl fun m _ => ?_
    rw [pentTermP, map_add, coeff_Xpow_mul_C, coeff_Xpow_mul_C, map_add, pentE_pos, pentE_neg]
    congr 1
    · by_cases hh : a = (m + 1) * (3 * m + 2) / 2
      · rw [if_pos hh, if_pos hh.symm, evm1_T_eq_sgn]
      · rw [if_neg hh, if_neg (Ne.symm hh), map_zero]
    · by_cases hh : a = (m + 1) * (3 * m + 4) / 2
      · rw [if_pos hh, if_pos hh.symm, evm1_T_eq_sgn]
      · rw [if_neg hh, if_neg (Ne.symm hh), map_zero]

lemma sgn_add (a b : ℤ) : sgn (a + b) = sgn a * sgn b := by
  unfold sgn
  by_cases ha : Even a <;> by_cases hb : Even b <;> simp [ha, hb, Int.even_add]

lemma sgn_three_mul (s : ℤ) : sgn (3 * s) = sgn s := by
  unfold sgn
  simp [Int.even_mul, show ¬ Even (3 : ℤ) by decide]

lemma sgn_tsum (t : ℤ × ℤ × ℤ) : sgn (tsum t) = sgn t.1 * sgn t.2.1 * sgn t.2.2 := by
  unfold tsum; rw [sgn_add, sgn_add]

lemma pent_bound {t : ℤ} {k : ℤ} (h : t * (3 * t - 1) ≤ 6 * k + 2) : -(k + 1) ≤ t ∧ t ≤ k + 1 := by
  have h0 := pent_nonneg t
  constructor <;> nlinarith [sq_nonneg (t + k + 1), sq_nonneg (t - k - 1)]

lemma pentE_sum_iff (t : ℤ × ℤ × ℤ) (m : ℕ) :
    pentE t.1 + pentE t.2.1 + pentE t.2.2 = m ↔ 3 * dsum t + 2 * tsum t = 2 * m := by
  have h1 := two_mul_pentE t.1
  have h2 := two_mul_pentE t.2.1
  have h3 := two_mul_pentE t.2.2
  have e : 3 * dsum t + 2 * tsum t
      = t.1 * (3 * t.1 - 1) + t.2.1 * (3 * t.2.1 - 1) + t.2.2 * (3 * t.2.2 - 1) := by
    simp only [dsum, tsum]; ring
  rw [e]
  constructor
  · intro h; rw [← h1, ← h2, ← h3]; push_cast [← h]; ring
  · intro h
    have : (2 : ℤ) * ((pentE t.1 + pentE t.2.1 + pentE t.2.2 : ℕ) : ℤ) = 2 * m := by
      push_cast; linarith
    exact_mod_cast (by linarith : ((pentE t.1 + pentE t.2.1 + pentE t.2.2 : ℕ) : ℤ) = m)

/-- solutions of `3·dsum + 2·tsum = 6k+e` (`e ≤ 2`) live in `box3 (k+1)`. -/
lemma mem_box3_of_pent {t : ℤ × ℤ × ℤ} {k e : ℤ} (he : e ≤ 2) (h : 3 * dsum t + 2 * tsum t = 6 * k + e) :
    t ∈ box3 (k.toNat + 1) := by
  have e' : 3 * dsum t + 2 * tsum t
      = t.1 * (3 * t.1 - 1) + t.2.1 * (3 * t.2.1 - 1) + t.2.2 * (3 * t.2.2 - 1) := by
    simp only [dsum, tsum]; ring
  have p1 := pent_nonneg t.1; have p2 := pent_nonneg t.2.1; have p3 := pent_nonneg t.2.2
  have b1 := pent_bound (t := t.1) (k := k) (by linarith)
  have b2 := pent_bound (t := t.2.1) (k := k) (by linarith)
  have b3 := pent_bound (t := t.2.2) (k := k) (by linarith)
  have hk : 0 ≤ 6 * k + 2 := by linarith
  simp only [box3, Finset.mem_product, Finset.mem_Icc]
  push_cast
  refine ⟨⟨by omega, by omega⟩, ⟨by omega, by omega⟩, ⟨by omega, by omega⟩⟩

lemma sum_box3_shrink {M : Type*} [AddCommMonoid M] (B c : ℕ) (hB : c ≤ B)
    (P : ℤ × ℤ × ℤ → Prop) [DecidablePred P] (g : ℤ × ℤ × ℤ → M)
    (hP : ∀ t, P t → t ∈ box3 c) :
    ∑ t ∈ box3 B, (if P t then g t else 0) = ∑ t ∈ box3 c, (if P t then g t else 0) := by
  symm
  apply Finset.sum_subset
  · intro t ht
    simp only [box3, Finset.mem_product, Finset.mem_Icc] at ht ⊢
    push_cast at ht ⊢; omega
  · intro t _ hnot
    rw [if_neg (fun hp => hnot (hP t hp))]

/-! ## Stage 4c: Winquist's specialization `E·R₀`, `E·R₁` -/

/-- the slice series `R_N(q) = Σ_K RZ N K q^K`. -/
noncomputable def RS (N : ℤ) : PowerSeries ℤ := mk fun j => RZ N j

/-- convolution with Euler's product. -/
lemma coeff_qfacInf_mul_RS (r : ℤ) (k : ℕ) :
    coeff k (qfacInf * RS r) = ∑ s ∈ Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1),
      sgn s * RZ r ((k : ℤ) - pentE s) := by
  rw [PowerSeries.coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
  simp only [RS, coeff_mk]
  rw [Finset.sum_congr rfl fun i hi => by
        rw [coeff_qfacInf_box i k (by simp at hi; omega), Finset.sum_mul]]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun s _ => ?_
  simp only [ite_mul, zero_mul]
  rw [Finset.sum_ite_eq]
  split_ifs with hs
  · simp only [Finset.mem_range] at hs
    rw [Nat.cast_sub (by omega)]
  · rw [RZ_neg]
    · ring
    · simp only [Finset.mem_range, not_lt] at hs; push_cast; omega

lemma sum_Icc_neg_reindex (B : ℤ) (f : ℤ → ℤ) :
    ∑ s ∈ Finset.Icc (-B) B, f s = ∑ s ∈ Finset.Icc (-B) B, f (-s) := by
  refine Finset.sum_nbij' (fun x => -x) (fun x => -x) ?_ ?_ ?_ ?_ ?_
  · intro x hx; simp only [Finset.mem_coe, Finset.mem_Icc, Finset.coe_Icc, Set.mem_Icc] at hx ⊢; omega
  · intro x hx; simp only [Finset.mem_coe, Finset.mem_Icc, Finset.coe_Icc, Set.mem_Icc] at hx ⊢; omega
  · intro x _; simp
  · intro x _; simp
  · intro x _; simp

/-- the regrouping step shared by both residues: `Σ_s sgn s · R_{3s+r}(k−s)` is a signed triple sum. -/
lemma sum_sgn_RZ_eq (k : ℕ) (r : ℤ) (hr : r = 0 ∨ r = 1) :
    ∑ s ∈ Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1), sgn s * RZ (3 * s + r) ((k : ℤ) - s)
      = sgn r * ∑ t ∈ box3 (k + 1),
          if 3 * dsum t + 2 * tsum t = 6 * k + 2 * r then sgn t.1 * sgn t.2.1 * sgn t.2.2 else 0 := by
  have hbig : ∀ s ∈ Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1),
      RZ (3 * s + r) ((k : ℤ) - s) = ∑ t ∈ box3 (2 * k + 2),
        if dsum t = 2 * ((k : ℤ) - s) ∧ tsum t = 3 * s + r then 1 else 0 := by
    intro s hs
    simp only [Finset.mem_Icc] at hs
    exact RZ_eq_of_box _ _ _ (by omega)
  rw [Finset.sum_congr rfl fun s hs => by rw [hbig s hs, Finset.mul_sum]]
  rw [Finset.sum_comm]
  have hsplit : ∀ t ∈ box3 (2 * k + 2),
      (∑ s ∈ Finset.Icc (-((k : ℤ) + 1)) ((k : ℤ) + 1),
        sgn s * if dsum t = 2 * ((k : ℤ) - s) ∧ tsum t = 3 * s + r then 1 else 0)
      = sgn r * if 3 * dsum t + 2 * tsum t = 6 * k + 2 * r then sgn t.1 * sgn t.2.1 * sgn t.2.2 else 0 := by
    intro t _
    by_cases hc : 3 * dsum t + 2 * tsum t = 6 * k + 2 * r
    · rw [if_pos hc]
      have hm := mem_box3_of_pent (k := (k : ℤ)) (e := 2 * r) (by omega) hc
      simp only [box3, Finset.mem_product, Finset.mem_Icc] at hm
      push_cast at hm
      obtain ⟨s0, hs0⟩ : ∃ s0, tsum t = 3 * s0 + r := ⟨(tsum t - r) / 3, by
        have : (3 : ℤ) ∣ tsum t - r := by
          have hd : 3 * dsum t + 2 * tsum t - 6 * k - 2 * r = 0 := by linarith
          exact ⟨dsum t + tsum t - 2 * k - r, by linarith⟩
        omega⟩
      have hd0 : dsum t = 2 * ((k : ℤ) - s0) := by linarith
      rw [Finset.sum_eq_single s0]
      · rw [if_pos ⟨hd0, hs0⟩, mul_one, ← sgn_tsum, hs0, sgn_add, sgn_three_mul]
        rcases hr with rfl | rfl
        · simp [sgn]
        · simp only [sgn]; split_ifs <;> norm_num
      · intro b _ hb
        rw [if_neg]; · ring
        rintro ⟨_, h2⟩; exact hb (by linarith)
      · intro hs0m
        exfalso; apply hs0m
        simp only [Finset.mem_Icc, tsum] at hs0 ⊢
        constructor <;> omega
    · rw [if_neg hc, mul_zero]
      refine Finset.sum_eq_zero fun s _ => ?_
      rw [if_neg]; · ring
      rintro ⟨h1, h2⟩; exact hc (by rw [h1, h2]; ring)
  rw [Finset.sum_congr rfl hsplit, ← Finset.mul_sum]
  rw [sum_box3_shrink (2 * k + 2) (k + 1) (by omega)
      (fun t => 3 * dsum t + 2 * tsum t = 6 * k + 2 * r) (fun t => sgn t.1 * sgn t.2.1 * sgn t.2.2)
      (fun t ht => by
        have := mem_box3_of_pent (k := (k : ℤ)) (e := 2 * r) (by omega) ht
        simpa using this)]

/-- the signed triple sum is a coefficient of `(q;q)³`. -/
lemma coeff_qfacInf_cube_eq (m k : ℕ) (e : ℤ) (he : e = 0 ∨ e = 1) (hm : (m : ℤ) = 3 * k + e) :
    coeff m (qfacInf ^ 3) = ∑ t ∈ box3 (k + 1),
      if 3 * dsum t + 2 * tsum t = 6 * k + 2 * e then sgn t.1 * sgn t.2.1 * sgn t.2.2 else 0 := by
  rw [coeff_cube_of_box qfacInf pentE sgn _ m (fun i hi => coeff_qfacInf_box i m hi)]
  have hbox : (Finset.Icc (-((m : ℤ) + 1)) ((m : ℤ) + 1)) ×ˢ (Finset.Icc (-((m : ℤ) + 1)) ((m : ℤ) + 1)) ×ˢ
      (Finset.Icc (-((m : ℤ) + 1)) ((m : ℤ) + 1)) = box3 (m + 1) := by
    simp only [box3]; push_cast; rfl
  rw [hbox]
  have hcond : ∀ t : ℤ × ℤ × ℤ, (pentE t.1 + pentE t.2.1 + pentE t.2.2 = m) ↔
      (3 * dsum t + 2 * tsum t = 6 * k + 2 * e) := by
    intro t; rw [pentE_sum_iff, hm]; constructor <;> intro h <;> linarith
  simp_rw [hcond]
  apply sum_box3_shrink (m + 1) (k + 1) (by omega)
  intro t ht
  have := mem_box3_of_pent (k := (k : ℤ)) (e := 2 * e) (by omega) ht
  simpa using this

/-- **(I0′)**: `[q^k](E·R₀) = [q^{3k}] E³`. -/
theorem coeff_E_RS0 (k : ℕ) : coeff k (qfacInf * RS 0) = coeff (3 * k) (qfacInf ^ 3) := by
  rw [coeff_qfacInf_mul_RS]
  have h : ∀ s : ℤ, RZ 0 ((k : ℤ) - pentE s) = RZ (3 * s + 0) ((k : ℤ) - s) := by
    intro s
    rw [add_zero, RZ_three_mul]
    congr 1
    have h1 := two_mul_pentE s; have h2 := half_ss1 s
    linarith
  simp_rw [h]
  rw [sum_sgn_RZ_eq k 0 (Or.inl rfl), coeff_qfacInf_cube_eq (3 * k) k 0 (Or.inl rfl) (by push_cast; ring)]
  simp [sgn]

/-- **(I1′)**: `[q^k](E·R₁) = −[q^{3k+1}] E³`. -/
theorem coeff_E_RS1 (k : ℕ) : coeff k (qfacInf * RS 1) = -coeff (3 * k + 1) (qfacInf ^ 3) := by
  rw [coeff_qfacInf_mul_RS, sum_Icc_neg_reindex]
  have h : ∀ s : ℤ, sgn (-s) * RZ 1 ((k : ℤ) - pentE (-s)) = sgn s * RZ (3 * s + 1) ((k : ℤ) - s) := by
    intro s
    rw [sgn_neg, RZ_three_mul_add_one]
    congr 2
    have h1 := two_mul_pentE (-s); have h2 := half_s3m s
    linarith
  simp_rw [h]
  rw [sum_sgn_RZ_eq k 1 (Or.inr rfl), coeff_qfacInf_cube_eq (3 * k + 1) k 1 (Or.inr rfl) (by push_cast; ring)]
  simp [sgn]


/-! ## Stage 5a: the char-11 Frobenius chain (ported from the mod-7 file) -/

open MockTheta5.Bailey

noncomputable def Ψ11 : PowerSeries ℤ →+* PowerSeries (ZMod 11) := PowerSeries.map (Int.castRingHom (ZMod 11))
noncomputable def E11 : PowerSeries ℤ →+* PowerSeries ℤ := (PowerSeries.expand 11 (by norm_num)).toRingHom
noncomputable def expand11 : PowerSeries (ZMod 11) →+* PowerSeries (ZMod 11) :=
  (PowerSeries.expand 11 (by norm_num)).toRingHom

instance fact_prime_eleven : Fact (Nat.Prime 11) := ⟨by decide⟩
instance charP_powerSeries_zmod11 : CharP (PowerSeries (ZMod 11)) 11 :=
  charP_of_injective_ringHom (PowerSeries.C_injective (R := ZMod 11)) 11

theorem E11_X : E11 X = X ^ 11 := PowerSeries.expand_X 11 (by norm_num)

theorem coeff_Ψ11 (n : ℕ) (f : PowerSeries ℤ) : coeff n (Ψ11 f) = (Int.castRingHom (ZMod 11)) (coeff n f) := PowerSeries.coeff_map _ _ _

theorem coeff_expand11 (n : ℕ) (f : PowerSeries (ZMod 11)) : coeff n (expand11 f) = if 11 ∣ n then coeff (n / 11) f else 0 := by
  have h : expand11 f = PowerSeries.expand 11 (by norm_num) f := rfl
  rw [h, PowerSeries.coeff_expand]

theorem E11_qfac (n : ℕ) : E11 (qfac n) = ∏ i ∈ Finset.range n, (1 - X ^ (11 * i + 11)) := by
  rw [qfac, map_prod]
  refine Finset.prod_congr rfl (fun i _ => ?_)
  rw [map_sub, map_one, map_pow, E11_X, ← pow_mul, show 11 * (i + 1) = 11 * i + 11 from by ring]

theorem frob_factor11 (e : ℕ) : ((1 : PowerSeries (ZMod 11)) - X ^ e) ^ 11 = 1 - X ^ (11 * e) := by rw [sub_pow_char_of_commute _ (Commute.all _ _), one_pow, ← pow_mul, Nat.mul_comm e 11]

theorem coeff_E11_qfacInf {i N : ℕ} (h : i + 1 ≤ 11 * N) : coeff i (E11 qfacInf) = coeff i (E11 (qfac N)) := by
  have hdvd : (X : PowerSeries ℤ) ^ (i + 1) ∣ (E11 qfacInf - E11 (qfac N)) := by
    obtain ⟨g, hg⟩ : (X : PowerSeries ℤ) ^ N ∣ (qfacInf - qfac N) := by
      rw [PowerSeries.X_pow_dvd_iff]; intro j hj
      rw [map_sub, coeff_qfacInf (show j + 1 ≤ N by omega), sub_self]
    rw [← map_sub, hg, map_mul, map_pow, E11_X, ← pow_mul]
    exact dvd_mul_of_dvd_left (pow_dvd_pow X (by omega)) _
  obtain ⟨c, hc⟩ := hdvd
  have hz : coeff i (E11 qfacInf) - coeff i (E11 (qfac N)) = 0 := by
    rw [← map_sub, hc]; exact MockTheta5.mt_coeff_Xpow_mul_zero _ _ i (by omega)
  exact sub_eq_zero.mp hz

theorem frobenius_qfac11 (N : ℕ) : (Ψ11 (qfac N)) ^ 11 = Ψ11 (E11 (qfac N)) := by
  rw [E11_qfac, Ψ11, qfac, map_prod, map_prod, ← Finset.prod_pow]
  refine Finset.prod_congr rfl (fun i _ => ?_)
  simp only [map_sub, map_one, map_pow, PowerSeries.map_X]
  rw [frob_factor11 (i + 1), show 11 * (i + 1) = 11 * i + 11 from by ring]

theorem coeff_congr11 {f g : PowerSeries (ZMod 11)} {m : ℕ} (h : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (f - g)) : coeff m f = coeff m g := by
  obtain ⟨c, hc⟩ := h
  have hz : coeff m f - coeff m g = 0 := by rw [← map_sub, hc, coeff_X_pow_mul']; simp
  exact sub_eq_zero.mp hz

theorem frobenius_qfacInf11 : (Ψ11 qfacInf) ^ 11 = Ψ11 (E11 qfacInf) := by
  ext m
  have hqf : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (Ψ11 qfacInf - Ψ11 (qfac (m + 1))) := by
    rw [PowerSeries.X_pow_dvd_iff]; intro i hi
    rw [map_sub, Ψ11, PowerSeries.coeff_map, PowerSeries.coeff_map,
        coeff_qfacInf (show i + 1 ≤ m + 1 by omega), sub_self]
  have h11 : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ ((Ψ11 qfacInf) ^ 11 - (Ψ11 (qfac (m + 1))) ^ 11) :=
    dvd_trans hqf (sub_dvd_pow_sub_pow _ _ 11)
  have hE : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (Ψ11 (E11 qfacInf) - Ψ11 (E11 (qfac (m + 1)))) := by
    rw [PowerSeries.X_pow_dvd_iff]; intro i hi
    rw [map_sub, Ψ11, PowerSeries.coeff_map, PowerSeries.coeff_map,
        coeff_E11_qfacInf (show i + 1 ≤ 11 * (m + 1) by omega), sub_self]
  rw [coeff_congr11 h11, frobenius_qfac11, coeff_congr11 hE]

theorem g11_unit : IsUnit (Ψ11 qfacInf) := isUnit_qfacInf.map Ψ11

theorem g11_pow11 : (Ψ11 qfacInf) ^ 11 = expand11 (Ψ11 qfacInf) := by
  rw [frobenius_qfacInf11]
  ext n
  rw [coeff_Ψ11, coeff_expand11]
  have h : E11 qfacInf = PowerSeries.expand 11 (by norm_num) qfacInf := rfl
  rw [h, PowerSeries.coeff_expand]
  split_ifs with hd
  · rw [coeff_Ψ11]
  · simp

theorem Ψ11_partitionGF_mul : Ψ11 partitionGF * Ψ11 qfacInf = 1 := by rw [← map_mul, partitionGF, Ring.inverse_mul_cancel _ isUnit_qfacInf, map_one]

theorem P_eq11 : Ψ11 partitionGF = Ψ11 (qfacInf ^ 10) * expand11 (Ψ11 partitionGF) := by
  set g := Ψ11 qfacInf with hg
  set P := Ψ11 partitionGF with hP
  have hu : IsUnit g := g11_unit
  have hPg : P * g = 1 := Ψ11_partitionGF_mul
  have hgP : g * P = 1 := by rw [mul_comm]; exact hPg
  have hgexp : g ^ 11 * expand11 P = 1 := by rw [g11_pow11, ← map_mul, hgP, map_one]
  have hgx : g * (g ^ 10 * expand11 P) = 1 := by
    rw [← mul_assoc, show g * g ^ 10 = g ^ 11 from by ring]; exact hgexp
  have hPinv : P = Ring.inverse g := by
    calc P = P * 1 := (mul_one _).symm
      _ = P * (g * Ring.inverse g) := by rw [Ring.mul_inverse_cancel g hu]
      _ = (P * g) * Ring.inverse g := by ring
      _ = Ring.inverse g := by rw [hPg, one_mul]
  rw [show Ψ11 (qfacInf ^ 10) = g ^ 10 from by rw [map_pow]]
  conv_lhs => rw [hPinv]
  symm
  calc g ^ 10 * expand11 P = 1 * (g ^ 10 * expand11 P) := (one_mul _).symm
    _ = (Ring.inverse g * g) * (g ^ 10 * expand11 P) := by rw [Ring.inverse_mul_cancel g hu]
    _ = Ring.inverse g * (g * (g ^ 10 * expand11 P)) := by ring
    _ = Ring.inverse g := by rw [hgx, mul_one]

theorem partition_congruence_mod11_of (H : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0) (n : ℕ) : coeff (11 * n + 6) (Ψ11 partitionGF) = 0 := by
  rw [P_eq11, PowerSeries.coeff_mul]
  refine Finset.sum_eq_zero (fun p hp => ?_)
  obtain ⟨a, b⟩ := p
  have hab : a + b = 11 * n + 6 := Finset.mem_antidiagonal.mp hp
  by_cases hb : 11 ∣ b
  · obtain ⟨j, rfl⟩ := hb
    rw [H a (by omega), zero_mul]
  · rw [coeff_expand11, if_neg hb, mul_zero]

theorem eleven_dvd_coeff_partitionGF_of (H : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0) (n : ℕ) : (11 : ℤ) ∣ coeff (11 * n + 6) partitionGF := by
  have h : ((coeff (11 * n + 6) partitionGF : ℤ) : ZMod 11) = 0 := by
    have hh := partition_congruence_mod11_of H n; rwa [coeff_Ψ11] at hh
  exact_mod_cast (ZMod.intCast_zmod_eq_zero_iff_dvd _ 11).mp h

theorem eleven_dvd_partition_card_of (H : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0) (n : ℕ) : 11 ∣ Fintype.card (Nat.Partition (11 * n + 6)) := by
  have h := eleven_dvd_coeff_partitionGF_of H n
  rw [coeff_partitionGF_eq_card] at h
  exact_mod_cast h

theorem heart11 : ∀ x y : ZMod 11, x ^ 2 + y ^ 2 = 0 → x = 0 ∧ y = 0 := by decide

theorem int_heart11 {x y : ℤ} (h : (11 : ℤ) ∣ x ^ 2 + y ^ 2) : (11 : ℤ) ∣ x ∧ (11 : ℤ) ∣ y := by
  have h1 : ((x ^ 2 + y ^ 2 : ℤ) : ZMod 11) = 0 := by
    rw [ZMod.intCast_zmod_eq_zero_iff_dvd]; exact_mod_cast h
  push_cast at h1
  obtain ⟨hx, hy⟩ := heart11 _ _ h1
  rw [ZMod.intCast_zmod_eq_zero_iff_dvd] at hx hy
  exact ⟨by exact_mod_cast hx, by exact_mod_cast hy⟩

/-! ## Stage 5b: every `(q;q)·R_N` coefficient at `a ≡ 6 (mod 11)` is divisible by 11 -/

lemma RS_eq_X_pow_mul (N r c : ℤ) (hc : 0 ≤ c) (h : ∀ j : ℤ, RZ N j = RZ r (j - c)) :
    RS N = X ^ c.toNat * RS r := by
  ext j
  rw [coeff_X_pow_mul']
  simp only [RS, coeff_mk]
  rw [h]
  split_ifs with hj
  · rw [Nat.cast_sub hj]; congr 1; omega
  · rw [RZ_neg]; omega

lemma coeff_E_X_pow_mul (c a : ℕ) (F : PowerSeries ℤ) :
    coeff a (qfacInf * (X ^ c * F)) = if c ≤ a then coeff (a - c) (qfacInf * F) else 0 := by
  rw [show qfacInf * (X ^ c * F) = X ^ c * (qfacInf * F) by ring, coeff_X_pow_mul']

lemma eleven_dvd_jacobi_term {a n : ℕ} {y : ℤ} (ha : a % 11 = 6)
    (hsq : (2 * (n : ℤ) + 1) ^ 2 + y ^ 2 = 24 * a + 10) :
    (11 : ℤ) ∣ (-1 : ℤ) ^ n * (2 * (n : ℤ) + 1) := by
  have h11 : (11 : ℤ) ∣ (2 * (n : ℤ) + 1) ^ 2 + y ^ 2 := by rw [hsq]; omega
  exact Dvd.dvd.mul_left (int_heart11 h11).1 _

lemma eleven_dvd_coeff_jacobi {m : ℕ} {a : ℕ} {y : ℤ} (ha : a % 11 = 6)
    (hsq : ∀ n : ℕ, (n : ℤ) * ((n : ℤ) + 1) = 2 * (m : ℤ) → (2 * (n : ℤ) + 1) ^ 2 + y ^ 2 = 24 * a + 10) :
    (11 : ℤ) ∣ coeff m jacobiCubeSum := by
  by_cases h0 : coeff m jacobiCubeSum = 0
  · rw [h0]; exact dvd_zero _
  obtain ⟨n, hn, hv⟩ := coeff_jacobiCubeSum_value h0
  rw [hv]
  exact eleven_dvd_jacobi_term ha (hsq n hn)

lemma eleven_dvd_coeff_E_RS (N : ℤ) (a : ℕ) (ha : a % 11 = 6) :
    (11 : ℤ) ∣ coeff a (qfacInf * RS N) := by
  obtain ⟨s, r, hr, rfl⟩ : ∃ s r : ℤ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ N = 3 * s + r :=
    ⟨N / 3, N % 3, by omega, by omega⟩
  rcases hr with rfl | rfl | rfl
  · -- `N = 3s`: shift `c = 3·s(s−1)/2`, partner `y = 6s − 3`
    have hc2 := half_ss1 s
    have hc0 : 0 ≤ 3 * (s * (s - 1) / 2) := by have := dT_nonneg s; omega
    rw [RS_eq_X_pow_mul _ 0 (3 * (s * (s - 1) / 2)) hc0 (fun j => by rw [add_zero, RZ_three_mul]),
      coeff_E_X_pow_mul]
    split_ifs with hca
    · rw [coeff_E_RS0, jacobi_cube_identity]
      refine eleven_dvd_coeff_jacobi (y := 6 * s - 3) ha fun n hn => ?_
      have hct : ((3 * (s * (s - 1) / 2)).toNat : ℤ) = 3 * (s * (s - 1) / 2) := Int.toNat_of_nonneg hc0
      push_cast [Nat.cast_sub hca] at hn
      rw [hct] at hn
      nlinarith [hn, hc2]
    · exact dvd_zero _
  · -- `N = 3s+1`: shift `c = s(3s−1)/2`, partner `y = 6s − 1`
    have hc2 := half_s3m s
    have hc0 : 0 ≤ s * (3 * s - 1) / 2 := by have := pent_nonneg s; omega
    rw [RS_eq_X_pow_mul _ 1 (s * (3 * s - 1) / 2) hc0 (fun j => RZ_three_mul_add_one s j),
      coeff_E_X_pow_mul]
    split_ifs with hca
    · rw [coeff_E_RS1, jacobi_cube_identity, dvd_neg]
      refine eleven_dvd_coeff_jacobi (y := 6 * s - 1) ha fun n hn => ?_
      have hct : ((s * (3 * s - 1) / 2).toNat : ℤ) = s * (3 * s - 1) / 2 := Int.toNat_of_nonneg hc0
      push_cast [Nat.cast_sub hca] at hn
      rw [hct] at hn
      nlinarith [hn, hc2]
    · exact dvd_zero _
  · -- `N = 3s+2`: shift `c = s(3s+1)/2`, partner `y = 6s + 1`
    have hc2 := half_s3p s
    have hc0 : 0 ≤ s * (3 * s + 1) / 2 := by
      have : 0 ≤ s * (3 * s + 1) := by rcases le_or_gt s (-1) with h | h <;> nlinarith
      omega
    rw [RS_eq_X_pow_mul _ 1 (s * (3 * s + 1) / 2) hc0 (fun j => RZ_three_mul_add_two s j),
      coeff_E_X_pow_mul]
    split_ifs with hca
    · rw [coeff_E_RS1, jacobi_cube_identity, dvd_neg]
      refine eleven_dvd_coeff_jacobi (y := 6 * s + 1) ha fun n hn => ?_
      have hct : ((s * (3 * s + 1) / 2).toNat : ℤ) = s * (3 * s + 1) / 2 := Int.toNat_of_nonneg hc0
      push_cast [Nat.cast_sub hca] at hn
      rw [hct] at hn
      nlinarith [hn, hc2]
    · exact dvd_zero _

/-! ## Stage 5c: assembly -/

lemma RZ_eq_zero_of_large (N : ℤ) (j : ℕ) (hN : N ∉ Finset.Icc (-(3 * ((j : ℤ) + 1))) (3 * ((j : ℤ) + 1))) :
    RZ N j = 0 := by
  unfold RZ
  refine Finset.sum_eq_zero fun t ht => if_neg ?_
  rintro ⟨_, h2⟩
  apply hN
  simp only [box3, Finset.mem_product, Finset.mem_Icc] at ht
  push_cast at ht
  simp only [Finset.mem_Icc, tsum] at h2 ⊢
  simp only [Int.toNat_natCast] at ht
  constructor <;> omega

lemma six_coeff_qfac9_uniform (j a : ℕ) (hj : j ≤ a) :
    6 * coeff j (qfacInf ^ 9) = ∑ N ∈ Finset.Icc (-(3 * ((a : ℤ) + 1))) (3 * ((a : ℤ) + 1)),
      w3 N * RZ N j := by
  rw [six_coeff_qfac9]
  apply Finset.sum_subset
  · intro N hN; simp only [Finset.mem_Icc] at hN ⊢; omega
  · intro N _ hN; rw [RZ_eq_zero_of_large N j hN, mul_zero]

theorem eleven_dvd_coeff_qfac10 (a : ℕ) (ha : a % 11 = 6) : (11 : ℤ) ∣ coeff a (qfacInf ^ 10) := by
  have key : 6 * coeff a (qfacInf ^ 10) = ∑ N ∈ Finset.Icc (-(3 * ((a : ℤ) + 1))) (3 * ((a : ℤ) + 1)),
      w3 N * coeff a (qfacInf * RS N) := by
    rw [show qfacInf ^ 10 = qfacInf * qfacInf ^ 9 by ring, PowerSeries.coeff_mul, Finset.mul_sum]
    simp_rw [PowerSeries.coeff_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun p hp => ?_
    have hp2 : p.2 ≤ a := by have := Finset.mem_antidiagonal.mp hp; omega
    rw [show 6 * (coeff p.1 qfacInf * coeff p.2 (qfacInf ^ 9))
          = coeff p.1 qfacInf * (6 * coeff p.2 (qfacInf ^ 9)) by ring,
      six_coeff_qfac9_uniform p.2 a hp2, Finset.mul_sum]
    refine Finset.sum_congr rfl fun N _ => ?_
    simp only [RS, coeff_mk]
    ring
  have hdvd : (11 : ℤ) ∣ 6 * coeff a (qfacInf ^ 10) := by
    rw [key]
    exact Finset.dvd_sum fun N _ => Dvd.dvd.mul_left (eleven_dvd_coeff_E_RS N a ha) _
  omega

/-- the key vanishing, now a theorem (formerly the Dyson hypothesis `H`). -/
theorem coeff_Ψ11_qfac10_eq_zero : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0 := by
  intro a ha
  rw [coeff_Ψ11, eq_intCast]
  exact (ZMod.intCast_zmod_eq_zero_iff_dvd _ 11).mpr (eleven_dvd_coeff_qfac10 a ha)

/-- **Ramanujan's third partition congruence**: `11 ∣ p(11n+6)`, where `p(n)` is the number of partitions
of `n` (Mathlib's `Nat.Partition`). Unconditional. -/
theorem eleven_dvd_partition_card (n : ℕ) : 11 ∣ Fintype.card (Nat.Partition (11 * n + 6)) :=
  eleven_dvd_partition_card_of coeff_Ψ11_qfac10_eq_zero n

end MockTheta5.JTP
