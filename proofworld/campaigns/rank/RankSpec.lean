/-
# (2.18) specialized at `z = q^k`, base `q⁵`

`(q^k;q⁵)_∞ (q^{5−k};q⁵)_∞ (q⁵;q⁵)_∞ · Σ_n q^{5n²}/((q^k;q⁵)_{n+1}(q^{5−k};q⁵)_n)
   = Σ_{a≥0} (−1)^a (q^{5T(a)+ka} + [a≠0] q^{5T(a)−ka}) · E₅(PT_a)`,
proved like `RankHR1` but in `ℤ⟦q⟧` with the step-5 Pochhammers `Pinf`. For `k = 1, 2` the left side is
`J_{5,1}(1+φ)` and `J_{5,2}(1+ψ)` with Ramanujan's mock theta functions `φ, ψ` (Lost Notebook p. 20).
-/
import RamanujanTau.RankRamanujan5

set_option autoImplicit false

namespace RankProof
open PowerSeries Finset MockTheta5.Bailey
open MockTheta5.JTP (qfacInf isUnit_qfacInf E5)

section Spec
variable (k : ℕ)

lemma E5_eq_Ea (f : PowerSeries ℤ) : E5 f = Ea 5 (by norm_num) f := rfl

lemma Pinf_shift (b a : ℕ) (hb : 1 ≤ b) (ha : 1 ≤ a) (n : ℕ) :
    Pinf b a = Pfin b a n * Pinf (b + a * n) a := by
  symm
  refine Pinf_ext hb ha fun N => ?_
  have h1 := X_pow_dvd_Pinf_sub (b + a * n) a (by omega) ha N
  have h2 := X_pow_dvd_Pfin_sub b a N hb ha (n + N) (by omega)
  have hsplit : Pfin b a (n + N) = Pfin b a n * Pfin (b + a * n) a N := by
    rw [Pfin, Pfin, Pfin, prod_range_add]; congr 1
    exact prod_congr rfl fun i _ => by congr 2; ring
  rw [show Pfin b a n * Pinf (b + a * n) a - Pfin b a N
      = Pfin b a n * (Pinf (b + a * n) a - Pfin (b + a * n) a N) + (Pfin b a (n + N) - Pfin b a N) by
    rw [hsplit]; ring]
  exact dvd_add (dvd_mul_of_dvd_right h1 _) h2

lemma isUnit_Pfin (b a n : ℕ) (hb : 1 ≤ b) : IsUnit (Pfin b a n) := by
  rw [isUnit_iff_constantCoeff, Pfin, map_prod]
  rw [prod_eq_one fun i _ => by
    rw [map_sub, map_one, map_pow, constantCoeff_X, zero_pow (by omega), sub_zero]]
  exact isUnit_one

/-- `Σ_{n≤N} q^{5n²}/((q^k;q⁵)_{n+1}(q^{5−k};q⁵)_n)`. -/
noncomputable def PhiTr (N : ℕ) : PowerSeries ℤ :=
  ∑ n ∈ range (N + 1), X ^ (5 * n ^ 2) * Ring.inverse (Pfin k 5 (n + 1)) * Ring.inverse (Pfin (5 - k) 5 n)

/-- Ramanujan's series `1 + φ` (`k = 1`), `1 + ψ` (`k = 2`). -/
noncomputable def Phi : PowerSeries ℤ := mk fun c => coeff c (PhiTr k c)

lemma PhiTr_dvd {M N : ℕ} (h : M ≤ N) : (X : PowerSeries ℤ) ^ (M + 1) ∣ PhiTr k N - PhiTr k M := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [PhiTr, PhiTr, show M + d + 1 = (M + 1) + d by ring, sum_range_add, add_sub_cancel_left]
  refine dvd_sum fun n _ => ?_
  rw [mul_assoc]
  have : M + 1 ≤ (M + 1 + n) ^ 2 := le_trans (by omega) (Nat.le_self_pow two_ne_zero _)
  exact dvd_mul_of_dvd_left (pow_dvd_pow X (by omega)) _

lemma Phi_dvd (N : ℕ) : (X : PowerSeries ℤ) ^ (N + 1) ∣ Phi k - PhiTr k N := by
  rw [X_pow_dvd_iff]; intro c hc
  have := (X_pow_dvd_iff.mp (PhiTr_dvd k (show c ≤ N by omega))) c (by omega)
  rw [map_sub, sub_eq_zero] at this
  rw [map_sub, Phi, coeff_mk, this, sub_self]

/-- absorbing the finite Pochhammers. -/
lemma absorb (hk : 1 ≤ k) (hk5 : k ≤ 4) (n : ℕ) :
    Pinf k 5 * Pinf (5 - k) 5 * (X ^ (5 * n ^ 2) * Ring.inverse (Pfin k 5 (n + 1)) * Ring.inverse (Pfin (5 - k) 5 n))
      = X ^ (5 * n ^ 2) * Pinf (k + 5 * (n + 1)) 5 * Pinf (5 - k + 5 * n) 5 := by
  rw [Pinf_shift k 5 hk (by norm_num) (n + 1), Pinf_shift (5 - k) 5 (by omega) (by norm_num) n]
  have h1 := Ring.mul_inverse_cancel _ (isUnit_Pfin k 5 (n + 1) hk)
  have h2 := Ring.mul_inverse_cancel _ (isUnit_Pfin (5 - k) 5 n (by omega))
  linear_combination X ^ (5 * n ^ 2) * Pinf (k + 5 * (n + 1)) 5 * Pinf (5 - k + 5 * n) 5
    * (Pfin (5 - k) 5 n * Ring.inverse (Pfin (5 - k) 5 n) * h1 + h2)

/-- the exponent of the `(n, i, j)` term. -/
def ex (n i j : ℕ) : ℕ := 5 * (n ^ 2 + i.choose 2 + j.choose 2 + (n + 1) * (i + j)) + k * i - k * j

noncomputable def tauS (c n i j : ℕ) : ℤ := coeff c (X ^ ex k n i j * E5 (Ring.inverse (qfac i) * Ring.inverse (qfac j)))

lemma Ea_inv (i : ℕ) : Ea 5 (by norm_num) (Ring.inverse (qfac i)) = E5 (Ring.inverse (qfac i)) := rfl

/-- Euler-expanding both products. -/
lemma euler_n (hk : 1 ≤ k) (hk5 : k ≤ 4) (c n : ℕ) :
    coeff c (X ^ (5 * n ^ 2) * Pinf (k + 5 * (n + 1)) 5 * Pinf (5 - k + 5 * n) 5)
      = ∑ i ∈ range (c + 1), ∑ j ∈ range (c + 1), (-1) ^ (i + j) * tauS k c n i j := by
  rw [coeffZ_congr (Pinf_euler_dvd (5 - k + 5 * n) 5 (by omega) (by norm_num) c), mul_right_comm,
    coeffZ_congr (Pinf_euler_dvd (k + 5 * (n + 1)) 5 (by omega) (by norm_num) c)]
  simp only [mul_sum, sum_mul, map_sum]
  refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
  rw [tauS, ← coeff_C_mul, Ea_inv, Ea_inv]
  congr 1
  have hex : ex k n i j = 5 * n ^ 2 + (5 * i.choose 2 + (k + 5 * (n + 1)) * i) + (5 * j.choose 2 + (5 - k + 5 * n) * j) := by
    unfold ex
    have hA : 5 * (n ^ 2 + i.choose 2 + j.choose 2 + (n + 1) * (i + j))
        = 5 * n ^ 2 + 5 * i.choose 2 + 5 * j.choose 2 + 5 * ((n + 1) * i) + 5 * ((n + 1) * j) := by ring
    have hB : (k + 5 * (n + 1)) * i = k * i + 5 * ((n + 1) * i) := by ring
    have hC : (5 - k + 5 * n) * j + k * j = 5 * ((n + 1) * j) := by
      rw [← add_mul, show 5 - k + 5 * n + k = 5 * (n + 1) by omega]; ring
    rw [hA, hB]
    generalize k * i = ki at *
    generalize k * j = kj at *
    generalize (n + 1) * i = ni at *
    generalize (n + 1) * j = nj at *
    generalize (5 - k + 5 * n) * j = mj at *
    omega
  rw [hex, pow_add, pow_add, map_mul, pow_add (-1 : ℤ), map_mul]
  ring

lemma E5_dvd {m : ℕ} {f : PowerSeries ℤ} (h : (X : PowerSeries ℤ) ^ m ∣ f) : (X : PowerSeries ℤ) ^ m ∣ E5 f :=
  (pow_dvd_pow X (Nat.le_mul_of_pos_left m (by norm_num) : m ≤ 5 * m)).trans (by rw [E5_eq_Ea]; exact Ea_dvd 5 _ h)

/-- one block: `Σ_{n,j} q^{s+5((j+n)²+a(j+n)+j)}/((q⁵;q⁵)_{j+a}(q⁵;q⁵)_j) = q^s E₅(Σ_m q^{m²+am} β*_m)`. -/
lemma block (c a s : ℕ) (hs : a ≤ s) (f : ℕ → ℕ → ℤ)
    (hf : ∀ n j, f n j = coeff c (X ^ (s + 5 * ((j + n) ^ 2 + a * (j + n) + j))
      * E5 (Ring.inverse (qfac j) * Ring.inverse (qfac (j + a))))) :
    ∑ n ∈ range (c + 1), ∑ j ∈ range (c + 1 - a), f n j = coeff c (X ^ s * E5 (LHSz a)) := by
  have hz : ∀ n j, c < s + 5 * ((j + n) ^ 2 + a * (j + n) + j) → f n j = 0 := fun n j h => by
    rw [hf]; exact coeffZ_Xpow_zero h _
  have hdvd : (X : PowerSeries ℤ) ^ (c + 1) ∣ E5 (LHSz a)
      - E5 (∑ m ∈ range (c + 1), X ^ (m ^ 2 + a * m) * SLz a m) := by
    rw [← map_sub]; exact E5_dvd (CrankProof.LHSz_dvd a c)
  rw [coeffZ_congr hdvd]
  calc ∑ n ∈ range (c + 1), ∑ j ∈ range (c + 1 - a), f n j
      = ∑ j ∈ range (c + 1), ∑ n ∈ range (c + 1 - j), f n j := by
        rw [sum_comm]
        have e1 : ∑ j ∈ range (c + 1 - a), ∑ n ∈ range (c + 1), f n j
            = ∑ j ∈ range (c + 1), ∑ n ∈ range (c + 1), f n j := by
          refine sum_subset (range_subset_range.mpr (by omega)) fun j _ hj => ?_
          simp only [mem_range, not_lt] at hj
          refine sum_eq_zero fun n _ => hz n j ?_
          generalize (j + n) ^ 2 + a * (j + n) = Y
          omega
        rw [e1]
        refine sum_congr rfl fun j _ => ?_
        symm
        refine sum_subset (range_subset_range.mpr (by omega)) fun n _ hn => ?_
        simp only [mem_range, not_lt] at hn
        refine hz n j ?_
        have : j + n ≤ (j + n) ^ 2 := Nat.le_self_pow two_ne_zero _
        generalize (j + n) ^ 2 = Y at *
        generalize a * (j + n) = Z
        omega
    _ = ∑ m ∈ range (c + 1), ∑ l ∈ range (m + 1), f (m - l) l := by
        rw [← sum_range_diag_flip (c + 1) (fun l n => f n l)]
    _ = _ := by
        rw [map_sum, mul_sum, map_sum]
        refine sum_congr rfl fun m _ => ?_
        rw [SLz, mul_sum, map_sum, mul_sum, map_sum]
        refine sum_congr rfl fun l hl => ?_
        have hl' := mem_range.mp hl
        rw [hf, show l + (m - l) = m by omega]
        congr 1
        simp only [map_mul, map_pow, MockTheta5.JTP.E5_X, ← pow_mul, ← pow_add]
        rw [show s + 5 * (m ^ 2 + a * m + l) = s + ((m ^ 2 + a * m) * 5 + l * 5) by ring]
        ring_nf

lemma ex_branch1 (n j a : ℕ) :
    ex k n (j + a) j = 5 * (a * (a + 1) / 2) + k * a + 5 * ((j + n) ^ 2 + a * (j + n) + j) := by
  have h := CrankProof.exp_HR1 a j n
  unfold ex
  rw [show k * (j + a) = k * j + k * a by ring]
  have : 5 * (n ^ 2 + (j + a).choose 2 + j.choose 2 + (n + 1) * (j + a + j))
      = 5 * (a * (a + 1) / 2) + 5 * ((j + n) ^ 2 + a * (j + n) + j) := by
    rw [← mul_add, ← add_assoc, ← h]; ring
  omega

lemma ex_branch2 (hk5 : k ≤ 4) (n i a : ℕ) :
    ex k n i (i + a) = (5 * (a * (a + 1) / 2) - k * a) + 5 * ((i + n) ^ 2 + a * (i + n) + i) := by
  have h := CrankProof.exp_HR1 a i n
  have hT : a ≤ a * (a + 1) / 2 := MockTheta5.JTP.tri_ge a
  have hka : k * a ≤ 5 * (a * (a + 1) / 2) := by nlinarith
  unfold ex
  rw [show k * (i + a) = k * i + k * a by ring]
  have : 5 * (n ^ 2 + i.choose 2 + (i + a).choose 2 + (n + 1) * (i + (i + a)))
      = 5 * (a * (a + 1) / 2) + 5 * ((i + n) ^ 2 + a * (i + n) + i) := by
    linarith [h]
  omega

/-- the two weights `q^{5T(a) ± ka}` (the specialization of `z^{±a} q^{T(a)}` in base `q⁵`). -/
def hiE (a : ℕ) : ℕ := 5 * (a * (a + 1) / 2) + k * a
def loE (a : ℕ) : ℕ := 5 * (a * (a + 1) / 2) - k * a

/-- the specialized Hecke–Rogers side, truncated. -/
noncomputable def SpecTr (N : ℕ) : PowerSeries ℤ :=
  ∑ a ∈ range (N + 1), (X ^ hiE k a + (if a = 0 then 0 else X ^ loE k a)) * C ((-1 : ℤ) ^ a) * E5 (LHSz a)

lemma neg_one_pow_twice (i a : ℕ) : (-1 : ℤ) ^ (i + a + i) = (-1) ^ a := by
  rw [show i + a + i = a + 2 * i by ring, pow_add, pow_mul]; norm_num

theorem spec_coeff (hk : 1 ≤ k) (hk5 : k ≤ 4) (c : ℕ) :
    coeff c (Pinf k 5 * Pinf (5 - k) 5 * Phi k) = coeff c (SpecTr k c) := by
  have hT : ∀ a, a ≤ a * (a + 1) / 2 := MockTheta5.JTP.tri_ge
  rw [coeffZ_congr (Phi_dvd k c), PhiTr, mul_sum, map_sum]
  simp only [absorb k hk hk5, euler_n k hk hk5]
  -- split each square
  simp only [sum_square_splitZ (fun i j => (-1) ^ (i + j) * tauS k c _ i j) (c + 1)]
  rw [sum_add_distrib, SpecTr, map_sum]
  simp only [add_mul, map_add, sum_add_distrib]
  congr 1
  · -- branch 1: `i = j + a`
    rw [sum_comm]
    refine sum_congr rfl fun a _ => ?_
    simp only [neg_one_pow_twice, ← mul_sum]
    rw [mul_comm (X ^ hiE k a), mul_assoc, coeff_C_mul]
    congr 1
    refine block c a (hiE k a) (by unfold hiE; nlinarith [hT a]) _ fun n j => ?_
    rw [tauS, ex_branch1, mul_comm (Ring.inverse (qfac (j + a)))]
    rfl
  · -- branch 2: `j = i + (b + 1)`
    have hF : ∀ b, ∑ n ∈ range (c + 1), ∑ i ∈ range (c + 1 - b - 1),
        (-1) ^ (i + (i + b + 1)) * tauS k c n i (i + b + 1)
        = coeff c (X ^ loE k (b + 1) * C ((-1 : ℤ) ^ (b + 1)) * E5 (LHSz (b + 1))) := by
      intro b
      have hs : ∀ i, (-1 : ℤ) ^ (i + (i + b + 1)) = (-1) ^ (b + 1) := fun i => by
        rw [show i + (i + b + 1) = i + (b + 1) + i by ring, neg_one_pow_twice]
      simp only [hs, ← mul_sum]
      rw [mul_comm (X ^ loE k (b + 1)), mul_assoc, coeff_C_mul, show c + 1 - b - 1 = c + 1 - (b + 1) by omega]
      congr 1
      refine block c (b + 1) (loE k (b + 1)) ?_ _ fun n i => ?_
      · unfold loE; have := hT (b + 1)
        have h2 : k * (b + 1) ≤ 4 * (b + 1) := Nat.mul_le_mul_right _ hk5
        omega
      · rw [tauS, show i + b + 1 = i + (b + 1) by ring, ex_branch2 k hk5]
        rfl
    rw [sum_comm]
    simp only [hF]
    have hz : coeff c (X ^ loE k (c + 1) * C ((-1 : ℤ) ^ (c + 1)) * E5 (LHSz (c + 1))) = 0 := by
      rw [mul_assoc]; apply coeffZ_Xpow_zero; unfold loE
      have h1 := hT (c + 1)
      have h2 : k * (c + 1) ≤ 4 * (c + 1) := Nat.mul_le_mul_right _ hk5
      omega
    rw [sum_range_succ, hz, add_zero, sum_range_succ', if_pos rfl, zero_mul, zero_mul, map_zero, add_zero]
    exact sum_congr rfl fun b _ => by rw [if_neg (by omega)]

end Spec

end RankProof
