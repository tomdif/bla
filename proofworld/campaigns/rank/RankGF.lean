/-
# Dyson's rank: the generating function (largest-part form)

`Σ_{λ ⊢ n} z^{rank λ} = [qⁿ] (1 + Σ_{m≥1} z^{m−1} q^m / (q/z;q)_m)`,  `rank = largest part − #parts`.
Proof as for the crank: split by the largest part `m`, and write "largest part `= m`" as
"parts `≤ m`" minus "parts `≤ m−1`", each a `Nat.Partition.genFun` product `1/(q/z;q)_m`.
-/
import RamanujanTau.CrankAndrewsGarvan

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset
open scoped PowerSeries.WithPiTopology

variable {n : ℕ}

/-- **Dyson's rank**: largest part minus number of parts. -/
def rank (l : n.Partition) : ℤ := (largest l : ℤ) - (l.parts.card : ℤ)

/-- weight: parts `≤ m`, each contributing `y`. -/
noncomputable def rW (y : ℂ) (m i c : ℕ) : ℂ := if 1 ≤ i ∧ i ≤ m then y ^ c else 0

lemma weight_r (y : ℂ) (m : ℕ) (l : n.Partition) :
    l.parts.toFinsupp.prod (rW y m) = if ∀ i ∈ l.parts, i ≤ m then y ^ l.parts.card else 0 := by
  rw [toFinsupp_prod_eq]
  simp only [rW]
  split_ifs with h
  · rw [Finset.prod_congr rfl fun i hi => if_pos ⟨l.parts_pos (Multiset.mem_toFinset.mp hi),
      h i (Multiset.mem_toFinset.mp hi)⟩, Finset.prod_pow_eq_pow_sum, Multiset.toFinset_sum_count_eq]
  · push_neg at h
    obtain ⟨i, hi, hm⟩ := h
    exact Finset.prod_eq_zero (Multiset.mem_toFinset.mpr hi) (if_neg fun h' => by omega)

noncomputable def rSeq (y : ℂ) (m i : ℕ) : PowerSeries ℂ := if i + 1 ≤ m then 1 - C y * X ^ (i + 1) else 1

lemma rSeq_prod (y : ℂ) (m : ℕ) : ∀ N, ∏ i ∈ range N, rSeq y m i = poch y 1 (min N m) := by
  intro N
  induction N with
  | zero => simp [poch_zero]
  | succ N ih =>
      rw [prod_range_succ, ih, rSeq]
      by_cases h : N + 1 ≤ m
      · rw [if_pos h, show min (N + 1) m = min N m + 1 by omega, poch_succ, show 1 + min N m = N + 1 by omega]
      · rw [if_neg h, mul_one]
        congr 1; omega

lemma rW_factor (y : ℂ) (m i : ℕ) :
    ((1 : PowerSeries ℂ) + ∑' c, rW y m (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * rSeq y m i = 1 := by
  by_cases h : i + 1 ≤ m
  · have hw : ∀ c, rW y m (i + 1) (c + 1) = y ^ (c + 1) := fun c => by
      unfold rW; rw [if_pos ⟨by omega, h⟩]
    have hd : rSeq y m i = 1 - C y * X ^ (i + 1) := by unfold rSeq; rw [if_pos h]
    simp only [hw, hd]
    exact geo_factor y (p := i + 1) (by omega)
  · have hw : ∀ c, rW y m (i + 1) (c + 1) = 0 := fun c => by unfold rW; rw [if_neg (by omega)]
    have hd : rSeq y m i = 1 := by unfold rSeq; rw [if_neg h]
    simp only [hw, hd, zero_smul, tsum_zero, add_zero, mul_one]

/-- `genFun(rW y m) · (yq;q)_m = 1`. -/
theorem genFun_rW_mul (y : ℂ) (m : ℕ) : Nat.Partition.genFun (rW y m) * poch y 1 m = 1 := by
  have hp : HasProd (rSeq y m) (poch y 1 m) := by
    have he : ∀ i, rSeq y m i = 1 + (rSeq y m i - 1) := fun i => by ring
    rw [show rSeq y m = fun i => 1 + (rSeq y m i - 1) from funext he]
    refine hasProd_of_stable _ (fun i => ?_) _ (fun k => ⟨m, fun N hN => ?_⟩)
    · simp only [rSeq]; split_ifs
      · exact ⟨-C y, by ring⟩
      · simp
    · simp only [add_sub_cancel]
      rw [rSeq_prod y m N, min_eq_right hN]
  have h := (Nat.Partition.hasProd_genFun (rW y m)).mul hp
  rw [show (fun i => ((1 : PowerSeries ℂ) + ∑' c, rW y m (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * rSeq y m i)
      = fun _ => 1 from funext (rW_factor y m)] at h
  exact h.unique hasProd_one

lemma genFun_rW_eq (y : ℂ) (m : ℕ) : Nat.Partition.genFun (rW y m) = Ring.inverse (poch y 1 m) := by
  have h := genFun_rW_mul y m
  have hu : IsUnit (poch y 1 m) := isUnit_poch y (le_refl 1) m
  calc Nat.Partition.genFun (rW y m)
      = Nat.Partition.genFun (rW y m) * (poch y 1 m * Ring.inverse (poch y 1 m)) := by
        rw [Ring.mul_inverse_cancel _ hu, mul_one]
    _ = Ring.inverse (poch y 1 m) := by rw [← mul_assoc, h, one_mul]

lemma largest_facts' (l : n.Partition) (hn : 1 ≤ n) :
    1 ≤ largest l ∧ largest l ≤ n ∧ ∀ m, (∀ i ∈ l.parts, i ≤ m) ↔ largest l ≤ m := by
  have hne : l.parts ≠ 0 := by
    intro h; have := l.parts_sum; rw [h, Multiset.sum_zero] at this; omega
  have hne' : l.parts.toFinset.Nonempty := by
    obtain ⟨x, hx⟩ := Multiset.exists_mem_of_ne_zero hne; exact ⟨x, Multiset.mem_toFinset.mpr hx⟩
  obtain ⟨L, hLmem', hL⟩ := Finset.exists_mem_eq_sup l.parts.toFinset hne' id
  have hLmem : L ∈ l.parts := Multiset.mem_toFinset.mp hLmem'
  have hsup : largest l = L := by rw [largest, hL]; rfl
  refine ⟨hsup ▸ l.parts_pos hLmem, hsup ▸ (l.parts_sum ▸ Multiset.le_sum_of_mem hLmem), fun m => ?_⟩
  constructor
  · intro h; rw [hsup]; exact h L hLmem
  · intro h i hi
    have hle : i ≤ L := by
      have := Finset.le_sup (f := id) (Multiset.mem_toFinset.mpr hi); rw [hL] at this; exact this
    exact le_trans hle (hsup ▸ h)

lemma zpow_rank_decomp {z : ℂ} (hz : z ≠ 0) (hn : 1 ≤ n) (l : n.Partition) :
    z ^ rank l = ∑ m ∈ Icc 1 n, z ^ m * (l.parts.toFinsupp.prod (rW z⁻¹ m)
      - l.parts.toFinsupp.prod (rW z⁻¹ (m - 1))) := by
  simp only [weight_r]
  obtain ⟨hL1, hLn, hiff⟩ := largest_facts' l hn
  simp only [hiff]
  rw [Finset.sum_eq_single (largest l)]
  · rw [if_pos le_rfl, if_neg (by omega), sub_zero, rank, zpow_sub₀ hz, zpow_natCast, zpow_natCast, inv_pow,
      div_eq_mul_inv]
  · intro m hm hne
    have hm2 := Finset.mem_Icc.mp hm
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [if_neg (by omega), if_neg (by omega), sub_self, mul_zero]
    · rw [if_pos (by omega), if_pos (by omega), sub_self, mul_zero]
  · intro hnot; exact absurd (Finset.mem_Icc.mpr ⟨hL1, hLn⟩) hnot

lemma inv_poch_sub (y : ℂ) {m : ℕ} (hm : 1 ≤ m) :
    Ring.inverse (poch y 1 m) - Ring.inverse (poch y 1 (m - 1)) = C y * X ^ m * Ring.inverse (poch y 1 m) := by
  obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
  have hu := isUnit_poch y (le_refl 1) k
  have hu' := isUnit_poch y (le_refl 1) (k + 1)
  rw [show k + 1 - 1 = k by omega]
  have hs : poch y 1 (k + 1) = poch y 1 k * (1 - C y * X ^ (1 + k)) := poch_succ y 1 k
  have h1 : poch y 1 k * Ring.inverse (poch y 1 k) = 1 := Ring.mul_inverse_cancel _ hu
  have h2 : poch y 1 (k + 1) * Ring.inverse (poch y 1 (k + 1)) = 1 := Ring.mul_inverse_cancel _ hu'
  apply hu'.mul_right_cancel
  have e1 : Ring.inverse (poch y 1 (k + 1)) * poch y 1 (k + 1) = 1 := Ring.inverse_mul_cancel _ hu'
  have e2 : Ring.inverse (poch y 1 k) * poch y 1 (k + 1) = 1 - C y * X ^ (1 + k) := by
    rw [hs, ← mul_assoc, Ring.inverse_mul_cancel _ hu, one_mul]
  calc (Ring.inverse (poch y 1 (k + 1)) - Ring.inverse (poch y 1 k)) * poch y 1 (k + 1)
      = Ring.inverse (poch y 1 (k + 1)) * poch y 1 (k + 1) - Ring.inverse (poch y 1 k) * poch y 1 (k + 1) :=
        sub_mul _ _ _
    _ = 1 - (1 - C y * X ^ (1 + k)) := by rw [e1, e2]
    _ = C y * X ^ (k + 1) * Ring.inverse (poch y 1 (k + 1)) * poch y 1 (k + 1) := by
        rw [mul_assoc, e1]; ring

/-- **the rank generating function** (largest-part form), coefficientwise:
`Σ_{λ⊢n} z^{rank λ} = Σ_{m=1}^{n} [qⁿ] z^{m−1} q^m/(q/z;q)_m` for `n ≥ 1`. -/
theorem rank_sum_eq {z : ℂ} (hz : z ≠ 0) (hn : 1 ≤ n) :
    ∑ l : n.Partition, z ^ rank l
      = ∑ m ∈ Icc 1 n, coeff n (C (z ^ (m - 1)) * X ^ m * Ring.inverse (poch z⁻¹ 1 m)) := by
  have h1 : ∑ l : n.Partition, z ^ rank l = ∑ m ∈ Icc 1 n, z ^ m * (coeff n (Nat.Partition.genFun (rW z⁻¹ m))
      - coeff n (Nat.Partition.genFun (rW z⁻¹ (m - 1)))) := by
    simp only [Nat.Partition.coeff_genFun, ← Finset.sum_sub_distrib, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun l _ => zpow_rank_decomp hz hn l
  rw [h1]
  refine Finset.sum_congr rfl fun m hm => ?_
  have hm1 := (Finset.mem_Icc.mp hm).1
  rw [genFun_rW_eq, genFun_rW_eq, ← map_sub, inv_poch_sub _ hm1, ← coeff_C_mul,
    show C (z ^ m) * (C z⁻¹ * X ^ m * Ring.inverse (poch z⁻¹ 1 m))
      = C (z ^ m * z⁻¹) * X ^ m * Ring.inverse (poch z⁻¹ 1 m) by rw [map_mul]; ring]
  congr 3
  rw [pow_sub₀ _ hz hm1, pow_one]

/-! ## G3: the finite identity `Σ_j q^{j²+j} y^j [n, j] (yq^{j+2};q)_{n−j} = 1` for every `y`
(a polynomial in `y` vanishing at all `y = q^b` by `F_eq_one`). -/

section G3
open MockTheta5.Bailey

lemma gaussBinom_symm : ∀ a b : ℕ, gaussBinom (a + b) a = gaussBinom (a + b) b
  | 0, b => by simp
  | a + 1, 0 => by simp
  | a + 1, b + 1 => by
      have h1 := gaussBinom_succ_succ (a + b + 1) a
      have h2 := gaussBinom_pascal1 (a + b + 1) b
      have i1 := gaussBinom_symm a (b + 1)
      have i2 := gaussBinom_symm (a + 1) b
      rw [show a + (b + 1) = a + b + 1 by ring] at i1
      rw [show a + 1 + b = a + b + 1 by ring] at i2
      rw [show a + 1 + (b + 1) = a + b + 1 + 1 by ring, h1, h2, i1, i2, show a + b + 1 - b = a + 1 by omega]

/-- the polynomial (in `Y`) whose vanishing is the identity. -/
noncomputable def Ppoly (n : ℕ) : Polynomial (PowerSeries ℤ) :=
  ∑ j ∈ range (n + 1), Polynomial.C (X ^ (j ^ 2 + j) * gaussBinom n j) * Polynomial.X ^ j
      * ∏ i ∈ range (n - j), (1 - Polynomial.X * Polynomial.C (X ^ (j + 2 + i))) - 1

lemma Ppoly_eval (n b : ℕ) : (Ppoly n).eval (X ^ b) = MockTheta5.Bailey.F n (b + 1) - 1 := by
  rw [Ppoly, MockTheta5.Bailey.F, Polynomial.eval_sub, Polynomial.eval_one, Polynomial.eval_finsetSum]
  congr 1
  refine sum_congr rfl fun j _ => ?_
  rw [Polynomial.eval_mul, Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_pow, Polynomial.eval_X,
    Polynomial.eval_prod, MockTheta5.Bailey.rfac]
  simp only [Polynomial.eval_sub, Polynomial.eval_one, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_C]
  have e1 : (X : PowerSeries ℤ) ^ (j ^ 2 + j) * gaussBinom n j * (X ^ b) ^ j
      = X ^ (j ^ 2 + (b + 1) * j) * gaussBinom n j := by
    rw [← pow_mul, mul_comm (X ^ (j ^ 2 + j) * gaussBinom n j), ← mul_assoc, ← pow_add]
    rw [show b * j + (j ^ 2 + j) = j ^ 2 + (b + 1) * j by ring]
  rw [e1]
  congr 1
  refine prod_congr rfl fun i _ => ?_
  rw [← pow_add, show b + (j + 2 + i) = b + 1 + j + 1 + i by ring]

lemma Ppoly_eq_zero (n : ℕ) : Ppoly n = 0 := by
  apply Polynomial.eq_zero_of_infinite_isRoot
  have hsub : Set.range (fun b : ℕ => (X : PowerSeries ℤ) ^ b) ⊆ {x | (Ppoly n).IsRoot x} := by
    rintro _ ⟨b, rfl⟩
    show (Ppoly n).eval (X ^ b) = 0
    rw [Ppoly_eval, MockTheta5.Bailey.F_eq_one, sub_self]
  refine Set.Infinite.mono hsub (Set.infinite_range_of_injective fun b c h => ?_)
  have := congrArg (coeff b) h
  by_contra hbc
  simp [coeff_X_pow, hbc] at this

/-- **G3** in `ℂ⟦X⟧`: `Σ_{j≤n} y^j q^{j²+j} [n,j] ∏_{i<n−j}(1 − y q^{j+2+i}) = 1` for all `y ∈ ℂ`. -/
theorem G3 (y : ℂ) (n : ℕ) :
    ∑ j ∈ range (n + 1), C (y ^ j) * X ^ (j ^ 2 + j) * MockTheta5.JTP.ψC (gaussBinom n j)
      * ∏ i ∈ range (n - j), (1 - C y * X ^ (j + 2 + i)) = 1 := by
  have h := congrArg (fun p => p.eval₂ MockTheta5.JTP.ψC (C y)) (Ppoly_eq_zero n)
  simp only [Polynomial.eval₂_zero] at h
  rw [Ppoly, Polynomial.eval₂_sub, Polynomial.eval₂_one, Polynomial.eval₂_finset_sum, sub_eq_zero] at h
  refine Eq.trans ?_ h
  refine sum_congr rfl fun j _ => ?_
  simp only [Polynomial.eval₂_mul, Polynomial.eval₂_C, Polynomial.eval₂_X_pow, Polynomial.eval₂_X,
    Polynomial.eval₂_finset_prod, Polynomial.eval₂_sub, Polynomial.eval₂_one, Polynomial.eval₂_pow, map_mul, map_pow,
    MockTheta5.JTP.ψC, PowerSeries.map_X]
  ring

end G3

/-! ## G2: `1/(xq;q)_k = Σ_j x^j q^j [k+j−1, j]` (truncated form) -/

section G2
open MockTheta5.Bailey
local notation "ψ" => MockTheta5.JTP.ψC

/-- the `N`-truncated `q`-binomial series. -/
noncomputable def Tk (x : ℂ) (k N : ℕ) : PowerSeries ℂ :=
  ∑ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + j - 1) j)

lemma pascal_k (k j : ℕ) (hj : 1 ≤ j) :
    gaussBinom (k + j) j = gaussBinom (k + j - 1) j + X ^ k * gaussBinom (k + j - 1) (j - 1) := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  have h := gaussBinom_pascal1 (k + i) i
  rw [show k + (i + 1) = k + i + 1 by ring, show k + i + 1 - 1 = k + i by omega, show i + 1 - 1 = i by omega,
    h, show k + i - i = k by omega]

lemma Tk_step (x : ℂ) (k N : ℕ) :
    Tk x (k + 1) N * (1 - C x * X ^ (k + 1)) - Tk x k N
      = -(C (x ^ (N + 1)) * X ^ (N + 1 + k) * ψ (gaussBinom (k + N) N)) := by
  unfold Tk
  -- the j-th difference via Pascal, and the shifted subtracted sum
  have hP : ∀ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + 1 + j - 1) j)
      - C (x ^ j) * X ^ j * ψ (gaussBinom (k + j - 1) j)
      = if j = 0 then 0 else C (x ^ j) * X ^ (j + k) * ψ (gaussBinom (k + j - 1) (j - 1)) := by
    intro j _
    split_ifs with h0
    · subst h0; simp
    · rw [show k + 1 + j - 1 = k + j by omega, pascal_k k j (by omega)]
      simp only [map_add, map_mul, map_pow, PowerSeries.map_X, pow_add]; ring
  have e1 : ∑ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + 1 + j - 1) j) * (1 - C x * X ^ (k + 1))
      = ∑ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + 1 + j - 1) j)
        - ∑ j ∈ range (N + 1), C (x ^ (j + 1)) * X ^ (j + 1 + k) * ψ (gaussBinom (k + j) j) := by
    rw [← sum_sub_distrib]
    refine sum_congr rfl fun j _ => ?_
    rw [show k + 1 + j - 1 = k + j by omega, pow_succ x j, map_mul]; ring
  rw [Finset.sum_mul, e1]
  have e2 : ∑ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + 1 + j - 1) j)
      - ∑ j ∈ range (N + 1), C (x ^ j) * X ^ j * ψ (gaussBinom (k + j - 1) j)
      = ∑ i ∈ range N, C (x ^ (i + 1)) * X ^ (i + 1 + k) * ψ (gaussBinom (k + i) i) := by
    rw [← sum_sub_distrib, sum_congr rfl hP, sum_range_succ', if_pos rfl, add_zero]
    refine sum_congr rfl fun i _ => ?_
    rw [if_neg (by omega), show k + (i + 1) - 1 = k + i by omega, show i + 1 - 1 = i by omega]
  have e3 : ∑ j ∈ range (N + 1), C (x ^ (j + 1)) * X ^ (j + 1 + k) * ψ (gaussBinom (k + j) j)
      = ∑ i ∈ range N, C (x ^ (i + 1)) * X ^ (i + 1 + k) * ψ (gaussBinom (k + i) i)
        + C (x ^ (N + 1)) * X ^ (N + 1 + k) * ψ (gaussBinom (k + N) N) := sum_range_succ _ _
  linear_combination e2 - e3

lemma isUnit_one_sub (c : ℂ) (k : ℕ) : IsUnit (1 - C c * X ^ (k + 1) : PowerSeries ℂ) := by
  rw [isUnit_iff_constantCoeff]; simp

/-- **G2**: `X^{N+1} ∣ 1/(xq;q)_k − T_k(N)`. -/
theorem inv_poch_dvd (x : ℂ) : ∀ k N, (X : PowerSeries ℂ) ^ (N + 1) ∣ Ring.inverse (poch x 1 k) - Tk x k N
  | 0, N => by
      have : Tk x 0 N = 1 := by
        unfold Tk
        rw [sum_range_succ', sum_eq_zero fun j _ => by
          rw [show 0 + (j + 1) - 1 = j by omega, gaussBinom_eq_zero_of_lt (by omega), map_zero, mul_zero]]
        simp
      rw [this, poch_zero, Ring.inverse_one, sub_self]; exact dvd_zero _
  | k + 1, N => by
      obtain ⟨d, hd⟩ := inv_poch_dvd x k N
      have hu := isUnit_one_sub x k
      have hpk := isUnit_poch x (le_refl 1) k
      have hs : poch x 1 (k + 1) = poch x 1 k * (1 - C x * X ^ (k + 1)) := by
        rw [poch_succ, show 1 + k = k + 1 by ring]
      have hinv : Ring.inverse (poch x 1 (k + 1)) = Ring.inverse (poch x 1 k) * Ring.inverse (1 - C x * X ^ (k + 1)) := by
        rw [hs, inverse_mul' hpk hu]
      have hstep := Tk_step x k N
      refine ⟨Ring.inverse (1 - C x * X ^ (k + 1)) * (d + C (x ^ (N + 1)) * X ^ k * ψ (gaussBinom (k + N) N)), ?_⟩
      have hcancel : Tk x (k + 1) N * (1 - C x * X ^ (k + 1)) * Ring.inverse (1 - C x * X ^ (k + 1))
          = Tk x (k + 1) N := by rw [mul_assoc, Ring.mul_inverse_cancel _ hu, mul_one]
      rw [hinv, show Ring.inverse (poch x 1 k) = Tk x k N + X ^ (N + 1) * d by rw [← hd]; ring,
        show Tk x k N = Tk x (k + 1) N * (1 - C x * X ^ (k + 1))
          + C (x ^ (N + 1)) * X ^ (N + 1 + k) * ψ (gaussBinom (k + N) N) by linear_combination -hstep]
      linear_combination hcancel

end G2

/-! ## G4: the Durfee form equals the largest-part form -/

section G4
open MockTheta5.Bailey
local notation "ψ" => MockTheta5.JTP.ψC

lemma poch_split (c : ℂ) (a b : ℕ) :
    poch c 1 (a + b) = poch c 1 a * ∏ i ∈ range b, (1 - C c * X ^ (a + 1 + i)) := by
  rw [poch, prod_range_add, poch]
  congr 1
  refine prod_congr rfl fun i _ => by rw [show 1 + (a + i) = a + 1 + i by ring]

/-- G3 in the form used below: `Σ_{k≤m} y^k q^{k²+m−k}[m−1,m−k]/(yq;q)_k = y q^m/(yq;q)_m` (`m ≥ 1`). -/
lemma G3inv (y : ℂ) {m : ℕ} (hm : 1 ≤ m) :
    ∑ k ∈ range (m + 1), C (y ^ k) * X ^ (k ^ 2 + m - k) * ψ (gaussBinom (m - 1) (m - k))
      * Ring.inverse (poch y 1 k) = C y * X ^ m * Ring.inverse (poch y 1 m) := by
  have h := G3 y (m - 1)
  have hu := isUnit_poch y (le_refl 1) m
  set W : PowerSeries ℂ := C y * X ^ m * Ring.inverse (poch y 1 m) with hW
  have hterm : ∀ j ∈ range (m - 1 + 1), C (y ^ j) * X ^ (j ^ 2 + j) * ψ (gaussBinom (m - 1) j)
      * (∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) * W
      = C (y ^ (j + 1)) * X ^ ((j + 1) ^ 2 + m - (j + 1)) * ψ (gaussBinom (m - 1) (m - (j + 1)))
        * Ring.inverse (poch y 1 (j + 1)) := by
    intro j hj
    have hj' := mem_range.mp hj
    have hsplit : poch y 1 m = poch y 1 (j + 1) * ∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i)) := by
      have := poch_split y (j + 1) (m - 1 - j)
      rw [show j + 1 + (m - 1 - j) = m by omega] at this
      rw [this]
    have hsym : gaussBinom (m - 1) j = gaussBinom (m - 1) (m - (j + 1)) := by
      have := gaussBinom_symm j (m - 1 - j)
      rwa [show j + (m - 1 - j) = m - 1 by omega, show m - 1 - j = m - (j + 1) by omega] at this
    have hu' := isUnit_poch y (le_refl 1) (j + 1)
    have hPr : IsUnit (∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) :=
      isUnit_of_mul_isUnit_right (hsplit ▸ hu)
    have hP : (∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) * Ring.inverse (poch y 1 m)
        = Ring.inverse (poch y 1 (j + 1)) := by
      rw [hsplit, inverse_mul' hu' hPr, mul_comm (Ring.inverse (poch y 1 (j + 1))), ← mul_assoc,
        Ring.mul_inverse_cancel _ hPr, one_mul]
    rw [← hsym, show (j + 1) ^ 2 + m - (j + 1) = j ^ 2 + j + m by
      have : (j + 1) ^ 2 = j ^ 2 + 2 * j + 1 := by ring
      omega]
    calc C (y ^ j) * X ^ (j ^ 2 + j) * ψ (gaussBinom (m - 1) j)
          * (∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) * W
        = C (y ^ j) * C y * (X ^ (j ^ 2 + j) * X ^ m) * ψ (gaussBinom (m - 1) j)
          * ((∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) * Ring.inverse (poch y 1 m)) := by
          rw [hW]; ring
      _ = _ := by rw [hP, ← map_mul, ← pow_succ, ← pow_add]
  have hsum : ∑ j ∈ range (m - 1 + 1), C (y ^ j) * X ^ (j ^ 2 + j) * ψ (gaussBinom (m - 1) j)
      * (∏ i ∈ range (m - 1 - j), (1 - C y * X ^ (j + 2 + i))) * W = W := by
    rw [← sum_mul, h, one_mul]
  rw [sum_congr rfl hterm, show m - 1 + 1 = m by omega] at hsum
  rw [sum_range_succ', ← hsum]
  simp only [pow_zero, map_one, one_mul, zero_add, show m - 0 = m by omega,
    gaussBinom_eq_zero_of_lt (show m - 1 < m by omega), map_zero, mul_zero, zero_mul, add_zero]

lemma coeffC_congr {n : ℕ} {f g g' : PowerSeries ℂ} (h : (X : PowerSeries ℂ) ^ (n + 1) ∣ g - g') :
    coeff n (f * g) = coeff n (f * g') := by
  obtain ⟨d, hd⟩ := h
  have : f * g - f * g' = X ^ (n + 1) * (f * d) := by rw [← mul_sub, hd]; ring
  have h0 : coeff n (f * g - f * g') = 0 := by rw [this, coeff_X_pow_mul', if_neg (by omega)]
  rwa [map_sub, sub_eq_zero] at h0

lemma coeffC_Xpow_zero {n d : ℕ} (h : n < d) (f : PowerSeries ℂ) : coeff n (X ^ d * f) = 0 := by
  rw [coeff_X_pow_mul', if_neg (by omega)]

/-- the Durfee form `Σ_k q^{k²}(xy)^k/((xq;q)_k(yq;q)_k)`, coefficientwise. -/
noncomputable def Dser (x y : ℂ) : PowerSeries ℂ :=
  mk fun n => coeff n (∑ k ∈ range (n + 1),
    C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k))

/-- the largest-part form `1 + Σ_{m≥1} x^m y q^m/(yq;q)_m`, coefficientwise. -/
noncomputable def Lser (x y : ℂ) : PowerSeries ℂ :=
  mk fun n => coeff n (1 + ∑ m ∈ Icc 1 n, C (x ^ m * y) * X ^ m * Ring.inverse (poch y 1 m))

/-- **Durfee = largest part**: `Σ_k q^{k²}(xy)^k/((xq;q)_k(yq;q)_k) = 1 + Σ_{m≥1} x^m y q^m/(yq;q)_m`. -/
theorem D_eq_L (x y : ℂ) : Dser x y = Lser x y := by
  ext n
  rw [Dser, Lser, coeff_mk, coeff_mk]
  set f : ℕ → ℕ → ℂ := fun k j => coeff n (C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch y 1 k)
    * (C (x ^ j) * X ^ j * ψ (gaussBinom (k + j - 1) j))) with hf
  -- left side as a double sum
  have hL : coeff n (∑ k ∈ range (n + 1),
      C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k))
      = ∑ k ∈ range (n + 1), ∑ j ∈ range (n + 1), f k j := by
    rw [map_sum]
    refine sum_congr rfl fun k _ => ?_
    rw [show C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch x 1 k) * Ring.inverse (poch y 1 k)
        = C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch y 1 k) * Ring.inverse (poch x 1 k) by ring,
      coeffC_congr (inv_poch_dvd x k n), Tk, mul_sum, map_sum]
  -- right side as a double sum
  set V : ℕ → PowerSeries ℂ := fun m => C (x ^ m) * ∑ k ∈ range (m + 1),
    C (y ^ k) * X ^ (k ^ 2 + m - k) * ψ (gaussBinom (m - 1) (m - k)) * Ring.inverse (poch y 1 k) with hV
  have hV0 : V 0 = 1 := by simp [hV, poch_zero]
  have hVm : ∀ m, 1 ≤ m → V m = C (x ^ m * y) * X ^ m * Ring.inverse (poch y 1 m) := by
    intro m hm; rw [hV]; dsimp only; rw [G3inv y hm, map_mul]; ring
  have hR : coeff n (1 + ∑ m ∈ Icc 1 n, C (x ^ m * y) * X ^ m * Ring.inverse (poch y 1 m))
      = ∑ m ∈ range (n + 1), ∑ k ∈ range (m + 1), f k (m - k) := by
    have e1 : (1 : PowerSeries ℂ) + ∑ m ∈ Icc 1 n, C (x ^ m * y) * X ^ m * Ring.inverse (poch y 1 m)
        = ∑ m ∈ range (n + 1), V m := by
      rw [sum_range_succ', hV0, add_comm]
      congr 1
      rw [← Finset.Ico_add_one_right_eq_Icc, sum_Ico_eq_sum_range, show n + 1 - 1 = n by omega]
      exact sum_congr rfl fun i _ => by rw [hVm (i + 1) (by omega), show 1 + i = i + 1 by ring]
    rw [e1, map_sum]
    refine sum_congr rfl fun m _ => ?_
    rw [hV]; dsimp only; rw [mul_sum, map_sum]
    refine sum_congr rfl fun k hk => ?_
    have hk' := mem_range.mp hk
    rw [hf]; dsimp only
    congr 1
    rw [show k + (m - k) - 1 = m - 1 by omega, show x ^ m = x ^ k * x ^ (m - k) by
      rw [← pow_add, show k + (m - k) = m by omega], show k ^ 2 + m - k = k ^ 2 + (m - k) by omega, pow_add]
    simp only [map_mul, mul_pow]
    ring
  rw [hL, hR, sum_range_diag_flip]
  refine sum_congr rfl fun k hk => ?_
  symm
  refine sum_subset (range_subset_range.mpr (by omega)) fun j _ hj => ?_
  simp only [mem_range, not_lt] at hj
  have hk' := mem_range.mp hk
  rw [hf]; dsimp only
  rw [show C ((x * y) ^ k) * X ^ (k ^ 2) * Ring.inverse (poch y 1 k)
      * (C (x ^ j) * X ^ j * ψ (gaussBinom (k + j - 1) j))
      = X ^ (k ^ 2 + j) * (C ((x * y) ^ k) * Ring.inverse (poch y 1 k) * C (x ^ j)
        * ψ (gaussBinom (k + j - 1) j)) by rw [pow_add]; ring]
  have : k ≤ k ^ 2 := Nat.le_self_pow two_ne_zero k
  exact coeffC_Xpow_zero (by omega) _

/-- **The rank generating function in Durfee form**: for `z ≠ 0`,
`Σ_{λ⊢n} z^{rank λ} = [qⁿ] Σ_k q^{k²}/((zq;q)_k (q/z;q)_k)`. -/
theorem rank_durfee {z : ℂ} (hz : z ≠ 0) (n : ℕ) :
    ∑ l : n.Partition, z ^ rank l = coeff n (Dser z z⁻¹) := by
  rw [D_eq_L, Lser, coeff_mk]
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [Finset.Icc_self, show ¬ (1 ≤ 0) from by omega, Finset.Icc_eq_empty, not_false_eq_true,
      Finset.sum_empty, add_zero, map_one, coeff_zero_eq_constantCoeff]
    rw [Fintype.sum_unique]; simp [rank, largest]
  · rw [rank_sum_eq hz hn, map_add, coeff_one, if_neg (by omega), zero_add, map_sum]
    refine Finset.sum_congr rfl fun m hm => ?_
    rw [pow_sub₀ _ hz (Finset.mem_Icc.mp hm).1, pow_one]

end G4

end CrankProof
