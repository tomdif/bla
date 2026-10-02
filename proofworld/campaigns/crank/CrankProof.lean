/-
# Dyson's crank (Andrews–Garvan) — work in progress

M1: the specialized q-binomial theorem
    `(Σ_j (zq;q)_j (q/z)^j / (q;q)_j) · (q/z;q)_∞ = (q²;q)_∞`     in `ℂ⟦X⟧`, `z ≠ 0`,
by the functional equation `(1−y)G(y) = (1−ay)G(qy)` and an `X`-adic limit.
-/
import RamanujanTau.RamanujanMostBeautiful
import Mathlib.RingTheory.Polynomial.Cyclotomic.Roots
import Mathlib.FieldTheory.KummerExtension

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset

/-! ## Finite and infinite `q`-Pochhammer symbols in `ℂ⟦X⟧` -/

/-- `(c q^s; q)_n = ∏_{i<n} (1 − c q^{s+i})`. -/
noncomputable def poch (c : ℂ) (s n : ℕ) : PowerSeries ℂ := ∏ i ∈ range n, (1 - C c * X ^ (s + i))

lemma poch_zero (c : ℂ) (s : ℕ) : poch c s 0 = 1 := by simp [poch]

lemma poch_succ (c : ℂ) (s n : ℕ) : poch c s (n + 1) = poch c s n * (1 - C c * X ^ (s + n)) := by
  rw [poch, prod_range_succ]; rfl

lemma X_pow_dvd_poch_sub (c : ℂ) (s N : ℕ) : ∀ M, N ≤ M → (X : PowerSeries ℂ) ^ (s + N) ∣ poch c s M - poch c s N := by
  intro M hM
  induction M, hM using Nat.le_induction with
  | base => simp
  | succ M hNM ih =>
      rw [poch_succ, show poch c s M * (1 - C c * X ^ (s + M)) - poch c s N
          = (poch c s M - poch c s N) - poch c s M * C c * X ^ (s + M) by ring]
      exact dvd_sub ih (Dvd.dvd.mul_left (pow_dvd_pow X (by omega)) _)

/-- `(c q^s; q)_∞`, by coefficient stabilization (valid for `s ≥ 1`). -/
noncomputable def pochInf (c : ℂ) (s : ℕ) : PowerSeries ℂ := mk fun k => coeff k (poch c s (k + 1))

lemma coeff_eq_of_X_pow_dvd {f g : PowerSeries ℂ} {K k : ℕ} (h : (X : PowerSeries ℂ) ^ K ∣ f - g) (hk : k < K) :
    coeff k f = coeff k g := by
  obtain ⟨d, hd⟩ := h
  have := congrArg (coeff k) hd
  rw [map_sub, coeff_X_pow_mul', if_neg (by omega)] at this
  exact sub_eq_zero.mp this

lemma coeff_pochInf (c : ℂ) {s : ℕ} (hs : 1 ≤ s) {k N : ℕ} (h : k + 1 ≤ N) :
    coeff k (pochInf c s) = coeff k (poch c s N) := by
  rw [pochInf, coeff_mk]
  exact (coeff_eq_of_X_pow_dvd (X_pow_dvd_poch_sub c s (k + 1) N h) (by omega)).symm

lemma X_pow_dvd_pochInf_sub (c : ℂ) {s : ℕ} (hs : 1 ≤ s) (N : ℕ) :
    (X : PowerSeries ℂ) ^ N ∣ pochInf c s - poch c s N := by
  rw [PowerSeries.X_pow_dvd_iff]
  intro k hk
  rw [map_sub, coeff_pochInf c hs (show k + 1 ≤ N by omega), sub_self]

lemma constantCoeff_poch (c : ℂ) {s : ℕ} (hs : 1 ≤ s) (n : ℕ) : constantCoeff (poch c s n) = 1 := by
  rw [poch, map_prod]
  refine prod_eq_one fun i _ => ?_
  rw [map_sub, map_one, map_mul, map_pow, constantCoeff_X, zero_pow (by omega), mul_zero, sub_zero]

lemma isUnit_poch (c : ℂ) {s : ℕ} (hs : 1 ≤ s) (n : ℕ) : IsUnit (poch c s n) := by
  rw [PowerSeries.isUnit_iff_constantCoeff, constantCoeff_poch c hs]; exact isUnit_one

/-! ## The `q`-binomial series `G(y) = Σ_j c_j y^j` with `c_j = (αq;q)_j/(q;q)_j`, at `y = β q^m`

One argument gives both identities we need:
* `α = z, β = z⁻¹`: the specialized `q`-binomial theorem `G(q/z)·(q/z;q)_∞ = (q²;q)_∞`;
* `α = 0, β = z`: Euler's identity `Σ_j z^j q^j/(q;q)_j · (zq;q)_∞ = 1`. -/

variable (α β : ℂ)

/-- `c_j = (αq;q)_j / (q;q)_j`. -/
noncomputable def qcoef (j : ℕ) : PowerSeries ℂ := poch α 1 j * Ring.inverse (poch 1 1 j)

lemma inv_mul_cancel_right' {A B : PowerSeries ℂ} (hA : IsUnit A) (hB : IsUnit B) :
    Ring.inverse (A * B) * B = Ring.inverse A := by
  rw [Ring.mul_inverse_rev' (Commute.all _ _), mul_comm (Ring.inverse B), mul_assoc,
    Ring.inverse_mul_cancel _ hB, mul_one]

/-- the coefficient recursion `c_{j+1}(1 − q^{j+1}) = c_j(1 − α q^{j+1})`. -/
lemma qcoef_rec (j : ℕ) : qcoef α (j + 1) * (1 - X ^ (j + 1)) = qcoef α j * (1 - C α * X ^ (j + 1)) := by
  have h1 : (1 - X ^ (j + 1) : PowerSeries ℂ) = 1 - C 1 * X ^ (1 + j) := by rw [map_one, one_mul, add_comm]
  have hu : IsUnit (1 - C (1 : ℂ) * X ^ (1 + j) : PowerSeries ℂ) := by
    rw [PowerSeries.isUnit_iff_constantCoeff]; simp
  rw [qcoef, qcoef, poch_succ, poch_succ, h1, mul_assoc, mul_assoc (poch α 1 j),
    inv_mul_cancel_right' (isUnit_poch 1 le_rfl j) hu, add_comm 1 j]
  ring

/-- the variable `y_m = β q^m`. -/
noncomputable def yv (m : ℕ) : PowerSeries ℂ := C β * X ^ m

/-- partial sums `S_J(y_m) = Σ_{j<J} c_j y_m^j`. -/
noncomputable def Spart (m J : ℕ) : PowerSeries ℂ := ∑ j ∈ range J, qcoef α j * yv β m ^ j

/-- the series `G(y_m)`. -/
noncomputable def Gser (m : ℕ) : PowerSeries ℂ := mk fun k => coeff k (Spart α β m (k + 1))

lemma X_pow_dvd_yv_pow (m j : ℕ) : (X : PowerSeries ℂ) ^ (m * j) ∣ yv β m ^ j :=
  ⟨C β ^ j, by rw [yv, mul_pow, ← pow_mul]; ring⟩

lemma X_pow_dvd_Spart_sub {m : ℕ} (hm : 1 ≤ m) (J : ℕ) :
    ∀ J', J ≤ J' → (X : PowerSeries ℂ) ^ J ∣ Spart α β m J' - Spart α β m J := by
  intro J' hJ
  induction J', hJ using Nat.le_induction with
  | base => simp
  | succ J' hJJ ih =>
      rw [Spart, sum_range_succ, ← Spart,
        show Spart α β m J' + qcoef α J' * yv β m ^ J' - Spart α β m J
          = (Spart α β m J' - Spart α β m J) + qcoef α J' * yv β m ^ J' by ring]
      refine dvd_add ih (Dvd.dvd.mul_left (dvd_trans (pow_dvd_pow X ?_) (X_pow_dvd_yv_pow β m J')) _)
      nlinarith

lemma coeff_Gser {m : ℕ} (hm : 1 ≤ m) {k J : ℕ} (h : k + 1 ≤ J) :
    coeff k (Gser α β m) = coeff k (Spart α β m J) := by
  rw [Gser, coeff_mk]
  exact (coeff_eq_of_X_pow_dvd (X_pow_dvd_Spart_sub α β hm (k + 1) J h) (by omega)).symm

lemma X_pow_dvd_Gser_sub {m : ℕ} (hm : 1 ≤ m) (J : ℕ) :
    (X : PowerSeries ℂ) ^ J ∣ Gser α β m - Spart α β m J := by
  rw [PowerSeries.X_pow_dvd_iff]
  intro k hk
  rw [map_sub, coeff_Gser α β hm (show k + 1 ≤ J by omega), sub_self]

lemma X_pow_dvd_Spart_sub_one {m : ℕ} (hm : 1 ≤ m) : ∀ J, (X : PowerSeries ℂ) ^ m ∣ Spart α β m (J + 1) - 1 := by
  intro J
  induction J with
  | zero => simp [Spart, qcoef, poch_zero]
  | succ J ih =>
      rw [Spart, sum_range_succ, ← Spart,
        show Spart α β m (J + 1) + qcoef α (J + 1) * yv β m ^ (J + 1) - 1
          = (Spart α β m (J + 1) - 1) + qcoef α (J + 1) * yv β m ^ (J + 1) by ring]
      exact dvd_add ih (Dvd.dvd.mul_left
        (dvd_trans (pow_dvd_pow X (by nlinarith)) (X_pow_dvd_yv_pow β m (J + 1))) _)

lemma X_pow_dvd_Gser_sub_one {m : ℕ} (hm : 1 ≤ m) : (X : PowerSeries ℂ) ^ m ∣ Gser α β m - 1 := by
  have h1 := X_pow_dvd_Gser_sub α β hm m
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have h2 := X_pow_dvd_Spart_sub_one α β hm m'
  have := dvd_add h1 h2
  rwa [sub_add_sub_cancel] at this

/-! ### the functional equation `(1 − y)G(y) = (1 − αβ q^{m+1})G(qy)` -/

variable {α β}

lemma yv_mul (m : ℕ) : C α * X * yv β m = C (α * β) * X ^ (m + 1) := by
  rw [yv, map_mul, pow_succ']; ring

lemma X_pow_mul_yv_pow (m j : ℕ) : (X : PowerSeries ℂ) ^ j * yv β m ^ j = yv β (m + 1) ^ j := by
  rw [yv, yv, ← mul_pow]; congr 1; ring

lemma fe_partial (m J : ℕ) :
    (1 - yv β m) * Spart α β m (J + 1) - (1 - C (α * β) * X ^ (m + 1)) * Spart α β (m + 1) (J + 1)
      = -(qcoef α J * (1 - C α * X ^ (J + 1)) * yv β m ^ (J + 1)) := by
  have ha : C (α * β) * (X : PowerSeries ℂ) ^ (m + 1) = C α * X * yv β m := (yv_mul m).symm
  induction J with
  | zero =>
      simp only [Spart, sum_range_one, qcoef, poch_zero, Ring.inverse_one, mul_one, pow_zero, zero_add, pow_one]
      rw [ha]; ring
  | succ J ih =>
      have hs : ∀ m', Spart α β m' (J + 1 + 1) = Spart α β m' (J + 1) + qcoef α (J + 1) * yv β m' ^ (J + 1) := by
        intro m'; rw [Spart, sum_range_succ]; rfl
      have hy : yv β (m + 1) ^ (J + 1) = X ^ (J + 1) * yv β m ^ (J + 1) := (X_pow_mul_yv_pow m (J + 1)).symm
      have hrec := qcoef_rec α J
      rw [hs, hs, hy, ha]
      rw [ha] at ih
      linear_combination ih + (yv β m ^ (J + 1)) * hrec

/-- **the functional equation** `(1 − y_m) G(y_m) = (1 − αβ q^{m+1}) G(y_{m+1})` (`m ≥ 1`). -/
theorem fe {m : ℕ} (hm : 1 ≤ m) :
    (1 - yv β m) * Gser α β m = (1 - C (α * β) * X ^ (m + 1)) * Gser α β (m + 1) := by
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  set J := k + 1
  have h1 : (X : PowerSeries ℂ) ^ J ∣ (1 - yv β m) * Gser α β m - (1 - yv β m) * Spart α β m (J + 1) := by
    rw [← mul_sub]
    exact Dvd.dvd.mul_left (dvd_trans (pow_dvd_pow X (by omega)) (X_pow_dvd_Gser_sub α β hm (J + 1))) _
  have h2 : (X : PowerSeries ℂ) ^ J ∣ (1 - C (α * β) * X ^ (m + 1)) * Gser α β (m + 1)
      - (1 - C (α * β) * X ^ (m + 1)) * Spart α β (m + 1) (J + 1) := by
    rw [← mul_sub]
    exact Dvd.dvd.mul_left (dvd_trans (pow_dvd_pow X (by omega)) (X_pow_dvd_Gser_sub α β (by omega) (J + 1))) _
  have h3 : (X : PowerSeries ℂ) ^ J ∣ (1 - yv β m) * Spart α β m (J + 1)
      - (1 - C (α * β) * X ^ (m + 1)) * Spart α β (m + 1) (J + 1) := by
    rw [fe_partial]
    exact dvd_neg.mpr (Dvd.dvd.mul_left (dvd_trans (pow_dvd_pow X (by nlinarith)) (X_pow_dvd_yv_pow β m (J + 1))) _)
  have := dvd_add (dvd_sub h1 h2) h3
  rw [show (1 - yv β m) * Gser α β m - (1 - yv β m) * Spart α β m (J + 1)
        - ((1 - C (α * β) * X ^ (m + 1)) * Gser α β (m + 1) - (1 - C (α * β) * X ^ (m + 1)) * Spart α β (m + 1) (J + 1))
        + ((1 - yv β m) * Spart α β m (J + 1) - (1 - C (α * β) * X ^ (m + 1)) * Spart α β (m + 1) (J + 1))
      = (1 - yv β m) * Gser α β m - (1 - C (α * β) * X ^ (m + 1)) * Gser α β (m + 1) by ring] at this
  exact coeff_eq_of_X_pow_dvd (g := 0) (by simpa using this) (by omega) |>.trans (map_zero _)

lemma fe_iter : ∀ N : ℕ, Gser α β 1 * poch β 1 N = poch (α * β) 2 N * Gser α β (N + 1) := by
  intro N
  induction N with
  | zero => simp [poch_zero]
  | succ N ih =>
      rw [poch_succ, poch_succ, ← mul_assoc, ih, mul_assoc, mul_comm (Gser α β (N + 1)),
        show (1 - C β * X ^ (1 + N)) = 1 - yv β (N + 1) by rw [yv, add_comm],
        fe (by omega), show (2 + N) = N + 1 + 1 by ring]
      ring

/-- **the `q`-binomial theorem at `x = βq`, `a = αq`**: `G(βq)·(βq;q)_∞ = (αβq²;q)_∞`. -/
theorem qbinomial (α β : ℂ) : Gser α β 1 * pochInf β 1 = pochInf (α * β) 2 := by
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  set N := k + 1
  have h1 : (X : PowerSeries ℂ) ^ N ∣ Gser α β 1 * pochInf β 1 - Gser α β 1 * poch β 1 N := by
    rw [← mul_sub]; exact Dvd.dvd.mul_left (X_pow_dvd_pochInf_sub _ le_rfl N) _
  have h2 : (X : PowerSeries ℂ) ^ N ∣ poch (α * β) 2 N * Gser α β (N + 1) - poch (α * β) 2 N := by
    rw [show poch (α * β) 2 N * Gser α β (N + 1) - poch (α * β) 2 N = poch (α * β) 2 N * (Gser α β (N + 1) - 1) by ring]
    exact Dvd.dvd.mul_left (dvd_trans (pow_dvd_pow X (by omega)) (X_pow_dvd_Gser_sub_one α β (by omega))) _
  have h3 : (X : PowerSeries ℂ) ^ N ∣ poch (α * β) 2 N - pochInf (α * β) 2 := by
    rw [← neg_sub]; exact dvd_neg.mpr (X_pow_dvd_pochInf_sub _ (by norm_num) N)
  have := dvd_add (dvd_add h1 h2) h3
  rw [fe_iter N, show Gser α β 1 * pochInf β 1 - poch (α * β) 2 N * Gser α β (N + 1)
        + (poch (α * β) 2 N * Gser α β (N + 1) - poch (α * β) 2 N) + (poch (α * β) 2 N - pochInf (α * β) 2)
      = Gser α β 1 * pochInf β 1 - pochInf (α * β) 2 by ring] at this
  exact coeff_eq_of_X_pow_dvd (g := 0) (by simpa using this) (by omega) |>.trans (map_zero _)

lemma poch_zero_const (s n : ℕ) : poch 0 s n = 1 := by simp [poch]

lemma pochInf_zero_const (s : ℕ) : pochInf 0 s = 1 := by
  ext k; rw [pochInf, coeff_mk, poch_zero_const]

/-- **Euler's identity** `Σ_j z^j q^j/(q;q)_j · (zq;q)_∞ = 1`. -/
theorem euler_identity (z : ℂ) : Gser 0 z 1 * pochInf z 1 = 1 := by
  rw [qbinomial, zero_mul, pochInf_zero_const]

/-- **the specialized `q`-binomial theorem**: `G(q/z)·(q/z;q)_∞ = (q²;q)_∞` (`z ≠ 0`). -/
theorem qbinomial_special {z : ℂ} (hz : z ≠ 0) : Gser z z⁻¹ 1 * pochInf z⁻¹ 1 = pochInf 1 2 := by
  rw [qbinomial, mul_inv_cancel₀ hz]


/-! ## M2a: infinite products as `HasProd` (X-adic) -/

open scoped PowerSeries.WithPiTopology
open Filter Topology

lemma X_pow_dvd_order {e : PowerSeries ℂ} {k : ℕ} (h : (X : PowerSeries ℂ) ^ k ∣ e) : (k : ℕ∞) ≤ e.order := by
  rw [PowerSeries.X_pow_dvd_iff] at h
  exact PowerSeries.nat_le_order _ _ h

/-- a product `∏ (1 + e i)` with `X^{i+1} ∣ e i`, whose range partial products stabilize to `D`. -/
lemma hasProd_of_stable (e : ℕ → PowerSeries ℂ) (he : ∀ i, (X : PowerSeries ℂ) ^ (i + 1) ∣ e i)
    (D : PowerSeries ℂ) (hD : ∀ k, ∃ N0, ∀ N ≥ N0, coeff k (∏ i ∈ range N, (1 + e i)) = coeff k D) :
    HasProd (fun i => 1 + e i) D := by
  have hm : Multipliable (fun i => 1 + e i) := by
    apply WithPiTopology.multipliable_one_add_of_tendsto_order_atTop_nhds_top
    refine ENat.tendsto_nhds_top_iff_natCast_lt.mpr fun n => eventually_atTop.mpr ⟨n, fun m hm => ?_⟩
    exact lt_of_lt_of_le (by exact_mod_cast Nat.lt_succ_of_le hm) (X_pow_dvd_order (he m))
  have ht : Tendsto (fun N => ∏ i ∈ range N, (1 + e i)) atTop (𝓝 D) := by
    rw [WithPiTopology.tendsto_iff_coeff_tendsto]
    intro k
    obtain ⟨N0, hN0⟩ := hD k
    exact tendsto_atTop_of_eventually_const (i₀ := N0) fun N hN => hN0 N hN
  have h1 := tendsto_nhds_unique hm.hasProd.tendsto_prod_nat ht
  rw [← h1]; exact hm.hasProd

/-- the geometric `genFun` factor: `(1 + Σ_c a^{c+1} X^{p(c+1)})·(1 − aX^p) = 1`. -/
lemma geo_factor (a : ℂ) {p : ℕ} (hp : 1 ≤ p) :
    ((1 : PowerSeries ℂ) + ∑' c, (a ^ (c + 1)) • X ^ (p * (c + 1))) * (1 - C a * X ^ p) = 1 := by
  have hcc : (C a * X ^ p : PowerSeries ℂ).constantCoeff = 0 := by
    rw [map_mul, map_pow, constantCoeff_X, zero_pow (by omega), mul_zero]
  have hgeo : ((1 : PowerSeries ℂ) + ∑' c, (a ^ (c + 1)) • X ^ (p * (c + 1))) = ∑' c, (C a * X ^ p) ^ c := by
    rw [(WithPiTopology.summable_pow_of_constantCoeff_eq_zero hcc).tsum_eq_zero_add, pow_zero]
    refine congrArg (1 + ·) (tsum_congr fun c => ?_)
    rw [smul_eq_C_mul, mul_pow, ← map_pow, ← pow_mul]
  rw [hgeo]
  exact WithPiTopology.tsum_pow_mul_one_sub_of_constantCoeff_eq_zero hcc

/-! ## M2b: the crank and the per-partition weights -/

variable {n : ℕ}

/-- `ω(λ)`: the number of ones. -/
def ones (l : n.Partition) : ℕ := l.parts.count 1
/-- `μ(λ)`: the number of parts larger than `ω(λ)`. -/
def muC (l : n.Partition) : ℕ := (l.parts.filter fun i => l.parts.count 1 < i).card
/-- the largest part (`0` for the empty partition). -/
def largest (l : n.Partition) : ℕ := l.parts.toFinset.sup id
/-- **Dyson's crank** (Andrews–Garvan): the largest part if there are no ones, else `μ − ω`. -/
def crank (l : n.Partition) : ℤ := if ones l = 0 then (largest l : ℤ) else (muC l : ℤ) - (ones l : ℤ)

lemma toFinsupp_prod_eq (P : Multiset ℕ) (f : ℕ → ℕ → ℂ) :
    P.toFinsupp.prod f = ∏ i ∈ P.toFinset, f i (P.count i) := by
  simp [Finsupp.prod, Multiset.toFinsupp_support, Multiset.toFinsupp_apply]

lemma card_filter_eq_sum_count (P : Multiset ℕ) (p : ℕ → Prop) [DecidablePred p] :
    (P.filter p).card = ∑ i ∈ P.toFinset.filter p, P.count i := by
  rw [← Multiset.toFinset_sum_count_eq, Multiset.toFinset_filter]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [Multiset.count_filter_of_pos (Finset.mem_filter.mp hi).2]

/-- weights for "exactly `j` ones" (`fW`) and "no ones" (`gW`), parts `> j` weighted by `z`. -/
noncomputable def fW (z : ℂ) (j i c : ℕ) : ℂ := if i = 1 then (if c = j then z⁻¹ ^ j else 0) else if i ≤ j then 1 else z ^ c
noncomputable def gW (z : ℂ) (j i c : ℕ) : ℂ := if i = 1 then 0 else if i ≤ j then 1 else z ^ c

lemma prod_rest_eq (z : ℂ) (P : Multiset ℕ) (j : ℕ) (hj : 1 ≤ j) :
    ∏ i ∈ P.toFinset.erase 1, (if i ≤ j then 1 else z ^ P.count i) = z ^ (P.filter fun i => j < i).card := by
  rw [card_filter_eq_sum_count, ← Finset.prod_pow_eq_pow_sum, Finset.prod_ite, Finset.prod_const_one, one_mul]
  apply Finset.prod_congr
  · ext i; simp only [Finset.mem_filter, Finset.mem_erase, Multiset.mem_toFinset, not_le]
    constructor
    · rintro ⟨⟨_, hi⟩, h⟩; exact ⟨hi, h⟩
    · rintro ⟨hi, h⟩; exact ⟨⟨by omega, hi⟩, h⟩
  · intros; rfl

lemma prod_split_one (P : Multiset ℕ) (f : ℕ → ℕ → ℂ) :
    ∏ i ∈ P.toFinset, f i (P.count i)
      = (if 1 ∈ P then f 1 (P.count 1) else 1) * ∏ i ∈ P.toFinset.erase 1, f i (P.count i) := by
  split_ifs with h
  · exact (Finset.mul_prod_erase _ (fun i => f i (P.count i)) (Multiset.mem_toFinset.mpr h)).symm
  · rw [one_mul, Finset.erase_eq_of_notMem (by simpa using h)]

lemma weight_fg (z : ℂ) {j : ℕ} (hj : 1 ≤ j) (l : n.Partition) :
    l.parts.toFinsupp.prod (fW z j) - l.parts.toFinsupp.prod (gW z j)
      = if ones l = j then z ^ muC l * z⁻¹ ^ j else 0 := by
  rw [toFinsupp_prod_eq, toFinsupp_prod_eq, prod_split_one, prod_split_one]
  have hrest : ∀ w : ℕ → ℕ → ℂ, (∀ i, i ≠ 1 → ∀ c, w i c = if i ≤ j then 1 else z ^ c) →
      ∏ i ∈ l.parts.toFinset.erase 1, w i (l.parts.count i) = z ^ (l.parts.filter fun i => j < i).card := by
    intro w hw
    rw [← prod_rest_eq z l.parts j hj]
    exact Finset.prod_congr rfl fun i hi => hw i (Finset.ne_of_mem_erase hi) _
  rw [hrest (fW z j) (fun i hi c => by simp [fW, hi]), hrest (gW z j) (fun i hi c => by simp [gW, hi])]
  simp only [fW, gW, if_true]
  by_cases h1 : 1 ∈ l.parts
  · rw [if_pos h1, if_pos h1, zero_mul, sub_zero]
    by_cases hc : l.parts.count 1 = j
    · rw [if_pos hc, ones, if_pos hc, muC, hc]; ring
    · rw [if_neg hc, ones, if_neg hc, zero_mul]
  · rw [if_neg h1, if_neg h1, sub_self, ones, if_neg]
    rw [Multiset.count_eq_zero.mpr h1]; omega

/-- weight for "all parts in `[2, m]`". -/
noncomputable def hW (m i c : ℕ) : ℂ := if 2 ≤ i ∧ i ≤ m then 1 else 0

lemma weight_h (m : ℕ) (l : n.Partition) :
    l.parts.toFinsupp.prod (hW m) = if ∀ i ∈ l.parts, 2 ≤ i ∧ i ≤ m then 1 else 0 := by
  rw [toFinsupp_prod_eq]
  simp only [hW]
  split_ifs with h
  · exact Finset.prod_eq_one fun i hi => if_pos (h i (Multiset.mem_toFinset.mp hi))
  · push_neg at h
    obtain ⟨i, hi, hni⟩ := h
    exact Finset.prod_eq_zero (Multiset.mem_toFinset.mpr hi) (if_neg fun ⟨h1, h2⟩ => by have := hni h1; omega)


/-! ## M2c: the `genFun` product identities -/

lemma tsum_zero_factor (p : ℕ) : ((1 : PowerSeries ℂ) + ∑' c, ((0 : ℂ)) • X ^ (p * (c + 1))) = 1 := by simp

lemma tsum_single_factor (w : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    ((1 : PowerSeries ℂ) + ∑' c, (if c + 1 = j then w else 0) • X ^ (1 * (c + 1))) = 1 + C w * X ^ j := by
  congr 1
  rw [tsum_eq_single (j - 1) (fun c hc => by rw [if_neg (by omega), zero_smul])]
  rw [if_pos (by omega), one_mul, show j - 1 + 1 = j by omega, smul_eq_C_mul]

/-- the "`(q²;q)_{j−1}`-then-`(zq^{j+1};q)`" factor sequence: `d_i` for the part `i+1`. -/
noncomputable def dSeq (z : ℂ) (j i : ℕ) : PowerSeries ℂ :=
  if i = 0 then 1 else if i + 1 ≤ j then 1 - X ^ (i + 1) else 1 - C z * X ^ (i + 1)

lemma dSeq_prod_le (z : ℂ) (j : ℕ) : ∀ N, N ≤ j → ∏ i ∈ range N, dSeq z j i = poch 1 2 (N - 1) := by
  intro N hN
  induction N with
  | zero => simp [poch_zero]
  | succ N ih =>
      rw [prod_range_succ, ih (by omega), dSeq]
      rcases Nat.eq_zero_or_pos N with rfl | hN0
      · simp [poch_zero]
      · rw [if_neg (by omega), if_pos (by omega), show N + 1 - 1 = (N - 1) + 1 by omega, poch_succ,
          map_one, one_mul, show 2 + (N - 1) = N + 1 by omega]

lemma dSeq_prod_ge (z : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    ∀ N, j ≤ N → ∏ i ∈ range N, dSeq z j i = poch 1 2 (j - 1) * poch z (j + 1) (N - j) := by
  intro N hN
  induction N, hN using Nat.le_induction with
  | base => rw [dSeq_prod_le z j j le_rfl, Nat.sub_self, poch_zero, mul_one]
  | succ N hjN ih =>
      rw [prod_range_succ, ih, dSeq, if_neg (by omega), if_neg (by omega), show N + 1 - j = (N - j) + 1 by omega,
        poch_succ, show j + 1 + (N - j) = N + 1 by omega]
      ring

lemma hasProd_dSeq (z : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    HasProd (dSeq z j) (poch 1 2 (j - 1) * pochInf z (j + 1)) := by
  have he : ∀ i, dSeq z j i = 1 + (dSeq z j i - 1) := fun i => by ring
  rw [show dSeq z j = fun i => 1 + (dSeq z j i - 1) from funext he]
  refine hasProd_of_stable _ (fun i => ?_) _ (fun k => ⟨j + k + 1, fun N hN => ?_⟩)
  · simp only [dSeq]
    split_ifs
    · simp
    · exact ⟨-1, by ring⟩
    · exact ⟨-C z, by ring⟩
  · simp only [add_sub_cancel]
    rw [dSeq_prod_ge z hj N (by omega)]
    have hd : (X : PowerSeries ℂ) ^ (k + 1) ∣ pochInf z (j + 1) - poch z (j + 1) (N - j) :=
      dvd_trans (pow_dvd_pow X (show k + 1 ≤ N - j by omega)) (X_pow_dvd_pochInf_sub z (show 1 ≤ j + 1 by omega) (N - j))
    exact coeff_eq_of_X_pow_dvd (K := k + 1)
      (by rw [← mul_sub, ← neg_sub, mul_neg]; exact dvd_neg.mpr (Dvd.dvd.mul_left hd _)) (by omega)

lemma isUnit_pochInf (c : ℂ) {s : ℕ} (hs : 1 ≤ s) : IsUnit (pochInf c s) := by
  rw [PowerSeries.isUnit_iff_constantCoeff, ← coeff_zero_eq_constantCoeff_apply, coeff_pochInf c hs le_rfl,
    coeff_zero_eq_constantCoeff_apply, constantCoeff_poch c hs]
  exact isUnit_one

lemma gW_factor (z : ℂ) (j i : ℕ) :
    ((1 : PowerSeries ℂ) + ∑' c, gW z j (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * dSeq z j i = 1 := by
  by_cases h0 : i = 0
  · subst h0; simp [gW, dSeq]
  · have h1 : i + 1 ≠ 1 := by omega
    by_cases hij : i + 1 ≤ j
    · have hw : ∀ c, gW z j (i + 1) (c + 1) = (1 : ℂ) ^ (c + 1) := fun c => by
        unfold gW; rw [if_neg h1, if_pos hij, one_pow]
      have hd : dSeq z j i = 1 - C 1 * X ^ (i + 1) := by
        unfold dSeq; rw [if_neg h0, if_pos hij, map_one, one_mul]
      simp only [hw, hd]
      exact geo_factor (1 : ℂ) (p := i + 1) (by omega)
    · have hw : ∀ c, gW z j (i + 1) (c + 1) = z ^ (c + 1) := fun c => by
        unfold gW; rw [if_neg h1, if_neg hij]
      have hd : dSeq z j i = 1 - C z * X ^ (i + 1) := by
        unfold dSeq; rw [if_neg h0, if_neg hij]
      simp only [hw, hd]
      exact geo_factor z (p := i + 1) (by omega)

/-- `genFun g_j · (q²;q)_{j−1}(zq^{j+1};q)_∞ = 1`. -/
theorem genFun_gW_mul (z : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    Nat.Partition.genFun (gW z j) * (poch 1 2 (j - 1) * pochInf z (j + 1)) = 1 := by
  have h := (Nat.Partition.hasProd_genFun (gW z j)).mul (hasProd_dSeq z hj)
  have h1 : (fun i => ((1 : PowerSeries ℂ) + ∑' c, gW z j (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * dSeq z j i)
      = fun _ => 1 := funext (gW_factor z j)
  rw [h1] at h
  exact h.unique hasProd_one

/-- `genFun f_j · (q²;q)_{j−1}(zq^{j+1};q)_∞ = 1 + z^{−j} q^j`. -/
theorem genFun_fW_mul (z : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    Nat.Partition.genFun (fW z j) * (poch 1 2 (j - 1) * pochInf z (j + 1)) = 1 + C (z⁻¹ ^ j) * X ^ j := by
  have h := (Nat.Partition.hasProd_genFun (fW z j)).mul (hasProd_dSeq z hj)
  have h1 : (fun i => ((1 : PowerSeries ℂ) + ∑' c, fW z j (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * dSeq z j i)
      = fun i => if i = 0 then 1 + C (z⁻¹ ^ j) * X ^ j else 1 := by
    funext i
    by_cases h0 : i = 0
    · subst h0
      simp only [if_true, dSeq, mul_one, fW, zero_add]
      exact tsum_single_factor _ hj
    · rw [if_neg h0]
      have hfg : ∀ c, fW z j (i + 1) (c + 1) = gW z j (i + 1) (c + 1) := fun c => by
        unfold fW gW; rw [if_neg (show i + 1 ≠ 1 by omega), if_neg (show i + 1 ≠ 1 by omega)]
      simp only [hfg]
      exact gW_factor z j i
  rw [h1] at h
  exact h.unique (hasProd_ite_eq 0 _)

/-- **the "exactly `j` ones" generating function** `genFun f_j − genFun g_j = z^{−j}q^j/((q²;q)_{j−1}(zq^{j+1};q)_∞)`. -/
theorem genFun_fW_sub_gW (z : ℂ) {j : ℕ} (hj : 1 ≤ j) :
    Nat.Partition.genFun (fW z j) - Nat.Partition.genFun (gW z j)
      = C (z⁻¹ ^ j) * X ^ j * Ring.inverse (poch 1 2 (j - 1) * pochInf z (j + 1)) := by
  have hu : IsUnit (poch 1 2 (j - 1) * pochInf z (j + 1)) :=
    (isUnit_poch 1 (by norm_num) _).mul (isUnit_pochInf z (by omega))
  set D := poch 1 2 (j - 1) * pochInf z (j + 1)
  have hf := genFun_fW_mul z hj
  have hg := genFun_gW_mul z hj
  calc Nat.Partition.genFun (fW z j) - Nat.Partition.genFun (gW z j)
      = (Nat.Partition.genFun (fW z j) * D - Nat.Partition.genFun (gW z j) * D) * Ring.inverse D := by
        rw [← sub_mul, mul_assoc, Ring.mul_inverse_cancel _ hu, mul_one]
    _ = C (z⁻¹ ^ j) * X ^ j * Ring.inverse D := by rw [hf, hg]; ring

/-- the "parts in `[2,m]`" factor sequence. -/
noncomputable def hSeq (m i : ℕ) : PowerSeries ℂ := if 2 ≤ i + 1 ∧ i + 1 ≤ m then 1 - X ^ (i + 1) else 1

lemma hSeq_prod (m : ℕ) : ∀ N, ∏ i ∈ range N, hSeq m i = poch 1 2 (min N m - 1) := by
  intro N
  induction N with
  | zero => simp [poch_zero]
  | succ N ih =>
      rw [prod_range_succ, ih, hSeq]
      by_cases h : 2 ≤ N + 1 ∧ N + 1 ≤ m
      · rw [if_pos h, show min (N + 1) m - 1 = (min N m - 1) + 1 by omega, poch_succ, map_one, one_mul,
          show 2 + (min N m - 1) = N + 1 by omega]
      · rw [if_neg h, mul_one]
        congr 1; omega

lemma hW_factor (m i : ℕ) :
    ((1 : PowerSeries ℂ) + ∑' c, hW m (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * hSeq m i = 1 := by
  by_cases h : 2 ≤ i + 1 ∧ i + 1 ≤ m
  · have hw : ∀ c, hW m (i + 1) (c + 1) = (1 : ℂ) ^ (c + 1) := fun c => by
      unfold hW; rw [if_pos h, one_pow]
    have hd : hSeq m i = 1 - C 1 * X ^ (i + 1) := by unfold hSeq; rw [if_pos h, map_one, one_mul]
    simp only [hw, hd]
    exact geo_factor (1 : ℂ) (p := i + 1) (by omega)
  · have hw : ∀ c, hW m (i + 1) (c + 1) = 0 := fun c => by unfold hW; rw [if_neg h]
    have hd : hSeq m i = 1 := by unfold hSeq; rw [if_neg h]
    simp only [hw, hd, zero_smul, tsum_zero, add_zero, mul_one]

/-- `genFun h_m · (q²;q)_{m−1} = 1`. -/
theorem genFun_hW_mul (m : ℕ) : Nat.Partition.genFun (hW m) * poch 1 2 (m - 1) = 1 := by
  have hp : HasProd (hSeq m) (poch 1 2 (m - 1)) := by
    have he : ∀ i, hSeq m i = 1 + (hSeq m i - 1) := fun i => by ring
    rw [show hSeq m = fun i => 1 + (hSeq m i - 1) from funext he]
    refine hasProd_of_stable _ (fun i => ?_) _ (fun k => ⟨m, fun N hN => ?_⟩)
    · simp only [hSeq]; split_ifs
      · exact ⟨-1, by ring⟩
      · simp
    · simp only [add_sub_cancel]
      rw [hSeq_prod m N, min_eq_right hN]
  have h := (Nat.Partition.hasProd_genFun (hW m)).mul hp
  rw [show (fun i => ((1 : PowerSeries ℂ) + ∑' c, hW m (i + 1) (c + 1) • X ^ ((i + 1) * (c + 1))) * hSeq m i)
      = fun _ => 1 from funext (hW_factor m)] at h
  exact h.unique hasProd_one


/-! ## M3a: partition facts -/

lemma card_parts_le (l : n.Partition) : l.parts.card ≤ n := by
  have h := Multiset.sum_map_le_sum_map (s := l.parts) (fun _ => (1 : ℕ)) (fun i => i) (fun i hi => l.parts_pos hi)
  simpa [l.parts_sum] using h

lemma ones_le (l : n.Partition) : ones l ≤ n :=
  le_trans (Multiset.count_le_card 1 l.parts) (card_parts_le l)

lemma largest_facts (l : n.Partition) (hn : 1 ≤ n) (h0 : ones l = 0) :
    2 ≤ largest l ∧ largest l ≤ n ∧ ∀ m, (∀ i ∈ l.parts, 2 ≤ i ∧ i ≤ m) ↔ largest l ≤ m := by
  have hne : l.parts ≠ 0 := by
    intro h; have := l.parts_sum; rw [h, Multiset.sum_zero] at this; omega
  have hne' : l.parts.toFinset.Nonempty := by
    obtain ⟨x, hx⟩ := Multiset.exists_mem_of_ne_zero hne; exact ⟨x, Multiset.mem_toFinset.mpr hx⟩
  obtain ⟨L, hLmem', hL⟩ := Finset.exists_mem_eq_sup l.parts.toFinset hne' id
  have hLmem : L ∈ l.parts := Multiset.mem_toFinset.mp hLmem'
  have h1 : (1 : ℕ) ∉ l.parts := by rw [← Multiset.count_eq_zero]; exact h0
  have hge2 : ∀ i ∈ l.parts, 2 ≤ i := fun i hi => by
    have := l.parts_pos hi; have : i ≠ 1 := fun h => h1 (h ▸ hi); omega
  have hsup : largest l = L := by rw [largest, hL]; rfl
  refine ⟨hsup ▸ hge2 L hLmem, hsup ▸ (l.parts_sum ▸ Multiset.le_sum_of_mem hLmem), fun m => ?_⟩
  constructor
  · intro h; rw [hsup]; exact (h L hLmem).2
  · intro h i hi
    have hle : i ≤ L := by
      have := Finset.le_sup (f := id) (Multiset.mem_toFinset.mpr hi); rw [hL] at this; exact this
    exact ⟨hge2 i hi, le_trans hle (hsup ▸ h)⟩

lemma one_mem_of_ones (l : n.Partition) (h : ones l ≠ 0) : (1 : ℕ) ∈ l.parts := by
  rw [← Multiset.count_ne_zero]; exact h

/-! ## M3b: the per-partition decomposition of `z^{crank}` -/

lemma zpow_crank_decomp {z : ℂ} (hz : z ≠ 0) (hn : 1 ≤ n) (l : n.Partition) :
    z ^ crank l = (∑ m ∈ Icc 2 n, z ^ m * (l.parts.toFinsupp.prod (hW m) - l.parts.toFinsupp.prod (hW (m - 1))))
      + ∑ j ∈ Icc 1 n, (l.parts.toFinsupp.prod (fW z j) - l.parts.toFinsupp.prod (gW z j)) := by
  rw [Finset.sum_congr rfl fun j hj => weight_fg z (Finset.mem_Icc.mp hj).1 l]
  simp only [weight_h]
  by_cases h0 : ones l = 0
  · obtain ⟨hL2, hLn, hiff⟩ := largest_facts l hn h0
    rw [Finset.sum_eq_zero (fun j hj => if_neg (by have := (Finset.mem_Icc.mp hj).1; omega)), add_zero]
    simp only [hiff]
    rw [Finset.sum_eq_single (largest l)]
    · rw [if_pos le_rfl, if_neg (by omega), sub_zero, mul_one, crank, if_pos h0, zpow_natCast]
    · intro m hm hne
      have hm2 := Finset.mem_Icc.mp hm
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · rw [if_neg (by omega), if_neg (by omega), sub_self, mul_zero]
      · rw [if_pos (by omega), if_pos (by omega), sub_self, mul_zero]
    · intro hnot; exact absurd (Finset.mem_Icc.mpr ⟨hL2, hLn⟩) hnot
  · have h1 := one_mem_of_ones l h0
    have hno : ∀ m, ¬ (∀ i ∈ l.parts, 2 ≤ i ∧ i ≤ m) := fun m h => by have := (h 1 h1).1; omega
    simp only [hno, if_false, sub_self, mul_zero, Finset.sum_const_zero, zero_add]
    rw [Finset.sum_eq_single (ones l)]
    · rw [if_pos rfl, crank, if_neg h0, zpow_sub₀ hz, zpow_natCast, zpow_natCast, div_eq_mul_inv, inv_pow]
    · intro j _ hne; rw [if_neg (Ne.symm hne)]
    · intro hnot; exact absurd (Finset.mem_Icc.mpr ⟨by omega, ones_le l⟩) hnot

/-- **the combinatorial crank sum**, split into `genFun` coefficients. -/
theorem crank_sum_eq {z : ℂ} (hz : z ≠ 0) (hn : 1 ≤ n) :
    ∑ l : n.Partition, z ^ crank l
      = (∑ m ∈ Icc 2 n, z ^ m * (coeff n (Nat.Partition.genFun (hW m)) - coeff n (Nat.Partition.genFun (hW (m - 1)))))
        + ∑ j ∈ Icc 1 n, coeff n (Nat.Partition.genFun (fW z j) - Nat.Partition.genFun (gW z j)) := by
  simp only [Nat.Partition.coeff_genFun, map_sub, ← Finset.sum_sub_distrib, Finset.mul_sum]
  rw [Finset.sum_comm (s := Icc 2 n), Finset.sum_comm (s := Icc 1 n), ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun l _ => zpow_crank_decomp hz hn l

/-! ## M3c: the analytic side -/

lemma inverse_mul' {A B : PowerSeries ℂ} (_hA : IsUnit A) (_hB : IsUnit B) :
    Ring.inverse (A * B) = Ring.inverse A * Ring.inverse B := by
  rw [Ring.mul_inverse_rev' (Commute.all _ _), mul_comm]

lemma pochInf_split (c : ℂ) {s : ℕ} (hs : 1 ≤ s) (j : ℕ) : pochInf c s = poch c s j * pochInf c (s + j) := by
  ext k
  have hsplit : ∀ N, poch c s (j + N) = poch c s j * poch c (s + j) N := by
    intro N; rw [poch, poch, poch, prod_range_add]; congr 1
    exact prod_congr rfl fun i _ => by rw [add_assoc]
  rw [coeff_pochInf c hs (show k + 1 ≤ j + (k + 1) by omega), hsplit]
  exact coeff_eq_of_X_pow_dvd (K := k + 1)
    (by rw [← mul_sub, ← neg_sub, mul_neg]; exact dvd_neg.mpr (Dvd.dvd.mul_left
      (X_pow_dvd_pochInf_sub c (by omega) (k + 1)) _)) (by omega)

lemma poch_one_split (m : ℕ) (hm : 1 ≤ m) : poch 1 1 m = (1 - X) * poch 1 2 (m - 1) := by
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  induction m' with
  | zero => simp [poch, map_one]
  | succ m' ih =>
      rw [poch_succ, ih (by omega), show m' + 1 + 1 - 1 = m' + 1 by omega, show m' + 1 - 1 = m' by omega,
        poch_succ, show 1 + (m' + 1) = 2 + m' by ring]
      ring

lemma isUnit_one_sub_X : IsUnit (1 - X : PowerSeries ℂ) := by
  rw [PowerSeries.isUnit_iff_constantCoeff]; simp

/-- `(1−q)·Euler(z)·G(q/z) = (q;q)_∞/((zq;q)_∞(q/z;q)_∞)`. -/
lemma crankGF_eq {z : ℂ} (hz : z ≠ 0) :
    pochInf 1 1 * Ring.inverse (pochInf z 1 * pochInf z⁻¹ 1) = (1 - X) * Gser 0 z 1 * Gser z z⁻¹ 1 := by
  have hu1 := isUnit_pochInf z (le_refl 1)
  have hu2 := isUnit_pochInf z⁻¹ (le_refl 1)
  have hu3 := isUnit_pochInf 1 (show 1 ≤ 2 by norm_num)
  have he : Gser 0 z 1 = Ring.inverse (pochInf z 1) := by
    have := euler_identity z
    calc Gser 0 z 1 = Gser 0 z 1 * (pochInf z 1 * Ring.inverse (pochInf z 1)) := by
          rw [Ring.mul_inverse_cancel _ hu1, mul_one]
      _ = Ring.inverse (pochInf z 1) := by rw [← mul_assoc, this, one_mul]
  have hq : Ring.inverse (pochInf z⁻¹ 1) = Gser z z⁻¹ 1 * Ring.inverse (pochInf 1 2) := by
    have := qbinomial_special hz
    calc Ring.inverse (pochInf z⁻¹ 1)
        = (Gser z z⁻¹ 1 * pochInf z⁻¹ 1) * Ring.inverse (pochInf 1 2) * Ring.inverse (pochInf z⁻¹ 1) := by
          rw [this, Ring.mul_inverse_cancel _ hu3, one_mul]
      _ = Gser z z⁻¹ 1 * Ring.inverse (pochInf 1 2) * (pochInf z⁻¹ 1 * Ring.inverse (pochInf z⁻¹ 1)) := by ring
      _ = Gser z z⁻¹ 1 * Ring.inverse (pochInf 1 2) := by rw [Ring.mul_inverse_cancel _ hu2, mul_one]
  have h11 : pochInf 1 1 = (1 - X) * pochInf 1 2 := by
    rw [pochInf_split 1 (le_refl 1) 1, poch, prod_range_one, map_one, one_mul, pow_one]
  rw [inverse_mul' hu1 hu2, hq, ← he, h11]
  calc (1 - X) * pochInf 1 2 * (Gser 0 z 1 * (Gser z z⁻¹ 1 * Ring.inverse (pochInf 1 2)))
      = (1 - X) * Gser 0 z 1 * Gser z z⁻¹ 1 * (pochInf 1 2 * Ring.inverse (pochInf 1 2)) := by ring
    _ = _ := by rw [Ring.mul_inverse_cancel _ hu3, mul_one]


/-! ## M3d: coefficient matching — the Andrews–Garvan theorem -/

lemma genFun_hW_eq (m : ℕ) : Nat.Partition.genFun (hW m) = Ring.inverse (poch 1 2 (m - 1)) := by
  have h := genFun_hW_mul m
  have hu := isUnit_poch 1 (show 1 ≤ 2 by norm_num) (m - 1)
  calc Nat.Partition.genFun (hW m)
      = Nat.Partition.genFun (hW m) * (poch 1 2 (m - 1) * Ring.inverse (poch 1 2 (m - 1))) := by
        rw [Ring.mul_inverse_cancel _ hu, mul_one]
    _ = Ring.inverse (poch 1 2 (m - 1)) := by rw [← mul_assoc, h, one_mul]

lemma sum_range_split2 (f : ℕ → ℂ) {n : ℕ} (hn : 1 ≤ n) :
    ∑ m ∈ range (n + 1), f m = f 0 + f 1 + ∑ m ∈ Icc 2 n, f m := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  rw [Finset.sum_range_succ', Finset.sum_range_succ', show Icc 2 (k + 1) = Ico 2 (k + 2) by
    ext x; simp only [Finset.mem_Icc, Finset.mem_Ico]; omega, Finset.sum_Ico_eq_sum_range,
    show k + 2 - 2 = k by omega]
  simp only [show ∀ x : ℕ, x + 1 + 1 = 2 + x from fun x => by ring]
  ring

lemma sum_range_split1 (f : ℕ → ℂ) (n : ℕ) : ∑ j ∈ range (n + 1), f j = f 0 + ∑ j ∈ Icc 1 n, f j := by
  rw [Finset.sum_range_succ', show Icc 1 n = Ico 1 (n + 1) by ext x; simp only [Finset.mem_Icc, Finset.mem_Ico]; omega,
    Finset.sum_Ico_eq_sum_range, show n + 1 - 1 = n by omega]
  simp only [show ∀ x : ℕ, x + 1 = 1 + x from fun x => by ring]
  ring

lemma coeff_trunc_Gser (α β : ℂ) (A : PowerSeries ℂ) (n : ℕ) :
    coeff n (A * Gser α β 1) = ∑ j ∈ range (n + 1), coeff n (A * (qcoef α j * yv β 1 ^ j)) := by
  rw [← map_sum, ← Finset.mul_sum, ← Spart]
  have hd : (X : PowerSeries ℂ) ^ (n + 1) ∣ A * Gser α β 1 - A * Spart α β 1 (n + 1) := by
    rw [← mul_sub]; exact Dvd.dvd.mul_left (X_pow_dvd_Gser_sub α β (le_refl 1) (n + 1)) _
  exact coeff_eq_of_X_pow_dvd hd (by omega)

/-- the `j ≥ 1` analytic terms are the "exactly `j` ones" generating functions. -/
lemma term_j_eq {z : ℂ} {j : ℕ} (hj : 1 ≤ j) :
    (1 - X) * Gser 0 z 1 * (qcoef z j * yv z⁻¹ 1 ^ j)
      = C (z⁻¹ ^ j) * X ^ j * Ring.inverse (poch 1 2 (j - 1) * pochInf z (j + 1)) := by
  have he : Gser 0 z 1 = Ring.inverse (pochInf z 1) := by
    have := euler_identity z
    calc Gser 0 z 1 = Gser 0 z 1 * (pochInf z 1 * Ring.inverse (pochInf z 1)) := by
          rw [Ring.mul_inverse_cancel _ (isUnit_pochInf z (le_refl 1)), mul_one]
      _ = Ring.inverse (pochInf z 1) := by rw [← mul_assoc, this, one_mul]
  have hz1 := isUnit_poch z (le_refl 1) j
  have hz2 := isUnit_pochInf z (show 1 ≤ j + 1 by omega)
  have hp2 := isUnit_poch 1 (show 1 ≤ 2 by norm_num) (j - 1)
  rw [he, pochInf_split z (le_refl 1) j, show 1 + j = j + 1 by ring, inverse_mul' hz1 hz2, qcoef,
    poch_one_split j hj, inverse_mul' isUnit_one_sub_X hp2, inverse_mul' hp2 hz2, yv, pow_one, mul_pow, ← map_pow]
  calc (1 - X) * (Ring.inverse (poch z 1 j) * Ring.inverse (pochInf z (j + 1)))
        * (poch z 1 j * (Ring.inverse (1 - X) * Ring.inverse (poch 1 2 (j - 1))) * (C (z⁻¹ ^ j) * X ^ j))
      = ((1 - X) * Ring.inverse (1 - X)) * (Ring.inverse (poch z 1 j) * poch z 1 j)
          * (C (z⁻¹ ^ j) * X ^ j * (Ring.inverse (poch 1 2 (j - 1)) * Ring.inverse (pochInf z (j + 1)))) := by ring
    _ = _ := by
        rw [Ring.mul_inverse_cancel _ isUnit_one_sub_X, Ring.inverse_mul_cancel _ hz1]; ring

/-- the `m ≥ 2` Euler terms are the "no ones, largest part `m`" generating functions. -/
lemma term_m_eq (z : ℂ) {m : ℕ} (hm : 2 ≤ m) :
    (1 - X) * (qcoef 0 m * yv z 1 ^ m)
      = C (z ^ m) * (Ring.inverse (poch 1 2 (m - 1)) - Ring.inverse (poch 1 2 (m - 1 - 1))) := by
  set A := poch 1 2 (m - 1 - 1)
  have hA : IsUnit A := isUnit_poch 1 (show 1 ≤ 2 by norm_num) _
  have hu : IsUnit (1 - X ^ m : PowerSeries ℂ) := by
    rw [PowerSeries.isUnit_iff_constantCoeff, map_sub, map_one, map_pow, constantCoeff_X, zero_pow (by omega),
      sub_zero]
    exact isUnit_one
  have hP2 : poch 1 2 (m - 1) = A * (1 - X ^ m) := by
    rw [show m - 1 = (m - 1 - 1) + 1 by omega, poch_succ, map_one, one_mul, show 2 + (m - 1 - 1) = m by omega]
  have hP1 : poch 1 1 m = (1 - X) * (A * (1 - X ^ m)) := by rw [poch_one_split m (by omega), hP2]
  have key : Ring.inverse (1 - X ^ m : PowerSeries ℂ) - 1 = X ^ m * Ring.inverse (1 - X ^ m) := by
    calc Ring.inverse (1 - X ^ m : PowerSeries ℂ) - 1
        = Ring.inverse (1 - X ^ m) - (1 - X ^ m) * Ring.inverse (1 - X ^ m) := by rw [Ring.mul_inverse_cancel _ hu]
      _ = X ^ m * Ring.inverse (1 - X ^ m) := by ring
  rw [qcoef, poch_zero_const, one_mul, hP1, hP2, inverse_mul' isUnit_one_sub_X (hA.mul hu), inverse_mul' hA hu,
    yv, pow_one, mul_pow, ← map_pow]
  calc (1 - X) * (Ring.inverse (1 - X) * (Ring.inverse A * Ring.inverse (1 - X ^ m)) * (C (z ^ m) * X ^ m))
      = ((1 - X) * Ring.inverse (1 - X)) * (C (z ^ m) * Ring.inverse A * (X ^ m * Ring.inverse (1 - X ^ m))) := by ring
    _ = C (z ^ m) * (Ring.inverse A * Ring.inverse (1 - X ^ m) - Ring.inverse A) := by
        rw [Ring.mul_inverse_cancel _ isUnit_one_sub_X, one_mul, ← key]; ring

/-- **The Andrews–Garvan crank theorem.** For `n ≥ 2` and `z ≠ 0`,
`Σ_{λ ⊢ n} z^{crank λ} = [qⁿ] (q;q)_∞ / ((zq;q)_∞ (z⁻¹q;q)_∞)`. -/
theorem crank_generating_function {z : ℂ} (hz : z ≠ 0) {n : ℕ} (hn : 2 ≤ n) :
    ∑ l : n.Partition, z ^ crank l = coeff n (pochInf 1 1 * Ring.inverse (pochInf z 1 * pochInf z⁻¹ 1)) := by
  rw [crank_sum_eq hz (by omega), crankGF_eq hz, coeff_trunc_Gser, sum_range_split1]
  -- the `j = 0` term is `(1−q)·Euler(z)`, expanded in its own sum
  have h0 : coeff n ((1 - X) * Gser 0 z 1 * (qcoef z 0 * yv z⁻¹ 1 ^ 0)) = coeff n ((1 - X) * Gser 0 z 1) := by
    simp [qcoef, poch_zero]
  rw [h0, coeff_trunc_Gser, sum_range_split2 _ (by omega : 1 ≤ n)]
  have hm0 : coeff n ((1 - X) * (qcoef 0 0 * yv z 1 ^ 0)) = 0 := by
    simp [qcoef, poch_zero, coeff_one, show n ≠ 0 by omega, show n ≠ 1 by omega, coeff_X]
  have hm1 : coeff n ((1 - X) * (qcoef 0 1 * yv z 1 ^ 1)) = 0 := by
    have hu : IsUnit (1 - X : PowerSeries ℂ) := isUnit_one_sub_X
    have hp : poch 0 1 1 * Ring.inverse (poch 1 1 1) = Ring.inverse (1 - X) := by
      rw [poch_zero_const, one_mul]; simp [poch]
    rw [qcoef, hp, pow_one, yv, pow_one, ← mul_assoc, Ring.mul_inverse_cancel _ hu, one_mul,
      coeff_C_mul, coeff_X, if_neg (by omega), mul_zero]
  rw [hm0, hm1, zero_add, zero_add]
  congr 1
  · refine Finset.sum_congr rfl fun m hm => ?_
    rw [term_m_eq z (Finset.mem_Icc.mp hm).1, coeff_C_mul, genFun_hW_eq, genFun_hW_eq, map_sub]
  · refine Finset.sum_congr rfl fun j hj => ?_
    rw [genFun_fW_sub_gW z (Finset.mem_Icc.mp hj).1, ← term_j_eq (Finset.mem_Icc.mp hj).1]


/-! ## M4a: evaluating the triangular JTP at a complex unit -/

/-- evaluation `z ↦ u` of Laurent polynomials into `ℂ`. -/
noncomputable def evU (u : ℂˣ) : LaurentPolynomial ℤ →+* ℂ := LaurentPolynomial.eval₂ (Int.castRingHom ℂ) u

lemma evU_T (u : ℂˣ) (m : ℤ) : evU u (LaurentPolynomial.T m) = (u : ℂ) ^ m := by
  simp [evU, LaurentPolynomial.eval₂_T]

lemma evU_C (u : ℂˣ) (a : ℤ) : evU u (LaurentPolynomial.C a) = (a : ℂ) := by
  simp [evU, LaurentPolynomial.eval₂_C]

lemma evU_invert (u : ℂˣ) (p : LaurentPolynomial ℤ) : evU u (MockTheta5.JTP.invertHom p) = evU u⁻¹ p := by
  have h : (evU u).toAddMonoidHom.comp MockTheta5.JTP.invertHom.toAddMonoidHom = (evU u⁻¹).toAddMonoidHom := by
    apply Finsupp.addHom_ext
    intro m b
    show evU u (MockTheta5.JTP.invertHom (Finsupp.single m b)) = evU u⁻¹ (Finsupp.single m b)
    rw [MockTheta5.JTP.invert_single,
      show (Finsupp.single (-m) b : LaurentPolynomial ℤ) = AddMonoidAlgebra.single (-m) b from rfl,
      show (Finsupp.single m b : LaurentPolynomial ℤ) = AddMonoidAlgebra.single m b from rfl,
      LaurentPolynomial.single_eq_C_mul_T, LaurentPolynomial.single_eq_C_mul_T,
      map_mul, map_mul, evU_C, evU_C, evU_T, evU_T, Units.val_inv_eq_inv_val, inv_zpow', zpow_neg]
  exact DFunLike.congr_fun h p

lemma map_evU_triProdQ1 (u : ℂˣ) (N : ℕ) :
    PowerSeries.map (evU u) (MockTheta5.JTP.triProdQ1 N) = poch (-(u : ℂ)) 1 N := by
  rw [MockTheta5.JTP.triProdQ1, map_prod, poch]
  refine Finset.prod_congr rfl fun i _ => ?_
  rw [map_add, map_one, map_mul, map_pow, PowerSeries.map_X, PowerSeries.map_C, evU_T, zpow_one, map_neg]
  ring_nf

lemma map_evU_triProdQ1Inf (u : ℂˣ) :
    PowerSeries.map (evU u) MockTheta5.JTP.triProdQ1Inf = pochInf (-(u : ℂ)) 1 := by
  ext k
  rw [PowerSeries.coeff_map, MockTheta5.JTP.coeff_triProdQ1Inf (le_refl (k + 1)), ← PowerSeries.coeff_map,
    map_evU_triProdQ1, pochInf, coeff_mk]

lemma map_evU_qfacInfL (u : ℂˣ) : PowerSeries.map (evU u) MockTheta5.JTP.qfacInfL = pochInf 1 1 := by
  ext k
  rw [MockTheta5.JTP.qfacInfL, PowerSeries.coeff_map, PowerSeries.coeff_map, evU_C,
    MockTheta5.JTP.coeff_qfacInf (le_refl (k + 1)), pochInf, coeff_mk]
  have : PowerSeries.map (Int.castRingHom ℂ) (MockTheta5.Bailey.qfac (k + 1)) = poch 1 1 (k + 1) := by
    rw [MockTheta5.Bailey.qfac, map_prod, poch]
    refine Finset.prod_congr rfl fun i _ => ?_
    rw [map_sub, map_one, map_pow, PowerSeries.map_X, map_one, one_mul, add_comm]
  rw [← this, PowerSeries.coeff_map]; rfl

/-- **the triangular JTP at `z = u`**: `(1+u)·(q;q)_∞·(−uq;q)_∞·(−u⁻¹q;q)_∞ = Σ_n uⁿ q^{n(n−1)/2}`. -/
theorem jtp_eval (u : ℂˣ) :
    (1 + (PowerSeries.C (u : ℂ))) * pochInf 1 1 * pochInf (-(u : ℂ)) 1 * pochInf (-((u⁻¹ : ℂˣ) : ℂ)) 1
      = PowerSeries.map (evU u) MockTheta5.JTP.triTheta := by
  rw [← MockTheta5.JTP.bilateral_triangular_JTP, MockTheta5.JTP.triProdQInf_eq_split, map_mul, map_mul, map_mul,
    map_add, map_one, PowerSeries.map_C, evU_T, zpow_one, map_evU_qfacInfL, map_evU_triProdQ1Inf]
  have hinv : PowerSeries.map (evU u) (PowerSeries.map MockTheta5.JTP.invertHom MockTheta5.JTP.triProdQ1Inf)
      = pochInf (-((u⁻¹ : ℂˣ) : ℂ)) 1 := by
    rw [← map_evU_triProdQ1Inf u⁻¹]
    ext k; rw [PowerSeries.coeff_map, PowerSeries.coeff_map, PowerSeries.coeff_map, evU_invert]
  rw [hinv]; ring

/-- coefficients of the evaluated theta series, as a box sum. -/
lemma coeff_map_evU_triTheta (u : ℂˣ) (i k : ℕ) (hik : i ≤ k) :
    coeff i (PowerSeries.map (evU u) MockTheta5.JTP.triTheta)
      = ∑ m ∈ Icc (-(k : ℤ)) ((k : ℤ) + 1), if MockTheta5.JTP.triE m = i then (u : ℂ) ^ m else 0 := by
  rw [PowerSeries.coeff_map, MockTheta5.JTP.coeff_triTheta_box i k hik, map_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  split_ifs <;> simp [evU_T]

/-! ## M4b: the product over the fifth roots of unity -/

open MockTheta5.JTP in
/-- `∏_{j<5} (ωʲq;q)_∞ = (q⁵;q⁵)_∞`. -/
theorem prod_pochInf_roots :
    pochInf 1 1 * pochInf ω5 1 * pochInf (ω5 ^ 2) 1 * pochInf (ω5 ^ 3) 1 * pochInf (ω5 ^ 4) 1
      = ψC (E5 qfacInf) := by
  have hw : (C ω5 : PowerSeries ℂ) ^ 4 + C ω5 ^ 3 + C ω5 ^ 2 + C ω5 + 1 = 0 := by
    rw [← map_pow, ← map_pow, ← map_pow, ← map_add, ← map_add, ← map_add, ← map_one C, ← map_add,
      phi_of_prim ω5_prim, map_zero]
  have hfin : ∀ N, poch 1 1 N * poch ω5 1 N * poch (ω5 ^ 2) 1 N * poch (ω5 ^ 3) 1 N * poch (ω5 ^ 4) 1 N
      = ψC (E5 (MockTheta5.Bailey.qfac N)) := by
    intro N
    rw [E5_qfac, map_prod, poch, poch, poch, poch, poch, ← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib,
      ← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun i _ => ?_
    have h := prod_one_sub_root (C ω5 : PowerSeries ℂ) (X ^ (1 + i)) hw
    rw [map_sub, map_one, map_pow, map_pow, map_pow, one_mul, show 5 * i + 5 = (1 + i) * 5 by ring, pow_mul,
      map_one, map_pow, map_pow, show ψC X = X from PowerSeries.map_X _, ← h]
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  set N := k + 1
  have hc : ∀ c : ℂ, (X : PowerSeries ℂ) ^ N ∣ pochInf c 1 - poch c 1 N := fun c => X_pow_dvd_pochInf_sub c le_rfl N
  have hL := MockTheta5.JTP.dvd_sub_mul' (MockTheta5.JTP.dvd_sub_mul' (MockTheta5.JTP.dvd_sub_mul'
    (MockTheta5.JTP.dvd_sub_mul' (hc 1) (hc ω5)) (hc (ω5 ^ 2))) (hc (ω5 ^ 3))) (hc (ω5 ^ 4))
  rw [hfin] at hL
  have hR : (X : PowerSeries ℂ) ^ N ∣ ψC (E5 qfacInf) - ψC (E5 (MockTheta5.Bailey.qfac N)) := by
    rw [← map_sub, ← map_sub]
    exact MockTheta5.JTP.X_pow_dvd_ψC (MockTheta5.JTP.X_pow_dvd_E5 (MockTheta5.JTP.X_pow_dvd_qfacInf_sub N N le_rfl))
  have := dvd_sub hL hR
  rw [show ∀ a b c : PowerSeries ℂ, a - c - (b - c) = a - b by intros; ring] at this
  exact coeff_eq_of_X_pow_dvd (g := 0) (by simpa using this) (by omega) |>.trans (map_zero _)

/-! ## M4c: the theta series at `u = −ζ²` vanishes in classes `2, 3, 4 (mod 5)` -/

open MockTheta5.JTP (ω5 ω5_prim ω5_pow5 ψC E5 qfacInf triE dis5 coeff_dis5 dis5_qfacInf_empty coeff_E5
  isUnit_qfacInf)

lemma ω5_ne : ω5 ≠ 0 := ω5_prim.ne_zero (by norm_num)

/-- `u = −ζ²`. -/
noncomputable def uZ : ℂ := -(ω5 ^ 2)

lemma uZ_ne : uZ ≠ 0 := neg_ne_zero.mpr (pow_ne_zero 2 ω5_ne)

noncomputable def uU : ℂˣ := Units.mk0 uZ uZ_ne

lemma uU_val : (uU : ℂ) = uZ := rfl

lemma uZ_pow5 : uZ ^ 5 = -1 := by
  rw [uZ, neg_pow, ← pow_mul, show 2 * 5 = 5 * 2 from rfl, pow_mul, ω5_pow5]; norm_num

lemma u_cancel {m : ℤ} (h : (5 : ℤ) ∣ m - 3) : uZ ^ m + uZ ^ (1 - m) = 0 := by
  obtain ⟨t, ht⟩ := h
  have h1 : uZ ^ (2 * m - 1) = -1 := by
    rw [show 2 * m - 1 = (5 : ℕ) * (2 * t + 1) by push_cast; omega, zpow_mul, zpow_natCast, uZ_pow5]
    exact Odd.neg_one_zpow ⟨t, by ring⟩
  have h2 : uZ ^ m = uZ ^ (1 - m) * uZ ^ (2 * m - 1) := by
    rw [← zpow_add₀ uZ_ne]; congr 1; ring
  rw [h2, h1]; ring

lemma tri5_ne2 : ∀ x : ZMod 5, x * (x - 1) ≠ 2 * 2 := by decide
lemma tri5_ne4 : ∀ x : ZMod 5, x * (x - 1) ≠ 2 * 4 := by decide
lemma tri5_eq3 : ∀ x : ZMod 5, x * (x - 1) = 2 * 3 → x = 3 := by decide

lemma triE_mod5 {m : ℤ} {n r : ℕ} (hr : r = 2 ∨ r = 3 ∨ r = 4) (h : triE m = 5 * n + r) :
    (5 : ℤ) ∣ m - 3 := by
  have h2 := MockTheta5.JTP.two_mul_triExp m
  rw [show (m * (m - 1) / 2).toNat = triE m from rfl, h] at h2
  have hc : (m : ZMod 5) * ((m : ZMod 5) - 1) = 2 * (r : ZMod 5) := by
    have := congrArg (Int.cast : ℤ → ZMod 5) h2
    push_cast at this
    rw [show (5 : ZMod 5) = 0 from rfl] at this
    rw [← this]; ring
  rcases hr with rfl | rfl | rfl
  · exact absurd (by simpa using hc) (tri5_ne2 _)
  · have hx := tri5_eq3 _ (by simpa using hc)
    have := (ZMod.intCast_zmod_eq_zero_iff_dvd (m - 3) 5).mp (by push_cast; rw [hx]; ring)
    exact_mod_cast this
  · exact absurd (by simpa using hc) (tri5_ne4 _)

/-- the evaluated theta series `Θ_u = Σ_m u^m q^{m(m−1)/2}` at `u = −ζ²`. -/
noncomputable abbrev ThU : PowerSeries ℂ := PowerSeries.map (evU uU) MockTheta5.JTP.triTheta

theorem theta_vanish (n r : ℕ) (hr : r = 2 ∨ r = 3 ∨ r = 4) : coeff (5 * n + r) ThU = 0 := by
  rw [coeff_map_evU_triTheta uU _ _ le_rfl]
  refine Finset.sum_involution (fun m _ => 1 - m) ?_ ?_ ?_ ?_
  · intro m _
    have hsym : triE (1 - m) = triE m := by
      unfold triE; rw [show (1 - m) * (1 - m - 1) = m * (m - 1) by ring]
    rw [hsym]
    split_ifs with h
    · rw [uU_val]; exact u_cancel (triE_mod5 hr h)
    · simp
  · intro m _ _; show 1 - m ≠ m; omega
  · intro m hm; simp only [Finset.mem_Icc] at hm ⊢; omega
  · intro m _; ring

/-! ## M4d: assembling `F(ζ) = (1−ζ²)⁻¹ · E·Θ · 1/E(q⁵)` -/

lemma map_inverse {R S : Type*} [CommRing R] [CommRing S] (φ : R →+* S) {u : R} (hu : IsUnit u) :
    φ (Ring.inverse u) = Ring.inverse (φ u) := by
  have h1 : φ u * φ (Ring.inverse u) = 1 := by rw [← map_mul, Ring.mul_inverse_cancel u hu, map_one]
  calc φ (Ring.inverse u) = (Ring.inverse (φ u) * φ u) * φ (Ring.inverse u) := by
        rw [Ring.inverse_mul_cancel _ (hu.map φ), one_mul]
    _ = Ring.inverse (φ u) * (φ u * φ (Ring.inverse u)) := by ring
    _ = Ring.inverse (φ u) := by rw [h1, mul_one]

lemma pochInf_one_eq : pochInf 1 1 = ψC qfacInf := by
  rw [← map_evU_qfacInfL uU, MockTheta5.JTP.qfacInfL]
  ext k; simp [PowerSeries.coeff_map]

lemma ω5_inv : ω5⁻¹ = ω5 ^ 4 :=
  inv_eq_of_mul_eq_one_right (by rw [← pow_succ', ω5_pow5])

lemma neg_uZ : -uZ = ω5 ^ 2 := by rw [uZ, neg_neg]

lemma neg_uU_inv : -((uU⁻¹ : ℂˣ) : ℂ) = ω5 ^ 3 := by
  rw [Units.val_inv_eq_inv_val, uU_val, uZ, inv_neg, neg_neg]
  exact inv_eq_of_mul_eq_one_right (by rw [← pow_add, ω5_pow5])

lemma one_sub_ω5sq_ne : (1 : ℂ) - ω5 ^ 2 ≠ 0 :=
  sub_ne_zero.mpr (Ne.symm (ω5_prim.pow_ne_one_of_pos_of_lt (by norm_num) (by norm_num)))

theorem crankF_root :
    pochInf 1 1 * Ring.inverse (pochInf ω5 1 * pochInf ω5⁻¹ 1)
      = C (1 - ω5 ^ 2)⁻¹ * (ψC qfacInf * ThU * ψC (E5 (Ring.inverse qfacInf))) := by
  have hJ := jtp_eval uU
  rw [neg_uU_inv, uU_val, neg_uZ, show (1 : PowerSeries ℂ) + C uZ = C (1 - ω5 ^ 2) by
    rw [uZ, map_sub, map_one, map_neg, sub_eq_add_neg]] at hJ
  have hW := prod_pochInf_roots
  rw [map_inverse E5 isUnit_qfacInf, map_inverse ψC (isUnit_qfacInf.map E5), ← hW, show ThU = _ from hJ.symm, ← pochInf_one_eq, ω5_inv]
  set A := pochInf 1 1
  set B := pochInf ω5 1 * pochInf (ω5 ^ 4) 1
  set D := pochInf (ω5 ^ 2) 1 * pochInf (ω5 ^ 3) 1
  have hAD : IsUnit (A * D) :=
    (isUnit_pochInf _ le_rfl).mul ((isUnit_pochInf _ le_rfl).mul (isUnit_pochInf _ le_rfl))
  have hW' : A * pochInf ω5 1 * pochInf (ω5 ^ 2) 1 * pochInf (ω5 ^ 3) 1 * pochInf (ω5 ^ 4) 1 = B * (A * D) := by
    simp only [A, B, D]; ring
  rw [hW', Ring.mul_inverse_rev B (A * D)]
  have hc : C (1 - ω5 ^ 2)⁻¹ * C (1 - ω5 ^ 2) = (1 : PowerSeries ℂ) := by
    rw [← map_mul, inv_mul_cancel₀ one_sub_ω5sq_ne, map_one]
  have hADi : (A * D) * Ring.inverse (A * D) = 1 := Ring.mul_inverse_cancel _ hAD
  calc A * Ring.inverse B = A * Ring.inverse B * ((C (1 - ω5 ^ 2)⁻¹ * C (1 - ω5 ^ 2)) * ((A * D) * Ring.inverse (A * D))) := by
        rw [hc, hADi, one_mul, mul_one]
    _ = _ := by simp only [A, D]; ring

theorem crank_sum_root_zero (n : ℕ) : ∑ l : (5 * n + 4).Partition, ω5 ^ crank l = 0 := by
  rw [crank_generating_function ω5_ne (by omega), crankF_root, coeff_C_mul]
  refine mul_eq_zero_of_right _ ?_
  rw [coeff_mul]
  refine Finset.sum_eq_zero fun ⟨a, b⟩ hab => ?_
  rw [Finset.mem_antidiagonal] at hab
  by_cases h5 : 5 ∣ b
  · refine mul_eq_zero_of_left ?_ _
    rw [coeff_mul]
    refine Finset.sum_eq_zero fun ⟨c, d⟩ hcd => ?_
    rw [Finset.mem_antidiagonal] at hcd
    rcases (by omega : c % 5 = 3 ∨ c % 5 = 4 ∨ d % 5 = 2 ∨ d % 5 = 3 ∨ d % 5 = 4) with h | h | h | h | h
    · refine mul_eq_zero_of_left ?_ _
      have := congrArg (coeff (c / 5)) (dis5_qfacInf_empty 3 (Or.inl rfl))
      rw [coeff_dis5, map_zero, show 5 * (c / 5) + 3 = c by omega] at this
      simp [PowerSeries.coeff_map, this]
    · refine mul_eq_zero_of_left ?_ _
      have := congrArg (coeff (c / 5)) (dis5_qfacInf_empty 4 (Or.inr rfl))
      rw [coeff_dis5, map_zero, show 5 * (c / 5) + 4 = c by omega] at this
      simp [PowerSeries.coeff_map, this]
    all_goals
      refine mul_eq_zero_of_right _ ?_
      rw [show d = 5 * (d / 5) + d % 5 by omega, h]
      exact theta_vanish _ _ (by omega)
  · refine mul_eq_zero_of_right _ ?_
    simp [PowerSeries.coeff_map, coeff_E5, h5]

/-! ## M4e: the cyclotomic step and equidistribution -/

lemma cyc_equal (N0 N1 N2 N3 N4 : ℕ)
    (h : (N0 : ℂ) + N1 * ω5 + N2 * ω5 ^ 2 + N3 * ω5 ^ 3 + N4 * ω5 ^ 4 = 0) :
    N0 = N4 ∧ N1 = N4 ∧ N2 = N4 ∧ N3 = N4 := by
  set q : Polynomial ℚ := Polynomial.C ((N0 : ℚ) - N4) + Polynomial.C ((N1 : ℚ) - N4) * Polynomial.X
    + Polynomial.C ((N2 : ℚ) - N4) * Polynomial.X ^ 2 + Polynomial.C ((N3 : ℚ) - N4) * Polynomial.X ^ 3 with hq
  have hphi := MockTheta5.JTP.phi_of_prim ω5_prim
  have hev : Polynomial.aeval ω5 q = 0 := by
    simp only [hq, map_add, map_mul, map_pow, Polynomial.aeval_C, Polynomial.aeval_X, eq_ratCast]
    push_cast
    linear_combination h - (N4 : ℂ) * hphi
  have hdvd : minpoly ℚ ω5 ∣ q := minpoly.dvd ℚ ω5 hev
  rw [← Polynomial.cyclotomic_eq_minpoly_rat ω5_prim (by norm_num)] at hdvd
  have hdeg : q.degree < (Polynomial.cyclotomic 5 ℚ).degree := by
    rw [Polynomial.degree_cyclotomic, Nat.totient_prime Nat.prime_five]
    refine lt_of_le_of_lt (b := ((3 : ℕ) : WithBot ℕ)) ?_ (by decide)
    rw [hq]; compute_degree
    all_goals norm_num
  have hq0 := Polynomial.eq_zero_of_dvd_of_degree_lt hdvd hdeg
  have c0 := congrArg (Polynomial.coeff · 0) hq0
  have c1 := congrArg (Polynomial.coeff · 1) hq0
  have c2 := congrArg (Polynomial.coeff · 2) hq0
  have c3 := congrArg (Polynomial.coeff · 3) hq0
  simp only [hq, Polynomial.coeff_add, Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, Polynomial.coeff_X,
    Polynomial.coeff_C, Polynomial.coeff_zero] at c0 c1 c2 c3
  norm_num [sub_eq_zero] at c0 c1 c2 c3
  exact ⟨by exact_mod_cast c0, by exact_mod_cast c1, by exact_mod_cast c2, by exact_mod_cast c3⟩

/-- the number of partitions of `n` with crank `≡ k (mod 5)`. -/
noncomputable def crankCount (n k : ℕ) : ℕ := (univ.filter fun l : n.Partition => (crank l % 5).toNat = k).card

lemma ω5_zpow_crank (c : ℤ) : ω5 ^ c = ω5 ^ (c % 5).toNat := by
  rw [← zpow_natCast, Int.toNat_of_nonneg (Int.emod_nonneg c (by norm_num))]
  conv_lhs => rw [← Int.mul_ediv_add_emod c 5]
  rw [zpow_add₀ ω5_ne, zpow_mul, show ((5 : ℤ)) = ((5 : ℕ) : ℤ) from rfl, zpow_natCast, ω5_pow5, one_zpow, one_mul]

lemma crank_sum_grouped (n : ℕ) :
    ∑ l : n.Partition, ω5 ^ crank l = ∑ k ∈ range 5, (crankCount n k : ℂ) * ω5 ^ k := by
  simp_rw [ω5_zpow_crank]
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range 5) (g := fun l : n.Partition => (crank l % 5).toNat)
    (fun l _ => by simp only [Finset.mem_range]; omega)]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_congr rfl (g := fun _ => ω5 ^ k) (fun l hl => by rw [(Finset.mem_filter.mp hl).2]),
    Finset.sum_const, nsmul_eq_mul, crankCount]

lemma crankCount_sum (n : ℕ) : ∑ k ∈ range 5, crankCount n k = Fintype.card n.Partition := by
  rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun l : n.Partition => (crank l % 5).toNat)
    (t := range 5) (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (crank l % 5).toNat < 5 by omega)))]
  rfl

/-- **Andrews–Garvan (mod 5).** The crank splits the partitions of `5n+4` into five equal classes. -/
theorem crank_equidistribution_mod5 (n : ℕ) {i : ℕ} (hi : i < 5) :
    5 * (univ.filter fun l : (5 * n + 4).Partition => crank l % 5 = i).card
      = Fintype.card (5 * n + 4).Partition := by
  have h := crank_sum_root_zero n
  rw [crank_sum_grouped] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, mul_one, pow_one] at h
  obtain ⟨e0, e1, e2, e3⟩ := cyc_equal _ _ _ _ _ h
  have hs := crankCount_sum (5 * n + 4)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  have hfilt : (univ.filter fun l : (5 * n + 4).Partition => crank l % 5 = i) =
      univ.filter fun l => (crank l % 5).toNat = i :=
    Finset.filter_congr fun l _ => by omega
  rw [hfilt, ← crankCount]
  interval_cases i <;> omega

/-! ## M5a: generic `p`-th-root machinery -/

lemma prod_one_sub_root_gen {ζ : ℂ} {p : ℕ} (hζ : IsPrimitiveRoot ζ p) (hp : 0 < p) (y : PowerSeries ℂ) :
    ∏ i ∈ range p, (1 - C (ζ ^ i) * y) = 1 - y ^ p := by
  have h := _root_.X_pow_sub_C_eq_prod (hζ.map_of_injective (PowerSeries.C_injective (R := ℂ))) hp
    (rfl : y ^ p = y ^ p)
  have h1 := congrArg (Polynomial.eval (1 : PowerSeries ℂ)) h
  simp only [Polynomial.eval_sub, Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_C, one_pow,
    Polynomial.eval_prod] at h1
  simp_rw [map_pow]; exact h1.symm

/-- `f(q) ↦ f(q^p)` on `ℤ⟦X⟧`. -/
noncomputable def Ep (p : ℕ) (hp : p ≠ 0) : PowerSeries ℤ →+* PowerSeries ℤ := (PowerSeries.expand p hp).toRingHom

lemma Ep_X (p : ℕ) (hp : p ≠ 0) : Ep p hp X = X ^ p := PowerSeries.expand_X p hp

lemma coeff_Ep_of_not_dvd {p : ℕ} (hp : p ≠ 0) (f : PowerSeries ℤ) {m : ℕ} (h : ¬ p ∣ m) :
    coeff m (Ep p hp f) = 0 := PowerSeries.coeff_expand_of_not_dvd p hp f h

lemma X_pow_dvd_Ep {p : ℕ} (hp : p ≠ 0) {K : ℕ} {f : PowerSeries ℤ} (h : X ^ K ∣ f) :
    (X : PowerSeries ℤ) ^ K ∣ Ep p hp f := by
  obtain ⟨d, rfl⟩ := h
  rw [map_mul, map_pow, Ep_X, ← pow_mul]
  exact dvd_mul_of_dvd_left (pow_dvd_pow X (Nat.le_mul_of_pos_left K (Nat.pos_of_ne_zero hp))) _

lemma dvd_sub_prod_gen {K : ℕ} (s : Finset ℕ) (F G : ℕ → PowerSeries ℂ)
    (h : ∀ j, (X : PowerSeries ℂ) ^ K ∣ F j - G j) :
    (X : PowerSeries ℂ) ^ K ∣ ∏ j ∈ s, F j - ∏ j ∈ s, G j := by
  refine Finset.induction_on s (by simp) ?_
  intro a s ha ih
  rw [prod_insert ha, prod_insert ha]
  exact MockTheta5.JTP.dvd_sub_mul' (h a) ih

lemma ψC_Ep_qfac {p : ℕ} (hp : p ≠ 0) (N : ℕ) :
    ψC (Ep p hp (MockTheta5.Bailey.qfac N)) = ∏ i ∈ range N, (1 - (X ^ (1 + i)) ^ p) := by
  rw [MockTheta5.Bailey.qfac, map_prod, map_prod]
  refine Finset.prod_congr rfl fun i _ => ?_
  rw [map_sub, map_one, map_pow, Ep_X, map_sub, map_one, map_pow, map_pow, show ψC X = X from PowerSeries.map_X _,
    ← pow_mul, ← pow_mul, mul_comm, add_comm]

/-- `∏_{j<p} (ζʲq;q)_∞ = (q^p;q^p)_∞`. -/
theorem prod_pochInf_roots_gen {ζ : ℂ} {p : ℕ} (hζ : IsPrimitiveRoot ζ p) (hp : p ≠ 0) :
    ∏ j ∈ range p, pochInf (ζ ^ j) 1 = ψC (Ep p hp qfacInf) := by
  have hfin : ∀ N, ∏ j ∈ range p, poch (ζ ^ j) 1 N = ψC (Ep p hp (MockTheta5.Bailey.qfac N)) := by
    intro N
    simp only [poch]
    rw [Finset.prod_comm, ψC_Ep_qfac]
    exact Finset.prod_congr rfl fun i _ => prod_one_sub_root_gen hζ (Nat.pos_of_ne_zero hp) _
  rw [← sub_eq_zero]
  ext k
  rw [map_zero]
  have hL := dvd_sub_prod_gen (K := k + 1) (range p) (fun j => pochInf (ζ ^ j) 1) (fun j => poch (ζ ^ j) 1 (k + 1))
    (fun j => X_pow_dvd_pochInf_sub _ le_rfl (k + 1))
  rw [hfin] at hL
  have hR : (X : PowerSeries ℂ) ^ (k + 1) ∣ ψC (Ep p hp qfacInf) - ψC (Ep p hp (MockTheta5.Bailey.qfac (k + 1))) := by
    rw [← map_sub, ← map_sub]
    exact MockTheta5.JTP.X_pow_dvd_ψC (X_pow_dvd_Ep hp (MockTheta5.JTP.X_pow_dvd_qfacInf_sub _ _ le_rfl))
  have := dvd_sub hL hR
  rw [show ∀ a b c : PowerSeries ℂ, a - c - (b - c) = a - b by intros; ring] at this
  exact coeff_eq_of_X_pow_dvd (g := 0) (by simpa using this) (by omega) |>.trans (map_zero _)

lemma u_cancel_gen {u : ℂ} (hu0 : u ≠ 0) {p : ℕ} (hu : u ^ p = -1) {m : ℤ}
    (h : (p : ℤ) ∣ 2 * m - 1) : u ^ m + u ^ (1 - m) = 0 := by
  obtain ⟨t, ht⟩ := h
  have hto : Odd t := by
    have : Odd ((p : ℤ) * t) := by rw [← ht]; exact ⟨m - 1, by ring⟩
    exact (Int.odd_mul.mp this).2
  have h1 : u ^ (2 * m - 1) = -1 := by
    rw [ht, zpow_mul, zpow_natCast, hu]; exact hto.neg_one_zpow
  have h2 : u ^ m = u ^ (1 - m) * u ^ (2 * m - 1) := by
    rw [← zpow_add₀ hu0]; congr 1; ring
  rw [h2, h1]; ring

/-- **theta vanishing**: if `u^p = −1` and every `m` with `m(m−1)/2 = b` has `p ∣ 2m−1`, the `q^b`
coefficient of `Σ_m u^m q^{m(m−1)/2}` vanishes (the terms `m`, `1−m` cancel). -/
theorem theta_vanish_gen (u : ℂˣ) {p : ℕ} (hu : (u : ℂ) ^ p = -1) (b : ℕ)
    (hb : ∀ m : ℤ, triE m = b → (p : ℤ) ∣ 2 * m - 1) :
    coeff b (PowerSeries.map (evU u) MockTheta5.JTP.triTheta) = 0 := by
  rw [coeff_map_evU_triTheta u _ _ le_rfl]
  refine Finset.sum_involution (fun m _ => 1 - m) ?_ ?_ ?_ ?_
  · intro m _
    have hsym : triE (1 - m) = triE m := by
      unfold triE; rw [show (1 - m) * (1 - m - 1) = m * (m - 1) by ring]
    rw [hsym]
    split_ifs with h
    · exact u_cancel_gen u.ne_zero hu (hb m h)
    · simp
  · intro m _ _; show 1 - m ≠ m; omega
  · intro m hm; simp only [Finset.mem_Icc] at hm ⊢; omega
  · intro m _; ring

/-! ## M5b: mod 7 -/

noncomputable def ω7 : ℂ := Complex.exp (2 * ↑Real.pi * Complex.I / ((7 : ℕ) : ℂ))

lemma ω7_prim : IsPrimitiveRoot ω7 7 := Complex.isPrimitiveRoot_exp 7 (by norm_num)
lemma ω7_pow7 : ω7 ^ 7 = 1 := ω7_prim.pow_eq_one
lemma ω7_ne : ω7 ≠ 0 := ω7_prim.ne_zero (by norm_num)
lemma ω7_inv : ω7⁻¹ = ω7 ^ 6 := inv_eq_of_mul_eq_one_right (by rw [← pow_succ', ω7_pow7])

noncomputable def uA : ℂˣ := Units.mk0 (-(ω7 ^ 2)) (neg_ne_zero.mpr (pow_ne_zero 2 ω7_ne))
noncomputable def uB : ℂˣ := Units.mk0 (-(ω7 ^ 3)) (neg_ne_zero.mpr (pow_ne_zero 3 ω7_ne))

lemma uA_pow7 : (uA : ℂ) ^ 7 = -1 := by
  show (-(ω7 ^ 2)) ^ 7 = -1
  rw [neg_pow, ← pow_mul, show 2 * 7 = 7 * 2 from rfl, pow_mul, ω7_pow7]; norm_num

lemma uB_pow7 : (uB : ℂ) ^ 7 = -1 := by
  show (-(ω7 ^ 3)) ^ 7 = -1
  rw [neg_pow, ← pow_mul, show 3 * 7 = 7 * 3 from rfl, pow_mul, ω7_pow7]; norm_num

lemma tri7_ne2 : ∀ x : ZMod 7, x * (x - 1) ≠ 2 * 2 := by decide
lemma tri7_ne4 : ∀ x : ZMod 7, x * (x - 1) ≠ 2 * 4 := by decide
lemma tri7_ne5 : ∀ x : ZMod 7, x * (x - 1) ≠ 2 * 5 := by decide
lemma tri7_eq6 : ∀ x : ZMod 7, x * (x - 1) = 2 * 6 → x = 4 := by decide

lemma triE_mod7 {m : ℤ} {n r : ℕ} (hr : r = 2 ∨ r = 4 ∨ r = 5 ∨ r = 6) (h : triE m = 7 * n + r) :
    (7 : ℤ) ∣ 2 * m - 1 := by
  have h2 := MockTheta5.JTP.two_mul_triExp m
  rw [show (m * (m - 1) / 2).toNat = triE m from rfl, h] at h2
  have hc : (m : ZMod 7) * ((m : ZMod 7) - 1) = 2 * (r : ZMod 7) := by
    have := congrArg (Int.cast : ℤ → ZMod 7) h2
    push_cast at this
    rw [show (7 : ZMod 7) = 0 from rfl] at this
    rw [← this]; ring
  rcases hr with rfl | rfl | rfl | rfl
  · exact absurd (by simpa using hc) (tri7_ne2 _)
  · exact absurd (by simpa using hc) (tri7_ne4 _)
  · exact absurd (by simpa using hc) (tri7_ne5 _)
  · have hx := tri7_eq6 _ (by simpa using hc)
    have := (ZMod.intCast_zmod_eq_zero_iff_dvd (2 * m - 1) 7).mp (by push_cast; rw [hx]; decide)
    exact_mod_cast this

lemma theta7_vanish (u : ℂˣ) (hu : (u : ℂ) ^ 7 = -1) (n r : ℕ) (hr : r = 2 ∨ r = 4 ∨ r = 5 ∨ r = 6) :
    coeff (7 * n + r) (PowerSeries.map (evU u) MockTheta5.JTP.triTheta) = 0 :=
  theta_vanish_gen u hu _ fun _ h => triE_mod7 hr h

lemma neg_inv_root {w : ℂ} {a b : ℕ} (h : w ^ (a + b) = 1) (hu : -(w ^ a) ≠ 0) :
    -((Units.mk0 (-(w ^ a)) hu)⁻¹ : ℂˣ).val = w ^ b := by
  rw [Units.val_inv_eq_inv_val, Units.val_mk0, inv_neg, neg_neg]
  exact inv_eq_of_mul_eq_one_right (by rw [← pow_add, h])

lemma one_add_C_neg (c : ℂ) : (1 : PowerSeries ℂ) + C (-c) = C (1 - c) := by
  rw [map_sub, map_one, map_neg, sub_eq_add_neg]

lemma one_sub_ω7_ne {j : ℕ} (h0 : 0 < j) (h7 : j < 7) : (1 : ℂ) - ω7 ^ j ≠ 0 :=
  sub_ne_zero.mpr (Ne.symm (ω7_prim.pow_ne_one_of_pos_of_lt (by omega) h7))

noncomputable abbrev ThA : PowerSeries ℂ := PowerSeries.map (evU uA) MockTheta5.JTP.triTheta
noncomputable abbrev ThB : PowerSeries ℂ := PowerSeries.map (evU uB) MockTheta5.JTP.triTheta

theorem crankF_root7 :
    pochInf 1 1 * Ring.inverse (pochInf ω7 1 * pochInf ω7⁻¹ 1)
      = C ((1 - ω7 ^ 2) * (1 - ω7 ^ 3))⁻¹ * (ThA * ThB * ψC (Ep 7 (by norm_num) (Ring.inverse qfacInf))) := by
  have hJA := jtp_eval uA
  have hJB := jtp_eval uB
  have nA : -((uA⁻¹ : ℂˣ) : ℂ) = ω7 ^ 5 := neg_inv_root (a := 2) (b := 5) (by rw [ω7_pow7]) _
  have nB : -((uB⁻¹ : ℂˣ) : ℂ) = ω7 ^ 4 := neg_inv_root (a := 3) (b := 4) (by rw [ω7_pow7]) _
  rw [show -(uA : ℂ) = ω7 ^ 2 from neg_neg _, nA,
    show (uA : ℂ) = -(ω7 ^ 2) from rfl, one_add_C_neg] at hJA
  rw [show -(uB : ℂ) = ω7 ^ 3 from neg_neg _, nB,
    show (uB : ℂ) = -(ω7 ^ 3) from rfl, one_add_C_neg] at hJB
  have hW := prod_pochInf_roots_gen ω7_prim (by norm_num : (7 : ℕ) ≠ 0)
  simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul, pow_zero, pow_one] at hW
  rw [map_inverse (Ep 7 _) isUnit_qfacInf, map_inverse ψC (isUnit_qfacInf.map _), ← hW,
    show ThA = _ from hJA.symm, show ThB = _ from hJB.symm, ω7_inv]
  set A := pochInf 1 1
  set B := pochInf ω7 1 * pochInf (ω7 ^ 6) 1
  set D := pochInf (ω7 ^ 2) 1 * pochInf (ω7 ^ 5) 1 * pochInf (ω7 ^ 3) 1 * pochInf (ω7 ^ 4) 1
  have hAD : IsUnit (A * D) :=
    (isUnit_pochInf _ le_rfl).mul ((((isUnit_pochInf _ le_rfl).mul (isUnit_pochInf _ le_rfl)).mul
      (isUnit_pochInf _ le_rfl)).mul (isUnit_pochInf _ le_rfl))
  have hW' : A * pochInf ω7 1 * pochInf (ω7 ^ 2) 1 * pochInf (ω7 ^ 3) 1 * pochInf (ω7 ^ 4) 1 * pochInf (ω7 ^ 5) 1
      * pochInf (ω7 ^ 6) 1 = B * (A * D) := by
    simp only [A, B, D]; ring
  rw [hW', Ring.mul_inverse_rev B (A * D)]
  have hc : C ((1 - ω7 ^ 2) * (1 - ω7 ^ 3))⁻¹ * (C (1 - ω7 ^ 2) * C (1 - ω7 ^ 3)) = (1 : PowerSeries ℂ) := by
    rw [← map_mul, ← map_mul, inv_mul_cancel₀ (mul_ne_zero (one_sub_ω7_ne (by norm_num) (by norm_num))
      (one_sub_ω7_ne (by norm_num) (by norm_num))), map_one]
  have hADi : (A * D) * Ring.inverse (A * D) = 1 := Ring.mul_inverse_cancel _ hAD
  calc A * Ring.inverse B
      = A * Ring.inverse B * ((C ((1 - ω7 ^ 2) * (1 - ω7 ^ 3))⁻¹ * (C (1 - ω7 ^ 2) * C (1 - ω7 ^ 3)))
          * ((A * D) * Ring.inverse (A * D))) := by rw [hc, hADi, one_mul, mul_one]
    _ = _ := by simp only [A, D]; ring

theorem crank_sum_root7_zero (n : ℕ) : ∑ l : (7 * n + 5).Partition, ω7 ^ crank l = 0 := by
  rw [crank_generating_function ω7_ne (by omega), crankF_root7, coeff_C_mul]
  refine mul_eq_zero_of_right _ ?_
  rw [coeff_mul]
  refine Finset.sum_eq_zero fun ⟨a, b⟩ hab => ?_
  rw [Finset.mem_antidiagonal] at hab
  by_cases h7 : 7 ∣ b
  · refine mul_eq_zero_of_left ?_ _
    rw [coeff_mul]
    refine Finset.sum_eq_zero fun ⟨c, d⟩ hcd => ?_
    rw [Finset.mem_antidiagonal] at hcd
    by_cases hc : c % 7 = 2 ∨ c % 7 = 4 ∨ c % 7 = 5 ∨ c % 7 = 6
    · refine mul_eq_zero_of_left ?_ _
      rw [show c = 7 * (c / 7) + c % 7 by omega]
      exact theta7_vanish uA uA_pow7 _ _ hc
    · refine mul_eq_zero_of_right _ ?_
      rw [show d = 7 * (d / 7) + d % 7 by omega]
      exact theta7_vanish uB uB_pow7 _ _ (by omega)
  · refine mul_eq_zero_of_right _ ?_
    simp only [PowerSeries.coeff_map, coeff_Ep_of_not_dvd _ _ h7, map_zero]

lemma cyc_equal7 (N : ℕ → ℕ) (h : ∑ k ∈ range 7, (N k : ℂ) * ω7 ^ k = 0) {k : ℕ} (hk : k < 6) : N k = N 6 := by
  set q : Polynomial ℚ := ∑ j ∈ range 6, Polynomial.C ((N j : ℚ) - N 6) * Polynomial.X ^ j with hq
  have hg := ω7_prim.geom_sum_eq_zero (by norm_num)
  have hev : Polynomial.aeval ω7 q = 0 := by
    simp only [hq, map_sum, map_mul, map_pow, Polynomial.aeval_C, Polynomial.aeval_X, eq_ratCast]
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h hg ⊢
    push_cast
    linear_combination h - (N 6 : ℂ) * hg
  have hdvd : minpoly ℚ ω7 ∣ q := minpoly.dvd ℚ ω7 hev
  rw [← Polynomial.cyclotomic_eq_minpoly_rat ω7_prim (by norm_num)] at hdvd
  have hdeg : q.degree < (Polynomial.cyclotomic 7 ℚ).degree := by
    rw [Polynomial.degree_cyclotomic, Nat.totient_prime Nat.prime_seven]
    refine lt_of_le_of_lt (b := ((5 : ℕ) : WithBot ℕ)) ?_ (by decide)
    rw [hq]; simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]; compute_degree
    all_goals norm_num
  have hq0 := Polynomial.eq_zero_of_dvd_of_degree_lt hdvd hdeg
  have ck := congrArg (Polynomial.coeff · k) hq0
  simp only [hq, Polynomial.finsetSum_coeff, Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, mul_ite, mul_one,
    mul_zero, Finset.sum_ite_eq, Finset.mem_range, if_pos hk, Polynomial.coeff_zero] at ck
  exact_mod_cast sub_eq_zero.mp ck

/-- the number of partitions of `n` with crank `≡ k (mod 7)`. -/
noncomputable def crankCount7 (n k : ℕ) : ℕ := (univ.filter fun l : n.Partition => (crank l % 7).toNat = k).card

lemma ω7_zpow_crank (c : ℤ) : ω7 ^ c = ω7 ^ (c % 7).toNat := by
  rw [← zpow_natCast, Int.toNat_of_nonneg (Int.emod_nonneg c (by norm_num))]
  conv_lhs => rw [← Int.mul_ediv_add_emod c 7]
  rw [zpow_add₀ ω7_ne, zpow_mul, show ((7 : ℤ)) = ((7 : ℕ) : ℤ) from rfl, zpow_natCast, ω7_pow7, one_zpow, one_mul]

lemma crank_sum_grouped7 (n : ℕ) :
    ∑ l : n.Partition, ω7 ^ crank l = ∑ k ∈ range 7, (crankCount7 n k : ℂ) * ω7 ^ k := by
  simp_rw [ω7_zpow_crank]
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range 7) (g := fun l : n.Partition => (crank l % 7).toNat)
    (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (crank l % 7).toNat < 7 by omega)))]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_congr rfl (g := fun _ => ω7 ^ k) (fun l hl => by rw [(Finset.mem_filter.mp hl).2]),
    Finset.sum_const, nsmul_eq_mul, crankCount7]

lemma crankCount7_sum (n : ℕ) : ∑ k ∈ range 7, crankCount7 n k = Fintype.card n.Partition := by
  rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun l : n.Partition => (crank l % 7).toNat)
    (t := range 7) (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (crank l % 7).toNat < 7 by omega)))]
  rfl

/-- **Andrews–Garvan (mod 7).** The crank splits the partitions of `7n+5` into seven equal classes. -/
theorem crank_equidistribution_mod7 (n : ℕ) {i : ℕ} (hi : i < 7) :
    7 * (univ.filter fun l : (7 * n + 5).Partition => crank l % 7 = i).card
      = Fintype.card (7 * n + 5).Partition := by
  have h := crank_sum_root7_zero n
  rw [crank_sum_grouped7] at h
  have e : ∀ k < 6, crankCount7 (7 * n + 5) k = crankCount7 (7 * n + 5) 6 := fun k hk => cyc_equal7 _ h hk
  have hs := crankCount7_sum (7 * n + 5)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  have hfilt : (univ.filter fun l : (7 * n + 5).Partition => crank l % 7 = i) =
      univ.filter fun l => (crank l % 7).toNat = i :=
    Finset.filter_congr fun l _ => by omega
  rw [hfilt, ← crankCount7]
  have e0 := e 0 (by norm_num); have e1 := e 1 (by norm_num); have e2 := e 2 (by norm_num)
  have e3 := e 3 (by norm_num); have e4 := e 4 (by norm_num); have e5 := e 5 (by norm_num)
  interval_cases i <;> omega

end CrankProof
