/-
# Dyson's crank mod 11 (Garvan), via Winquist's identity as a lattice reindexing
-/
import RamanujanTau.CrankAndrewsGarvan

set_option autoImplicit false

namespace CrankProof
open PowerSeries Finset
open scoped PowerSeries.WithPiTopology
open MockTheta5.JTP (ψC E5 qfacInf triE pentE sgn isUnit_qfacInf)

/-! ## L1: lattice sums `Σ_i w_i q^{e_i}` in `ℂ⟦X⟧` -/

section Lattice
variable {ι κ : Type*}

/-- `Σ_i w_i q^{e_i}`, as a `tsum` in the coefficientwise topology. -/
noncomputable def lat (w : ι → ℂ) (e : ι → ℕ) : PowerSeries ℂ := ∑' i, C (w i) * X ^ (e i)

/-- `e` has finite sublevel sets. -/
def Proper (e : ι → ℕ) : Prop := ∀ k, Set.Finite {i | e i ≤ k}

lemma summable_mono (w : ι → ℂ) {e : ι → ℕ} (he : Proper e) : Summable fun i => C (w i) * X ^ (e i) := by
  rw [PowerSeries.WithPiTopology.summable_iff_summable_coeff]
  intro k
  refine summable_of_hasFiniteSupport ((he k).subset fun i hi => ?_)
  simp only [Function.mem_support, coeff_C_mul_X_pow, ne_eq, ite_eq_right_iff, Classical.not_imp] at hi
  exact le_of_eq hi.1.symm

lemma coeff_lat (w : ι → ℂ) {e : ι → ℕ} (he : Proper e) (k : ℕ) :
    coeff k (lat w e) = ∑' i, if e i = k then w i else 0 := by
  rw [lat, (summable_mono w he).map_tsum (coeff k) (PowerSeries.WithPiTopology.continuous_coeff ℂ k)]
  refine tsum_congr fun i => ?_
  rw [coeff_C_mul_X_pow]
  by_cases h : e i = k
  · rw [if_pos h.symm, if_pos h]
  · rw [if_neg (Ne.symm h), if_neg h]

lemma proper_prod {e : ι → ℕ} {e' : κ → ℕ} (he : Proper e) (he' : Proper e') :
    Proper fun p : ι × κ => e p.1 + e' p.2 := fun k =>
  ((he k).prod (he' k)).subset fun p hp => by
    simp only [Set.mem_setOf_eq] at hp
    exact ⟨by simp only [Set.mem_setOf_eq]; omega, by simp only [Set.mem_setOf_eq]; omega⟩

lemma proper_equiv {e : ι → ℕ} (he : Proper e) (σ : κ ≃ ι) : Proper (e ∘ σ) := fun k =>
  ((he k).preimage σ.injective.injOn).subset fun _ hp => hp

lemma lat_mul (w : ι → ℂ) (w' : κ → ℂ) {e : ι → ℕ} {e' : κ → ℕ} (he : Proper e) (he' : Proper e') :
    lat w e * lat w' e' = lat (fun p : ι × κ => w p.1 * w' p.2) (fun p => e p.1 + e' p.2) := by
  rw [lat, lat, lat, Summable.tsum_mul_tsum (summable_mono w he) (summable_mono w' he')]
  · refine tsum_congr fun p => ?_
    rw [map_mul, pow_add]; ring
  · have := summable_mono (fun p : ι × κ => w p.1 * w' p.2) (proper_prod he he')
    refine this.congr fun p => ?_
    rw [map_mul, pow_add]; ring

lemma lat_equiv (σ : κ ≃ ι) (w : ι → ℂ) (e : ι → ℕ) : lat (w ∘ σ) (e ∘ σ) = lat w e :=
  Equiv.tsum_eq σ (fun i => C (w i) * X ^ (e i))

lemma lat_congr {w w' : ι → ℂ} {e e' : ι → ℕ} (h : ∀ i, w i = w' i ∧ e i = e' i) : lat w e = lat w' e' :=
  tsum_congr fun i => by rw [(h i).1, (h i).2]

lemma lat_shift (c : ℂ) (n : ℕ) (w : ι → ℂ) {e : ι → ℕ} (he : Proper e) :
    lat (fun i => c * w i) (fun i => n + e i) = C c * X ^ n * lat w e := by
  rw [lat, lat, ← (summable_mono w he).tsum_mul_left]
  refine tsum_congr fun i => ?_
  rw [map_mul, pow_add]; ring

/-- Fubini for lattice sums. -/
lemma lat_fubini (w : ι × κ → ℂ) {e : ι × κ → ℕ} (he : Proper e) :
    lat w e = ∑' i, lat (fun j => w (i, j)) (fun j => e (i, j)) := by
  have hsec : ∀ i, Proper fun j => e (i, j) := fun i k =>
    ((he k).preimage (fun _ _ _ _ h => (Prod.mk.inj h).2)).subset fun _ hp => hp
  exact (summable_mono w he).tsum_prod' fun i => summable_mono (fun j => w (i, j)) (hsec i)

/-- a sign-reversing, exponent-preserving involution kills a lattice sum. -/
lemma lat_eq_zero_of_invol (σ : ι ≃ ι) (w : ι → ℂ) (e : ι → ℕ)
    (h : ∀ i, w (σ i) = -w i ∧ e (σ i) = e i) : lat w e = 0 := by
  have h1 : lat w e = -lat w e := by
    conv_lhs => rw [← lat_equiv σ]
    rw [lat, lat, ← tsum_neg]
    refine tsum_congr fun i => ?_
    simp only [Function.comp_apply, (h i).1, (h i).2, map_neg, neg_mul]
  ext k
  have := congrArg (coeff k) h1
  rw [map_neg, map_zero] at *
  linear_combination this / 2

end Lattice

/-! ## L2: Euler's and Jacobi's series as lattice sums -/

lemma triE_bound {m : ℤ} {k : ℕ} (h : triE m ≤ k) : -(k : ℤ) ≤ m ∧ m ≤ k + 1 := by
  have h2 := MockTheta5.JTP.two_mul_triExp m
  rw [show (m * (m - 1) / 2).toNat = triE m from rfl] at h2
  have : (triE m : ℤ) ≤ k := by exact_mod_cast h
  constructor <;> nlinarith

lemma pentE_bound {s : ℤ} {k : ℕ} (h : pentE s ≤ k) : -((k : ℤ) + 1) ≤ s ∧ s ≤ k + 1 := by
  have h2 := MockTheta5.JTP.two_mul_pentE s
  have : (pentE s : ℤ) ≤ k := by exact_mod_cast h
  constructor <;> nlinarith

lemma proper_triE : Proper triE := fun k =>
  (Set.finite_Icc (-(k : ℤ)) (k + 1)).subset fun _ hm => triE_bound hm

lemma proper_pentE : Proper pentE := fun k =>
  (Set.finite_Icc (-((k : ℤ) + 1)) (k + 1)).subset fun _ hm => pentE_bound hm

lemma tsum_ite_eq_box {f : ℤ → ℂ} {e : ℤ → ℕ} {k : ℕ} {lo hi : ℤ}
    (hb : ∀ m, e m = k → lo ≤ m ∧ m ≤ hi) :
    (∑' m, if e m = k then f m else 0) = ∑ m ∈ Icc lo hi, if e m = k then f m else 0 :=
  tsum_eq_sum fun m hm => if_neg fun h => hm (Finset.mem_Icc.mpr (hb m h))

/-- Jacobi: `map (evU u) triTheta = Σ_m u^m q^{m(m−1)/2}`. -/
lemma theta_lat (u : ℂˣ) : PowerSeries.map (evU u) MockTheta5.JTP.triTheta = lat (fun m => (u : ℂ) ^ m) triE := by
  ext k
  rw [coeff_lat _ proper_triE, coeff_map_evU_triTheta u k k le_rfl,
    tsum_ite_eq_box fun m h => triE_bound (le_of_eq h)]

/-- Euler: `(q;q)_∞ = Σ_s (−1)^s q^{s(3s−1)/2}`. -/
lemma euler_lat : ψC qfacInf = lat (fun s => ((sgn s : ℤ) : ℂ)) pentE := by
  ext k
  rw [coeff_lat _ proper_pentE, tsum_ite_eq_box fun m h => pentE_bound (le_of_eq h), PowerSeries.coeff_map,
    MockTheta5.JTP.coeff_qfacInf_box k k le_rfl, map_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  split_ifs <;> simp

/-! ## L3: Winquist's identity as a reindexing of `ℤ⁴` -/

/-- `J(x) = Σ_m (−x)^m q^{m(m−1)/2}`. -/
noncomputable def Jx (x : ℂ) : PowerSeries ℂ := lat (fun m : ℤ => (-x) ^ m) triE

/-- `(q;q)_∞` as a lattice sum. -/
noncomputable def Elat : PowerSeries ℂ := lat (fun s : ℤ => ((sgn s : ℤ) : ℂ)) pentE

/-- the fibre exponent: `n = (N−s−t, M−s+t, s, t)`. -/
def eF (N M : ℤ) (p : ℤ × ℤ) : ℕ := triE (N - p.1 - p.2) + triE (M - p.1 + p.2) + triE p.1 + triE p.2

def c1 (N M : ℤ) : ℤ := (N + M) / 3
def c2 (N M : ℤ) : ℤ := (N - M + 1) / 3
def good (N M : ℤ) : Prop := (N + M) % 3 ≠ 2 ∧ (N - M) % 3 ≠ 1
instance (N M : ℤ) : Decidable (good N M) := by unfold good; infer_instance

noncomputable def εW (N M : ℤ) : ℂ := (-1 : ℂ) ^ (N + M - c1 N M + c2 N M)
def eW (N M : ℤ) : ℕ := eF N M (c1 N M, c2 N M)

/-- Winquist's series `W(a,b) = Σ_{(N,M) good} ε a^N b^M q^{e(N,M)}`. -/
noncomputable def Wlat (a b : ℂ) : PowerSeries ℂ :=
  lat (fun p : ℤ × ℤ => if good p.1 p.2 then εW p.1 p.2 * (a ^ p.1 * b ^ p.2) else 0) (fun p => eW p.1 p.2)

lemma two_triE (m : ℤ) : 2 * (triE m : ℤ) = m * (m - 1) := MockTheta5.JTP.two_mul_triExp m

lemma eF_cast (N M : ℤ) (p : ℤ × ℤ) : 2 * (eF N M p : ℤ) =
    (N - p.1 - p.2) * (N - p.1 - p.2 - 1) + (M - p.1 + p.2) * (M - p.1 + p.2 - 1) + p.1 * (p.1 - 1)
      + p.2 * (p.2 - 1) := by
  unfold eF; push_cast
  linear_combination two_triE (N - p.1 - p.2) + two_triE (M - p.1 + p.2) + two_triE p.1 + two_triE p.2

lemma neg_one_zpow_even {x y : ℤ} (k : ℤ) (h : x = y + 2 * k) : (-1 : ℂ) ^ x = (-1 : ℂ) ^ y := by
  rw [h, zpow_add₀ (by norm_num), zpow_mul]; norm_num

lemma neg_one_zpow_odd {x y : ℤ} (k : ℤ) (h : x = y + 2 * k + 1) : (-1 : ℂ) ^ x = -(-1 : ℂ) ^ y := by
  rw [h, zpow_add₀ (by norm_num), zpow_add₀ (by norm_num), zpow_mul]; norm_num

lemma sgn_cast (s : ℤ) : ((sgn s : ℤ) : ℂ) = (-1 : ℂ) ^ s := by
  unfold sgn
  by_cases h : Even s
  · rw [if_pos h, h.neg_one_zpow]; norm_num
  · rw [if_neg h, (Int.not_even_iff_odd.mp h).neg_one_zpow]; norm_num

/-- the affine bijection `u ↦ c ± u`. -/
def aff (c : ℤ) (pos : Bool) : ℤ ≃ ℤ := if pos then Equiv.addLeft c else Equiv.subLeft c

lemma aff_apply (c : ℤ) (pos : Bool) (u : ℤ) : aff c pos u = if pos then c + u else c - u := by
  cases pos <;> rfl

lemma proper_eF (N M : ℤ) : Proper (eF N M) := fun k =>
  (((Set.finite_Icc (-(k : ℤ)) (k + 1)).prod (Set.finite_Icc (-(k : ℤ)) (k + 1)))).subset fun p hp => by
    simp only [Set.mem_setOf_eq, eF] at hp
    have h3 := triE_bound (show triE p.1 ≤ k by omega)
    have h4 := triE_bound (show triE p.2 ≤ k by omega)
    exact ⟨Set.mem_Icc.mpr h3, Set.mem_Icc.mpr h4⟩

/-- **the fibre lemma**: each `(N,M)`-fibre is `±q^e·(q;q)_∞²` or `0`. -/
theorem fib_eq (N M : ℤ) :
    lat (fun p : ℤ × ℤ => (-1 : ℂ) ^ (N + M - p.1 + p.2)) (eF N M)
      = if good N M then C (εW N M) * X ^ (eW N M) * (Elat * Elat) else 0 := by
  split_ifs with hg
  · obtain ⟨hg1, hg2⟩ := hg
    have hr1 : (N + M) % 3 = 0 ∨ (N + M) % 3 = 1 := by omega
    have hr2 : (N - M) % 3 = 0 ∨ (N - M) % 3 = 2 := by omega
    set b1 : Bool := decide ((N + M) % 3 = 1)
    set b2 : Bool := decide ((N - M) % 3 = 0)
    rw [← lat_equiv ((aff (c1 N M) b1).prodCongr (aff (c2 N M) b2)), Elat, lat_mul _ _ proper_pentE proper_pentE,
      ← lat_shift _ _ _ (proper_prod proper_pentE proper_pentE)]
    refine lat_congr fun q => ?_
    obtain ⟨u, v⟩ := q
    simp only [Function.comp_apply, Equiv.prodCongr_apply, Prod.map, aff_apply, sgn_cast]
    have hc1 := Int.emod_add_ediv (N + M) 3
    have hc2 := Int.emod_add_ediv (N - M + 1) 3
    unfold c1 c2 at *
    have hW := eF_cast N M ((N + M) / 3, (N - M + 1) / 3)
    have hP1 := MockTheta5.JTP.two_mul_pentE u
    have hP2 := MockTheta5.JTP.two_mul_pentE v
    rcases hr1 with h1 | h1 <;> rcases hr2 with h2 | h2
    · have e1 : b1 = false := by simp [b1, h1]
      have e2 : b2 = true := by simp [b2, h2]
      simp only [e1, e2, Bool.false_eq_true, if_false, if_true]
      have hF := eF_cast N M ((N + M) / 3 - u, (N - M + 1) / 3 + v)
      refine ⟨?_, ?_⟩
      · rw [εW, c1, c2, ← zpow_add₀ (by norm_num), ← zpow_add₀ (by norm_num)]
        exact neg_one_zpow_even (0) (by ring)
      · have : (eF N M ((N + M) / 3 - u, (N - M + 1) / 3 + v) : ℤ) = eW N M + (pentE u + pentE v) := by
          unfold eW c1 c2
          dsimp only at hF hW
          have q1 : 3 * ((N + M) / 3) = N + M := by omega
          have q2 : 3 * ((N - M + 1) / 3) = N - M := by omega
          have hWd : (eW N M : ℤ) = eF N M ((N + M) / 3, (N - M + 1) / 3) := rfl
          have h2x : 2 * (eF N M ((N + M) / 3 - u, (N - M + 1) / 3 + v) : ℤ) = 2 * (eW N M + (pentE u + pentE v) : ℤ) := by
            linear_combination hF - hW - hP1 - hP2 + (2 * ((-u))) * q1 + (2 * (v)) * q2 - 2 * hWd
          omega
        exact_mod_cast this
    · have e1 : b1 = false := by simp [b1, h1]
      have e2 : b2 = false := by simp [b2]; omega
      simp only [e1, e2, Bool.false_eq_true, if_false]
      have hF := eF_cast N M ((N + M) / 3 - u, (N - M + 1) / 3 - v)
      refine ⟨?_, ?_⟩
      · rw [εW, c1, c2, ← zpow_add₀ (by norm_num), ← zpow_add₀ (by norm_num)]
        exact neg_one_zpow_even (-v) (by ring)
      · have : (eF N M ((N + M) / 3 - u, (N - M + 1) / 3 - v) : ℤ) = eW N M + (pentE u + pentE v) := by
          unfold eW c1 c2
          dsimp only at hF hW
          have q1 : 3 * ((N + M) / 3) = N + M := by omega
          have q2 : 3 * ((N - M + 1) / 3) = N - M + 1 := by omega
          have hWd : (eW N M : ℤ) = eF N M ((N + M) / 3, (N - M + 1) / 3) := rfl
          have h2x : 2 * (eF N M ((N + M) / 3 - u, (N - M + 1) / 3 - v) : ℤ) = 2 * (eW N M + (pentE u + pentE v) : ℤ) := by
            linear_combination hF - hW - hP1 - hP2 + (2 * ((-u))) * q1 + (2 * ((-v))) * q2 - 2 * hWd
          omega
        exact_mod_cast this
    · have e1 : b1 = true := by simp [b1, h1]
      have e2 : b2 = true := by simp [b2, h2]
      simp only [e1, e2, if_true]
      have hF := eF_cast N M ((N + M) / 3 + u, (N - M + 1) / 3 + v)
      refine ⟨?_, ?_⟩
      · rw [εW, c1, c2, ← zpow_add₀ (by norm_num), ← zpow_add₀ (by norm_num)]
        exact neg_one_zpow_even (-u) (by ring)
      · have : (eF N M ((N + M) / 3 + u, (N - M + 1) / 3 + v) : ℤ) = eW N M + (pentE u + pentE v) := by
          unfold eW c1 c2
          dsimp only at hF hW
          have q1 : 3 * ((N + M) / 3) = N + M - 1 := by omega
          have q2 : 3 * ((N - M + 1) / 3) = N - M := by omega
          have hWd : (eW N M : ℤ) = eF N M ((N + M) / 3, (N - M + 1) / 3) := rfl
          have h2x : 2 * (eF N M ((N + M) / 3 + u, (N - M + 1) / 3 + v) : ℤ) = 2 * (eW N M + (pentE u + pentE v) : ℤ) := by
            linear_combination hF - hW - hP1 - hP2 + (2 * (u)) * q1 + (2 * (v)) * q2 - 2 * hWd
          omega
        exact_mod_cast this
    · have e1 : b1 = true := by simp [b1, h1]
      have e2 : b2 = false := by simp [b2]; omega
      simp only [e1, e2, Bool.false_eq_true, if_false, if_true]
      have hF := eF_cast N M ((N + M) / 3 + u, (N - M + 1) / 3 - v)
      refine ⟨?_, ?_⟩
      · rw [εW, c1, c2, ← zpow_add₀ (by norm_num), ← zpow_add₀ (by norm_num)]
        exact neg_one_zpow_even (-u - v) (by ring)
      · have : (eF N M ((N + M) / 3 + u, (N - M + 1) / 3 - v) : ℤ) = eW N M + (pentE u + pentE v) := by
          unfold eW c1 c2
          dsimp only at hF hW
          have q1 : 3 * ((N + M) / 3) = N + M - 1 := by omega
          have q2 : 3 * ((N - M + 1) / 3) = N - M + 1 := by omega
          have hWd : (eW N M : ℤ) = eF N M ((N + M) / 3, (N - M + 1) / 3) := rfl
          have h2x : 2 * (eF N M ((N + M) / 3 + u, (N - M + 1) / 3 - v) : ℤ) = 2 * (eW N M + (pentE u + pentE v) : ℤ) := by
            linear_combination hF - hW - hP1 - hP2 + (2 * (u)) * q1 + (2 * ((-v))) * q2 - 2 * hWd
          omega
        exact_mod_cast this
  · unfold good at hg
    rcases not_and_or.mp hg with h | h
    · have h1 : 3 * c1 N M = N + M - 2 := by unfold c1; omega
      refine lat_eq_zero_of_invol ((Equiv.subLeft (2 * c1 N M + 1)).prodCongr (Equiv.refl ℤ)) _ _ fun q => ?_
      obtain ⟨s, t⟩ := q
      simp only [Equiv.prodCongr_apply, Prod.map, Equiv.subLeft_apply, Equiv.refl_apply]
      refine ⟨neg_one_zpow_odd (s - c1 N M - 1) (by ring), ?_⟩
      have hA := eF_cast N M (2 * c1 N M + 1 - s, t)
      have hB := eF_cast N M (s, t)
      dsimp only at hA hB
      have : (eF N M (2 * c1 N M + 1 - s, t) : ℤ) = eF N M (s, t) := by
        have h2x : 2 * (eF N M (2 * c1 N M + 1 - s, t) : ℤ) = 2 * (eF N M (s, t) : ℤ) := by
          linear_combination hA - hB + (4 * c1 N M - 4 * s + 2) * h1
        omega
      exact_mod_cast this
    · have h2 : 3 * c2 N M = N - M - 1 := by unfold c2; omega
      refine lat_eq_zero_of_invol ((Equiv.refl ℤ).prodCongr (Equiv.subLeft (2 * c2 N M + 1))) _ _ fun q => ?_
      obtain ⟨s, t⟩ := q
      simp only [Equiv.prodCongr_apply, Prod.map, Equiv.subLeft_apply, Equiv.refl_apply]
      refine ⟨neg_one_zpow_odd (c2 N M - t) (by ring), ?_⟩
      have hA := eF_cast N M (s, 2 * c2 N M + 1 - t)
      have hB := eF_cast N M (s, t)
      dsimp only at hA hB
      have : (eF N M (s, 2 * c2 N M + 1 - t) : ℤ) = eF N M (s, t) := by
        have h2x : 2 * (eF N M (s, 2 * c2 N M + 1 - t) : ℤ) = 2 * (eF N M (s, t) : ℤ) := by
          linear_combination hA - hB + (4 * c2 N M - 4 * t + 2) * h2
        omega
      exact_mod_cast this

/-- `(N, M, s, t) ↦ (N−s−t, M−s+t, s, t)`. -/
def σ4 : (ℤ × ℤ) × (ℤ × ℤ) ≃ ((ℤ × ℤ) × ℤ) × ℤ where
  toFun q := (((q.1.1 - q.2.1 - q.2.2, q.1.2 - q.2.1 + q.2.2), q.2.1), q.2.2)
  invFun n := ((n.1.1.1 + n.1.2 + n.2, n.1.1.2 + n.1.2 - n.2), (n.1.2, n.2))
  left_inv q := by ext <;> simp <;> ring
  right_inv n := by ext <;> simp <;> ring

lemma proper_eW : Proper fun p : ℤ × ℤ => eW p.1 p.2 := fun k =>
  ((Set.finite_Icc (-3 * (k : ℤ) - 3) (3 * k + 3)).prod (Set.finite_Icc (-3 * (k : ℤ) - 3) (3 * k + 3))).subset
    fun p hp => by
      simp only [Set.mem_setOf_eq, eW, eF] at hp
      have h1 := triE_bound (show triE (p.1 - c1 p.1 p.2 - c2 p.1 p.2) ≤ k by omega)
      have h2 := triE_bound (show triE (p.2 - c1 p.1 p.2 + c2 p.1 p.2) ≤ k by omega)
      have h3 := triE_bound (show triE (c1 p.1 p.2) ≤ k by omega)
      have h4 := triE_bound (show triE (c2 p.1 p.2) ≤ k by omega)
      exact ⟨Set.mem_Icc.mpr ⟨by omega, by omega⟩, Set.mem_Icc.mpr ⟨by omega, by omega⟩⟩

lemma weight4 {a b : ℂ} (ha : a ≠ 0) (hb : b ≠ 0) (N M s t : ℤ) :
    (-a) ^ (N - s - t) * (-b) ^ (M - s + t) * (-(a * b)) ^ s * (-(a / b)) ^ t
      = a ^ N * b ^ M * (-1 : ℂ) ^ (N + M - s + t) := by
  have h1 : (-1 : ℂ) ≠ 0 := by norm_num
  have e : (-a) ^ (N - s - t) * (-b) ^ (M - s + t) * (-(a * b)) ^ s * (-(a / b)) ^ t
      = ((((-1 : ℂ) ^ (N - s - t) * (-1) ^ (M - s + t)) * (-1) ^ s) * (-1) ^ t)
        * ((a ^ (N - s - t) * a ^ s) * a ^ t) * ((b ^ (M - s + t) * b ^ s) * b ^ (-t)) := by
    simp only [neg_eq_neg_one_mul a, neg_eq_neg_one_mul b, neg_eq_neg_one_mul (a * b),
      div_eq_mul_inv, neg_eq_neg_one_mul (a * b⁻¹), mul_zpow, inv_zpow, zpow_neg]
    ring
  rw [e, ← zpow_add₀ h1, ← zpow_add₀ h1, ← zpow_add₀ h1, ← zpow_add₀ ha, ← zpow_add₀ ha, ← zpow_add₀ hb,
    ← zpow_add₀ hb, show N - s - t + s + t = N by ring, show M - s + t + s + -t = M by ring,
    show N - s - t + (M - s + t) + s + t = N + M - s + t by ring]
  ring

/-- **Winquist's identity** (as a lattice reindexing):
`J(a)J(b)J(ab)J(a/b) = W(a,b)·(q;q)_∞²`. -/
theorem winquist {a b : ℂ} (ha : a ≠ 0) (hb : b ≠ 0) :
    Jx a * Jx b * Jx (a * b) * Jx (a / b) = Wlat a b * (Elat * Elat) := by
  have hp2 := proper_prod proper_triE proper_triE
  have hp3 := proper_prod hp2 proper_triE
  have hp4 := proper_prod hp3 proper_triE
  rw [Jx, Jx, Jx, Jx, lat_mul _ _ proper_triE proper_triE, lat_mul _ _ hp2 proper_triE,
    lat_mul _ _ hp3 proper_triE, ← lat_equiv σ4]
  have hP := proper_equiv hp4 σ4
  rw [lat_congr (w' := fun q : (ℤ × ℤ) × (ℤ × ℤ) => (a ^ q.1.1 * b ^ q.1.2) * (-1 : ℂ) ^ (q.1.1 + q.1.2 - q.2.1 + q.2.2))
    (e' := fun q => 0 + eF q.1.1 q.1.2 q.2) fun q => ⟨by simp only [Function.comp_apply, σ4, Equiv.coe_fn_mk]; exact weight4 ha hb _ _ _ _,
      by simp only [Function.comp_apply, σ4, Equiv.coe_fn_mk, eF]; omega⟩]
  have hP' : Proper fun q : (ℤ × ℤ) × (ℤ × ℤ) => 0 + eF q.1.1 q.1.2 q.2 := by
    convert hP using 1; funext q; simp only [Function.comp_apply, σ4, Equiv.coe_fn_mk, eF]; omega
  rw [lat_fubini _ hP', Wlat, lat, ← (summable_mono _ proper_eW).tsum_mul_right]
  refine tsum_congr fun NM => ?_
  dsimp only
  rw [lat_shift _ _ _ (proper_eF NM.1 NM.2), fib_eq]
  split_ifs
  · rw [map_mul, pow_zero, mul_one, map_mul, map_mul]; ring
  · simp

/-! ## L4: the class `11n+6` of `W(ζ⁵, ζ²)` vanishes -/

/-- `6·e(N,M) = N² + M² − 3N − M` on the good classes. -/
lemma eW_formula {N M : ℤ} (hg : good N M) : 6 * (eW N M : ℤ) = N ^ 2 + M ^ 2 - 3 * N - M := by
  obtain ⟨hg1, hg2⟩ := hg
  have hW := eF_cast N M (c1 N M, c2 N M)
  dsimp only at hW
  have hWd : (eW N M : ℤ) = eF N M ((N + M) / 3, (N - M + 1) / 3) := rfl
  have h2x : 12 * (eW N M : ℤ) = 2 * (N ^ 2 + M ^ 2 - 3 * N - M) := by
    unfold c1 c2 at hW
    have hr1 : (N + M) % 3 = 0 ∨ (N + M) % 3 = 1 := by omega
    have hr2 : (N - M) % 3 = 0 ∨ (N - M) % 3 = 2 := by omega
    set x := (N + M) / 3 with hx
    set y := (N - M + 1) / 3 with hy
    rcases hr1 with h1 | h1 <;> rcases hr2 with h2 | h2
    · have q1 : 3 * x = N + M := by omega
      have q2 : 3 * y = N - M := by omega
      linear_combination 6 * hW + 12 * hWd + (-2 * M - 2 * N + 6 * x + 2) * q1 + (2 * M - 2 * N + 6 * y - 2) * q2
    · have q1 : 3 * x = N + M := by omega
      have q2 : 3 * y = N - M + 1 := by omega
      linear_combination 6 * hW + 12 * hWd + (-2 * M - 2 * N + 6 * x + 2) * q1 + (2 * M - 2 * N + 6 * y) * q2
    · have q1 : 3 * x = N + M - 1 := by omega
      have q2 : 3 * y = N - M := by omega
      linear_combination 6 * hW + 12 * hWd + (-2 * M - 2 * N + 6 * x) * q1 + (2 * M - 2 * N + 6 * y - 2) * q2
    · have q1 : 3 * x = N + M - 1 := by omega
      have q2 : 3 * y = N - M + 1 := by omega
      linear_combination 6 * hW + 12 * hWd + (-2 * M - 2 * N + 6 * x) * q1 + (2 * M - 2 * N + 6 * y) * q2
  omega

lemma tsum_eq_zero_of_invol {ι : Type*} (σ : ι ≃ ι) (f : ι → ℂ) (h : ∀ i, f (σ i) = -f i) : ∑' i, f i = 0 := by
  have h1 : ∑' i, f i = -∑' i, f i := by
    conv_lhs => rw [← Equiv.tsum_eq σ]
    rw [← tsum_neg]; exact tsum_congr h
  linear_combination h1 / 2

lemma sum_sq_mod11 : ∀ x y : ZMod 11, x ^ 2 + y ^ 2 = 0 → x = 0 := by decide

noncomputable def ω11 : ℂ := Complex.exp (2 * ↑Real.pi * Complex.I / ((11 : ℕ) : ℂ))

lemma ω11_prim : IsPrimitiveRoot ω11 11 := Complex.isPrimitiveRoot_exp 11 (by norm_num)
lemma ω11_pow11 : ω11 ^ 11 = 1 := ω11_prim.pow_eq_one
lemma ω11_ne : ω11 ≠ 0 := ω11_prim.ne_zero (by norm_num)

lemma ω11_zpow_mul11 (m : ℤ) : ω11 ^ (11 * m) = 1 := by
  rw [zpow_mul, show ((11 : ℤ)) = ((11 : ℕ) : ℤ) from rfl, zpow_natCast, ω11_pow11, one_zpow]

lemma good_reflect (N M : ℤ) : good (3 - N) M ↔ good N M := by unfold good; omega

lemma parity_reflect (N M : ℤ) (h1 : (N + M) % 3 ≠ 2) (h2 : (N - M) % 3 ≠ 1) :
    (3 - N + M - (3 - N + M) / 3 + (3 - N - M + 1) / 3 - (N + M - (N + M) / 3 + (N - M + 1) / 3)) % 2 = 1 := by
  obtain ⟨x, r1, hx, hr1⟩ : ∃ x r1, N + M = 3 * x + r1 ∧ (r1 = 0 ∨ r1 = 1) :=
    ⟨(N + M) / 3, (N + M) % 3, by omega, by omega⟩
  obtain ⟨y, r2, hy, hr2⟩ : ∃ y r2, N - M = 3 * y + r2 ∧ (r2 = 0 ∨ r2 = 2) :=
    ⟨(N - M) / 3, (N - M) % 3, by omega, by omega⟩
  have e1 : (N + M) / 3 = x := by omega
  have e2 : (3 - N - M + 1) / 3 = 1 - x := by omega
  rcases hr2 with rfl | rfl
  · have e3 : (N - M + 1) / 3 = y := by omega
    have e4 : (3 - N + M) / 3 = 1 - y := by omega
    rw [e1, e2, e3, e4]; omega
  · have e3 : (N - M + 1) / 3 = y + 1 := by omega
    have e4 : (3 - N + M) / 3 = -y := by omega
    rw [e1, e2, e3, e4]; omega

lemma εW_reflect {N M : ℤ} (hg : good N M) : εW (3 - N) M = -εW N M := by
  unfold εW c1 c2
  have hp := parity_reflect N M hg.1 hg.2
  refine neg_one_zpow_odd ((3 - N + M - (3 - N + M) / 3 + (3 - N - M + 1) / 3
    - (N + M - (N + M) / 3 + (N - M + 1) / 3) - 1) / 2) ?_
  omega

theorem W_class6 (n : ℕ) : coeff (11 * n + 6) (Wlat (ω11 ^ 5) (ω11 ^ 2)) = 0 := by
  rw [Wlat, coeff_lat _ proper_eW]
  refine tsum_eq_zero_of_invol ((Equiv.subLeft 3).prodCongr (Equiv.refl ℤ)) _ fun p => ?_
  obtain ⟨N, M⟩ := p
  simp only [Equiv.prodCongr_apply, Prod.map, Equiv.subLeft_apply, Equiv.refl_apply]
  by_cases hg : good N M
  · have hg' : good (3 - N) M := (good_reflect N M).mpr hg
    have hf := eW_formula hg
    have hf' := eW_formula hg'
    have he : eW (3 - N) M = eW N M := by
      have : 6 * (eW (3 - N) M : ℤ) = 6 * eW N M := by linear_combination hf' - hf
      omega
    rw [he, if_pos hg, if_pos hg']
    by_cases hk : eW N M = 11 * n + 6
    · rw [if_pos hk, if_pos hk]
      -- `11 ∣ 2N − 3`
      have hsq : ((2 * N - 3 : ℤ) : ZMod 11) ^ 2 + ((2 * M - 1 : ℤ) : ZMod 11) ^ 2 = 0 := by
        have h24 : (2 * N - 3) ^ 2 + (2 * M - 1) ^ 2 = 11 * (24 * (n : ℤ) + 14) := by
          have : (eW N M : ℤ) = 11 * n + 6 := by exact_mod_cast hk
          linear_combination (-4) * hf + 24 * this
        have := congrArg (Int.cast : ℤ → ZMod 11) h24
        push_cast at this ⊢
        rw [this, show (11 : ZMod 11) = 0 from rfl, zero_mul]
      have h11 := (ZMod.intCast_zmod_eq_zero_iff_dvd (2 * N - 3) 11).mp (sum_sq_mod11 _ _ hsq)
      obtain ⟨j, hj⟩ := h11
      -- the phase is reflection-invariant
      have hph : (ω11 ^ 5) ^ (3 - N) = (ω11 ^ 5) ^ N := by
        rw [← zpow_natCast, ← zpow_mul, ← zpow_mul, show ((5 : ℕ) : ℤ) * (3 - N) = ((5 : ℕ) : ℤ) * N + 11 * (-5 * j) by
          push_cast; linear_combination (-5 : ℤ) * hj, zpow_add₀ ω11_ne, ω11_zpow_mul11, mul_one]
      rw [hph, εW_reflect hg]; ring
    · rw [if_neg hk, if_neg hk, neg_zero]
  · have hg' : ¬ good (3 - N) M := fun h => hg ((good_reflect N M).mp h)
    simp only [if_neg hg, if_neg hg', ite_self, neg_zero]

/-! ## L5: the crank at `ζ₁₁` -/

lemma ω11_inv : ω11⁻¹ = ω11 ^ 10 := inv_eq_of_mul_eq_one_right (by rw [← pow_succ', ω11_pow11])

lemma ω11_pow_inv {i j : ℕ} (h : i + j = 11) : (ω11 ^ i)⁻¹ = ω11 ^ j :=
  inv_eq_of_mul_eq_one_right (by rw [← pow_add, h, ω11_pow11])

lemma one_sub_ω11_ne {j : ℕ} (h0 : 0 < j) (h11 : j < 11) : (1 : ℂ) - ω11 ^ j ≠ 0 :=
  sub_ne_zero.mpr (Ne.symm (ω11_prim.pow_ne_one_of_pos_of_lt (by omega) h11))

/-- `J(x) = (1−x)·(q;q)_∞·(xq;q)_∞·(q/x;q)_∞`. -/
lemma Jx_prod {x : ℂ} (hx : x ≠ 0) : Jx x = C (1 - x) * pochInf 1 1 * pochInf x 1 * pochInf x⁻¹ 1 := by
  have h := jtp_eval (Units.mk0 (-x) (neg_ne_zero.mpr hx))
  rw [theta_lat] at h
  simp only [Units.val_mk0, Units.val_inv_eq_inv_val, neg_neg, inv_neg, one_add_C_neg] at h
  rw [Jx, ← h]

lemma Elat_eq : Elat = pochInf 1 1 := by rw [Elat, ← euler_lat, pochInf_one_eq]

noncomputable def cc11 : ℂ := (1 - ω11 ^ 5) * (1 - ω11 ^ 2) * (1 - ω11 ^ 7) * (1 - ω11 ^ 3)

lemma cc11_ne : cc11 ≠ 0 := by
  unfold cc11
  exact mul_ne_zero (mul_ne_zero (mul_ne_zero (one_sub_ω11_ne (by norm_num) (by norm_num))
    (one_sub_ω11_ne (by norm_num) (by norm_num))) (one_sub_ω11_ne (by norm_num) (by norm_num)))
    (one_sub_ω11_ne (by norm_num) (by norm_num))

/-- Winquist at `(a, b) = (ζ⁵, ζ²)`, divided by `(q;q)_∞²`. -/
theorem W_eq : Wlat (ω11 ^ 5) (ω11 ^ 2) = C cc11 * pochInf 1 1 ^ 2 *
    (pochInf (ω11 ^ 5) 1 * pochInf (ω11 ^ 6) 1 * pochInf (ω11 ^ 2) 1 * pochInf (ω11 ^ 9) 1
      * pochInf (ω11 ^ 7) 1 * pochInf (ω11 ^ 4) 1 * pochInf (ω11 ^ 3) 1 * pochInf (ω11 ^ 8) 1) := by
  have h := winquist (pow_ne_zero 5 ω11_ne) (pow_ne_zero 2 ω11_ne)
  rw [show ω11 ^ 5 * ω11 ^ 2 = ω11 ^ 7 by rw [← pow_add],
    show ω11 ^ 5 / ω11 ^ 2 = ω11 ^ 3 by rw [div_eq_iff (pow_ne_zero _ ω11_ne), ← pow_add],
    Jx_prod (pow_ne_zero _ ω11_ne), Jx_prod (pow_ne_zero _ ω11_ne), Jx_prod (pow_ne_zero _ ω11_ne),
    Jx_prod (pow_ne_zero _ ω11_ne), ω11_pow_inv (show 5 + 6 = 11 by rfl), ω11_pow_inv (show 2 + 9 = 11 by rfl),
    ω11_pow_inv (show 7 + 4 = 11 by rfl), ω11_pow_inv (show 3 + 8 = 11 by rfl), Elat_eq] at h
  have hu : IsUnit (pochInf 1 1 * pochInf 1 1) := (isUnit_pochInf _ le_rfl).mul (isUnit_pochInf _ le_rfl)
  refine hu.mul_right_cancel (h.symm.trans ?_)
  rw [cc11, map_mul, map_mul, map_mul]; ring

theorem crankF_root11 :
    pochInf 1 1 * Ring.inverse (pochInf ω11 1 * pochInf ω11⁻¹ 1)
      = C cc11⁻¹ * (Wlat (ω11 ^ 5) (ω11 ^ 2) * ψC (Ep 11 (by norm_num) (Ring.inverse qfacInf))) := by
  have hW := prod_pochInf_roots_gen ω11_prim (by norm_num : (11 : ℕ) ≠ 0)
  simp only [Finset.prod_range_succ, Finset.prod_range_zero, one_mul, pow_zero, pow_one] at hW
  rw [map_inverse (Ep 11 _) isUnit_qfacInf, map_inverse ψC (isUnit_qfacInf.map _), ← hW, W_eq, ω11_inv]
  set A := pochInf 1 1
  set B := pochInf ω11 1 * pochInf (ω11 ^ 10) 1
  set D := pochInf (ω11 ^ 5) 1 * pochInf (ω11 ^ 6) 1 * pochInf (ω11 ^ 2) 1 * pochInf (ω11 ^ 9) 1
      * pochInf (ω11 ^ 7) 1 * pochInf (ω11 ^ 4) 1 * pochInf (ω11 ^ 3) 1 * pochInf (ω11 ^ 8) 1
  have hAD : IsUnit (A * D) := by
    simp only [A, D]
    repeat' first | exact isUnit_pochInf _ le_rfl | apply IsUnit.mul
  have hW' : A * pochInf ω11 1 * pochInf (ω11 ^ 2) 1 * pochInf (ω11 ^ 3) 1 * pochInf (ω11 ^ 4) 1
      * pochInf (ω11 ^ 5) 1 * pochInf (ω11 ^ 6) 1 * pochInf (ω11 ^ 7) 1 * pochInf (ω11 ^ 8) 1
      * pochInf (ω11 ^ 9) 1 * pochInf (ω11 ^ 10) 1 = B * (A * D) := by
    simp only [A, B, D]; ring
  rw [hW', Ring.mul_inverse_rev B (A * D)]
  have hc : C cc11⁻¹ * C cc11 = (1 : PowerSeries ℂ) := by rw [← map_mul, inv_mul_cancel₀ cc11_ne, map_one]
  have hADi : (A * D) * Ring.inverse (A * D) = 1 := Ring.mul_inverse_cancel _ hAD
  calc A * Ring.inverse B
      = A * Ring.inverse B * ((C cc11⁻¹ * C cc11) * ((A * D) * Ring.inverse (A * D))) := by
        rw [hc, hADi, one_mul, mul_one]
    _ = _ := by simp only [A, D]; ring

theorem crank_sum_root11_zero (n : ℕ) : ∑ l : (11 * n + 6).Partition, ω11 ^ crank l = 0 := by
  rw [crank_generating_function ω11_ne (by omega), crankF_root11, coeff_C_mul]
  refine mul_eq_zero_of_right _ ?_
  rw [coeff_mul]
  refine Finset.sum_eq_zero fun ⟨a, b⟩ hab => ?_
  rw [Finset.mem_antidiagonal] at hab
  by_cases h11 : 11 ∣ b
  · refine mul_eq_zero_of_left ?_ _
    rw [show a = 11 * (a / 11) + 6 by omega]
    exact W_class6 _
  · refine mul_eq_zero_of_right _ ?_
    simp only [PowerSeries.coeff_map, coeff_Ep_of_not_dvd _ _ h11, map_zero]

lemma cyc_equal11 (N : ℕ → ℕ) (h : ∑ k ∈ range 11, (N k : ℂ) * ω11 ^ k = 0) {k : ℕ} (hk : k < 10) :
    N k = N 10 := by
  set q : Polynomial ℚ := ∑ j ∈ range 10, Polynomial.C ((N j : ℚ) - N 10) * Polynomial.X ^ j with hq
  have hg := ω11_prim.geom_sum_eq_zero (by norm_num)
  have hev : Polynomial.aeval ω11 q = 0 := by
    simp only [hq, map_sum, map_mul, map_pow, Polynomial.aeval_C, Polynomial.aeval_X, eq_ratCast]
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h hg ⊢
    push_cast
    linear_combination h - (N 10 : ℂ) * hg
  have hdvd : minpoly ℚ ω11 ∣ q := minpoly.dvd ℚ ω11 hev
  rw [← Polynomial.cyclotomic_eq_minpoly_rat ω11_prim (by norm_num)] at hdvd
  have hdeg : q.degree < (Polynomial.cyclotomic 11 ℚ).degree := by
    rw [Polynomial.degree_cyclotomic, Nat.totient_prime (by decide : Nat.Prime 11)]
    refine lt_of_le_of_lt (b := ((9 : ℕ) : WithBot ℕ)) ?_ (by decide)
    rw [hq]; simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]; compute_degree
    all_goals norm_num
  have hq0 := Polynomial.eq_zero_of_dvd_of_degree_lt hdvd hdeg
  have ck := congrArg (Polynomial.coeff · k) hq0
  simp only [hq, Polynomial.finsetSum_coeff, Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, mul_ite, mul_one,
    mul_zero, Finset.sum_ite_eq, Finset.mem_range, if_pos hk, Polynomial.coeff_zero] at ck
  exact_mod_cast sub_eq_zero.mp ck

/-- the number of partitions of `n` with crank `≡ k (mod 11)`. -/
noncomputable def crankCount11 (n k : ℕ) : ℕ := (univ.filter fun l : n.Partition => (crank l % 11).toNat = k).card

lemma ω11_zpow_crank (c : ℤ) : ω11 ^ c = ω11 ^ (c % 11).toNat := by
  rw [← zpow_natCast, Int.toNat_of_nonneg (Int.emod_nonneg c (by norm_num))]
  conv_lhs => rw [← Int.mul_ediv_add_emod c 11]
  rw [zpow_add₀ ω11_ne, ω11_zpow_mul11, one_mul]

lemma crank_sum_grouped11 (n : ℕ) :
    ∑ l : n.Partition, ω11 ^ crank l = ∑ k ∈ range 11, (crankCount11 n k : ℂ) * ω11 ^ k := by
  simp_rw [ω11_zpow_crank]
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range 11) (g := fun l : n.Partition => (crank l % 11).toNat)
    (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (crank l % 11).toNat < 11 by omega)))]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_congr rfl (g := fun _ => ω11 ^ k) (fun l hl => by rw [(Finset.mem_filter.mp hl).2]),
    Finset.sum_const, nsmul_eq_mul, crankCount11]

lemma crankCount11_sum (n : ℕ) : ∑ k ∈ range 11, crankCount11 n k = Fintype.card n.Partition := by
  rw [← Finset.card_univ, Finset.card_eq_sum_card_fiberwise (f := fun l : n.Partition => (crank l % 11).toNat)
    (t := range 11) (fun l _ => Finset.mem_coe.mpr (Finset.mem_range.mpr (show (crank l % 11).toNat < 11 by omega)))]
  rfl

/-- **Garvan (mod 11)** — Dyson's conjecture for the crank: the partitions of `11n+6` split into eleven
crank classes mod 11 of equal size. -/
theorem crank_equidistribution_mod11 (n : ℕ) {i : ℕ} (hi : i < 11) :
    11 * (univ.filter fun l : (11 * n + 6).Partition => crank l % 11 = i).card
      = Fintype.card (11 * n + 6).Partition := by
  have h := crank_sum_root11_zero n
  rw [crank_sum_grouped11] at h
  have e : ∀ k < 10, crankCount11 (11 * n + 6) k = crankCount11 (11 * n + 6) 10 := fun k hk => cyc_equal11 _ h hk
  have hs := crankCount11_sum (11 * n + 6)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  have hfilt : (univ.filter fun l : (11 * n + 6).Partition => crank l % 11 = i) =
      univ.filter fun l => (crank l % 11).toNat = i :=
    Finset.filter_congr fun l _ => by omega
  rw [hfilt, ← crankCount11]
  have e0 := e 0 (by norm_num); have e1 := e 1 (by norm_num); have e2 := e 2 (by norm_num)
  have e3 := e 3 (by norm_num); have e4 := e 4 (by norm_num); have e5 := e 5 (by norm_num)
  have e6 := e 6 (by norm_num); have e7 := e 7 (by norm_num); have e8 := e 8 (by norm_num)
  have e9 := e 9 (by norm_num)
  interval_cases i <;> omega

end CrankProof
