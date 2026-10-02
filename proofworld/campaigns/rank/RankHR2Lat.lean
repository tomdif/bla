/-
# Dyson's rank, step 2: the lattice identity behind (2.19), coefficientwise

For each exponent `k`, the involution `τ` (RankHR2Core) pairs off the points of norm
`3U² + V² = 24k + 4 + 24a²`, so the signed counts of the four point classes cancel:
`#RP − #RM − #LP + #LM = 0`.
-/
import RamanujanTau.RankHR2Core

set_option autoImplicit false

namespace RankProof
open Finset

section Finite
variable (a k : ℕ)

/-- the target norm `24k + 4 + 24a²`. -/
def nT : ℤ := 24 * (k : ℤ) + 4 + 24 * (a : ℤ) ^ 2

/-- all lattice points of norm `nT a k` (inside a box that contains them). -/
noncomputable def Sk : Finset (ℤ × ℤ) :=
  ((Icc (-nT a k) (nT a k)) ×ˢ (Icc (-nT a k) (nT a k))).filter fun p => nrm3 p = nT a k

lemma mem_Sk_iff (p : ℤ × ℤ) : p ∈ Sk a k ↔ nrm3 p = nT a k := by
  unfold Sk
  rw [mem_filter, mem_product, mem_Icc, mem_Icc]
  constructor
  · exact fun h => h.2
  · intro h
    refine ⟨⟨⟨?_, ?_⟩, ?_, ?_⟩, h⟩ <;> unfold nrm3 at h <;> nlinarith [sq_nonneg (p.1 - 1), sq_nonneg (p.1 + 1),
      sq_nonneg (p.2 - 1), sq_nonneg (p.2 + 1), sq_nonneg p.1, sq_nonneg p.2]

/-- **the finite lattice identity**: `Σ_{nrm3 p = nT} (w_R − w_L)(p) = 0`. -/
theorem sum_wD_Sk : ∑ p ∈ Sk a k, wD a p = 0 := by
  refine sum_involution (fun p _ => tau a p) (fun p _ => ?_) (fun p _ hne => ?_) (fun p hp => ?_)
    (fun p _ => (tau_spec p).1)
  · rw [(tau_spec p).2]; ring
  · intro hfix
    replace hfix : tau a p = p := hfix
    apply hne
    have h := (tau_spec (a := a) p).2
    rw [hfix] at h
    omega
  · rw [mem_Sk_iff] at hp ⊢; rw [tau_nrm3, hp]

/-- the four signed counts. -/
theorem counts_cancel :
    ((Sk a k).filter fun p => RP a p.1 p.2).card + ((Sk a k).filter fun p => LM a p.1 p.2).card
      = ((Sk a k).filter fun p => RM a p.1 p.2).card + ((Sk a k).filter fun p => LP a p.1 p.2).card := by
  have h := sum_wD_Sk a k
  unfold wD at h
  simp only [sum_add_distrib, sum_sub_distrib, sum_boole] at h
  omega

end Finite

/-! ### parametrizations of the four point classes -/

section Param
variable (a k : ℕ)

/-- `L⁺`: `q^{(a−b)² + |b|(|b|+1) + 2(3r²+3|b|r+r)}`, point `(2b−4a, 12r+6|b|+2)`. -/
def cLP (x : ℤ × ℕ) : Prop :=
  ((a : ℤ) - x.1) ^ 2 + |x.1| * (|x.1| + 1) + 2 * (3 * (x.2 : ℤ) ^ 2 + 3 * |x.1| * x.2 + x.2) = k

instance (x : ℤ × ℕ) : Decidable (cLP a k x) := by unfold cLP; infer_instance

lemma LP_iff (U V : ℤ) : LP a U V ↔ U % 2 = 0 ∧ 6 * |(U + 4 * a) / 2| + 2 ≤ V
    ∧ (V - 6 * |(U + 4 * a) / 2| - 2) % 12 = 0 := by
  unfold LP
  rcases le_or_gt 0 ((U + 4 * (a : ℤ)) / 2) with h | h
  · rw [abs_of_nonneg h]; constructor
    · rintro ⟨h1, ⟨-, h2, h3⟩ | ⟨h2, -⟩⟩
      · exact ⟨h1, h2, h3⟩
      · omega
    · rintro ⟨h1, h2, h3⟩; exact ⟨h1, Or.inl ⟨h, h2, h3⟩⟩
  · rw [abs_of_neg h]; constructor
    · rintro ⟨h1, ⟨h2, -⟩ | ⟨-, h2, h3⟩⟩
      · omega
      · exact ⟨h1, by linarith, by rw [show V - 6 * -((U + 4 * a) / 2) - 2 = V + 6 * ((U + 4 * a) / 2) - 2 by ring]; exact h3⟩
    · rintro ⟨h1, h2, h3⟩
      exact ⟨h1, Or.inr ⟨h, by linarith, by rw [show V + 6 * ((U + 4 * a) / 2) - 2 = V - 6 * -((U + 4 * a) / 2) - 2 by ring]; exact h3⟩⟩

lemma card_LP (N : ℕ) (hk : k ≤ N) :
    ((Icc (-(N : ℤ)) N ×ˢ range (N + 1)).filter (cLP a k)).card = ((Sk a k).filter fun p => LP a p.1 p.2).card := by
  refine card_nbij' (fun x => (2 * x.1 - 4 * a, 12 * (x.2 : ℤ) + 6 * |x.1| + 2))
    (fun p => ((p.1 + 4 * a) / 2, ((p.2 - 6 * |(p.1 + 4 * a) / 2| - 2) / 12).toNat)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hx
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cLP] at hx ⊢
    rw [mem_Sk_iff, LP_iff, show (2 * b - 4 * a + 4 * a) / 2 = b by omega]
    refine ⟨?_, by omega, by omega, by omega⟩
    rw [nrm3, nT, ← hx.2]
    have : |b| ^ 2 = b ^ 2 := sq_abs b
    dsimp only
    linear_combination 12 * this
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cLP] at hp ⊢
    rw [mem_Sk_iff, LP_iff] at hp
    obtain ⟨hN, h1, h2, h3⟩ := hp
    set b := (U + 4 * (a : ℤ)) / 2 with hb
    have hU : U = 2 * b - 4 * a := by omega
    have hr0 : 0 ≤ (V - 6 * |b| - 2) / 12 := by omega
    have hV : V = 12 * (((V - 6 * |b| - 2) / 12).toNat : ℤ) + 6 * |b| + 2 := by
      rw [Int.toNat_of_nonneg hr0]; omega
    set r := ((V - 6 * |b| - 2) / 12).toNat
    have he : ((a : ℤ) - b) ^ 2 + |b| * (|b| + 1) + 2 * (3 * (r : ℤ) ^ 2 + 3 * |b| * r + r) = k := by
      have : |b| ^ 2 = b ^ 2 := sq_abs b
      unfold nrm3 nT at hN; dsimp only at hN
      rw [hU, hV] at hN
      nlinarith
    have hb0 : 0 ≤ |b| := abs_nonneg b
    have hr : (0 : ℤ) ≤ r := by positivity
    refine ⟨⟨⟨?_, ?_⟩, ?_⟩, he⟩
    · have : |b| ≤ k := by nlinarith [sq_nonneg ((a : ℤ) - b)]
      have := neg_abs_le b; omega
    · have : |b| ≤ k := by nlinarith [sq_nonneg ((a : ℤ) - b)]
      have := le_abs_self b; omega
    · have : (r : ℤ) ≤ k := by nlinarith [sq_nonneg ((a : ℤ) - b)]
      omega
  · rintro ⟨b, r⟩ _
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [show (2 * b - 4 * a + 4 * a) / 2 = b by omega,
      show (12 * (r : ℤ) + 6 * |b| + 2 - 6 * |b| - 2) / 12 = r by omega, Int.toNat_natCast]
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_Sk_iff, LP_iff] at hp
    obtain ⟨-, h1, h2, h3⟩ := hp
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [Int.toNat_of_nonneg (by omega)]; omega

/-- `L⁻`: `q^{(a−b)² + |b|(|b|+1) + 2(3r²+3|b|r−r−|b|)}` (`r ≥ 1`), point `(2b−4a, 12r+6|b|−2)`. -/
def cLM (x : ℤ × ℕ) : Prop := 1 ≤ x.2 ∧
  ((a : ℤ) - x.1) ^ 2 + |x.1| * (|x.1| + 1) + 2 * (3 * (x.2 : ℤ) ^ 2 + 3 * |x.1| * x.2 - x.2 - |x.1|) = k

instance (x : ℤ × ℕ) : Decidable (cLM a k x) := by unfold cLM; infer_instance

lemma LM_iff (U V : ℤ) : LM a U V ↔ U % 2 = 0 ∧ 6 * |(U + 4 * a) / 2| + 10 ≤ V
    ∧ (V - 6 * |(U + 4 * a) / 2| + 2) % 12 = 0 := by
  unfold LM
  rcases le_or_gt 0 ((U + 4 * (a : ℤ)) / 2) with h | h
  · rw [abs_of_nonneg h]; constructor
    · rintro ⟨h1, ⟨-, h2, h3⟩ | ⟨h2, -⟩⟩
      · exact ⟨h1, h2, h3⟩
      · omega
    · rintro ⟨h1, h2, h3⟩; exact ⟨h1, Or.inl ⟨h, h2, h3⟩⟩
  · rw [abs_of_neg h]; constructor
    · rintro ⟨h1, ⟨h2, -⟩ | ⟨-, h2, h3⟩⟩
      · omega
      · exact ⟨h1, by linarith, by rw [show V - 6 * -((U + 4 * a) / 2) + 2 = V + 6 * ((U + 4 * a) / 2) + 2 by ring]; exact h3⟩
    · rintro ⟨h1, h2, h3⟩
      exact ⟨h1, Or.inr ⟨h, by linarith, by rw [show V + 6 * ((U + 4 * a) / 2) + 2 = V - 6 * -((U + 4 * a) / 2) + 2 by ring]; exact h3⟩⟩

lemma card_LM (N : ℕ) (hk : k ≤ N) :
    ((Icc (-(N : ℤ)) N ×ˢ range (N + 1)).filter (cLM a k)).card = ((Sk a k).filter fun p => LM a p.1 p.2).card := by
  refine card_nbij' (fun x => (2 * x.1 - 4 * a, 12 * (x.2 : ℤ) + 6 * |x.1| - 2))
    (fun p => ((p.1 + 4 * a) / 2, ((p.2 - 6 * |(p.1 + 4 * a) / 2| + 2) / 12).toNat)) ?_ ?_ ?_ ?_
  · rintro ⟨b, r⟩ hx
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cLM] at hx ⊢
    rw [mem_Sk_iff, LM_iff, show (2 * b - 4 * a + 4 * a) / 2 = b by omega]
    refine ⟨?_, by omega, by omega, by omega⟩
    rw [nrm3, nT, ← hx.2.2]
    have : |b| ^ 2 = b ^ 2 := sq_abs b
    dsimp only
    linear_combination 12 * this
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cLM] at hp ⊢
    rw [mem_Sk_iff, LM_iff] at hp
    obtain ⟨hN, h1, h2, h3⟩ := hp
    set b := (U + 4 * (a : ℤ)) / 2 with hb
    have hU : U = 2 * b - 4 * a := by omega
    have hr0 : 0 ≤ (V - 6 * |b| + 2) / 12 := by omega
    have hV : V = 12 * (((V - 6 * |b| + 2) / 12).toNat : ℤ) + 6 * |b| - 2 := by
      rw [Int.toNat_of_nonneg hr0]; omega
    have hr1 : 1 ≤ ((V - 6 * |b| + 2) / 12).toNat := by omega
    set r := ((V - 6 * |b| + 2) / 12).toNat
    have he : ((a : ℤ) - b) ^ 2 + |b| * (|b| + 1) + 2 * (3 * (r : ℤ) ^ 2 + 3 * |b| * r - r - |b|) = k := by
      have : |b| ^ 2 = b ^ 2 := sq_abs b
      unfold nrm3 nT at hN; dsimp only at hN
      rw [hU, hV] at hN
      nlinarith
    have hb0 : 0 ≤ |b| := abs_nonneg b
    have hr : (1 : ℤ) ≤ r := by exact_mod_cast hr1
    have hbk : |b| ≤ k := by nlinarith [sq_nonneg ((a : ℤ) - b)]
    have hrk : (r : ℤ) ≤ k := by nlinarith [sq_nonneg ((a : ℤ) - b)]
    refine ⟨⟨⟨?_, ?_⟩, ?_⟩, hr1, he⟩
    · have := neg_abs_le b; omega
    · have := le_abs_self b; omega
    · omega
  · rintro ⟨b, r⟩ hx
    simp only [coe_filter, Set.mem_setOf_eq, cLM] at hx
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [show (2 * b - 4 * a + 4 * a) / 2 = b by omega,
      show (12 * (r : ℤ) + 6 * |b| - 2 - 6 * |b| + 2) / 12 = r by omega, Int.toNat_natCast]
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_Sk_iff, LM_iff] at hp
    obtain ⟨-, h1, h2, h3⟩ := hp
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [Int.toNat_of_nonneg (by omega)]; omega

/-- `R⁺`: `ψ`-index `j` (`q^{2j²+j}`) and `n = a + t` (`q^{n(3n+1)/2 − a²}`), point `(4j+1, 6n+1)`. -/
def cRP (x : ℤ × ℕ) : Prop :=
  4 * x.1 ^ 2 + 2 * x.1 + (a : ℤ) * (a + 1) + 6 * a * x.2 + (x.2 : ℤ) * (3 * x.2 + 1) = 2 * k

instance (x : ℤ × ℕ) : Decidable (cRP a k x) := by unfold cRP; infer_instance

lemma card_RP (N : ℕ) (hk : k ≤ N) :
    ((Icc (-(N : ℤ)) N ×ˢ range (N + 1)).filter (cRP a k)).card = ((Sk a k).filter fun p => RP a p.1 p.2).card := by
  refine card_nbij' (fun x => (4 * x.1 + 1, 6 * ((a : ℤ) + x.2) + 1))
    (fun p => ((p.1 - 1) / 4, ((p.2 - 1) / 6 - a).toNat)) ?_ ?_ ?_ ?_
  · rintro ⟨j, t⟩ hx
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cRP] at hx ⊢
    rw [mem_Sk_iff]
    refine ⟨?_, by unfold RP; omega⟩
    rw [nrm3, nT]
    dsimp only
    linear_combination 12 * hx.2
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cRP] at hp ⊢
    rw [mem_Sk_iff] at hp
    obtain ⟨hN, h1, h2, h3⟩ := hp
    have hU : U = 4 * ((U - 1) / 4) + 1 := by omega
    have hV : V = 6 * ((a : ℤ) + (((V - 1) / 6 - a).toNat : ℤ)) + 1 := by
      rw [Int.toNat_of_nonneg (by omega)]; omega
    set j := (U - 1) / 4
    set t := ((V - 1) / 6 - (a : ℤ)).toNat
    have he : 4 * j ^ 2 + 2 * j + (a : ℤ) * (a + 1) + 6 * a * t + (t : ℤ) * (3 * t + 1) = 2 * k := by
      unfold nrm3 nT at hN; dsimp only at hN
      rw [hU, hV] at hN
      nlinarith
    have ht : (0 : ℤ) ≤ t := by positivity
    have ha : (0 : ℤ) ≤ a := by positivity
    have hjk : j ≤ k := by nlinarith
    have hjk' : -(k : ℤ) ≤ j := by nlinarith
    have htk : (t : ℤ) ≤ k := by nlinarith
    exact ⟨⟨⟨by omega, by omega⟩, by omega⟩, he⟩
  · rintro ⟨j, t⟩ _
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [show (6 * ((a : ℤ) + t) + 1 - 1) / 6 - a = t by omega, Int.toNat_natCast]
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_Sk_iff] at hp
    obtain ⟨-, h1, h2, h3⟩ := hp
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [Int.toNat_of_nonneg (by omega)]; omega

/-- `R⁻`: as `R⁺` with the extra `q^{2n+1}`, point `(4j+1, 6n+5)`. -/
def cRM (x : ℤ × ℕ) : Prop :=
  4 * x.1 ^ 2 + 2 * x.1 + (a : ℤ) * (a + 1) + 6 * a * x.2 + (x.2 : ℤ) * (3 * x.2 + 1)
    + 2 * (2 * a + 2 * x.2 + 1) = 2 * k

instance (x : ℤ × ℕ) : Decidable (cRM a k x) := by unfold cRM; infer_instance

lemma card_RM (N : ℕ) (hk : k ≤ N) :
    ((Icc (-(N : ℤ)) N ×ˢ range (N + 1)).filter (cRM a k)).card = ((Sk a k).filter fun p => RM a p.1 p.2).card := by
  refine card_nbij' (fun x => (4 * x.1 + 1, 6 * ((a : ℤ) + x.2) + 5))
    (fun p => ((p.1 - 1) / 4, ((p.2 - 5) / 6 - a).toNat)) ?_ ?_ ?_ ?_
  · rintro ⟨j, t⟩ hx
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cRM] at hx ⊢
    rw [mem_Sk_iff]
    refine ⟨?_, by unfold RM; omega⟩
    rw [nrm3, nT]
    dsimp only
    linear_combination 12 * hx.2
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_product, mem_Icc, mem_range, cRM] at hp ⊢
    rw [mem_Sk_iff] at hp
    obtain ⟨hN, h1, h2, h3⟩ := hp
    have hU : U = 4 * ((U - 1) / 4) + 1 := by omega
    have hV : V = 6 * ((a : ℤ) + (((V - 5) / 6 - a).toNat : ℤ)) + 5 := by
      rw [Int.toNat_of_nonneg (by omega)]; omega
    set j := (U - 1) / 4
    set t := ((V - 5) / 6 - (a : ℤ)).toNat
    have he : 4 * j ^ 2 + 2 * j + (a : ℤ) * (a + 1) + 6 * a * t + (t : ℤ) * (3 * t + 1)
        + 2 * (2 * a + 2 * t + 1) = 2 * k := by
      unfold nrm3 nT at hN; dsimp only at hN
      rw [hU, hV] at hN
      nlinarith
    have ht : (0 : ℤ) ≤ t := by positivity
    have ha : (0 : ℤ) ≤ a := by positivity
    have hjk : j ≤ k := by nlinarith
    have hjk' : -(k : ℤ) ≤ j := by nlinarith
    have htk : (t : ℤ) ≤ k := by nlinarith
    exact ⟨⟨⟨by omega, by omega⟩, by omega⟩, he⟩
  · rintro ⟨j, t⟩ _
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [show (6 * ((a : ℤ) + t) + 5 - 5) / 6 - a = t by omega, Int.toNat_natCast]
  · rintro ⟨U, V⟩ hp
    simp only [coe_filter, Set.mem_setOf_eq, mem_Sk_iff] at hp
    obtain ⟨-, h1, h2, h3⟩ := hp
    simp only [Prod.mk.injEq]
    refine ⟨by omega, ?_⟩
    rw [Int.toNat_of_nonneg (by omega)]; omega

/-- **the parametrized count identity** (any box `N ≥ k`). -/
theorem param_counts (N : ℕ) (hk : k ≤ N) :
    let B := Icc (-(N : ℤ)) N ×ˢ range (N + 1)
    (B.filter (cRP a k)).card + (B.filter (cLM a k)).card = (B.filter (cRM a k)).card + (B.filter (cLP a k)).card := by
  intro B
  rw [card_RP a k N hk, card_LM a k N hk, card_RM a k N hk, card_LP a k N hk]
  exact counts_cancel a k

end Param

end RankProof
