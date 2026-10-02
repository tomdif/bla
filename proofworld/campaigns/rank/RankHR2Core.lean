/-
# Dyson's rank, step 2 core: the lattice involution behind Garvan's (2.19)
(see ROUTE.md). Points `(U, V) ∈ ℤ²`, norm `3U² + V²`.
-/
import Mathlib
import RamanujanTau.CrankWinquistMod11

set_option autoImplicit false

namespace RankProof

def LP (a : ℕ) (U V : ℤ) : Prop :=
  U % 2 = 0 ∧ ((0 ≤ (U + 4 * a) / 2 ∧ 6 * ((U + 4 * a) / 2) + 2 ≤ V ∧ (V - 6 * ((U + 4 * a) / 2) - 2) % 12 = 0)
    ∨ ((U + 4 * a) / 2 < 0 ∧ -6 * ((U + 4 * a) / 2) + 2 ≤ V ∧ (V + 6 * ((U + 4 * a) / 2) - 2) % 12 = 0))
def LM (a : ℕ) (U V : ℤ) : Prop :=
  U % 2 = 0 ∧ ((0 ≤ (U + 4 * a) / 2 ∧ 6 * ((U + 4 * a) / 2) + 10 ≤ V ∧ (V - 6 * ((U + 4 * a) / 2) + 2) % 12 = 0)
    ∨ ((U + 4 * a) / 2 < 0 ∧ -6 * ((U + 4 * a) / 2) + 10 ≤ V ∧ (V + 6 * ((U + 4 * a) / 2) + 2) % 12 = 0))
def RP (a : ℕ) (U V : ℤ) : Prop := U % 4 = 1 ∧ V % 6 = 1 ∧ 6 * (a : ℤ) + 1 ≤ V
def RM (a : ℕ) (U V : ℤ) : Prop := U % 4 = 1 ∧ V % 6 = 5 ∧ 6 * (a : ℤ) + 5 ≤ V
def Lq (a : ℕ) (U V : ℤ) : Prop := LP a U V ∨ LM a U V
def Rq (a : ℕ) (U V : ℤ) : Prop := RP a U V ∨ RM a U V

instance (a : ℕ) (U V : ℤ) : Decidable (LP a U V) := by unfold LP; infer_instance
instance (a : ℕ) (U V : ℤ) : Decidable (LM a U V) := by unfold LM; infer_instance
instance (a : ℕ) (U V : ℤ) : Decidable (RP a U V) := by unfold RP; infer_instance
instance (a : ℕ) (U V : ℤ) : Decidable (RM a U V) := by unfold RM; infer_instance
instance (a : ℕ) (U V : ℤ) : Decidable (Lq a U V) := by unfold Lq; infer_instance
instance (a : ℕ) (U V : ℤ) : Decidable (Rq a U V) := by unfold Rq; infer_instance

/-- the rotations by `(1 ± √−3)/2` on `z = V + U√−3`, `U ↦ −U`, and normalization to `U ≡ 1 (mod 4)`. -/
def rotpF (p : ℤ × ℤ) : ℤ × ℤ := ((p.1 + p.2) / 2, (p.2 - 3 * p.1) / 2)
def rotmF (p : ℤ × ℤ) : ℤ × ℤ := ((p.1 - p.2) / 2, (p.2 + 3 * p.1) / 2)
def negU (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def nrP (p : ℤ × ℤ) : ℤ × ℤ := if p.1 % 4 = 1 then p else negU p

def LqP (a : ℕ) (p : ℤ × ℤ) : Prop := Lq a p.1 p.2
def RqP (a : ℕ) (p : ℤ × ℤ) : Prop := Rq a p.1 p.2
instance (a : ℕ) (p : ℤ × ℤ) : Decidable (LqP a p) := by unfold LqP; infer_instance
instance (a : ℕ) (p : ℤ × ℤ) : Decidable (RqP a p) := by unfold RqP; infer_instance

/-- the involution on `ℤ²`. -/
def tau (a : ℕ) (p : ℤ × ℤ) : ℤ × ℤ :=
  if LqP a p then nrP (rotpF p)
  else if RqP a p then
    (if LqP a (rotmF p) then rotmF p
     else if LqP a (rotmF (negU p)) then rotmF (negU p)
     else if p.2 % 4 = 1 then nrP (rotpF p) else nrP (rotmF p))
  else p

/-- signed weight `w_R − w_L`. -/
def wD (a : ℕ) (p : ℤ × ℤ) : ℤ :=
  (if RP a p.1 p.2 then 1 else 0) - (if RM a p.1 p.2 then 1 else 0) - (if LP a p.1 p.2 then 1 else 0)
    + (if LM a p.1 p.2 then 1 else 0)

lemma Lq_even {a : ℕ} {U V : ℤ} (h : Lq a U V) : U % 2 = 0 ∧ V % 2 = 0 := by unfold Lq LP LM at h; omega
lemma Rq_odd {a : ℕ} {U V : ℤ} (h : Rq a U V) : U % 2 = 1 ∧ V % 2 = 1 := by unfold Rq RP RM at h; omega
lemma not_Lq_of_Rq {a : ℕ} {U V : ℤ} (h : Rq a U V) : ¬ Lq a U V := fun h' => by
  have := Lq_even h'; have := Rq_odd h; omega

lemma wD_LP {a : ℕ} {U V : ℤ} (h : LP a U V) : wD a (U, V) = -1 := by
  have hM : ¬ LM a U V := by unfold LP at h; unfold LM; omega
  have hR : ¬ Rq a U V := fun h' => not_Lq_of_Rq h' (Or.inl h)
  unfold Rq at hR
  simp only [wD, if_pos h, if_neg hM, if_neg (fun h' => hR (Or.inl h')), if_neg (fun h' => hR (Or.inr h'))]
  norm_num
lemma wD_LM {a : ℕ} {U V : ℤ} (h : LM a U V) : wD a (U, V) = 1 := by
  have hP : ¬ LP a U V := by unfold LM at h; unfold LP; omega
  have hR : ¬ Rq a U V := fun h' => not_Lq_of_Rq h' (Or.inr h)
  unfold Rq at hR
  simp only [wD, if_pos h, if_neg hP, if_neg (fun h' => hR (Or.inl h')), if_neg (fun h' => hR (Or.inr h'))]
  norm_num
lemma wD_RP {a : ℕ} {U V : ℤ} (h : RP a U V) : wD a (U, V) = 1 := by
  have hM : ¬ RM a U V := by unfold RP at h; unfold RM; omega
  have hL := not_Lq_of_Rq (Or.inl h : Rq a U V)
  unfold Lq at hL
  simp only [wD, if_pos h, if_neg hM, if_neg (fun h' => hL (Or.inl h')), if_neg (fun h' => hL (Or.inr h'))]
  norm_num
lemma wD_RM {a : ℕ} {U V : ℤ} (h : RM a U V) : wD a (U, V) = -1 := by
  have hP : ¬ RP a U V := by unfold RM at h; unfold RP; omega
  have hL := not_Lq_of_Rq (Or.inr h : Rq a U V)
  unfold Lq at hL
  simp only [wD, if_pos h, if_neg hP, if_neg (fun h' => hL (Or.inl h')), if_neg (fun h' => hL (Or.inr h'))]
  norm_num
lemma wD_zero {a : ℕ} {U V : ℤ} (hL : ¬ Lq a U V) (hR : ¬ Rq a U V) : wD a (U, V) = 0 := by
  unfold Lq at hL; unfold Rq at hR
  simp only [wD, if_neg (fun h' => hL (Or.inl h')), if_neg (fun h' => hL (Or.inr h')),
    if_neg (fun h' => hR (Or.inl h')), if_neg (fun h' => hR (Or.inr h'))]
  norm_num


/-! ### arithmetic facts, each an `omega` problem -/
section Arith
variable (a : ℕ)
set_option maxHeartbeats 1000000

lemma F1P (p : ℤ × ℤ) (h : LP a p.1 p.2) : RP a (nrP (rotpF p)).1 (nrP (rotpF p)).2 := by
  obtain ⟨U, V⟩ := p; unfold nrP negU rotpF; split_ifs <;> (simp only at *; unfold LP at h; unfold RP; omega)
lemma F1M (p : ℤ × ℤ) (h : LM a p.1 p.2) : RM a (nrP (rotpF p)).1 (nrP (rotpF p)).2 := by
  obtain ⟨U, V⟩ := p; unfold nrP negU rotpF; split_ifs <;> (simp only at *; unfold LM at h; unfold RM; omega)

lemma F2a (p : ℤ × ℤ) (h : LqP a p) (hs : (rotpF p).1 % 4 = 1) : rotmF (rotpF p) = p := by
  obtain ⟨U, V⟩ := p; unfold LqP Lq LP LM at h; simp only [rotmF, rotpF] at *; ext <;> simp only <;> omega
lemma F2b (p : ℤ × ℤ) (h : LqP a p) (hs : (rotpF p).1 % 4 ≠ 1) :
    ¬ LqP a (rotmF (negU (rotpF p))) ∧ rotmF (negU (negU (rotpF p))) = p := by
  obtain ⟨U, V⟩ := p; unfold LqP Lq LP LM at *; simp only [rotmF, rotpF, negU] at *
  refine ⟨by omega, ?_⟩; ext <;> simp only <;> omega

lemma F3 (p : ℤ × ℤ) (hR : RqP a p) : rotpF (rotmF p) = p ∧ nrP p = p := by
  obtain ⟨U, V⟩ := p; unfold RqP Rq RP RM at hR; simp only [rotmF, rotpF, nrP, negU] at *
  refine ⟨by ext <;> simp only <;> omega, by rw [if_pos (by omega)]⟩
lemma F4 (p : ℤ × ℤ) (hR : RqP a p) : rotpF (rotmF (negU p)) = negU p ∧ nrP (negU p) = p := by
  obtain ⟨U, V⟩ := p; unfold RqP Rq RP RM at hR; simp only [rotmF, rotpF, nrP, negU] at *
  refine ⟨by ext <;> simp only <;> omega, by rw [if_neg (by omega)]; simp⟩

lemma nrP_fst (q : ℤ × ℤ) (h : q.1 % 2 = 1) : (nrP q).1 % 4 = 1 ∧ (nrP q).2 = q.2 := by
  obtain ⟨x, y⟩ := q
  simp only [nrP, negU] at *
  split_ifs with hx
  · exact ⟨hx, rfl⟩
  · exact ⟨by omega, rfl⟩

lemma F5aP (p : ℤ × ℤ) (hR : RP a p.1 p.2) (h1 : ¬ LqP a (rotmF p)) (hV : p.2 % 4 = 1) :
    RM a (nrP (rotpF p)).1 (nrP (rotpF p)).2 := by
  have hv : (rotpF p).2 % 6 = 5 ∧ 6 * (a : ℤ) + 5 ≤ (rotpF p).2 := by
    obtain ⟨U, V⟩ := p; unfold RP at hR; unfold LqP Lq LP LM at h1; simp only [rotmF, rotpF] at *; omega
  have ho : (rotpF p).1 % 2 = 1 := by
    obtain ⟨U, V⟩ := p; unfold RP at hR; simp only [rotpF] at *; omega
  obtain ⟨h4, h5⟩ := nrP_fst _ ho
  unfold RM; rw [h5]; exact ⟨h4, hv.1, hv.2⟩

lemma F5aM (p : ℤ × ℤ) (hR : RM a p.1 p.2) (h1 : ¬ LqP a (rotmF p)) (hV : p.2 % 4 = 1) :
    RP a (nrP (rotpF p)).1 (nrP (rotpF p)).2 := by
  obtain ⟨U, V⟩ := p; unfold RM at hR; unfold LqP Lq LP LM at h1
  simp only [rotmF, rotpF, nrP, negU] at *; split_ifs <;> (unfold RP; simp only; omega)
lemma F5bP (p : ℤ × ℤ) (hR : RP a p.1 p.2) (h2 : ¬ LqP a (rotmF (negU p))) (hV : p.2 % 4 ≠ 1) :
    RM a (nrP (rotmF p)).1 (nrP (rotmF p)).2 := by
  have hv : (rotmF p).2 % 6 = 5 ∧ 6 * (a : ℤ) + 5 ≤ (rotmF p).2 := by
    obtain ⟨U, V⟩ := p; unfold RP at hR; unfold LqP Lq LP LM at h2; simp only [rotmF, negU] at *; omega
  have ho : (rotmF p).1 % 2 = 1 := by
    obtain ⟨U, V⟩ := p; unfold RP at hR; simp only [rotmF] at *; omega
  obtain ⟨h4, h5⟩ := nrP_fst _ ho
  unfold RM; rw [h5]; exact ⟨h4, hv.1, hv.2⟩

lemma F5bM (p : ℤ × ℤ) (hR : RM a p.1 p.2) (h2 : ¬ LqP a (rotmF (negU p))) (hV : p.2 % 4 ≠ 1) :
    RP a (nrP (rotmF p)).1 (nrP (rotmF p)).2 := by
  obtain ⟨U, V⟩ := p; unfold RM at hR; unfold LqP Lq LP LM at h2
  simp only [rotmF, rotpF, nrP, negU] at *; split_ifs <;> (unfold RP; simp only; omega)

lemma F6 (p : ℤ × ℤ) (hR : RqP a p) :
    let q := if p.2 % 4 = 1 then nrP (rotpF p) else nrP (rotmF p)
    ¬ LqP a (rotmF q) ∧ ¬ LqP a (rotmF (negU q))
      ∧ (if q.2 % 4 = 1 then nrP (rotpF q) else nrP (rotmF q)) = p := by
  obtain ⟨U, V⟩ := p; unfold RqP Rq RP RM at hR
  simp only [rotmF, rotpF, nrP, negU] at *
  split_ifs <;> (unfold LqP Lq LP LM; simp only; refine ⟨by omega, by omega, ?_⟩; ext <;> simp only <;> omega)

end Arith


/-! ### the involution -/

section Assembly
variable {a : ℕ}

lemma wDP_LP {q : ℤ × ℤ} (h : LP a q.1 q.2) : wD a q = -1 := by obtain ⟨x, y⟩ := q; exact wD_LP h
lemma wDP_LM {q : ℤ × ℤ} (h : LM a q.1 q.2) : wD a q = 1 := by obtain ⟨x, y⟩ := q; exact wD_LM h
lemma wDP_RP {q : ℤ × ℤ} (h : RP a q.1 q.2) : wD a q = 1 := by obtain ⟨x, y⟩ := q; exact wD_RP h
lemma wDP_RM {q : ℤ × ℤ} (h : RM a q.1 q.2) : wD a q = -1 := by obtain ⟨x, y⟩ := q; exact wD_RM h

lemma notLqP_of {q : ℤ × ℤ} (h : RqP a q) : ¬ LqP a q := not_Lq_of_Rq h

/-- `τ` of a right point, unfolded. -/
lemma tau_R {q : ℤ × ℤ} (h : RqP a q) :
    tau a q = (if LqP a (rotmF q) then rotmF q
      else if LqP a (rotmF (negU q)) then rotmF (negU q)
      else if q.2 % 4 = 1 then nrP (rotpF q) else nrP (rotmF q)) := by
  rw [tau, if_neg (notLqP_of h), if_pos h]

/-- the image of an L-point, and the way back. -/
lemma tau_of_L {p : ℤ × ℤ} (hL : LqP a p) :
    RqP a (nrP (rotpF p)) ∧ tau a (nrP (rotpF p)) = p
      ∧ wD a (nrP (rotpF p)) = -wD a p := by
  have hRq : RqP a (nrP (rotpF p)) := by
    rcases hL with hP | hM
    · exact Or.inl (F1P a p hP)
    · exact Or.inr (F1M a p hM)
  refine ⟨hRq, ?_, ?_⟩
  · rw [tau_R hRq]
    by_cases hs : (rotpF p).1 % 4 = 1
    · have hn : nrP (rotpF p) = rotpF p := by rw [nrP, if_pos hs]
      rw [hn, F2a a p hL hs, if_pos hL]
    · have hn : nrP (rotpF p) = negU (rotpF p) := by rw [nrP, if_neg hs]
      obtain ⟨hb1, hb2⟩ := F2b a p hL hs
      rw [hn, if_neg hb1, hb2, if_pos hL]
  · rcases hL with hP | hM
    · rw [wDP_LP hP, wDP_RP (F1P a p hP)]; norm_num
    · rw [wDP_LM hM, wDP_RM (F1M a p hM)]

/-- **`τ` is a sign-reversing involution.** -/
theorem tau_spec (p : ℤ × ℤ) : tau a (tau a p) = p ∧ wD a (tau a p) = -wD a p := by
  by_cases hL : LqP a p
  · have h := tau_of_L hL
    have ht : tau a p = nrP (rotpF p) := by rw [tau, if_pos hL]
    rw [ht]; exact ⟨h.2.1, h.2.2⟩
  by_cases hR : RqP a p
  · by_cases h1 : LqP a (rotmF p)
    · have ht : tau a p = rotmF p := by rw [tau_R hR, if_pos h1]
      have h := tau_of_L h1
      rw [(F3 a p hR).1, (F3 a p hR).2] at h
      rw [ht]
      exact ⟨by rw [tau, if_pos h1, (F3 a p hR).1, (F3 a p hR).2], by rw [h.2.2]; ring⟩
    by_cases h2 : LqP a (rotmF (negU p))
    · have ht : tau a p = rotmF (negU p) := by rw [tau_R hR, if_neg h1, if_pos h2]
      have h := tau_of_L h2
      rw [(F4 a p hR).1, (F4 a p hR).2] at h
      rw [ht]
      exact ⟨by rw [tau, if_pos h2, (F4 a p hR).1, (F4 a p hR).2], by rw [h.2.2]; ring⟩
    -- the rest: paired among themselves
    have ht : tau a p = (if p.2 % 4 = 1 then nrP (rotpF p) else nrP (rotmF p)) := by
      rw [tau_R hR, if_neg h1, if_neg h2]
    have hRq : RqP a (if p.2 % 4 = 1 then nrP (rotpF p) else nrP (rotmF p)) := by
      rcases hR with hP | hM
      · split_ifs with hV
        · exact Or.inr (F5aP a p hP h1 hV)
        · exact Or.inr (F5bP a p hP h2 hV)
      · split_ifs with hV
        · exact Or.inl (F5aM a p hM h1 hV)
        · exact Or.inl (F5bM a p hM h2 hV)
    obtain ⟨g1, g2, g3⟩ := F6 a p hR
    rw [ht]
    refine ⟨by rw [tau_R hRq, if_neg g1, if_neg g2, g3], ?_⟩
    rcases hR with hP | hM
    · rw [wDP_RP hP]
      split_ifs with hV
      · rw [wDP_RM (F5aP a p hP h1 hV)]
      · rw [wDP_RM (F5bP a p hP h2 hV)]
    · rw [wDP_RM hM]
      split_ifs with hV
      · rw [wDP_RP (F5aM a p hM h1 hV)]; norm_num
      · rw [wDP_RP (F5bM a p hM h2 hV)]; norm_num
  · have ht : tau a p = p := by rw [tau, if_neg hL, if_neg hR]
    obtain ⟨x, y⟩ := p
    rw [ht, ht]; exact ⟨rfl, by rw [wD_zero hL hR]; norm_num⟩

end Assembly


/-! ### norm preservation and the series identity -/

def nrm3 (p : ℤ × ℤ) : ℤ := 3 * p.1 ^ 2 + p.2 ^ 2

lemma nrm3_rotp {p : ℤ × ℤ} (h : (p.1 + p.2) % 2 = 0) : nrm3 (rotpF p) = nrm3 p := by
  obtain ⟨U, V⟩ := p
  obtain ⟨k, hk⟩ : ∃ k, U + V = 2 * k := ⟨(U + V) / 2, by simp only at h; omega⟩
  simp only [nrm3, rotpF]
  rw [show (U + V) / 2 = k by omega, show (V - 3 * U) / 2 = k - 2 * U by omega]
  have : V = 2 * k - U := by omega
  subst this; ring

lemma nrm3_rotm {p : ℤ × ℤ} (h : (p.1 + p.2) % 2 = 0) : nrm3 (rotmF p) = nrm3 p := by
  obtain ⟨U, V⟩ := p
  obtain ⟨k, hk⟩ : ∃ k, U - V = 2 * k := ⟨(U - V) / 2, by simp only at h; omega⟩
  simp only [nrm3, rotmF]
  rw [show (U - V) / 2 = k by omega, show (V + 3 * U) / 2 = 2 * U - k by omega]
  have : V = U - 2 * k := by omega
  subst this; ring

lemma nrm3_negU (p : ℤ × ℤ) : nrm3 (negU p) = nrm3 p := by simp [nrm3, negU]
lemma nrm3_nrP (p : ℤ × ℤ) : nrm3 (nrP p) = nrm3 p := by
  unfold nrP; split_ifs
  · rfl
  · exact nrm3_negU p

lemma par_of_L {a : ℕ} {p : ℤ × ℤ} (h : LqP a p) : (p.1 + p.2) % 2 = 0 := by
  have := Lq_even h; omega
lemma par_of_R {a : ℕ} {p : ℤ × ℤ} (h : RqP a p) : (p.1 + p.2) % 2 = 0 := by
  have := Rq_odd h; omega

theorem tau_nrm3 (a : ℕ) (p : ℤ × ℤ) : nrm3 (tau a p) = nrm3 p := by
  unfold tau
  split_ifs with hL hR h1 h2 hV
  · rw [nrm3_nrP, nrm3_rotp (par_of_L hL)]
  · rw [nrm3_rotm (par_of_R hR)]
  · rw [nrm3_rotm (by have := par_of_R hR; simp only [negU]; omega), nrm3_negU]
  · rw [nrm3_nrP, nrm3_rotp (par_of_R hR)]
  · rw [nrm3_nrP, nrm3_rotm (par_of_R hR)]
  · rfl

/-- the exponent `(3U² + V² − 4 − 24a²)/24`. -/
def eN (a : ℕ) (p : ℤ × ℤ) : ℕ := ((nrm3 p - 4 - 24 * (a : ℤ) ^ 2) / 24).toNat

open CrankProof in
/-- **the lattice identity behind (2.19)**: `Σ_p (w_R − w_L)(p) q^{e(p)} = 0`. -/
theorem lat_wD_zero (a : ℕ) : lat (fun p => (wD a p : ℂ)) (eN a) = 0 := by
  refine lat_eq_zero_of_invol (Function.Involutive.toPerm (tau a) fun p => (tau_spec p).1) _ _ fun p => ?_
  simp only [Function.Involutive.coe_toPerm]
  refine ⟨by rw [(tau_spec p).2]; push_cast; ring, by rw [eN, eN, tau_nrm3]⟩

end RankProof
