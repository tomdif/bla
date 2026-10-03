# Campaign: Dyson's rank conjecture mod 7 (Atkin–Swinnerton-Dyer)

Goal: [q^{7n+5}] R(ζ₇;q) = 0, i.e. R₅ = 0 in R(ζ₇;q) = Σ_{k<7} q^k R_k(q⁷).

Strategy (scaling the mod-5 Lost Notebook route, campaigns/rank):
1. Hypothesis H7: every orbit piece G_{c,o} = U_{7,c}(Σ_{a: |a| mod 7 = o} Hblk_a) of (2.18) lies in
   span{ theta products of weight 1 } ⊕ span{ q^s J_{7,j} Φ_k }, Φ_k = Σ q^{7n²}/((q^k;q⁷)_{n+1}(q^{7−k};q⁷)_n).
   (At p = 5 this held: G01 = J52 φ, G12 = −J51 φ, G31 = −J52 ψ/q, …)
2. If H7 holds, the 7 classes of F(ζ₇)·R(ζ₇) give 7 linear equations for R₀..R₆ with explicit RHS;
   solve, check R₅ = 0 (the mock Φ_k must cancel; the theta part reduces to product identities).
3. Lean: generalize RankSpec/RankP2 to base q⁷ (k = 1,2,3), F(ζ₇) dissection (Garvan 5.1), the solve.

Pipeline = proofworld: fit7.py PROPOSES identities (exact rational fit on 60 coefficients),
verification to 70+ coefficients KILLS overfits, survivors become Lean claims for the gate.

## Probe results (2026-10-03)
* **H7 holds** (fit7x.py → orbit_pieces_fit.txt): all 12 non-orbit-0 pieces G_{c,o} of (2.18) at ζ₇ are
  explicit combinations of J_{7,j}·Φ_k (k = 1..4) and theta quotients (weights ½, 1); orbit-0 pieces follow
  from the z = 1 relation. Exact reindexing identities found (reindex7.py):
  G(3,2) = M₁ = J₇,₁Φ₁;  G(0,3) = M₂ − J₇,₂;  q·G(2,1) = M₃ − J₇,₃;  q²·G(2,1) = M₄ − J₇,₄;  q³·G(0,3) = M₅ − J₇,₅
  (M_k = Σ_b w_b(q^k) Hblk_b(q⁷) = J_{7,k}Φ_k, the (2.18) specialization at z = q^k, base q⁷).
* **Explicit 7-dissection of R(ζ₇;q)** (fitR7.py → R7_dissection_fit.txt; ζ-coordinates in basis 1..ζ⁵):
  R₁ = J₇²/J₇,₁;  R₃, R₄ theta quotients (match Garvan);  R₀ ∋ Φ₁, R₂ ∋ q⁻¹Φ₃, R₆ ∋ q⁻¹Φ₂ (+ theta);  **R₅ = 0**.
  Theta parts: R₀ ~ J₇²J₇,₃/(J₇,₁J₇,₂), R₂ ~ J₇²J₇,₂/(J₇,₁J₇,₃), R₆ ~ q⁻¹(J₇²J₇,₂/J₇,₁² − J₇²J₇,₃²/(J₇,₁J₇,₂²)), …
Proof plan (guess-and-verify): R_g := these closed forms; show F(ζ₇)·R_g = H18(ζ₇) class by class using
(a) generalized orbit_mock bijections at p = 7 (k = 1..5, several prefactors), (b) F(ζ₇) 7-dissection (Garvan 5.1),
(c) the E² 7-dissection, (d) residual theta-quotient identities (to be enumerated). F unit ⇒ R = R_g ⇒ R₅ = 0.

## ⭐ Probe verdict: GO (2026-10-03)
`consist7.py`: with R_g = fitted dissection (R7_dissection_fit.txt), F(ζ₇) = J₇,₃ + q·s₁J₇,₂ − q³·s₂J₇,₁
(s₁ = ζ²+ζ³+ζ⁴+ζ⁵, s₂ = ζ³+ζ⁴; verified), E² classes (E2_dissection_fit.txt) and the 12 orbit pieces,
the class identities (F·R_g)_c = (H18(ζ₇))_c hold FORMALLY in all 7 classes — Φ₁, Φ₂, Φ₃ coefficients cancel
identically (using Φ₄ = qΦ₃+1−q, Φ₅ = q³Φ₂+1−q³, Φ₆ = q⁵Φ₁+1−q⁵) — modulo ONE theta identity:
    T :  J₇,₂³·J₇,₃ = J₇,₁·J₇,₃³ + q·J₇,₁³·J₇,₂      (checked to q³⁰⁰)
(Parser note: basis "q^{-s}·f" means q^{-s}(f − low terms); fixed in consist7.py.)

Lean architecture for rank mod 7 (all pieces identified):
 1. p = 7 versions of RankSpec (z = q^k, base q⁷) + RankP2 orbit bijections for the 12 pieces (fits list the forms).
 2. Φ_{7−k} = q^{?}Φ_k + 1 − q^{?} relations.
 3. F(ζ₇) 7-dissection (thL_dissect at p = 7) and the E² 7-dissection (from qfacInf_dissection7 / x,y,z).
 4. Identity T (likely from the x,y,z cube relations in Ramanujan7Identity.lean, or Weierstrass/quintuple).
 5. Assembly: F unit ⇒ R(ζ₇) = R_g ⇒ R₅ = 0 ⇒ cyclotomic ⇒ N(k,7,7n+5) = p(7n+5)/7.

## Lean progress + Part B structure (2026-10-03)
* DONE (RamanujanLean e6e6d38, RankMod7a.lean, axioms clean): F7_dissect (Garvan 5.1), xyz_J
  (x = J₇,₂/J₇,₁, y = J₇,₃/J₇,₂, z = J₇,₁/J₇,₃ — via F(ζ)F(ζ²)F(ζ³) = E²E(q⁷), NO quintuple product), identity_T.
* Branch maps at p = 7 (symbolic): M_k → G is type-PRESERVING (P→P, M→M), a = 7b ∓ 2k, r' = 7r + ρ;
  M₁→(3,2), M₂,M₅→(0,3), M₃,M₄→(2,1), M₆→(3,2) (shifts 0,0,−1,−2,−3,−5).
* Normalized orbit pieces (pieces_normalized.txt) contain ALL nine J₇,ⱼΦ_k:
  j = k  ← (2.18) at z = q^k;  j ≡ ±2k ← (2.20) at z = q^k [(z²q)(z⁻²q) → J₇,₂ₖ];  j ≡ ±3k ← a "z³" analogue
  ((1+z+z²)(z³q)(z⁻³q)(q)R(z;q) = Hecke series?).  Gating items: formal (2.20) and its m = 3 analogue.

## Hecke families m = 2, 3 (2026-10-03)
* hecke_m.py / hecke_template.py: H_m(z) := (1+…+z^{m−1})(z^m q)(z^{−m} q)(q)·R(z;q).
  m = 2 recovered as Garvan's (2.20) exactly (weights z^{n+1}+z^{−n} on V(n,j)). m = 3 EXISTS (sparse ±1
  z^c-coefficients = partial thetas 3t²+βt+γ, period-3 in c, symmetric c ↔ 2−c) but does NOT fit the
  (2.18)/(2.20) template — needs its own form.
* Uniform proof route: Appell–Lerch form. Per z^c:  E²·[z^c]H_m = Σ_s (−1)^s q^{C(s,2)} S_{c−ms},
  S_t = [z^t] Σ_n (−1)^n q^{n(3n+1)/2}/(1−zq^n). Verified for m = 1 (al_probe.py, c = 0..4), where
  E²·[z^c]H_1 = E·(−1)^c q^{T(c)} PT_c is ALREADY a proved object (bailey_k). NOT a pure bijection
  (#terms differ) → needs a sign-reversing involution (like RankHR2Core).
Remaining plan: (2a) AL form per z^c via involution (m = 1), then m = 2, 3 analogues; (2b) specialize at
z = q^k, base q⁷ → E(Q)·J₇,ₘₖΦ_k; (3) orbit-piece identities + assembly + cyclotomic finish.

## Option 2 (3D involution search) — NEGATIVE (2026-10-03)
cross3d.py / iso3d.py / iso3d_b.py: cross identity J₇,₁·G(1,1) = −J₇,₂·M₁ (true) is NOT a signed bijection
(518 vs 258 terms). Both sides live on the same ternary form 2Y² − 6A² + 3W² (168E + 101; b = 0 terms of M₁
sit at A = −2), but the best rational isometries (L→R sign-correct or L→L sign-reversing) cover ≤ 16/192
terms (≈ 8%). ⇒ the cross identities are not reindexing identities; they need genuine theory
(general Bailey lemma with ρ₁ = z, ρ₂ = 1/z → Appell–Lerch/Lambert form, or Hickerson–Mortenson f_{a,b,c}).
