# Dyson's rank (Atkin–Swinnerton-Dyer) — route analysis (2026-10-02)

Goal: N(k,5,5n+4) = p(5n+4)/5 and N(k,7,7n+5) = p(7n+5)/7 for all k (rank = largest part − #parts).

Verified numerically (route_numeric.py, dissect5.py):
* rank equidistribution holds (n < 40);
* Appell–Lerch form R(z;q) = (1−z)/E · Σ_{n∈ℤ} (−1)ⁿ q^{n(3n+1)/2}/(1−zqⁿ) matches the Durfee form
  Σ q^{n²}/((zq;q)_n(q/z;q)_n) (to q⁸⁰, generic z).

Reduction at ζ = ζ₅ (exact, elementary):
* 1/(1−ζqⁿ) = Σ_{k<5} ζᵏ q^{kn}/(1−q^{5n}) ⇒ (1−ζ)S(ζ) = 1 + Σ_k ζᵏ (T_k − T_{k−1}),
  T_k := Σ_{n≠0} (−1)ⁿ q^{n(3n+1)/2+kn}/(1−q^{5n}).
* Symmetry n ↦ −n: T_k = −T_{4−k} (so T₂ = 0, T₃ = −T₁, T₄ = −T₀).
* With 1/E = 𝒬·E(q²⁵)/E(q⁵)⁶ (𝒬 = Ramanujan's inversion polynomial in A, B, from
  RamanujanMostBeautiful.lean), rank equidistribution mod 5 ⇔ the two identities
      class₄(T₁·𝒬) = −1,     class₄(T₀·𝒬) = −2        (checked exactly to q^{5·78}).
  Equivalently class₄((1+5T₁)/E) = class₄((2+5T₀)/E) = 0.

Lean milestones:
1. Rank GF (combinatorial ↔ Durfee form) on Nat.Partition — like crank M2.
2. Appell–Lerch form (Bailey pair relative to z; repo has Bailey machinery).
3. THE HARD PART: the two Lambert-series identities above (ASD's theta-function content).
   Candidate routes: (a) lattice reindexing of indefinite (Hecke-type) cone sums, as for Winquist;
   (b) Hickerson–Mortenson n-dissection of Appell–Lerch sums m(x,q,z);
   (c) ASD's own functional-equation argument (hard to do formally).
4. Mod 7 analogue (T_k with 1/(1−q^{7n}), 7-dissection 𝒬₇ from Ramanujan7Identity.lean).

## Probe (option 1), 2026-10-02 — result: NEGATIVE for the lattice route
* Exponent residues: T₁ lives in classes {0,3,4} mod 5, T₀ in {0,1,2}. The targets become 3-term relations
  5T₁⁽⁰⁾ + (A³+2QB²)T₁⁽³⁾ + (A⁴−3QB)T₁⁽⁴⁾ = −1,  5T₀⁽⁰⁾ + (3A+QB⁴)T₀⁽¹⁾ + (2A²−QB³)T₀⁽²⁾ = −2.
* None of the six components is an eta quotient (product exponents c_n not periodic; dissect5.py),
  so each mixes a genuine mock (Appell–Lerch, level 3) part with theta parts; the identity holds only
  because the mock parts cancel across components. Clearing denominators gives "θ⁶ × cone-sum = θ⁶"
  in ~7 dimensions — no reindexing proof in sight.
* Literature route (Hickerson–Mortenson, arXiv 1309.1562): needs m(x,q,z) with the z-change identity
  (2.1d) (classically proved by elliptic-function pole/periodicity arguments) and their n-dissection
  Theorems 2.2/2.3, then closing theta identities. A formal-power-series proof of (2.1d) is the
  gating unknown.

## Better route found (2026-10-02): Garvan, "A new approach to the Dyson rank conjectures" (arXiv 2012.06676)
Avoids Appell–Lerch "change-of-z" entirely for (2.18). Proof of mod 5 needs:
* (2.18) (zq)(q/z)(q)·R(z;q) = Hecke–Rogers double sum HR(z)   [Garvan 2015, arXiv 1402.1884, Bailey pairs]
* (2.19) (zq)(q/z)(q)·R(z;q²) = Σ_{n≥0}Σ_{|j|≤n} (−1)^j z^j q^{n(3n+1)/2−j²}(1−q^{2n+1})   [Bradley-Thrush Thm 7.5]
* JTP-level 5-dissections (Lemma 3.1/3.2), then a 3×3 linear solve ⇒ R₃ = 0 ⇒ U_{5,3}R(ζ,q²) = 0 ⇔ mod-5 rank.
Verified numerically (hr_numeric.py): E·R_k = G_k + δ_{k0}E with G_k = Σ_{n≥1}(−1)^{n−1}q^{n(3n−1)/2+|k|n}(1−qⁿ),
and j(z)·C(z) = (1−z)·E·HR(z).
Key observation: with (zq)(q/z)(q) = Σ_{m≥1}(−1)^{m+1}q^{m(m−1)/2}Σ_{|i|<m}zⁱ, the z^a-coefficient of (2.18)/(2.19)
times E becomes an identity between explicit 2-D cone sums (the (1−qⁿ) factor cancels the geometric sum over i),
i.e. "pentagonal × partial theta = Hecke-type double sum" — lattice-framework territory (Garvan proves it with a
Bailey pair at a = q^k, Prop. 2.4/2.5 + limiting Bailey lemma; the repo has Bailey machinery).
Remaining combinatorial link: the rank-k GF formula (via Appell–Lerch form, or Dyson's map).

## Step 1 DONE in Lean (RankBailey.lean, 279 lines, axioms clean) — 2026-10-02
* Per-a form of (2.18) found and verified: [z^a]((zq)(q/z)(q)R(z;q)) = (−1)^a q^{a(a+1)/2}·E·Σ_j q^{j²+aj}β*_j,
  β*_j = Σ_{l≤j} q^l/((q)_l(q)_{l+a})  (from the Durfee form of R + Euler expansions; peradurfee.py).
* `SL_eq_SR` / `SLz_eq_SRz`: (α*, β*) is a Bailey pair relative to a = q^k, α*_r = q^{2r²+2kr+r} − q^{2r²+2kr−r−k}
  (telescoping certificate g(r) = −q^{n+1−r}B_r/((q)_{n+1−r}(q)_{n+r+k}); proved in any field, transferred via FractionRing).
* `bailey_k`: Σ_j q^{j²+kj}β*_j = (1/(q)_∞)·Σ_r q^{r²+kr}α*_r (coefficientwise; inner sums = `durfee_rect_base (2r+k)`).
Next: (a) rank GF Durfee form on Nat.Partition + [z^a] extraction ⇒ per-a (2.18); (b) a proof route for (2.19) per-a;
(c) U_{5,2/3/4} dissections + 3×3 solve ⇒ R₃ = 0; (d) cyclotomic ⇒ equidistribution.

## Step 2 core DONE in Lean (RankHR2Core.lean, axioms clean) — 2026-10-02
Per-a form of (2.19), derived from the Durfee form of R(z;q²) + the PROVED per-b (2.18) in base q²:
  Σ_{b∈ℤ} q^{(a−b)²+|b|(|b|+1)} PT_{|b|}(q²) = ψ(q)·q^{−a²}·Σ_{n≥|a|} q^{n(3n+1)/2}(1−q^{2n+1})   (hr2_reduced.py)
In coordinates (U,V) with 24e+4+24a² = 3U²+V² this is a signed lattice identity; proof = explicit involution τ on ℤ²:
  L-points ↦ ±rot(p) (rotation by (1+√−3)/2), remaining R-points paired by rot^{±1} chosen by V mod 4 (tau.py).
`tau_spec` (involution + sign flip, by ~12 omega lemmas), `tau_nrm3` (norm preserved), `lat_wD_zero`.
Remaining for (2.19): convert lat(w_R) and lat(w_L) to the two series (injective reparametrizations), then the
Durfee/Euler extraction linking them to (zq)(q/z)(q)R(z;q²).

## Rank generating function DONE in Lean (RankGF.lean, 445 lines, axioms clean) — 2026-10-02
`rank_durfee`: Σ_{λ⊢n} z^{rank λ} = [qⁿ] Σ_k q^{k²}/((zq;q)_k(q/z;q)_k) for all z ∈ ℂˣ (rank on Mathlib Nat.Partition).
Route: largest-part split via genFun (like the crank), then Durfee = largest-part form by
  G2 (truncated q-binomial expansion of 1/(xq;q)_k, induction + q-Pascal),
  G3 (Σ_j q^{j²+j}y^j[n,j](yq^{j+2};q)_{n−j} = 1: polynomial in y vanishing at y = q^b by the repo's F_eq_one),
  G4 (regroup by m = k + j, sum_range_diag_flip, Gaussian symmetry).
Remaining: (i) F(z)·D(z) = Σ_a z^a c_a via Euler expansions + `bailey_k`; (ii) same for R(z;q²) + `lat_wD_zero`;
(iii) evaluation at ζ₅, 5-dissections, 3×3 solve, cyclotomic ⇒ N(k,5,5n+4) = p(5n+4)/5.

## (2.18) and (2.19) DONE in Lean (axioms clean) — 2026-10-02
* RankHR1.lean `hecke_rogers_18`: [qⁿ] (zq)(q/z)·R(z;q) = Σ_{a≤n} w_a(z)·[qⁿ] (−1)^a q^{a(a+1)/2} ψ(LHSz a)
  (w_0 = 1, w_a = z^a + z^{−a}); Euler2 truncated + `sum_regroup` + `bailey_k` route.
* RankHR2Lat.lean: finite lattice identity per exponent k (`counts_cancel`, via `Finset.sum_involution` with τ),
  and the four parametrizations L±/R± ↔ point classes (`card_LP/LM/RP/RM`, `param_counts`).
* RankHR2Series.lean `lattice_trunc`: Σ_{|b|≤N} q^{(a−b)²+|b|(|b|+1)} PT_{|b|}(q²) ≡ ψ(q)·G_a (mod q^{N+1}),
  G_a = Σ_t q^{a(a+1)/2+3at+t(3t+1)/2}(1−q^{2a+2t+1}).
* RankHR2.lean `hecke_rogers_19`: [qⁿ] (q)(zq)(q/z)·R(z;q²) = Σ_{a≤n} w_a(z)·[qⁿ] (−1)^a G_a.
  Pieces: E2C (q↦q² over ℂ), OddInf c = (cq;q²)_∞ with odd Euler (`odd_euler_dvd`), odd JTP
  (`odd_jtp_coeff`: (zq;q²)(q/z;q²) = Σ w_a (−1)^a q^{a²}/(q²;q²)), generic `wz_prod` (convolution of two
  w-expansions over ℤ), `block_c` (Bailey + lattice + ψ = (q²;q²)²/(q;q)).
Next: endgame at ζ = ζ₅.

## ⭐ DYSON'S RANK CONJECTURE MOD 5 PROVED in Lean (2026-10-02/03), axioms clean
`CrankProof.rank_equidistribution_mod5 : 5 * #{λ ⊢ 5n+4 : rank λ ≡ i (5)} = p(5n+4)` (RankMod5.lean).
New files: RankTheta (step-a Pochhammers Pinf, Euler, J_{a,b} JTP `jtp_ab`), RankDissect (lattice theta,
generic 5-dissection `thL_dissect`, F(ζ) dissection, θ₄ dissection, E·E(q⁵)=F(ω)F(ω²), J52 = α J51),
RankMod5 (class selection via ZMod 5 `decide`, z=1 evaluations, eqs (3.15)-(3.17), product ids (3.19),
3×3 solve by one linear_combination + constant-term nonvanishing (s+1 ≠ 0), cyclotomic counting).

## Mod 7 status (2026-10-03): Garvan's route is OPEN (his Problem 1)
mod7_probe.py / mod7_pieces.py: at ζ₇ the only single-orbit theta-product classes are (2.18) class 4 and (2.19)
class 2; the latter involves R₁,R₃,R₄ only. Every class containing R₅ also contains a mock component (R₀/R₂/R₆).

## Ramanujan's Lost Notebook identity (4.1), mod 5
DONE (RankRamanujan5.lean, axioms clean): R₄ = 0, R₁ = J₅²/J_{5,1}, R₂ = (ζ+ζ⁴)J₅²/J_{5,2} (from the mod-5 solve
+ prod_id1/2). REMAINING R₀, R₃ (mock parts φ, ψ): r0_probe.py shows class 0 of (2.18) at ζ = theta part + 4·J_{5,2}·φ
(orbit-1 piece). Needs an identity tying a Hecke sub-sum to φ = R(q;q⁵)-type series. Garvan uses (2.20) at z = q, base q⁵.
NOTE: (2.20) does NOT factor as θ(z)·(2.18) — (z²q;q) ≠ (zq)(−zq); that factorization is (z²q²;q²), i.e. (2.21).
Infrastructure needed: (a) (2.20) (likely via Appell–Lerch form of R(z;q)); (b) two-variable lift ℤ[z,z⁻¹]⟦q⟧
(Laurent-polynomial-agree-on-ℂˣ ⇒ equal); (c) specialization z ↦ q^k, q ↦ q^5 on z-bounded series. Multi-session.
