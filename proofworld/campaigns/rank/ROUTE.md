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
