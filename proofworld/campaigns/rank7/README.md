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
