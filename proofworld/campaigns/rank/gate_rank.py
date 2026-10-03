"""Certify the rank campaign through the proofworld gate (planted controls + axiom audit + ledger)."""
import sys, os, json
sys.path.insert(0, os.path.expanduser("~/bla"))
from proofworld.gate import Claim, check

PRE = "import RamanujanTau.RankR03\n"
OPENS = "open PowerSeries Finset\n"
C = lambda n, s, p: Claim(n, s, p, trusted=True)
claims = [
    # the statistic, spelled out: rank = largest part − number of parts
    C("rank_is_dyson", "{n : ℕ} (l : n.Partition) : CrankProof.rank l = (CrankProof.largest l : ℤ) - (l.parts.card : ℤ)", "rfl"),
    C("rank_gf_durfee", "{z : ℂ} (hz : z ≠ 0) (n : ℕ) : ∑ l : n.Partition, z ^ CrankProof.rank l = coeff n (CrankProof.Dser z z⁻¹)",
      "CrankProof.rank_durfee hz n"),
    C("dyson_rank_mod5", "(n : ℕ) {i : ℕ} (hi : i < 5) : 5 * (univ.filter fun l : (5 * n + 4).Partition => CrankProof.rank l % 5 = i).card = Fintype.card (5 * n + 4).Partition",
      "CrankProof.rank_equidistribution_mod5 n hi"),
    C("garvan_2_18", "{z : ℂ} (hz : z ≠ 0) (n : ℕ) : coeff n (CrankProof.pochInf z 1 * CrankProof.pochInf z⁻¹ 1 * CrankProof.Dser z z⁻¹) = ∑ a ∈ range (n + 1), CrankProof.wz z a * coeff n (C ((-1) ^ a) * X ^ (a * (a + 1) / 2) * MockTheta5.JTP.ψC (RankProof.LHSz a))",
      "CrankProof.hecke_rogers_18 hz n"),
    C("garvan_2_19", "{z : ℂ} (hz : z ≠ 0) (n : ℕ) : coeff n (CrankProof.pochInf 1 1 * CrankProof.pochInf z 1 * CrankProof.pochInf z⁻¹ 1 * CrankProof.E2C (CrankProof.Dser z z⁻¹)) = ∑ a ∈ range (n + 1), CrankProof.wz z a * coeff n (MockTheta5.JTP.ψC (C ((-1 : ℤ) ^ a) * RankProof.Gsum a n))",
      "CrankProof.hecke_rogers_19 hz n"),
    C("lost_notebook_rank_dissection",
      ": CrankProof.Rk 0 = MockTheta5.JTP.ψC (MockTheta5.JTP.eQ ^ 2 * CrankProof.Jab 5 2 * Ring.inverse (CrankProof.Jab 5 1) ^ 2) + C (CrankProof.tζ - 2) * MockTheta5.JTP.ψC (RankProof.Phi 1 - 1) ∧ CrankProof.Rk 1 = MockTheta5.JTP.ψC (MockTheta5.JTP.eQ ^ 2 * Ring.inverse (CrankProof.Jab 5 1)) ∧ CrankProof.Rk 2 = C CrankProof.tζ * MockTheta5.JTP.ψC (MockTheta5.JTP.eQ ^ 2 * Ring.inverse (CrankProof.Jab 5 2)) ∧ X * CrankProof.Rk 3 = C (CrankProof.sζ - CrankProof.tζ) * MockTheta5.JTP.ψC (RankProof.Phi 2 - 1) + C (CrankProof.sζ + 1) * X * MockTheta5.JTP.ψC (MockTheta5.JTP.eQ ^ 2 * CrankProof.Jab 5 1 * Ring.inverse (CrankProof.Jab 5 2) ^ 2) ∧ CrankProof.Rk 4 = 0",
      "CrankProof.lost_notebook_rank_mod5"),
    C("Rk_is_dissection", "(k : ℕ) : CrankProof.Rk k = PowerSeries.mk fun n => coeff (5 * n + k) (CrankProof.Dser MockTheta5.JTP.ω5 MockTheta5.JTP.ω5⁻¹)", "rfl"),
    C("phi_def", "(k c : ℕ) : coeff c (RankProof.Phi k) = coeff c (∑ n ∈ range (c + 1), X ^ (5 * n ^ 2) * Ring.inverse (RankProof.Pfin k 5 (n + 1)) * Ring.inverse (RankProof.Pfin (5 - k) 5 n))",
      "by rw [RankProof.Phi, coeff_mk]; rfl"),
    C("orbit_mock_phi", ": X ^ (1 - 1) * MockTheta5.JTP.dis5 (2 * 1 - 1) (RankProof.Gorb 1) = CrankProof.Jab 5 1 * (1 - RankProof.Phi 1)",
      "RankProof.orbit_mock (Or.inl rfl)"),
    C("orbit_mock_psi", ": X ^ (2 - 1) * MockTheta5.JTP.dis5 (2 * 2 - 1) (RankProof.Gorb 2) = CrankProof.Jab 5 2 * (1 - RankProof.Phi 2)",
      "RankProof.orbit_mock (Or.inr rfl)"),
]
res = check(claims, preamble=PRE, project="~/RamanujanTau", opens=OPENS, timeout=1800, tag="rank-mod5")
for n, v in res.items():
    print(f"{v.status:8s} {n:32s} axioms={v.axioms} {v.detail[:200]}")
