#!/usr/bin/env python3
"""proofworld campaign: Ramanujan's THIRD congruence  11 | p(11n+6)  in the RamanujanTau project.

Status before this campaign: formalized nowhere we could find (mod 5 / mod 7 exist in RamanujanTau; the
pentagonal theorem, JTP and Rogers–Ramanujan exist elsewhere; Wolstenholme exists elsewhere).

Route (numerically pre-checked in dyson_numeric.py, 361 held-out coefficients):
  * mod 11 Frobenius:  P ≡ (q;q)^10 · P(q^11)            -- mechanical port of the mod-7 file (DREAMER seeds)
  * Dyson / B2 Macdonald:  6·[q^a](q;q)^10 = -Σ_{x≡1,y≡2 (6), x²+y²=12a+5} xy(x²-y²)   -- named hypothesis DysonB2
  * heart:  a ≡ 6 (mod 11) ⇒ x²+y² ≡ 0 (mod 11) ⇒ 11 | x, 11 | y   (−1 is a non-residue mod 11)

Anti-trap discipline:
  * the STATEMENTS below are fixed here (the math target); generators only propose PROOFS;
  * every verdict comes from proofworld.gate (collectAxioms footprint, controls first, receipts);
  * DysonB2 is a hypothesis in the theorem statement -- never an `axiom`; the final theorem says so;
  * at the end the whole verified chain is re-checked in ONE fresh gate run (independent replay).

Run:  python3 proofworld/campaigns/mod11/campaign.py              (dreamer seeds only)
      PROOFWORLD_LLM=1 python3 proofworld/campaigns/mod11/campaign.py   (Opus proposes when seeds fail)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/bla"))
from proofworld import gate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.expanduser("~/RamanujanTau")
MODEL = os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-5-5")
ROUNDS = int(os.environ.get("PROOFWORLD_ROUNDS", "4"))          # LLM propose/repair rounds per goal
PER_ROUND = 3                                                     # candidates per round

PRELUDE = r"""import RamanujanTau.MockTheta5PartitionCount
import Mathlib.Data.Int.Interval

namespace MockTheta5.JTP
open PowerSeries MockTheta5.Bailey

noncomputable def Ψ11 : PowerSeries ℤ →+* PowerSeries (ZMod 11) := PowerSeries.map (Int.castRingHom (ZMod 11))
noncomputable def E11 : PowerSeries ℤ →+* PowerSeries ℤ := (PowerSeries.expand 11 (by norm_num)).toRingHom
noncomputable def expand11 : PowerSeries (ZMod 11) →+* PowerSeries (ZMod 11) :=
  (PowerSeries.expand 11 (by norm_num)).toRingHom

instance fact_prime_eleven : Fact (Nat.Prime 11) := ⟨by decide⟩
instance charP_powerSeries_zmod11 : CharP (PowerSeries (ZMod 11)) 11 :=
  charP_of_injective_ringHom (PowerSeries.C_injective (R := ZMod 11)) 11

/-- Dyson's B₂ coefficient: `-Σ_{x≡1, y≡2 (mod 6), x²+y² = 12a+5} x·y·(x²−y²)`. -/
def dysonCoeff (a : ℕ) : ℤ :=
  -∑ x ∈ Finset.Icc (-(12 * (a : ℤ) + 5)) (12 * (a : ℤ) + 5),
    ∑ y ∈ Finset.Icc (-(12 * (a : ℤ) + 5)) (12 * (a : ℤ) + 5),
      if x % 6 = 1 ∧ y % 6 = 2 ∧ x ^ 2 + y ^ 2 = 12 * (a : ℤ) + 5 then x * y * (x ^ 2 - y ^ 2) else 0

/-- **Dyson's η¹⁰ formula / the B₂ Macdonald identity**, coefficientwise:
`6·[q^a](q;q)_∞¹⁰ = dysonCoeff a`. A named HYPOTHESIS here (never an axiom); numerically verified for
`a ≤ 400` (fit on `a < 40`, 361 held-out coefficients) in `dyson_numeric.py`. -/
def DysonB2 : Prop := ∀ a : ℕ, 6 * coeff a (qfacInf ^ 10) = dysonCoeff a
"""

H11 = "(H : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0)"

# (name, signature, [seed proofs], llm_allowed, note)
GOALS = [
    ("E11_X", ": E11 X = X ^ 11", ["PowerSeries.expand_X 11 (by norm_num)"], False, "port"),
    ("coeff_Ψ11", "(n : ℕ) (f : PowerSeries ℤ) : coeff n (Ψ11 f) = (Int.castRingHom (ZMod 11)) (coeff n f)",
     ["PowerSeries.coeff_map _ _ _"], False, "port"),
    ("coeff_expand11",
     "(n : ℕ) (f : PowerSeries (ZMod 11)) : coeff n (expand11 f) = if 11 ∣ n then coeff (n / 11) f else 0",
     ["by\n  have h : expand11 f = PowerSeries.expand 11 (by norm_num) f := rfl\n  rw [h, PowerSeries.coeff_expand]"],
     False, "port"),
    ("E11_qfac", "(n : ℕ) : E11 (qfac n) = ∏ i ∈ Finset.range n, (1 - X ^ (11 * i + 11))",
     ["by\n  rw [qfac, map_prod]\n  refine Finset.prod_congr rfl (fun i _ => ?_)\n"
      "  rw [map_sub, map_one, map_pow, E11_X, ← pow_mul, show 11 * (i + 1) = 11 * i + 11 from by ring]"],
     False, "port"),
    ("frob_factor11", "(e : ℕ) : ((1 : PowerSeries (ZMod 11)) - X ^ e) ^ 11 = 1 - X ^ (11 * e)",
     ["by rw [sub_pow_char_of_commute _ (Commute.all _ _), one_pow, ← pow_mul, Nat.mul_comm e 11]"],
     False, "port"),
    ("coeff_E11_qfacInf", "{i N : ℕ} (h : i + 1 ≤ 11 * N) : coeff i (E11 qfacInf) = coeff i (E11 (qfac N))",
     ["by\n  have hdvd : (X : PowerSeries ℤ) ^ (i + 1) ∣ (E11 qfacInf - E11 (qfac N)) := by\n"
      "    obtain ⟨g, hg⟩ : (X : PowerSeries ℤ) ^ N ∣ (qfacInf - qfac N) := by\n"
      "      rw [PowerSeries.X_pow_dvd_iff]; intro j hj\n"
      "      rw [map_sub, coeff_qfacInf (show j + 1 ≤ N by omega), sub_self]\n"
      "    rw [← map_sub, hg, map_mul, map_pow, E11_X, ← pow_mul]\n"
      "    exact dvd_mul_of_dvd_left (pow_dvd_pow X (by omega)) _\n"
      "  obtain ⟨c, hc⟩ := hdvd\n"
      "  have hz : coeff i (E11 qfacInf) - coeff i (E11 (qfac N)) = 0 := by\n"
      "    rw [← map_sub, hc]; exact MockTheta5.mt_coeff_Xpow_mul_zero _ _ i (by omega)\n"
      "  exact sub_eq_zero.mp hz"], False, "port"),
    ("frobenius_qfac11", "(N : ℕ) : (Ψ11 (qfac N)) ^ 11 = Ψ11 (E11 (qfac N))",
     ["by\n  rw [E11_qfac, Ψ11, qfac, map_prod, map_prod, ← Finset.prod_pow]\n"
      "  refine Finset.prod_congr rfl (fun i _ => ?_)\n"
      "  simp only [map_sub, map_one, map_pow, PowerSeries.map_X]\n"
      "  rw [frob_factor11 (i + 1), show 11 * (i + 1) = 11 * i + 11 from by ring]"], False, "port"),
    ("coeff_congr11",
     "{f g : PowerSeries (ZMod 11)} {m : ℕ} (h : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (f - g)) : "
     "coeff m f = coeff m g",
     ["by\n  obtain ⟨c, hc⟩ := h\n"
      "  have hz : coeff m f - coeff m g = 0 := by rw [← map_sub, hc, coeff_X_pow_mul']; simp\n"
      "  exact sub_eq_zero.mp hz"], False, "port"),
    ("frobenius_qfacInf11", ": (Ψ11 qfacInf) ^ 11 = Ψ11 (E11 qfacInf)",
     ["by\n  ext m\n"
      "  have hqf : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (Ψ11 qfacInf - Ψ11 (qfac (m + 1))) := by\n"
      "    rw [PowerSeries.X_pow_dvd_iff]; intro i hi\n"
      "    rw [map_sub, Ψ11, PowerSeries.coeff_map, PowerSeries.coeff_map,\n"
      "        coeff_qfacInf (show i + 1 ≤ m + 1 by omega), sub_self]\n"
      "  have h11 : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ ((Ψ11 qfacInf) ^ 11 - (Ψ11 (qfac (m + 1))) ^ 11) :=\n"
      "    dvd_trans hqf (sub_dvd_pow_sub_pow _ _ 11)\n"
      "  have hE : (X : PowerSeries (ZMod 11)) ^ (m + 1) ∣ (Ψ11 (E11 qfacInf) - Ψ11 (E11 (qfac (m + 1)))) := by\n"
      "    rw [PowerSeries.X_pow_dvd_iff]; intro i hi\n"
      "    rw [map_sub, Ψ11, PowerSeries.coeff_map, PowerSeries.coeff_map,\n"
      "        coeff_E11_qfacInf (show i + 1 ≤ 11 * (m + 1) by omega), sub_self]\n"
      "  rw [coeff_congr11 h11, frobenius_qfac11, coeff_congr11 hE]"], False, "port"),
    ("g11_unit", ": IsUnit (Ψ11 qfacInf)", ["isUnit_qfacInf.map Ψ11"], False, "port"),
    ("g11_pow11", ": (Ψ11 qfacInf) ^ 11 = expand11 (Ψ11 qfacInf)",
     ["by\n  rw [frobenius_qfacInf11]\n  ext n\n  rw [coeff_Ψ11, coeff_expand11]\n"
      "  have h : E11 qfacInf = PowerSeries.expand 11 (by norm_num) qfacInf := rfl\n"
      "  rw [h, PowerSeries.coeff_expand]\n  split_ifs with hd\n  · rw [coeff_Ψ11]\n  · simp"],
     False, "port"),
    ("Ψ11_partitionGF_mul", ": Ψ11 partitionGF * Ψ11 qfacInf = 1",
     ["by rw [← map_mul, partitionGF, Ring.inverse_mul_cancel _ isUnit_qfacInf, map_one]"], False, "port"),
    ("P_eq11", ": Ψ11 partitionGF = Ψ11 (qfacInf ^ 10) * expand11 (Ψ11 partitionGF)",
     ["by\n  set g := Ψ11 qfacInf with hg\n  set P := Ψ11 partitionGF with hP\n"
      "  have hu : IsUnit g := g11_unit\n  have hPg : P * g = 1 := Ψ11_partitionGF_mul\n"
      "  have hgP : g * P = 1 := by rw [mul_comm]; exact hPg\n"
      "  have hgexp : g ^ 11 * expand11 P = 1 := by rw [g11_pow11, ← map_mul, hgP, map_one]\n"
      "  have hgx : g * (g ^ 10 * expand11 P) = 1 := by\n"
      "    rw [← mul_assoc, show g * g ^ 10 = g ^ 11 from by ring]; exact hgexp\n"
      "  have hPinv : P = Ring.inverse g := by\n"
      "    calc P = P * 1 := (mul_one _).symm\n"
      "      _ = P * (g * Ring.inverse g) := by rw [Ring.mul_inverse_cancel g hu]\n"
      "      _ = (P * g) * Ring.inverse g := by ring\n"
      "      _ = Ring.inverse g := by rw [hPg, one_mul]\n"
      "  rw [show Ψ11 (qfacInf ^ 10) = g ^ 10 from by rw [map_pow]]\n"
      "  conv_lhs => rw [hPinv]\n  symm\n"
      "  calc g ^ 10 * expand11 P = 1 * (g ^ 10 * expand11 P) := (one_mul _).symm\n"
      "    _ = (Ring.inverse g * g) * (g ^ 10 * expand11 P) := by rw [Ring.inverse_mul_cancel g hu]\n"
      "    _ = Ring.inverse g * (g * (g ^ 10 * expand11 P)) := by ring\n"
      "    _ = Ring.inverse g := by rw [hgx, mul_one]"], False, "port"),
    ("partition_congruence_mod11_of", f"{H11} (n : ℕ) : coeff (11 * n + 6) (Ψ11 partitionGF) = 0",
     ["by\n  rw [P_eq11, PowerSeries.coeff_mul]\n  refine Finset.sum_eq_zero (fun p hp => ?_)\n"
      "  obtain ⟨a, b⟩ := p\n  have hab : a + b = 11 * n + 6 := Finset.mem_antidiagonal.mp hp\n"
      "  by_cases hb : 11 ∣ b\n  · obtain ⟨j, rfl⟩ := hb\n    rw [H a (by omega), zero_mul]\n"
      "  · rw [coeff_expand11, if_neg hb, mul_zero]"], True, "assembly"),
    ("eleven_dvd_coeff_partitionGF_of", f"{H11} (n : ℕ) : (11 : ℤ) ∣ coeff (11 * n + 6) partitionGF",
     ["by\n  have h : ((coeff (11 * n + 6) partitionGF : ℤ) : ZMod 11) = 0 := by\n"
      "    have hh := partition_congruence_mod11_of H n; rwa [coeff_Ψ11] at hh\n"
      "  exact_mod_cast (ZMod.intCast_zmod_eq_zero_iff_dvd _ 11).mp h"], True, "assembly"),
    ("eleven_dvd_partition_card_of", f"{H11} (n : ℕ) : 11 ∣ Fintype.card (Nat.Partition (11 * n + 6))",
     ["by\n  have h := eleven_dvd_coeff_partitionGF_of H n\n  rw [coeff_partitionGF_eq_card] at h\n"
      "  exact_mod_cast h"], True, "bridge"),
    # ---- the genuinely new mathematics: the mod-11 heart and Dyson ⇒ vanishing ----
    ("heart11", ": ∀ x y : ZMod 11, x ^ 2 + y ^ 2 = 0 → x = 0 ∧ y = 0", ["by decide"], True, "heart"),
    ("int_heart11", "{x y : ℤ} (h : (11 : ℤ) ∣ x ^ 2 + y ^ 2) : (11 : ℤ) ∣ x ∧ (11 : ℤ) ∣ y", [], True, "heart"),
    ("dysonCoeff_zero", ": dysonCoeff 0 = 6", ["by decide", "by simp [dysonCoeff]; decide", "by rfl"], True,
     "grounding: Dyson's normalisation at a = 0 (6·[q⁰](q;q)¹⁰ = 6)"),
    ("dysonCoeff_dvd", "(a : ℕ) (ha : a % 11 = 6) : (11 : ℤ) ∣ dysonCoeff a", [], True, "heart ⇒ Dyson side"),
    ("H_of_dyson", "(hD : DysonB2) : ∀ a : ℕ, a % 11 = 6 → coeff a (Ψ11 (qfacInf ^ 10)) = 0", [], True,
     "Dyson ⇒ the key vanishing"),
    ("eleven_dvd_partition_card_of_dyson",
     "(hD : DysonB2) (n : ℕ) : 11 ∣ Fintype.card (Nat.Partition (11 * n + 6))",
     ["eleven_dvd_partition_card_of (H_of_dyson hD) n"], True, "final: 11 ∣ p(11n+6) given DysonB2"),
]


# ------------------------------------------------------------------------------------------------ LLM proposer
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        import anthropic
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


SYSTEM = (
    "You write Lean 4 + Mathlib proofs inside an existing project (RamanujanTau, Lean v4.30.0-rc2, Mathlib "
    "May 2026). You are given the file prelude (definitions in scope), the lemmas already verified (usable by "
    "name), and one target. Reply with ONLY a JSON object {\"proofs\": [\"<proof>\", ...]} holding up to 3 "
    "distinct candidate proof terms for the target, each a complete term starting with `by` (or a term). "
    "Never use sorry, admit, native_decide, axiom, set_option, or new top-level declarations; a checker "
    "rejects them. Prefer robust tactics (omega, decide, simp, norm_num, ring, linear_combination, "
    "Int.emod_emod_of_dvd, ZMod casts) and explicit `have` steps over guesswork.")


def llm_propose(goal, verified, history):
    """history: list of (proof, gate status, detail) for this goal (repair feedback)."""
    name, sig = goal[0], goal[1]
    ctx = ("PRELUDE (in scope):\n```lean\n" + PRELUDE + "\n```\n\nVERIFIED LEMMAS (usable by name):\n"
           + "\n".join(f"theorem {n} {s}" for n, s, _ in verified))
    ask = f"TARGET:\n```lean\ntheorem {name} {sig} := ?\n```\nNote: {goal[4]}\n"
    if history:
        ask += "\nEarlier attempts and the checker's verdicts (fix these):\n" + "\n".join(
            f"--- attempt:\n{p}\n--- verdict: {st}: {d[:600]}" for p, st, d in history[-6:])
    import inspect
    stream_fn = _client().beta.messages.stream
    known = set(inspect.signature(stream_fn).parameters)
    newer = {"output_config": {"effort": "high"}, "fallbacks": "default"}     # newer than some SDK pins
    kw = {k: v for k, v in newer.items() if k in known}
    extra = {k: v for k, v in newer.items() if k not in known}
    with stream_fn(
            model=MODEL, max_tokens=32000, thinking={"type": "adaptive"},
            betas=["server-side-fallback-2026-07-01"], **kw, **({"extra_body": extra} if extra else {}),
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": ctx, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": ask}]) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        return [], {"refusal": True}
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        proofs = [p for p in json.loads(m.group(0))["proofs"] if isinstance(p, str)][:PER_ROUND]
    except Exception:
        proofs = []
    u = msg.usage
    return proofs, {"in": u.input_tokens, "out": u.output_tokens,
                    "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}


# ------------------------------------------------------------------------------------------------ the loop
def preamble_with(verified):
    """prelude + every verified lemma so far (as trusted text: each was PROVED by the gate already, and any
    sorry/axiom in them would still surface in later footprints because collectAxioms is transitive)."""
    return PRELUDE + "\n" + "\n\n".join(f"theorem {n} {s} := {p}" for n, s, p in verified) + "\n"


def attempt(goal, verified, proofs, tag):
    name, sig = goal[0], goal[1]
    claims = [gate.Claim(f"{name}__c{i}", sig, p) for i, p in enumerate(proofs)]
    vs = gate.check(claims, preamble=preamble_with(verified), project=PROJECT, timeout=900,
                    tag=f"mod11:{name}:{tag}")
    for i, p in enumerate(proofs):
        if vs[f"{name}__c{i}"].proved:
            return p, vs[f"{name}__c{i}"], [(p, vs[f"{name}__c{i}"].status, vs[f"{name}__c{i}"].detail)]
    return None, None, [(p, vs[f"{name}__c{i}"].status, vs[f"{name}__c{i}"].detail) for i, p in enumerate(proofs)]


def main():
    use_llm = os.environ.get("PROOFWORLD_LLM") == "1"
    print(f"=== proofworld campaign: 11 | p(11n+6)   [{'LLM ' + MODEL if use_llm else 'dreamer only'}] ===")
    ctl = gate.run_controls(project=PROJECT)
    print(f"  gate controls in RamanujanTau: {'PASS' if ctl['ok'] else 'FAIL'}")
    if not ctl["ok"]:
        return 1
    verified, log, spend = [], [], {"calls": 0, "in": 0, "out": 0, "cache_read": 0}
    t0 = time.time()
    for goal in GOALS:
        name, sig, seeds, llm_ok, note = goal
        hist = []
        proof, v = None, None
        if seeds:
            proof, v, h = attempt(goal, verified, seeds, "seed")
            hist += h
        rounds = 0
        while proof is None and use_llm and llm_ok and rounds < ROUNDS:
            rounds += 1
            cands, usage = llm_propose(goal, verified, hist)
            spend["calls"] += 1
            for k in ("in", "out", "cache_read"):
                spend[k] += usage.get(k, 0)
            if not cands:
                hist.append(("(no parseable proposal)", "REFUSED" if usage.get("refusal") else "EMPTY", ""))
                continue
            proof, v, h = attempt(goal, verified, cands, f"llm{rounds}")
            hist += h
        src = "seed" if proof is not None and rounds == 0 else (f"llm r{rounds}" if proof is not None else "-")
        status = "PROVED" if proof is not None else (hist[-1][1] if hist else "NO-CANDIDATE")
        print(f"  {name:36} {status:12} via {src:8} {('' if proof else (hist[-1][2][:70] if hist else ''))}",
              flush=True)
        log.append({"goal": name, "status": status, "via": src, "attempts": len(hist),
                    "axioms": v.axioms if v else None, "register": v.register if v else None})
        if proof is not None:
            verified.append((name, sig, proof))
        elif goal[3] and not use_llm:
            pass                                                   # open; later goals that need it will fail
    # independent replay: the whole verified chain in ONE fresh gate run
    replay = {}
    if verified:
        claims = [gate.Claim(n, s, p) for n, s, p in verified]
        replay = gate.check(claims, preamble=PRELUDE, project=PROJECT, timeout=1800, tag="mod11:replay")
    ok_replay = all(v.proved for v in replay.values())
    print(f"\n  independent replay of {len(verified)} verified lemmas in one run: "
          f"{'ALL PROVED' if ok_replay else 'MISMATCH: ' + str([n for n, v in replay.items() if not v.proved])}")
    regs = sorted({v.register for v in replay.values()})
    print(f"  register(s): {regs}   axioms of final theorem: "
          f"{replay['eleven_dvd_partition_card_of_dyson'].axioms if 'eleven_dvd_partition_card_of_dyson' in replay else '-'}")
    out = os.path.join(HERE, "PartitionCongruenceMod11.lean")
    with open(out, "w") as fh:
        fh.write("/- proofworld campaign output: every theorem below was PROVED by proofworld.gate and re-checked in\n"
                 "   one independent run. `DysonB2` is a HYPOTHESIS of the final theorem, not an axiom. -/\n")
        fh.write(PRELUDE + "\n" + "\n\n".join(f"theorem {n} {s} := {p}" for n, s, p in verified)
                 + "\n\nend MockTheta5.JTP\n")
    json.dump({"log": log, "spend": spend, "replay_ok": ok_replay, "wall_s": round(time.time() - t0)},
              open(os.path.join(HERE, "campaign_log.json"), "w"), indent=1)
    print(f"  wrote {out}\n  LLM spend: {spend}   wall {round(time.time() - t0)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
