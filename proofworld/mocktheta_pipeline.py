#!/usr/bin/env python3
"""proofworld.mocktheta_pipeline -- the proofworld loop applied to Ramanujan's 5th-order mock theta functions.

Generator PROPOSES a Lean proof; the Lean kernel DISPOSES. The curriculum is a ladder of layer-1 q-Pochhammer
lemmas over `PowerSeries ℤ` (built on the committed `RamanujanTau.MockTheta5Series` foundation) that lead
toward the *infinite* identities χ₀ = 2F₀ − φ₀(−q) etc. Each lemma's STATEMENT is fixed (the math target);
the model proposes only the PROOF, and a candidate is compiled (importing MockTheta5Series + Mathlib) so the
Lean KERNEL — not the model — owns correctness. Verified lemmas chain (later ones may use earlier ones) and are
persisted into `RamanujanTau/MockTheta5Lemmas.lean`; nothing unproven is ever written.

Anti-trap discipline (mirrors ramanujan_pipeline / leankernel): the model never scores itself, and an injected
FALSE lemma (canary) must be REJECTED by every proposed proof, or the gate is declared broken.

Offline: with no API key it runs in DREAMER mode (fixed candidate tactics) so the harness + gate + canary are
testable without the LLM. Gated LLM mode: PROOFWORLD_LLM=1 + ANTHROPIC_API_KEY → Opus proposes the proofs.
Run:  python3 -m proofworld.mocktheta_pipeline            (dreamer)
      PROOFWORLD_LLM=1 python3 -m proofworld.mocktheta_pipeline   (LLM-proposed proofs)
"""
from __future__ import annotations
import os, subprocess

REPO = os.path.expanduser("~/RamanujanTau")
ELAN = os.path.expanduser("~/.elan/bin")
MODEL = os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8")

PREAMBLE = "import RamanujanTau.MockTheta5Series\n\nnamespace MockTheta5\nopen PowerSeries\n\n"

# --- the curriculum: layer-1 q-Pochhammer lemmas (statement = the math target; the loop supplies the PROOF) ---
# each: name, Lean statement (no ':= proof'), and DREAMER seed tactics (used when the LLM gate is off).
CURRICULUM = [
    dict(name="mt_qpoch_one",
         stmt="theorem mt_qpoch_one (a : PowerSeries ℤ) : qpoch a 1 = 1 - a",
         seeds=["by simp [qpoch_succ]",
                "by rw [qpoch_succ, qpoch_zero, pow_zero, mul_one, one_mul]",
                "by rw [qpoch_succ, qpoch_zero]; ring_nf; simp"]),
    dict(name="mt_qpoch_inv_cancel",
         stmt="theorem mt_qpoch_inv_cancel (a : PowerSeries ℤ) (ha : constantCoeff a = 0) (n : ℕ) : "
              "Ring.inverse (qpoch a n) * qpoch a n = 1",
         seeds=["by exact Ring.inverse_mul_cancel _ (isUnit_qpoch a ha n)",
                "by rw [Ring.inverse_mul_cancel _ (isUnit_qpoch a ha n)]",
                "by exact (isUnit_qpoch a ha n).val_inv_mul"]),
    dict(name="mt_qpoch_mul_inv",
         stmt="theorem mt_qpoch_mul_inv (a : PowerSeries ℤ) (ha : constantCoeff a = 0) (n : ℕ) : "
              "qpoch a n * Ring.inverse (qpoch a n) = 1",
         seeds=["by exact Ring.mul_inverse_cancel _ (isUnit_qpoch a ha n)",
                "by rw [Ring.mul_inverse_cancel _ (isUnit_qpoch a ha n)]"]),
    # the KEY structural lemma — q-Pochhammer product split (a;q)_{m+n} = (a;q)_m · (a qᵐ; q)_n. Genuinely
    # nontrivial (induction + pow_add); offline seeds may not close it → honestly OPEN until the LLM proposes.
    dict(name="mt_qpoch_split",
         stmt="theorem mt_qpoch_split (a : PowerSeries ℤ) (m n : ℕ) : "
              "qpoch a (m + n) = qpoch a m * qpoch (a * X ^ m) n",
         seeds=["by induction n with | zero => simp | succ k ih => "
                "rw [Nat.add_succ, qpoch_succ, qpoch_succ, ih, pow_add]; ring",
                "by induction n with | zero => simp | succ k ih => "
                "rw [← Nat.add_one, ← Nat.add_assoc, qpoch_succ, qpoch_succ, ih, pow_add, mul_assoc]"]),
    # --- SUMMABLE-FAMILY rung: makes the infinite sums Σ_n X^{e(n)}·(unit) well-defined coefficient-wise ---
    # per-term vanishing: a term of order ≥ j contributes nothing to coefficient k < j.
    dict(name="mt_coeff_Xpow_mul_zero",
         stmt="theorem mt_coeff_Xpow_mul_zero (φ : PowerSeries ℤ) (j k : ℕ) (h : k < j) : "
              "coeff k ((X : PowerSeries ℤ) ^ j * φ) = 0",
         seeds=["by rw [coeff_X_pow_mul', if_neg (Nat.not_le.mpr h)]",
                "by rw [coeff_X_pow_mul']; exact if_neg (Nat.not_le.mpr h)",
                "by simp [coeff_X_pow_mul', Nat.not_le.mpr h]"]),
    # the PAYOFF: coefficient stabilization — once you include all terms reaching degree k (the first k+1),
    # the k-th coefficient of any larger partial sum is unchanged, so the infinite sum's coeff is THIS finite sum.
    dict(name="mt_coeff_sum_eq",
         stmt="theorem mt_coeff_sum_eq (f : ℕ → PowerSeries ℤ) (k : ℕ) "
              "(hf : ∀ n, k < n → coeff k (f n) = 0) (M : ℕ) (hM : k + 1 ≤ M) : "
              "coeff k (∑ n ∈ Finset.range M, f n) = ∑ n ∈ Finset.range (k + 1), coeff k (f n)",
         seeds=["by rw [map_sum]; symm; apply Finset.sum_subset "
                "(by intro x hx; exact Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hM)); "
                "intro n _ hn; exact hf n (by simp only [Finset.mem_range, not_lt] at hn; omega)",
                "by rw [map_sum]; symm; apply Finset.sum_subset (Finset.range_subset.mpr hM); "
                "intro n _ hn; exact hf n (by simp only [Finset.mem_range, not_lt] at hn; omega)"]),
]

CANARY = dict(name="mt_canary_false",
              stmt="theorem mt_canary_false (a : PowerSeries ℤ) : qpoch a 1 = 1 + a",
              seeds=["by simp [qpoch_succ]",
                     "by rw [qpoch_succ, qpoch_zero, pow_zero, mul_one, one_mul]"])


def kernel_check(verified_texts: list[str], stmt: str, proof: str, log=print) -> tuple[bool, str]:
    """THE GATE: compile a candidate (verified-so-far lemmas + the new lemma) importing MockTheta5Series.
    Success ⇔ the Lean kernel accepts it. A wrong proof fails to compile and is discarded."""
    body = "\n\n".join(verified_texts + [f"{stmt} := {proof}"])
    path = os.path.join(REPO, "_pw_mocktheta_cand.lean")
    open(path, "w").write(PREAMBLE + body + "\n\nend MockTheta5\n")
    env = dict(os.environ, PATH=ELAN + os.pathsep + os.environ.get("PATH", ""))
    try:
        r = subprocess.run(["lake", "env", "lean", "_pw_mocktheta_cand.lean"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr)
        ok = r.returncode == 0 and "error:" not in out
        return ok, out[:300]
    finally:
        try: os.remove(path)
        except OSError: pass


def opus_propose_proof(name: str, stmt: str, verified_texts: list[str], log=print) -> list[str]:
    """Gated LLM proposer: Opus proposes Lean proof scripts for the stated goal. Returns candidate proofs."""
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    import anthropic
    ctx = ("Available (already proven, in scope): qpoch a n := ∏ k ∈ Finset.range n, (1 - a*X^k); "
           "qpoch_zero : qpoch a 0 = 1; qpoch_succ : qpoch a (n+1) = qpoch a n * (1 - a*X^n); "
           "constantCoeff_qpoch (ha : constantCoeff a = 0) : constantCoeff (qpoch a n) = 1; "
           "isUnit_qpoch (ha : constantCoeff a = 0) : IsUnit (qpoch a n); plus all of Mathlib.\n"
           + ("Earlier proven lemmas you may use:\n" + "\n".join(verified_texts) if verified_texts else ""))
    ask = (f"In Lean 4 (Mathlib, namespace MockTheta5, `open PowerSeries`), prove:\n  {stmt}\n\n{ctx}\n\n"
           "Reply with 3 candidate proof scripts, one per line, each a complete Lean term starting with `by `. "
           "No prose, no code fences — just the `by ...` lines.")
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(model=MODEL, max_tokens=1500,
                                     messages=[{"role": "user", "content": ask}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        if "</think>" in text: text = text.split("</think>")[-1]
        return [ln.strip().strip("`") for ln in text.splitlines() if ln.strip().startswith("by ")][:4]
    except Exception as e:
        log(f"    (LLM error: {str(e)[:80]})"); return []


def attack(goal: dict, verified_texts: list[str], log=print) -> str | None:
    """propose proofs (LLM if gated, else dreamer seeds); the FIRST the kernel accepts closes the goal."""
    cands = opus_propose_proof(goal["name"], goal["stmt"], verified_texts, log) or goal["seeds"]
    src = "Opus" if (os.environ.get("PROOFWORLD_LLM") == "1" and os.environ.get("ANTHROPIC_API_KEY")) else "dreamer"
    log(f"\n  goal [{goal['name']}]  ({src} proposes {len(cands)} proof(s))")
    for proof in cands:
        ok, out = kernel_check(verified_texts, goal["stmt"], proof, log)
        mark = "✓ VERIFIED" if ok else "✗ rejected"
        err = "" if ok else "   (" + (next((l.split('error:',1)[1].strip()[:60] for l in out.splitlines() if 'error:' in l), out.strip()[:60])) + ")"
        log(f"    {proof[:60]:60} -> {mark}{err}")
        if ok:
            log(f"    => kernel accepted. truth owned by Lean, not the model.")
            return f"{goal['stmt']} := {proof}"
    log("    => no proposed proof closed it; honestly OPEN (nothing false accepted).")
    return None


def main():
    print("=" * 92)
    print("  proofworld → RamanujanTau :: the loop attacks 5th-order mock theta layer-1 lemmas")
    print("  (model proposes Lean PROOFS; the kernel owns the verdict; canary keeps the gate honest)")
    print("=" * 92)
    verified: list[tuple[str, str]] = []   # (name, full "stmt := proof" text)
    for goal in CURRICULUM:
        text = attack(goal, [t for _, t in verified])
        if text is not None:
            verified.append((goal["name"], text))

    # --- adversarial canary: the FALSE lemma must be rejected by EVERY proposed proof ---
    print(f"\n  [canary] attacking a deliberately FALSE lemma (qpoch a 1 = 1 + a) — must be REJECTED ...")
    canary_text = attack(CANARY, [t for _, t in verified])
    canary_killed = canary_text is None
    print(f"      canary {'KILLED by kernel ✓ (gate has teeth)' if canary_killed else 'SURVIVED ✗✗ — gate is BROKEN, nothing trusted'}")

    # --- persist ONLY kernel-verified lemmas (and only if the canary was killed) ---
    print("\n--- summary ---")
    for goal in CURRICULUM:
        tag = "VERIFIED" if any(n == goal["name"] for n, _ in verified) else "OPEN"
        print(f"  {goal['name']:22} {tag}")
    if verified and canary_killed:
        write_lemmas([t for _, t in verified], [n for n, _ in verified])
        print(f"\n  [persist] wrote {len(verified)} kernel-verified lemma(s) to MockTheta5Lemmas.lean.")
    elif not canary_killed:
        print("\n  [persist] SKIPPED — canary survived, so the gate is not trustworthy this run.")
    else:
        print("\n  [persist] nothing verified yet (LLM gate off / proofs open).")
    print("\n" + "=" * 92)
    print("  Setup complete: generator proposes, the Lean kernel disposes, the canary keeps it honest.")
    print("  Closing the layers = running this (PROOFWORLD_LLM=1) over many iterations + new curriculum rungs.")
    print("=" * 92)


def write_lemmas(texts: list[str], names: list[str]):
    hdr = ("/-! # 5th-order mock theta — layer-1 lemmas (proofworld-proposed, kernel-verified)\n\n"
           "q-Pochhammer building blocks over `PowerSeries ℤ`, toward the infinite identities. Each PROOF was\n"
           "proposed by the proofworld loop and accepted by the Lean kernel; nothing unproven appears here. -/\n")
    src = "import RamanujanTau.MockTheta5Series\n\n" + hdr + "\nnamespace MockTheta5\nopen PowerSeries\n\n" + \
          "\n\n".join(texts) + "\n\nend MockTheta5\n"
    open(os.path.join(REPO, "RamanujanTau", "MockTheta5Lemmas.lean"), "w").write(src)
    root = os.path.join(REPO, "RamanujanTau.lean"); s = open(root).read()
    if "MockTheta5Lemmas" not in s:
        open(root, "a").write("import RamanujanTau.MockTheta5Lemmas\n")


if __name__ == "__main__":
    main()
