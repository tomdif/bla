#!/usr/bin/env python3
"""proofworld.ramanujan_pipeline -- the proofworld loop applied to Ramanujan's τ, end to end.

Generator PROPOSES, the Lean kernel DISPOSES. Opus 4.8 *discovers* the prime-power closed forms
`τ(p⁴)`, `τ(p⁵)` (in terms of `τ(p)` and `p`); each proposal is compiled in the RamanujanTau Lean
project via a FIXED derivation from the Hecke recurrence (`rw [recurrence, lower closed forms]; ring`),
so the kernel — not the model — decides correctness: a wrong polynomial fails `ring` and is rejected.

Anti-trap discipline (unchanged): the model never scores itself. An injected WRONG closed form (canary)
must be REJECTED by the kernel; only kernel-verified forms are written into the repo.

Gated: needs PROOFWORLD_LLM=1 and ANTHROPIC_API_KEY (read from env only, never logged).
Run:  PROOFWORLD_LLM=1 python3 -m proofworld.ramanujan_pipeline
"""
from __future__ import annotations
import os, json, subprocess, re

REPO = os.path.expanduser("~/RamanujanTau")
ELAN = os.path.expanduser("~/.elan/bin")
MODEL = os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8")

# Fixed proof scaffolds: τ(p^k) from the recurrence at r=k-1, substituting the lower closed forms.
# The ONLY free part is the proposed RHS; `ring` is the kernel's verdict on the proposal.
SCAFFOLD = {
    4: ("3", "4", "2", "tau_prime_cube hp, tau_prime_sq hp"),     # r, k, r-1, lower-form rewrites
    5: ("4", "5", "3", "tau_prime_p4 hp, tau_prime_cube hp"),
}


def lean_candidate(theorems: list[str]) -> str:
    """Assemble a Lean file (in the RamanujanTau project) holding candidate theorems."""
    body = "\n\n".join(theorems)
    return ("import RamanujanTau.EulerFactor\n\nnamespace RamanujanTau\n\n"
            "variable [TauHeckeRecurrence]\n\n" + body + "\n\nend RamanujanTau\n")


def thm(name: str, k: int, rhs: str) -> str:
    r, kk, rm1, lowers = SCAFFOLD[k]
    return (f"theorem {name} {{p : ℕ}} (hp : p.Prime) : τ (p ^ {kk}) = {rhs} := by\n"
            f"  have h := TauHeckeRecurrence.hecke hp {r} (by norm_num)\n"
            f"  rw [show ({r} : ℕ) + 1 = {kk} from rfl, show ({r} : ℕ) - 1 = {rm1} from rfl] at h\n"
            f"  rw [h, {lowers}]; ring")


def kernel_check(theorems: list[str], log=print) -> tuple[bool, str]:
    """THE GATE: compile the candidate file with the Lean kernel. Success ⇔ kernel-verified."""
    path = os.path.join(REPO, "_pw_cand.lean")
    open(path, "w").write(lean_candidate(theorems))
    env = dict(os.environ, PATH=ELAN + os.pathsep + os.environ.get("PATH", ""))
    try:
        r = subprocess.run(["lake", "env", "lean", "_pw_cand.lean"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
        ok = r.returncode == 0
        return ok, (r.stdout + r.stderr)[:400]
    finally:
        try: os.remove(path)
        except OSError: pass


def opus_propose(log=print):
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        log("  gate OFF (need PROOFWORLD_LLM=1 + ANTHROPIC_API_KEY). Nothing called."); return None
    import anthropic
    client = anthropic.Anthropic()
    ask = (
        "In the theory of Ramanujan's tau function, the Hecke recurrence gives "
        "τ(p^{r+1}) = τ(p)·τ(p^r) − p^{11}·τ(p^{r-1}). With τ(p²)=τ(p)²−p¹¹ and τ(p³)=τ(p)³−2p¹¹τ(p), "
        "DERIVE the closed forms for τ(p⁴) and τ(p⁵) as polynomials in τ(p) and p. "
        "Write each as a Lean expression over ℤ using ONLY `τ p`, `(p : ℤ)`, integer literals, and + - * ^. "
        'Reply ONLY JSON: {"p4": "...", "p5": "..."}.  Example shape: "τ p ^ 4 - 3 * (p:ℤ)^11 * τ p ^ 2 + (p:ℤ)^22".'
    )
    msg = client.messages.create(model=MODEL, max_tokens=4000,
                                 messages=[{"role": "user", "content": ask}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if "</think>" in text: text = text.split("</think>")[-1]
    b = None
    try:
        b = json.loads(text)
    except Exception:
        m = re.findall(r"\{[^{}]*\"p4\"[^{}]*\}", text.replace("\n", " "))
        for cnd in reversed(m):
            try: b = json.loads(cnd); break
            except Exception: continue
    if b is None:
        log(f"  Opus output not parseable: {text[:90]!r}"); return None
    log(f"  Opus ({MODEL}) proposed:\n    τ(p⁴) = {b['p4']}\n    τ(p⁵) = {b['p5']}")
    return b


def main():
    print("=" * 90)
    print("  proofworld → RamanujanTau :: Opus DISCOVERS τ(p⁴),τ(p⁵); the Lean KERNEL owns the verdict")
    print("=" * 90)
    prop = opus_propose()
    if prop is None:
        return
    verified = {}

    # --- gate τ(p⁴) ---
    print("\n  [1] kernel-checking τ(p⁴) proposal ...")
    ok4, out4 = kernel_check([thm("tau_prime_p4", 4, prop["p4"])])
    print(f"      VERDICT: {'VERIFIED ✓' if ok4 else 'REJECTED ✗ — ' + out4.strip()[:160]}")
    if ok4:
        verified[4] = prop["p4"]

    # --- adversarial canary: a deliberately WRONG τ(p⁴) (off by +1). The kernel MUST reject it. ---
    print("\n  [canary] kernel-checking a deliberately WRONG τ(p⁴) (= proposal + 1) ...")
    okc, _ = kernel_check([thm("tau_prime_p4_bad", 4, f"({prop['p4']}) + 1")])
    print(f"      canary {'KILLED by kernel ✓ (gate has teeth)' if not okc else 'SURVIVED ✗✗ — gate is broken!'}")

    # --- gate τ(p⁵), chained on the VERIFIED τ(p⁴) ---
    if ok4:
        print("\n  [2] kernel-checking τ(p⁵) proposal (using the verified τ(p⁴)) ...")
        chain = [thm("tau_prime_p4", 4, verified[4]), thm("tau_prime_p5", 5, prop["p5"])]
        ok5, out5 = kernel_check(chain)
        print(f"      VERDICT: {'VERIFIED ✓' if ok5 else 'REJECTED ✗ — ' + out5.strip()[:160]}")
        if ok5:
            verified[5] = prop["p5"]

    # --- persist ONLY kernel-verified forms into the repo, then full-build ---
    if 4 in verified and 5 in verified:
        print("\n  [3] both kernel-verified — writing RamanujanTau/HeckePowers.lean and building ...")
        write_repo_file(verified)
        ok, tail = full_build()
        print(f"      full repo build: {'SUCCESS ✓' if ok else 'FAILED ✗'}\n      {tail}")
    else:
        print("\n  not both verified — nothing written (only kernel-certified forms are ever persisted).")

    print("\n" + "=" * 90)
    print("  The model proposed the mathematics; the Lean kernel decided what is true. A wrong form would fail")
    print("  `ring` and never enter the repo (the canary proves it). Generation scaled; the trust boundary did not.")
    print("=" * 90)


def write_repo_file(verified: dict):
    p4 = thm("tau_prime_p4", 4, verified[4])
    p5 = thm("tau_prime_p5", 5, verified[5])
    src = ("import RamanujanTau.EulerFactor\n\n"
           "/-! # Higher prime-power closed forms for `τ`, discovered via proofworld + kernel-gated\n\n"
           "`τ(p⁴)` and `τ(p⁵)` as polynomials in `τ(p)` and `p`, derived from the Hecke recurrence and the\n"
           "lower closed forms. These were PROPOSED by the proofworld LLM generator and VERIFIED by the Lean\n"
           "kernel (each via `rw [recurrence, lower forms]; ring`); only kernel-certified forms appear here.\n"
           "They continue the Euler-factor / Chebyshev-style structure: the coefficients are the Gegenbauer\n"
           "numbers of the recurrence `τ(p^{r+1}) = τ(p)·τ(p^r) − p¹¹·τ(p^{r-1})`. -/\n\n"
           "namespace RamanujanTau\n\nvariable [TauHeckeRecurrence]\n\n"
           "/-- **τ(p⁴)** closed form (proofworld-discovered, kernel-verified). -/\n" + p4 + "\n\n"
           "/-- **τ(p⁵)** closed form (proofworld-discovered, kernel-verified). -/\n" + p5 + "\n\n"
           "end RamanujanTau\n")
    open(os.path.join(REPO, "RamanujanTau", "HeckePowers.lean"), "w").write(src)
    # wire into root
    root = os.path.join(REPO, "RamanujanTau.lean"); s = open(root).read()
    if "HeckePowers" not in s:
        s = s.replace("import RamanujanTau.EulerFactor\n",
                       "import RamanujanTau.EulerFactor\nimport RamanujanTau.HeckePowers\n")
        open(root, "w").write(s)


def full_build():
    env = dict(os.environ, PATH=ELAN + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run(["lake", "build"], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "")


if __name__ == "__main__":
    main()
