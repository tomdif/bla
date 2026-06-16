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

# Fixed proof scaffolds: τ(p^k) from the recurrence at r=k-1, substituting the two lower closed forms
# τ(p^{k-1}), τ(p^{k-2}). The ONLY free part is the proposed RHS; `ring` is the kernel's verdict.
TARGETS = [4, 5, 6, 7]
SCAFFOLD = {
    4: ("3", "4", "2", "tau_prime_cube hp, tau_prime_sq hp"),     # r, k, r-1, lower-form rewrites
    5: ("4", "5", "3", "tau_prime_p4 hp, tau_prime_cube hp"),
    6: ("5", "6", "4", "tau_prime_p5 hp, tau_prime_p4 hp"),
    7: ("6", "7", "5", "tau_prime_p6 hp, tau_prime_p5 hp"),
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
    keys = ", ".join(f'"p{k}"' for k in TARGETS)
    ask = (
        "In the theory of Ramanujan's tau function, the Hecke recurrence gives "
        "τ(p^{r+1}) = τ(p)·τ(p^r) − p^{11}·τ(p^{r-1}). With τ(p²)=τ(p)²−p¹¹ and τ(p³)=τ(p)³−2p¹¹τ(p), "
        f"DERIVE the closed forms for {', '.join('τ(p^'+str(k)+')' for k in TARGETS)} as polynomials in τ(p) and p. "
        "Write each as a Lean expression over ℤ using ONLY `τ p`, `(p : ℤ)`, integer literals, and + - * ^. "
        f'Reply ONLY JSON with keys {keys}, e.g. '
        '{"p4": "τ p ^ 4 - 3 * (p:ℤ)^11 * τ p ^ 2 + (p:ℤ)^22", ...}.'
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
    if b is None or not all(f"p{k}" in b for k in TARGETS):
        log(f"  Opus output not parseable / incomplete: {text[:90]!r}"); return None
    log(f"  Opus ({MODEL}) proposed:")
    for k in TARGETS:
        log(f"    τ(p^{k}) = {b[f'p{k}']}")
    return b


def main():
    print("=" * 90)
    print("  proofworld → RamanujanTau :: Opus DISCOVERS τ(p⁴),τ(p⁵); the Lean KERNEL owns the verdict")
    print("=" * 90)
    prop = opus_propose()
    if prop is None:
        return
    verified = {}

    # --- chain-verify each target on its ALREADY-verified predecessors (the kernel gates every step) ---
    for k in TARGETS:
        print(f"\n  [{k}] kernel-checking τ(p^{k}) proposal (chained on verified p4..p{k-1}) ...")
        chain = [thm(f"tau_prime_p{j}", j, verified[j]) for j in verified]        # verified predecessors
        chain.append(thm(f"tau_prime_p{k}", k, prop[f"p{k}"]))                    # the new candidate
        ok, out = kernel_check(chain)
        print(f"      VERDICT: {'VERIFIED ✓' if ok else 'REJECTED ✗ — ' + out.strip()[:160]}")
        if ok:
            verified[k] = prop[f"p{k}"]
        else:
            print(f"      chain broken at p^{k} — stopping (downstream forms depend on it)."); break

    # --- adversarial canary at the deepest verified target: a WRONG form (off by +1) MUST be rejected ---
    if verified:
        kc = max(verified)
        print(f"\n  [canary] kernel-checking a deliberately WRONG τ(p^{kc}) (= proposal + 1) ...")
        chain = [thm(f"tau_prime_p{j}", j, verified[j]) for j in verified if j < kc]
        chain.append(thm(f"tau_prime_p{kc}", kc, f"({verified[kc]}) + 1"))
        okc, _ = kernel_check(chain)
        print(f"      canary {'KILLED by kernel ✓ (gate has teeth)' if not okc else 'SURVIVED ✗✗ — gate is broken!'}")

    # --- persist ONLY kernel-verified forms into the repo, then full-build ---
    if set(TARGETS).issubset(verified):
        print(f"\n  [persist] all of {['p'+str(k) for k in TARGETS]} kernel-verified — writing HeckePowers.lean ...")
        write_repo_file(verified)
        ok, tail = full_build()
        print(f"      full repo build: {'SUCCESS ✓' if ok else 'FAILED ✗'}\n      {tail}")
    else:
        print(f"\n  verified {sorted(verified)} of {TARGETS} — nothing written unless ALL verify "
              f"(only kernel-certified forms are ever persisted).")

    print("\n" + "=" * 90)
    print("  The model proposed the mathematics; the Lean kernel decided what is true. A wrong form would fail")
    print("  `ring` and never enter the repo (the canary proves it). Generation scaled; the trust boundary did not.")
    print("=" * 90)


def write_repo_file(verified: dict):
    thms = "\n\n".join(
        f"/-- **τ(p^{k})** closed form (proofworld-discovered, kernel-verified). -/\n"
        + thm(f"tau_prime_p{k}", k, verified[k]) for k in TARGETS)
    src = ("import RamanujanTau.EulerFactor\n\n"
           "/-! # Higher prime-power closed forms for `τ`, discovered via proofworld + kernel-gated\n\n"
           f"`τ(p⁴)`…`τ(p^{max(TARGETS)})` as polynomials in `τ(p)` and `p`, derived from the Hecke recurrence and\n"
           "the lower closed forms. Each was PROPOSED by the proofworld LLM generator and VERIFIED by the Lean\n"
           "kernel (via `rw [recurrence, lower forms]; ring`); only kernel-certified forms appear here. They\n"
           "continue the Euler-factor / Chebyshev-style structure: the coefficients are the Gegenbauer numbers\n"
           "of the recurrence `τ(p^{r+1}) = τ(p)·τ(p^r) − p¹¹·τ(p^{r-1})`. -/\n\n"
           "namespace RamanujanTau\n\nvariable [TauHeckeRecurrence]\n\n" + thms + "\n\nend RamanujanTau\n")
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
