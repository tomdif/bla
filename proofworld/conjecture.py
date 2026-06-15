#!/usr/bin/env python3
"""proofworld.conjecture -- the conjecture-and-prove loop: turn corpus DATA into new VERIFIED general laws.

This is corpus-driven ideation, kept honest by the same gate as everything else:

  mine instances (corpus)  ->  PROPOSE general laws (LLM + enumerator: "imagine new solutions")
  ->  NUMERICAL kill-test against the data (a cheap grounded falsifier -- an over-general claim dies here)
  ->  LEAN-verify the survivor on the Mathlib kernel (a real general theorem, not just the instances)
  ->  #print axioms confirms it is axiom-clean  ->  RE-ADMIT to the corpus (the library grows; insight compounds).

The headline behaviour: from the scattered facts σ₃(2)=9, σ₃(3)=28, σ₃(4)=73, ... the loop proposes
"σ_k(n) = n^k + 1". The DATA REFUTES the universal form (σ₃(4)=73 ≠ 4³+1=65), so it is rejected before any
expensive proof; the loop refines to the correct restricted law "σ_k(p) = p^k + 1 for PRIME p", which the kernel
then proves. Data corrects the over-generalization; the kernel certifies the survivor. Nothing unproven is believed.

Run:  python3 -m proofworld.conjecture        (live LLM proposals if PROOFWORLD_LLM=1; enumerator otherwise)
"""
from __future__ import annotations
import os, re, json, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_JSONL = os.path.join(HERE, "corpus", "corpus.jsonl")
DISCOVERED_JSONL = os.path.join(HERE, "corpus", "discovered.jsonl")
PROJECT_DIR, PROJECT_IMP = "~/RamanujanTau", "RamanujanTau"
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
SIG = re.compile(r"RamanujanTau\.sigma(\d+) (\d+) = (-?\d+)")


def is_prime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True

def mine_instances(corpus):
    """extract (k, n, value) from corpus sigma facts -> {k: {n: value}}."""
    data = {}
    for r in corpus:
        m = SIG.fullmatch(r["statement"].strip())
        if m:
            k, n, v = map(int, m.groups())
            data.setdefault(k, {})[n] = v
    return data


# ---------------- proposers (the creative step) ----------------
def enumerate_conjectures(data):
    """deterministic hypothesis class: per σ_k, the universal and the prime-restricted power-plus-one law."""
    out = []
    for k in sorted(data):
        out.append({"desc": f"σ_{k}(n) = n^{k} + 1  for ALL n", "k": k, "formula": f"n**{k}+1", "domain": "all"})
        out.append({"desc": f"σ_{k}(p) = p^{k} + 1  for PRIME p", "k": k, "formula": f"n**{k}+1", "domain": "prime"})
    return out

def llm_propose(data, log=print):
    """OPTIONAL live LLM ideation: given the instance data, propose general laws as structured JSON."""
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    import anthropic
    sample = {k: dict(sorted(v.items())) for k, v in data.items()}
    prompt = ("Here are computed values of divisor-power-sum functions sigma_k(n) = sum of d^k over divisors d of n:\n"
              f"{json.dumps(sample)}\n\n"
              "Propose general LAWS these satisfy. Reply ONLY a compact JSON array of objects "
              '{"k": <int>, "formula": "<python expr in n>", "domain": "all"|"prime"|"prime_power", "reason": "<=6 words"}. '
              "Use ** for power. Restrict the domain if a law only holds for primes. No prose, keep reasons very short.")
    try:
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-sonnet-4-6"), max_tokens=1500,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        mm = re.search(r"\[.*\]", text, re.DOTALL)          # robustly extract the JSON array from any prose/fences
        items = json.loads(mm.group(0) if mm else text)
        out = []
        for it in items:
            out.append({"desc": f"σ_{it['k']}: {it['formula']} [{it['domain']}] (LLM: {it.get('reason','')[:40]})",
                        "k": int(it["k"]), "formula": str(it["formula"]), "domain": it["domain"]})
        log(f"  LLM proposed {len(out)} candidate laws.")
        return out
    except Exception as e:
        log(f"  (LLM proposal skipped: {e})"); return []


def eval_formula(formula, n):
    return int(eval(formula, {"__builtins__": {}}, {"n": n, "p": n}))


# ---------------- grounded gates ----------------
def kill_test(conj, data):
    """cheap NUMERICAL falsifier: does the formula match EVERY corpus instance in its claimed domain?"""
    insts = data.get(conj["k"], {})
    checked = 0
    for n, v in sorted(insts.items()):
        if conj["domain"] == "prime" and not is_prime(n): continue
        if conj["domain"] == "prime_power" and not (is_prime_power(n)): continue
        try:
            if eval_formula(conj["formula"], n) != v:
                return ("refuted", n, v, eval_formula(conj["formula"], n), checked)
        except Exception:
            return ("uneval", n, v, None, checked)
        checked += 1
    return ("survives", None, None, None, checked)

def is_prime_power(n):
    if n < 2: return False
    for p in range(2, n + 1):
        if is_prime(p):
            m = n
            while m % p == 0: m //= p
            if m == 1: return True
    return False


def lean_verify_prime_formula(k, timeout=120):
    """verify σ_k(p) = p^k + 1 for prime p on the Mathlib kernel (only for the recognized provable template)."""
    src = (f"import {PROJECT_IMP}\nopen {PROJECT_IMP}\n"
           f"theorem conj (p : ℕ) (hp : p.Prime) : sigma{k} p = (p:ℤ)^{k} + 1 := by\n"
           f"  unfold sigma{k}; rw [hp.divisors, Finset.sum_pair hp.one_lt.ne]; push_cast; ring\n"
           f"#print axioms conj\n")
    d = os.path.expanduser(PROJECT_DIR)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "Conj.lean")
        with open(f, "w") as fh: fh.write(src)
        try:
            p = subprocess.run(["lake", "env", "lean", f], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, [], src
    out = p.stdout + p.stderr
    ok = (p.returncode == 0 and "error:" not in out)
    axline = next((l for l in out.splitlines() if "depends on axioms" in l), "")
    axs = [a for a in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", axline.split(":", 1)[1]) if a not in ("depends","on","axioms")] if ":" in axline else []
    return ok, axs, src

def is_prime_power_law(conj):
    """recognize the provable template: formula n**k+1 over the prime domain."""
    return conj["domain"] == "prime" and conj["formula"].replace(" ", "") in (f"n**{conj['k']}+1", f"p**{conj['k']}+1", f"1+n**{conj['k']}", f"1+p**{conj['k']}")


def main():
    corpus = [json.loads(l) for l in open(CORPUS_JSONL)] if os.path.exists(CORPUS_JSONL) else []
    data = mine_instances(corpus)
    print("=== proofworld.conjecture :: corpus DATA -> conjectured general law -> kernel-verified theorem ===\n")
    print(f"  mined instances from corpus: " + "; ".join(f"σ_{k} at n={sorted(v)}" for k, v in sorted(data.items())) + "\n")
    conjectures = enumerate_conjectures(data) + llm_propose(data)
    verified, discovered = 0, []
    seen = set()
    for c in conjectures:
        try:                                                  # dedup by MATHEMATICAL content, not formula spelling
            sig = tuple(eval_formula(c["formula"], n) for n in (2, 3, 5))
        except Exception:
            sig = (c["formula"],)
        key = (c["k"], c["domain"], sig)
        if key in seen: continue
        seen.add(key)
        verdict, n, v, got, checked = kill_test(c, data)
        if verdict == "refuted":
            print(f"  [{c['desc']}]\n      KILLED by data: at n={n}, corpus says {v} but formula gives {got} (cheap, pre-proof)")
            continue
        if verdict != "survives":
            print(f"  [{c['desc']}]  -> skipped ({verdict})"); continue
        print(f"  [{c['desc']}]\n      survives numerical kill-test ({checked} corpus instances) -> Lean:", end=" ")
        if is_prime_power_law(c):
            ok, axs, src = lean_verify_prime_formula(c["k"])
            clean = ok and bool(axs) and set(axs) <= STD_AXIOMS
            print(f"{'VERIFIED, axiom-clean ' + str(axs) if clean else ('proved but ' + str(axs)) if ok else 'proof FAILED'}")
            if clean:
                verified += 1
                discovered.append({"name": f"RamanujanTau.sigma{c['k']}_prime", "statement": f"∀ p, p.Prime → sigma{c['k']} p = (p:ℤ)^{c['k']} + 1",
                                   "axioms": axs, "project": "RamanujanTau", "domain": "modular forms / Ramanujan tau",
                                   "source": "proofworld.conjecture (data->law->kernel)"})
        else:
            print(f"numerically supported, no provable template wired (honest: proof TODO)")
    # RE-ADMIT verified laws to the growing corpus
    if discovered:
        os.makedirs(os.path.dirname(DISCOVERED_JSONL), exist_ok=True)
        with open(DISCOVERED_JSONL, "w") as fh:
            for r in discovered: fh.write(json.dumps(r) + "\n")
        print(f"\n  RE-ADMITTED {len(discovered)} kernel-verified general laws -> corpus/discovered.jsonl (the library grows)")
    print(f"\n  RESULT: {verified} NEW general laws conjectured from data and PROVEN on the kernel (each axiom-clean).")
    print("  The data refuted the over-general forms for free; the kernel certified the survivors. Insight, grounded:")
    print("  the model imagines from real verified structure, and nothing becomes a 'law' until the kernel proves it.")


if __name__ == "__main__":
    main()
