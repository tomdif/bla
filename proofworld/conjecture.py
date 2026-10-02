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
import os, re, json
from proofworld import gate, numeric

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_JSONL = os.path.join(HERE, "corpus", "corpus.jsonl")
DISCOVERED_JSONL = os.path.join(HERE, "corpus", "discovered.jsonl")
PROJECT_DIR, PROJECT_IMP = "~/RamanujanTau", "RamanujanTau"
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
    sample = {k: dict(sorted(v.items())) for k, v in data.items()}   # callers pass ONLY the proposer's `seen` split
    prompt = ("Here are computed values of divisor-power-sum functions sigma_k(n) = sum of d^k over divisors d of n:\n"
              f"{json.dumps(sample)}\n\n"
              "Propose general LAWS these satisfy. Reply ONLY a compact JSON array of objects "
              '{"k": <int>, "formula": "<python expr in n>", "domain": "all"|"prime"|"prime_power", "reason": "<=6 words"}. '
              "Use ** for power. Restrict the domain if a law only holds for primes. No prose, keep reasons very short.")
    try:
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8"), max_tokens=1500,
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
    """safe: AST-whitelisted integer arithmetic only (LLM output is never passed to eval())."""
    return numeric.compile_formula(formula, names=("n", "p"))(n)


DOMAINS = {"all": lambda n: True, "prime": lambda n: is_prime(n), "prime_power": lambda n: is_prime_power(n)}


# ---------------- grounded gates ----------------
def kill_test(conj, seen, held, min_support=3):
    """cheap NUMERICAL falsifier (proofworld.numeric): refuted by ANY instance, but it only counts as evidence
    when >= min_support HELD-OUT instances (never shown to the proposer) agree. Thin data -> INSUFFICIENT."""
    if conj["domain"] not in DOMAINS:
        return numeric.NumVerdict("UNEVALUABLE", detail=f"unknown domain {conj['domain']!r}")
    try:
        law = numeric.compile_formula(conj["formula"], names=("n", "p"))
    except numeric.UnsafeExpr as e:
        return numeric.NumVerdict("UNEVALUABLE", detail=f"unsafe formula: {e}")
    return numeric.heldout_kill_test(law, seen.get(conj["k"], {}), held.get(conj["k"], {}),
                                     DOMAINS[conj["domain"]], min_support=min_support)

def is_prime_power(n):
    if n < 2: return False
    for p in range(2, n + 1):
        if is_prime(p):
            m = n
            while m % p == 0: m //= p
            if m == 1: return True
    return False


def _sigma_claim(k):
    return gate.Claim(f"conj_sigma{k}", f"(p : ℕ) (hp : p.Prime) : sigma{k} p = (p:ℤ)^{k} + 1",
                      f"by\n  unfold sigma{k}; rw [hp.divisors, Finset.sum_pair hp.one_lt.ne]; push_cast; ring",
                      trusted=True)


def lean_verify_prime_formulas(ks, timeout=600):
    """verify σ_k(p) = p^k + 1 (prime p) for every k in ONE gated Mathlib run (one import, not one per law).
    Returns {k: Verdict}."""
    claims = {k: _sigma_claim(k) for k in sorted(set(ks))}
    if not claims:
        return {}
    vs = gate.check(list(claims.values()), preamble=f"import {PROJECT_IMP}\n", opens=f"open {PROJECT_IMP}",
                    project=PROJECT_DIR, timeout=timeout, tag="conjecture.sigma")
    return {k: vs[c.name] for k, c in claims.items()}


def lean_verify_prime_formula(k, timeout=300):
    """single-law convenience wrapper. Returns (ok, axioms, verdict)."""
    v = lean_verify_prime_formulas([k], timeout)[k]
    return v.proved, v.axioms, v

def is_prime_power_law(conj):
    """recognize the provable template: formula n**k+1 over the prime domain."""
    return conj["domain"] == "prime" and conj["formula"].replace(" ", "") in (f"n**{conj['k']}+1", f"p**{conj['k']}+1", f"1+n**{conj['k']}", f"1+p**{conj['k']}")


def main():
    corpus = [json.loads(l) for l in open(CORPUS_JSONL)] if os.path.exists(CORPUS_JSONL) else []
    data = mine_instances(corpus)
    print("=== proofworld.conjecture :: corpus DATA -> conjectured general law -> kernel-verified theorem ===\n")
    print(f"  mined instances from corpus: " + "; ".join(f"σ_{k} at n={sorted(v)}" for k, v in sorted(data.items())))
    # INDEPENDENCE: the proposer only ever sees `seen`; numeric evidence for a law comes from `held`
    split = {k: numeric.split(v, salt=f"sigma{k}") for k, v in data.items()}
    seen = {k: sv for k, (sv, _) in split.items()}
    held = {k: hv for k, (_, hv) in split.items()}
    print(f"  held out from the proposer: " + "; ".join(f"σ_{k} n={sorted(held[k])}" for k in sorted(held)) + "\n")
    nc = numeric.controls()
    if not nc["ok"]:
        print("  numeric kill-test FAILED its planted controls -- refusing to run."); return
    conjectures = enumerate_conjectures(data) + llm_propose(seen)
    verified, discovered, queued = 0, [], []
    seen_keys = set()
    for c in conjectures:
        try:                                                  # dedup by MATHEMATICAL content, not formula spelling
            sig = tuple(eval_formula(c["formula"], n) for n in (2, 3, 5))
        except (numeric.UnsafeExpr, ZeroDivisionError):
            sig = (c["formula"],)
        key = (c["k"], c["domain"], sig)
        if key in seen_keys: continue
        seen_keys.add(key)
        nv = kill_test(c, seen, held)
        if nv.status == "REFUTED":
            n, v, got = nv.counterexample
            print(f"  [{c['desc']}]\n      KILLED by data: at n={n}, corpus says {v} but formula gives {got} (cheap, pre-proof)")
            continue
        if nv.status == "UNEVALUABLE":
            print(f"  [{c['desc']}]  -> skipped ({nv.detail})"); continue
        evidence = (f"survives held-out kill-test ({nv.checked} unseen instances)" if nv.survives else
                    f"no refutation, but numeric evidence INSUFFICIENT ({nv.detail})")
        print(f"  [{c['desc']}]\n      {evidence}" + (" -> queued for the kernel" if is_prime_power_law(c) else
              f"; no provable template wired -- NOT a law (numeric: {nv.status}, register=numeric, not proved)"))
        if is_prime_power_law(c):
            queued.append((c, nv))
    # one batched kernel run for every queued law (Mathlib import paid once)
    verdicts = lean_verify_prime_formulas([c["k"] for c, _ in queued])
    print()
    for c, nv in queued:
        v = verdicts[c["k"]]
        print(f"  KERNEL [{c['desc']}] -> {'PROVED, axiom-clean ' + str(v.axioms) if v.proved else v.status + ': ' + v.detail[:80]}")
        if v.proved:
            verified += 1
            discovered.append({"name": f"RamanujanTau.sigma{c['k']}_prime", "statement": f"∀ p, p.Prime → sigma{c['k']} p = (p:ℤ)^{c['k']} + 1",
                               "axioms": v.axioms, "project": "RamanujanTau", "domain": "modular forms / Ramanujan tau",
                               "source": "proofworld.conjecture (data->law->kernel)",
                               "numeric": {"status": nv.status, "heldout_checked": nv.checked, **nv.provenance},
                               "gate": {"env": v.env, "source_sha": v.source_sha}})
    # RE-ADMIT verified laws to the growing corpus
    if discovered:
        os.makedirs(os.path.dirname(DISCOVERED_JSONL), exist_ok=True)
        with open(DISCOVERED_JSONL, "w") as fh:
            for r in discovered: fh.write(json.dumps(r) + "\n")
        print(f"\n  RE-ADMITTED {len(discovered)} kernel-verified general laws -> corpus/discovered.jsonl (the library grows)")
    print(f"\n  RESULT: {verified} NEW general laws conjectured from data and PROVED through the gate (each axiom-clean).")
    print("  The data refuted the over-general forms for free; the kernel certified the survivors. Insight, grounded:")
    print("  the model imagines from real verified structure, and nothing becomes a 'law' until the kernel proves it.")


if __name__ == "__main__":
    main()
