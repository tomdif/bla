#!/usr/bin/env python3
"""proofworld.cite -- use the imported verified corpus as the ESTABLISHED LIBRARY: retrieve relevant facts,
CITE them to prove a new goal, and verify in the project's Lean context -- with the new theorem checked to inherit
corpus-grade soundness (axiom-clean).

This closes the loop opened by corpus.py. A new goal is proved ONLY by citing axiom-clean corpus theorems; the
proof is kernel-verified; and `#print axioms` on the RESULT confirms its footprint is still clean -- so a proof
built on the corpus is itself corpus-grade and could be re-admitted. Premise selection here is an IDF-weighted
lexical retriever (a transparent, offline premise selector); swapping in an embedding ranker changes only score().

Pipeline:  goal -> premise_select (rank corpus facts by IDF overlap) -> dream proofs citing them ->
           Lean kernel verifies -> #print axioms confirms the result is axiom-clean.

Run:  python3 -m proofworld.cite
"""
from __future__ import annotations
import os, re, math, json
from proofworld import gate

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_JSONL = os.path.join(HERE, "corpus", "corpus.jsonl")
PROJECTS = {
    "RamanujanTau": {"dir": "~/RamanujanTau", "imp": "RamanujanTau"},
    "PlonkLean":    {"dir": "~/PlonkLean",    "imp": "PlonkLean"},
}
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def load_corpus():
    return [json.loads(l) for l in open(CORPUS_JSONL)] if os.path.exists(CORPUS_JSONL) else []

def tokenize(s):
    return set(TOKEN.findall(s))

def build_idf(corpus):
    df = {}
    for r in corpus:
        for t in tokenize(r["statement"]):
            df[t] = df.get(t, 0) + 1
    n = len(corpus) or 1
    return {t: math.log(1 + n / c) for t, c in df.items()}

def premise_select(goal, project, corpus, idf, k=6):
    """rank same-project corpus facts by IDF-weighted token overlap with the goal (rare symbols dominate)."""
    gt = tokenize(goal)
    scored = []
    for r in corpus:
        if r["project"] != project:
            continue
        overlap = gt & tokenize(r["statement"])
        score = sum(idf.get(t, 0.0) for t in overlap)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def lean_verify(imp, goal, tactic, project_dir, timeout=300):
    """one cited proof attempt through the trust gate. Returns (ok, axioms) -- ok means PROVED (clean footprint)."""
    v = gate.check([gate.Claim("pw_cite", f": {goal}", tactic)], preamble=f"import {imp}\n",
                   project=project_dir, timeout=timeout, tag="cite")["pw_cite"]
    return v.proved, v.axioms


def candidate_tactics(names):
    joined = ", ".join(names)
    return [f"by norm_num [{joined}]", f"by simp [{joined}]", f"by simp only [{joined}]",
            f"by rw [{joined}]; norm_num", f"by rw [{joined}]"]

def cite_and_verify(goal, project, corpus, idf, log=print):
    cfg = PROJECTS[project]
    picks = premise_select(goal, project, corpus, idf)
    log(f"\n  GOAL: {goal}")
    log(f"  premise-selection (IDF-ranked, top {len(picks)} of corpus):")
    for sc, r in picks:
        log(f"    {sc:5.1f}  {r['name']}   [{r['statement'][:48]}]")
    names = [r["name"] for _, r in picks]
    tacs = candidate_tactics(names)
    # every candidate in ONE gated Lean run (one project import instead of one per tactic)
    claims = [gate.Claim(f"pw_cite_{i}", f": {goal}", tac) for i, tac in enumerate(tacs)]
    vs = gate.check(claims, preamble=f"import {cfg['imp']}\n", project=cfg["dir"], tag="cite")
    for i, tac in enumerate(tacs):
        v = vs[f"pw_cite_{i}"]
        if v.proved:
            log(f"  PROVED by: {tac}")
            log(f"  kernel axioms of the NEW theorem: {v.axioms}  -> AXIOM-CLEAN (corpus-grade)")
            return True, tac, v.axioms, True
    worst = sorted({v.status for v in vs.values()})
    log(f"  no cited proof closed it (statuses: {worst}; try a larger k or different closer).")
    return False, None, None, False


def main():
    corpus = load_corpus()
    if not corpus:
        print("no corpus -- run `python3 -m proofworld.corpus` first"); return
    idf = build_idf(corpus)
    print("=== proofworld.cite :: prove NEW goals by CITING the verified corpus (established library) ===")
    print(f"  corpus: {len(corpus)} axiom-clean facts; premise selection = IDF-weighted lexical retrieval\n")
    # new goals that follow ONLY from cited corpus facts (the retriever must FIND the right ones among 773)
    goals = [
        ("RamanujanTau", "RamanujanTau.sigma3 2 * RamanujanTau.sigma7 2 = 1161"),     # 9 * 129
        ("RamanujanTau", "RamanujanTau.sigma3 3 + RamanujanTau.sigma9 2 = 541"),       # 28 + 513
    ]
    results = []
    for project, goal in goals:
        ok, tac, axs, clean = cite_and_verify(goal, project, corpus, idf)
        results.append(ok and clean)
    n = sum(results)
    print(f"\n  RESULT: {n}/{len(goals)} new goals proved by citing the corpus, each kernel-verified AND axiom-clean.")
    print("  SOUNDNESS: only axiom-clean corpus facts are cited; the new theorem's own axiom footprint is checked,")
    print("  so a proof built on the corpus is itself corpus-grade -- the library grows without ever admitting an")
    print("  unproven assumption. Retrieval proposes; the Lean kernel disposes.")


if __name__ == "__main__":
    main()
