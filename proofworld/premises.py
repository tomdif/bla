#!/usr/bin/env python3
"""proofworld.premises -- harvest (goal -> cited-lemmas) training pairs, and retrieval-augmented few-shot.

Premise selection is the cheapest, highest-leverage thing to learn for better ideation: given a goal, find the
lemmas worth citing. The ground-truth labels are FREE -- every verified proof already says which lemmas it used.
We extract them at the KERNEL level (walk each proof TERM for the theorems it references), which is sound and
complete at the term level, then filter out low-level plumbing (Eq.trans, congrArg, of_decide_eq_true, ...) so the
signal is real DOMAIN-lemma citations. Computational proofs (decide/norm_num) then yield no premises and are
dropped -- correctly, they carry no premise-selection signal.

Output: corpus/premises.jsonl -- records {theorem, statement, premises:[names], project}. These are training pairs
for a premise-selection embedding AND the retrieval pool for retrieval-augmented few-shot: given a NEW goal, we
retrieve the most similar SOLVED theorems and show the proposer (goal -> lemmas-it-cited) as in-context examples.
This is training-FREE adaptation -- the model conditions on real verified neighbours instead of guessing cold.

Usage:
  python3 -m proofworld.premises                # harvest pairs -> corpus/premises.jsonl
  python3 -m proofworld.premises fewshot "<goal>"   # build the RAG few-shot prompt for a goal (+ live LLM if gated)
"""
from __future__ import annotations
import os, re, sys, json, math, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_JSONL = os.path.join(HERE, "corpus", "premises.jsonl")
PROJECTS = [
    {"dir": "~/RamanujanTau", "imp": "RamanujanTau", "ns": "RamanujanTau", "domain": "modular forms / Ramanujan tau"},
    {"dir": "~/PlonkLean",    "imp": "PlonkLean",    "ns": "PlonkLean",    "domain": "PLONK zero-knowledge proof system"},
]
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

HARVEST = r'''import __IMPORT__
import Lean
open Lean
def stdAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]
def isClean (axs : Array Name) : Bool := axs.all (fun a => stdAxioms.contains a)
def noisePats : List String :=
  ["_proof_", "congr", ".eq_", ".match_", ".brec", ".below", "_cstage", ".noConfusion",
   ".sizeOf", ".inj", "._unfold", ".rec", ".cases", ".induct", "._sunfold", ".eq_def"]
def plumbPats : List String :=
  ["of_eq_true", "of_decide", "eq_self", "eq_true", "eq_false", "eq_mpr", "eq_mp", "Eq.trans",
   "Eq.refl", "Eq.mpr", "Eq.mp", "Eq.symm", "Iff.refl", "Iff.mpr", "Iff.mp", "id_eq", "ite_eq",
   "letFun", "funext", "propext", "trivial", "Decidable", "rfl",
   "Mathlib.Tactic", "Mathlib.Meta", "_simp_", "inst", ".intro", ".symm", "iff_self", "Lean."]
def strHas (hay needle : String) : Bool := (hay.splitOn needle).length > 1
def isNoise (n : Name) : Bool := let s := toString n; noisePats.any (fun p => strHas s p)
def isPlumb (n : Name) : Bool := let s := toString n; plumbPats.any (fun p => strHas s p)
#eval show Lean.Meta.MetaM Unit from do
  let env ← getEnv
  let mods := env.header.moduleNames
  let data := env.header.moduleData
  for i in [0:mods.size] do
    if (`__NS__).isPrefixOf mods[i]! then
      for cn in data[i]!.constNames do
        match env.find? cn with
        | some (.thmInfo ti) =>
            if !isNoise cn then
              let axs ← collectAxioms cn
              if isClean axs then
                let used := ti.value.getUsedConstants
                let prem := used.filter (fun u =>
                  (match env.find? u with | some (.thmInfo _) => true | _ => false) && !isNoise u && !isPlumb u)
                if prem.size > 0 then
                  let s := ((← Lean.Meta.ppExpr ti.type).pretty 1000000).replace "\n" " " |>.replace "\t" " "
                  IO.println s!"PAIR\t{cn}\t{String.intercalate "," (prem.toList.map toString)}\t{s}"
        | _ => pure ()
'''


def harvest(proj, timeout=400):
    d = os.path.expanduser(proj["dir"])
    src = HARVEST.replace("__IMPORT__", proj["imp"]).replace("__NS__", proj["ns"])
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "Prem.lean")
        with open(f, "w") as fh: fh.write(src)
        try:
            p = subprocess.run(["lake", "env", "lean", f], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return []
    recs = []
    for ln in p.stdout.splitlines():
        if ln.startswith("PAIR\t"):
            parts = ln.split("\t", 3)
            if len(parts) == 4:
                _, name, prem, stmt = parts
                recs.append({"theorem": name, "statement": stmt,
                             "premises": [x for x in prem.split(",") if x],
                             "project": proj["ns"], "domain": proj["domain"]})
    return recs


def build():
    os.makedirs(os.path.dirname(PAIRS_JSONL), exist_ok=True)
    all_recs = []
    print("=== proofworld.premises :: harvest (goal -> cited-lemmas) training pairs ===\n")
    for proj in PROJECTS:
        print(f"  harvesting {proj['ns']} ...", flush=True)
        recs = harvest(proj)
        npre = sum(len(r["premises"]) for r in recs)
        print(f"    {len(recs):4} theorems with >=1 meaningful premise  ({npre} goal-premise edges)")
        all_recs += recs
    with open(PAIRS_JSONL, "w") as fh:
        for r in all_recs: fh.write(json.dumps(r) + "\n")
    print(f"\n  wrote {len(all_recs)} training pairs -> {os.path.relpath(PAIRS_JSONL, HERE)}")
    print("  each pair = (goal statement) -> (domain lemmas its verified proof cited). Labels are kernel-sourced.")
    return all_recs


# ---------------- retrieval-augmented few-shot ----------------
def load_pairs():
    return [json.loads(l) for l in open(PAIRS_JSONL)] if os.path.exists(PAIRS_JSONL) else []

def idf(pairs):
    df = {}
    for r in pairs:
        for t in set(TOKEN.findall(r["statement"])):
            df[t] = df.get(t, 0) + 1
    n = len(pairs) or 1
    return {t: math.log(1 + n / c) for t, c in df.items()}

def retrieve(goal, pairs, weights, k=4):
    gt = set(TOKEN.findall(goal))
    scored = []
    for r in pairs:
        ov = gt & set(TOKEN.findall(r["statement"]))
        s = sum(weights.get(t, 0.0) for t in ov)
        if s > 0: scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]

def few_shot_prompt(goal, k=4):
    pairs = load_pairs()
    if not pairs:
        print("no pairs -- run `python3 -m proofworld.premises` first"); return
    w = idf(pairs); picks = retrieve(goal, pairs, w, k)
    print(f"=== retrieval-augmented few-shot for goal ===\n  GOAL: {goal}\n")
    print(f"  retrieved {len(picks)} similar SOLVED theorems (the in-context examples the proposer would see):\n")
    examples = []
    for sc, r in picks:
        print(f"  [{sc:4.1f}] {r['theorem'].split('.')[-1]}")
        print(f"        statement: {r['statement'][:90]}")
        print(f"        cited:     {', '.join(p.split('.')[-1] for p in r['premises'][:8])}")
        examples.append(r)
    # assemble the actual RAG prompt a proposer would receive
    blocks = "\n".join(f"- Goal: {r['statement']}\n  Lemmas its proof cited: {', '.join(r['premises'])}" for r in examples)
    prompt = (f"You are proposing which lemmas to cite to prove a goal. Here are similar SOLVED theorems and the "
              f"lemmas their proofs used:\n{blocks}\n\nNow for this GOAL, list the lemmas most worth citing:\n{goal}")
    print(f"\n  --- assembled RAG few-shot prompt ({len(prompt)} chars) ---")
    print("  " + prompt.replace("\n", "\n  ")[:700] + (" ..." if len(prompt) > 700 else ""))
    # optional: live proposer conditioned on the retrieved neighbours
    if os.environ.get("PROOFWORLD_LLM") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-sonnet-4-6"), max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        print(f"\n  --- LLM premise suggestion (conditioned on retrieved verified neighbours) ---\n  {out[:500]}")
    else:
        print("\n  (set PROOFWORLD_LLM=1 to see the live proposer's suggestion conditioned on these examples.)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "fewshot":
        few_shot_prompt(" ".join(sys.argv[2:]))
    else:
        build()
