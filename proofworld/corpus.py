#!/usr/bin/env python3
"""proofworld.corpus -- import the user's VERIFIED Lean corpus as a citable library, AXIOM-CLEAN ONLY.

A theorem is admitted only if its axiom footprint is clean -- a subset of {propext, Classical.choice, Quot.sound}
-- so NOTHING resting on a `sorry` or a project-local unproven `axiom` (an unproven hypothesis) ever becomes a
citable fact. This is the soundness gate from research_hard/research_lean, applied at corpus scale: provenance is
enforced by the kernel's own `collectAxioms`, not trusted.

For each project we run a Lean meta-harvester (via `lake env lean`, reusing the project's built oleans) that walks
the project's modules, computes every theorem's axiom footprint, drops compiler-generated noise, and emits the
clean theorems with their statements + footprint. Conditional theorems ([H] -> P) are kept -- they are genuine
verified implications; citing one still requires discharging H (proofworld's soundness gate handles that).

Pilot = the v4.30.0-rc2 cluster (same toolchain, oleans load). NB toolchains span v4.12-v4.30 across all projects;
each cluster is harvested in its own toolchain (you cannot mix incompatible oleans in one Lean process).

Usage:
  python3 -m proofworld.corpus            # harvest the pilot projects -> proofworld/corpus/corpus.jsonl + SUMMARY.md
  python3 -m proofworld.corpus search tau # substring-search the harvested corpus (premise-selection stub)
"""
from __future__ import annotations
import os, sys, json, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
CORPUS_JSONL = os.path.join(CORPUS_DIR, "corpus.jsonl")
SUMMARY_MD = os.path.join(CORPUS_DIR, "SUMMARY.md")

# pilot cluster: same toolchain (v4.30.0-rc2), own oleans load compatibly
PILOT = [
    {"dir": "~/RamanujanTau", "imp": "RamanujanTau", "ns": "RamanujanTau", "domain": "modular forms / Ramanujan tau"},
    {"dir": "~/PlonkLean",    "imp": "PlonkLean",    "ns": "PlonkLean",    "domain": "PLONK zero-knowledge proof system"},
]

# Lean meta-harvester (sentinels __IMPORT__/__NS__ filled per project; braces are Lean's so use .replace not .format)
HARVEST = r'''import __IMPORT__
import Lean
open Lean
def stdAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]
def isClean (axs : Array Name) : Bool := axs.all (fun a => stdAxioms.contains a)
def noisePats : List String :=
  ["_proof_", "congr", ".eq_", ".match_", ".brec", ".below", "_cstage", ".noConfusion",
   ".sizeOf", ".inj", "._unfold", ".rec", ".cases", ".induct", "._sunfold", ".eq_def"]
def strHas (hay needle : String) : Bool := (hay.splitOn needle).length > 1
def isNoise (n : Name) : Bool := let s := toString n; noisePats.any (fun p => strHas s p)
#eval show Lean.Meta.MetaM Unit from do
  let env ← getEnv
  let mods := env.header.moduleNames
  let data := env.header.moduleData
  let mut nReal := 0
  let mut nClean := 0
  let mut nSorry := 0
  let mut nDirty := 0
  for i in [0:mods.size] do
    if (`__NS__).isPrefixOf mods[i]! then
      for cn in data[i]!.constNames do
        match env.find? cn with
        | some (.thmInfo ti) =>
            if !isNoise cn then
              nReal := nReal + 1
              let axs ← collectAxioms cn
              if isClean axs then
                nClean := nClean + 1
                let s := ((← Lean.Meta.ppExpr ti.type).pretty 1000000).replace "\n" " " |>.replace "\t" " "
                IO.println s!"CLEAN\t{cn}\t{mods[i]!}\t{String.intercalate "," (axs.toList.map toString)}\t{s}"
              else if axs.contains ``sorryAx then nSorry := nSorry + 1
              else nDirty := nDirty + 1
        | _ => pure ()
  IO.println s!"STATS\t{nReal}\t{nClean}\t{nDirty}\t{nSorry}"
'''


def harvest_project(proj, timeout=300):
    """run the meta-harvester in the project (lake env lean) and return (records, stats)."""
    d = os.path.expanduser(proj["dir"])
    src = HARVEST.replace("__IMPORT__", proj["imp"]).replace("__NS__", proj["ns"])
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "Harvest.lean")
        with open(f, "w") as fh:
            fh.write(src)
        try:
            p = subprocess.run(["lake", "env", "lean", f], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [], {"error": "timeout"}
    records, stats = [], {}
    for ln in p.stdout.splitlines():
        if ln.startswith("CLEAN\t"):
            parts = ln.split("\t", 4)
            if len(parts) == 5:
                _, name, module, axs, stmt = parts
                records.append({"name": name, "statement": stmt, "module": module,
                                "axioms": [a for a in axs.split(",") if a],
                                "project": proj["ns"], "domain": proj["domain"]})
        elif ln.startswith("STATS\t"):
            _, nReal, nClean, nDirty, nSorry = ln.split("\t")
            stats = {"real_theorems": int(nReal), "clean": int(nClean), "axiom_dependent": int(nDirty), "sorry": int(nSorry)}
    if not records and p.returncode != 0:
        stats = stats or {"error": (p.stderr or p.stdout)[:200]}
    return records, stats


def build():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    all_recs, summary = [], []
    print("=== proofworld.corpus :: importing the verified corpus (AXIOM-CLEAN ONLY) ===\n")
    for proj in PILOT:
        print(f"  harvesting {proj['ns']:14} ({proj['domain']}) ...", flush=True)
        recs, stats = harvest_project(proj)
        all_recs += recs
        if "error" in stats:
            print(f"    ERROR: {stats['error']}"); summary.append((proj, stats)); continue
        print(f"    real theorems {stats['real_theorems']:4}  ->  CLEAN {stats['clean']:4} kept  |  "
              f"REJECTED {stats['axiom_dependent']} axiom-dependent + {stats['sorry']} sorry")
        summary.append((proj, stats))
    with open(CORPUS_JSONL, "w") as fh:
        for r in all_recs:
            fh.write(json.dumps(r) + "\n")
    with open(SUMMARY_MD, "w") as fh:
        fh.write("# proofworld verified corpus (axiom-clean only)\n\n")
        fh.write("Admitted iff axiom footprint ⊆ {propext, Classical.choice, Quot.sound} (kernel `collectAxioms`).\n")
        fh.write("Rejected = depends on a `sorry` or a project-local unproven `axiom`. Conditional theorems [H]→P kept.\n\n")
        fh.write("| project | domain | clean (kept) | axiom-dependent (rejected) | sorry (rejected) |\n")
        fh.write("|---|---|---|---|---|\n")
        for proj, st in summary:
            if "error" in st: fh.write(f"| {proj['ns']} | {proj['domain']} | ERROR | | |\n"); continue
            fh.write(f"| {proj['ns']} | {proj['domain']} | {st['clean']} | {st['axiom_dependent']} | {st['sorry']} |\n")
        fh.write(f"\n**Total citable facts: {len(all_recs)}** in `corpus.jsonl` (name, statement, module, axioms, project, domain).\n")
    print(f"\n  wrote {len(all_recs)} citable facts -> {os.path.relpath(CORPUS_JSONL, HERE)}")
    print(f"  summary -> {os.path.relpath(SUMMARY_MD, HERE)}")
    print(f"\n  GATE: every admitted fact is kernel-certified axiom-clean; nothing sorry- or hypothesis-tainted is citable.")
    return all_recs


def search(query):
    if not os.path.exists(CORPUS_JSONL):
        print("no corpus yet -- run `python3 -m proofworld.corpus` first"); return
    q = query.lower(); hits = 0
    print(f"=== corpus search: '{query}' (premise-selection stub: substring over statements+names) ===\n")
    with open(CORPUS_JSONL) as fh:
        for ln in fh:
            r = json.loads(ln)
            if q in r["statement"].lower() or q in r["name"].lower():
                hits += 1
                print(f"  [{r['project']}] {r['name']}\n      {r['statement'][:140]}")
    print(f"\n  {hits} matching axiom-clean facts.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        search(" ".join(sys.argv[2:]))
    else:
        build()
