#!/usr/bin/env python3
"""proofworld.decompose -- recursive proof decomposition on the Lean kernel (HTPS-style, soundness-gated).

The system DECIDES to break a goal into sub-lemmas and proves it bottom-up:

  prove(goal):
    1. TRY DIRECT: a cheap tactic budget. If one closes it (kernel-verified) -> a leaf, done.
    2. else DECOMPOSE: Opus proposes sub-lemma STATEMENTS + a final proof that combines them.
    3. RECURSE: prove each sub-lemma (try-direct, else decompose again) -- the loop OWNS the sub-proofs.
    4. GATE: assemble {proved sub-lemmas} + {goal via the final proof} into ONE Lean file; the KERNEL verifies the
       whole tree and #print axioms confirms it is axiom-clean. No sorry anywhere. A bad decomposition can only
       FAIL TO CLOSE (you try another); it can never manufacture a false proof -- the same soundness discipline.

The verifier tells it WHEN to decompose (direct failed) and WHETHER a split worked (the composition compiles).
Opus proposes the structure; z3/Lean own truth. Run: PROOFWORLD_LLM=1 python3 -m proofworld.decompose
"""
from __future__ import annotations
import os, re, json, glob, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "lean")                       # portable Mathlib project (lake exe cache get)
STD = {"propext", "Classical.choice", "Quot.sound"}
PREAMBLE = ("import Mathlib.Tactic\n"
            "def f : ℕ → ℕ\n  | 0 => 0\n  | (n+1) => f n + 2*n + 1\n")   # f n = n^2 (sum of odds)
GOAL = ("pw_goal", "(n : ℕ) : f n = n^2")
DIRECT = ["by rfl", "by simp", "by omega", "by decide", "by norm_num", "by simp [f]"]   # the cheap budget


def _lean(body, timeout=120):
    src = PREAMBLE + body
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "Dec.lean")
        open(p, "w").write(src)
        try:
            r = subprocess.run(["lake", "env", "lean", p], cwd=PROJECT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "", {99999}
    out = r.stdout + r.stderr
    errlines = {int(m.group(1)) for m in re.finditer(r"Dec\.lean:(\d+):\d+: error:", out)}
    return out, errlines


def _axioms(out, thm):
    ax = next((l for l in out.splitlines() if "depends on axioms" in l), "")
    return [a for a in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", ax.split(":", 1)[1]) if a not in ("depends", "on", "axioms")] if ":" in ax else []


def try_direct(stmt):
    """one Lean call: attempt every cheap tactic; return (tactic, axioms) for the first that closes clean."""
    base = len(PREAMBLE.splitlines())
    lines, where = [], {}
    for i, tac in enumerate(DIRECT):
        where[base + len(lines) + 1] = i
        lines.append(f"theorem pw_d{i} {stmt} := {tac}")
    out, errs = _lean("\n".join(lines) + "\n")
    for ln, i in where.items():
        if ln not in errs and "sorry" not in out:                  # clean compile of attempt i
            return DIRECT[i], []
    return None, None


def opus_decompose(name, stmt, err=None, log=print):
    if os.environ.get("PROOFWORLD_LLM") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    extra = f"\nYour previous attempt failed with Lean error:\n{err}\nFix it." if err else ""
    prompt = (
        "In Lean 4 with Mathlib, given this preamble:\n```\n" + PREAMBLE + "```\n"
        f"Prove the theorem  `theorem {name} {stmt}`  by DECOMPOSING it into helper lemmas.\n"
        "Reply ONLY with JSON: {\"lemmas\": [{\"name\": \"<id>\", \"sig\": \"(binders) : <prop>\"}], "
        "\"main_proof\": \"by <tactic block that proves the goal using the lemma names>\"}. "
        "Keep lemma sigs in the same '(binders) : prop' form. The lemmas should be simpler than the goal "
        "(e.g. a one-step unfolding fact provable by rfl)." + extra)
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("PROOFWORLD_LLM_MODEL", "claude-opus-4-8"), max_tokens=900,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception as e:
        log(f"    (Opus JSON parse failed: {e})"); return None


def prove(name, stmt, depth=0, indent="  ", log=print):
    log(f"{indent}GOAL [{name}]: {stmt}")
    tac, _ = try_direct(stmt)
    if tac:
        log(f"{indent}  try-direct: PROVED by `{tac}`  (leaf)")
        return {"ok": True, "name": name, "stmt": stmt, "kind": "direct", "proof": tac, "subs": []}
    log(f"{indent}  try-direct: FAILS (cheap tactics can't) -> DECOMPOSE")
    if depth >= 2:
        log(f"{indent}  max depth -> give up"); return {"ok": False}
    err = None
    for attempt in range(3):                                                # agentic retry: feed kernel errors back
        dec = opus_decompose(name, stmt, err)
        if not dec or "lemmas" not in dec:
            log(f"{indent}  no decomposition proposed"); return {"ok": False}
        log(f"{indent}  Opus proposes {len(dec['lemmas'])} sub-lemma(s) + a combining proof"
            + (f" (retry {attempt})" if attempt else ""))
        subs = []
        for L in dec["lemmas"]:
            r = prove(L["name"], L["sig"], depth + 1, indent + "    ")       # RECURSE
            if r["ok"]: subs.append(r)
        # GATE: assemble proved sub-lemmas + the goal (via main_proof); the KERNEL verifies the whole tree
        body = "".join(f"theorem {s['name']} {s['stmt']} := {s['proof']}\n" for s in subs)
        body += f"theorem {name} {stmt} := {dec['main_proof']}\n#print axioms {name}\n"
        out, errs = _lean(body)
        gline = len(PREAMBLE.splitlines()) + len(subs) + 1
        ok = (gline not in errs) and "sorry" not in out and "error:" not in out
        ax = _axioms(out, name); clean = ok and bool(ax) and set(ax) <= STD
        if clean:
            log(f"{indent}  GATE: composition VERIFIED by the kernel, axiom-clean {ax}")
            return {"ok": True, "name": name, "stmt": stmt, "kind": "decomp", "proof": dec["main_proof"], "subs": subs}
        err = next((l for l in out.splitlines() if "error:" in l), "unknown")[:300]
        log(f"{indent}  GATE: composition rejected by kernel ({err[:70]}) -> repair")
    return {"ok": False, "out": out}


def render(node, indent="    "):
    if not node.get("ok"): return
    tag = f"by {node['proof'].splitlines()[0]}" if node["kind"] == "direct" else "decompose:"
    print(f"{indent}{node['name']} : {node['stmt']}   <- {tag}")
    for s in node["subs"]:
        render(s, indent + "    ")


def main():
    print("=== proofworld.decompose :: recursive decomposition on the Lean kernel (try-direct -> split -> verify) ===\n")
    print(f"  preamble defines f with f n = n^2 (sum of odds).")
    res = prove(*GOAL)
    print("\n  --- proof tree (every node kernel-verified) ---")
    render(res) if res["ok"] else print("    (goal not proved within budget)")
    print(f"\n  RESULT: goal {'PROVED by recursive decomposition, axiom-clean' if res['ok'] else 'NOT proved'}.")
    print("  The verifier decided WHEN to decompose (direct failed) and WHETHER the split worked (kernel compiled it).")
    print("  Opus proposed the structure; the kernel owned truth at every node -- a bad split could only fail, never lie.")


if __name__ == "__main__":
    main()
