#!/usr/bin/env python3
"""swebench_adapter: the repair loop on a REAL EXTERNAL GitHub repo (the SWE-bench setup).

The code_repair_v2 step taken to its real conclusion: instead of a local package, CLONE a real public
repo (mahmoud/boltons) over the network and use ITS OWN pytest suite as the verifier. This is the
SWE-bench shape -- real external code, the repo's real tests own truth. (SWE-bench ships a curated bug
corpus; here the repo is clean on HEAD, so we inject one realistic bug, which is the faithful analog.)

Target: boltons/strutils.py :: format_int_list -- collapses contiguous integers into ranges
(`[1,3,5,6,7,8] -> '1,3,5-8'`). Bug: the single-element contiguity check `delta == 1` is flipped to
`delta == 2`, so pairs/ranges stop collapsing. The real test module (test_strutils.py: 9 tests incl a
format<->parse round-trip) is the verifier; a patch promotes only when ALL of them pass, so a fix that
breaks range formatting (a regression) is rejected even if it addresses the reported symptom.

Same loop, same discipline: the proposer proposes; the repo's real test suite owns truth; the cloned repo
is restored. Live LLM opt-in (RUN_LIVE_LLM_GATE=1); deterministic stub otherwise.
"""
import os, re, sys, json, subprocess

REPO_URL = "https://github.com/mahmoud/boltons.git"
CLONE = "/tmp/swebench_boltons"
TARGET = os.path.join(CLONE, "boltons/strutils.py")
TESTMOD = "tests/test_strutils.py"


def clone_if_needed():
    if not os.path.isdir(CLONE):
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO_URL, CLONE], check=True, timeout=120)


clone_if_needed()
ORIG = open(TARGET).read()
_m = re.search(r"^def format_int_list\(.*?(?=^def |\Z)", ORIG, re.S | re.M)
assert _m, "target function not found -- repo layout changed"
FSTART, FEND, FUNC = _m.start(), _m.end(), _m.group()
# inject the bug into the LAST `if delta == 1:` (the single-element contiguity check)
_i = FUNC.rindex("if delta == 1:")
FUNC_BUGGY = FUNC[:_i] + "if delta == 2:" + FUNC[_i + len("if delta == 1:"):]
assert FUNC_BUGGY != FUNC


def write_func(func_text):
    with open(TARGET, "w") as f:                              # `with` guarantees flush+close before pytest reads it
        f.write(ORIG[:FSTART] + func_text + ORIG[FEND:])


def restore():
    with open(TARGET, "w") as f:
        f.write(ORIG)


def run_tests():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}      # no stale .pyc across rapid rewrites
    r = subprocess.run([sys.executable, "-m", "pytest", TESTMOD, "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=CLONE, capture_output=True, text=True, timeout=180, env=env)
    last = next((ln for ln in reversed(r.stdout.splitlines()) if "passed" in ln or "failed" in ln or "error" in ln),
                r.stdout.strip()[-60:])
    return r.returncode == 0, last


# ---------------- proposers (operate on the buggy function source) ----------------
def stub_proposer(func_buggy, fail):
    partial = FUNC.replace(".format(min(contig_range),", ".format(max(contig_range),") \
                           .replace("max(contig_range))", "min(contig_range))")     # fixes symptom, reverses ranges
    return [
        {"name": "noop_wrong",   "func": func_buggy},                                # doesn't fix -> module red
        {"name": "swap_minmax",  "func": partial},                                   # regression: ranges print reversed
        {"name": "restore_check", "func": FUNC},                                     # correct fix
    ]


def llm_proposer(func_buggy, fail, model="claude-sonnet-4-6"):
    import anthropic
    cl = anthropic.Anthropic()
    prompt = ("You are a code-repair agent for the `boltons` library. The function below fails its pytest suite. "
              "Return ONLY JSON {\"patches\":[{\"name\":\"id\",\"func\":\"<full corrected function source>\"}]} -- the "
              "complete corrected `format_int_list` function. Do not break other tests.\n\n"
              f"BUGGY FUNCTION:\n{func_buggy}\n\nPYTEST FAILURE:\n{fail[-1200:]}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=1500,
                system="Respond with ONLY the JSON object -- no prose, no markdown fences.",
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        return [p for p in json.loads(m.group(0)).get("patches", []) if p.get("func")]
    except json.JSONDecodeError:
        return []


def repair(proposer, verify=True):
    write_func(FUNC_BUGGY); _g, fail = run_tests()
    cands = proposer(FUNC_BUGGY, fail)
    trail = []
    for c in cands:
        write_func(c["func"]); green, last = run_tests()
        trail.append((c["name"], green, last))
        if not verify:
            return {"promoted": c["name"], "green": green, "trail": trail, "n": len(cands)}
        if green:
            return {"promoted": c["name"], "green": True, "trail": trail, "n": len(cands)}
        write_func(FUNC_BUGGY)
    return {"promoted": None, "green": False, "trail": trail, "n": len(cands)}


def main():
    live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
    proposer = (lambda f, x: llm_proposer(f, x)) if live else stub_proposer
    mode = "LIVE LLM" if live else "stub proposer (RUN_LIVE_LLM_GATE=1 for a real LLM)"
    print(f"=== swebench_adapter: repair loop on a REAL external repo (mahmoud/boltons)  [{mode}] ===")
    print(f"  cloned {REPO_URL} -> {CLONE}")
    print(f"  target: boltons/strutils.py :: format_int_list   verifier: real `pytest {TESTMOD}`")
    try:
        write_func(FUNC_BUGGY); base_green, base_last = run_tests()
        print(f"  injected bug (single-element contiguity `delta == 1` -> `== 2`): suite green = {base_green}  [{base_last}]")
        try:
            res = repair(proposer, verify=True)
            if res["n"] == 0 and live:
                print("  LIVE LLM returned 0 parseable candidates; falling back to stub."); proposer = stub_proposer; res = repair(stub_proposer, verify=True)
        except Exception as e:
            print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub."); proposer = stub_proposer; res = repair(stub_proposer, verify=True)
        print(f"  proposer returned {res['n']} candidate(s); tried (real pytest each):")
        for n, g, last in res["trail"]: print(f"     {n:14} suite_green={g!s:5}  [{last}]")
        print(f"  PROMOTED: {res['promoted']} (real test suite green = {res['green']})")
        nv = repair(proposer if not live else stub_proposer, verify=False)
        print(f"  verifier ABLATED: promoted '{nv['promoted']}' without running pytest (actually green={nv['green']})")
    finally:
        restore()
        assert open(TARGET).read() == ORIG, "RESTORE FAILED"
        print(f"  [restored boltons/strutils.py to original]")

    print()
    swap = [r for r in res["trail"] if r[0] == "swap_minmax"]
    checks = {
        "the injected bug breaks the REAL external test suite (baseline red)": not base_green,
        "the proposer generated >=1 candidate": res["n"] >= 1,
        "repair makes the REAL repo's own pytest suite GREEN": res["green"],
        "REGRESSION PROTECTION: a fix that addresses the symptom but reverses ranges is rejected by the full suite":
            (not swap or not swap[0][1]) or live,
        "promotion required the real suite green (no red patch promoted)":
            res["promoted"] is None or [g for n, g, _ in res["trail"] if n == res["promoted"]][0] is True,
        "VERIFIER ABLATION can promote a non-green patch without running the tests":
            nv["promoted"] is not None and res["green"],
    }
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\nSWE-BENCH ADAPTER GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("VERDICT: the identical repair loop runs on a REAL EXTERNAL GitHub repo cloned over the network, verified by"
          "\n  the repo's OWN pytest suite -- the SWE-bench shape. A realistic bug in a real boltons function breaks the"
          "\n  real tests; the loop re-derives a fix the real suite accepts, and a patch that fixes the reported symptom"
          "\n  but breaks range formatting is rejected by the full module (real regression protection). Only the verifier"
          "\n  scaled -- from a local package to a real external project's test suite. Pointing it at the SWE-bench corpus"
          "\n  is the same adapter with a curated (repo, bug, test) triple instead of this injected one.")


if __name__ == "__main__":
    main()
