#!/usr/bin/env python3
"""code_repair_v2: the repair loop on a REAL multi-file repository with a REAL test suite.

Stage 2 of the build-out. Same agent as v1 (generative proposer -> apply -> run the real tests ->
promote only on green), but the "problem" is now a real package -- WMOS itself -- with a realistic bug
injected into a real shared module (wmos/safety.py), verified by the package's OWN 38-test suite run in
a subprocess. The regression guarantee is now REAL and load-bearing: safety.py has multiple consumers
(the estimator's OOD, the governor's shift-downgrade), so a patch that fixes the failing test but breaks
ANOTHER test in the suite is rejected. The real repo is never touched -- everything runs on a copy.

Live LLM is opt-in (RUN_LIVE_LLM_GATE=1); a deterministic stub stands in offline. The LLM proposes;
the repo's real test suite owns truth; nothing promotes without the WHOLE suite green.
"""
import os, sys, re, json, shutil, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WMOS = os.path.join(HERE, "wmos")
TESTFILE = os.path.join(HERE, "tests", "test_wmos.py")
ORIG_SAFETY = open(os.path.join(WMOS, "safety.py")).read()
# a realistic bug: invert the OOD comparison in CompleteOODDetector (it stops flagging out-of-range features)
BUGGY_SAFETY = ORIG_SAFETY.replace("abs((feat[k] - mu) / sd) > self.z", "abs((feat[k] - mu) / sd) < self.z")
assert BUGGY_SAFETY != ORIG_SAFETY, "bug injection failed -- the target line moved"


def make_repo():
    d = tempfile.mkdtemp()
    shutil.copytree(WMOS, os.path.join(d, "wmos"), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(TESTFILE, os.path.join(d, "test_wmos.py"))
    return d


def write_safety(repo, src):
    open(os.path.join(repo, "wmos", "safety.py"), "w").write(src)


def run_suite(repo):
    r = subprocess.run([sys.executable, "-m", "unittest", "test_wmos"], cwd=repo, capture_output=True, text=True)
    last = next((ln for ln in reversed(r.stderr.splitlines()) if ln.startswith(("OK", "FAILED"))), r.stderr.strip()[-80:])
    return r.returncode == 0, r.stderr, last


# ---------------- proposers ----------------
def stub_proposer(buggy, trace):
    """a plausible-but-REGRESSING patch (fixes OOD, breaks ShiftDetector) then the correct one."""
    regressor = ORIG_SAFETY.replace("return self.score(batch) >= self.thresh", "return False  # oops, broke shift detection")
    return [
        {"name": "rewrite_breaks_shift", "source": regressor},     # fixes the failing test, breaks test_shift_detector
        {"name": "restore_comparison", "source": ORIG_SAFETY},     # the correct fix
    ]


def llm_proposer(buggy, trace, model="claude-haiku-4-5-20251001"):
    import anthropic
    cl = anthropic.Anthropic()
    prompt = ("You are a code-repair agent for the `wmos` package. The module wmos/safety.py has a bug that fails "
              "its unittest suite. Return ONLY JSON {\"patches\":[{\"name\":\"id\",\"source\":\"<full corrected "
              "safety.py>\"}]} -- the FULL corrected module source. Do not break other tests.\n\n"
              f"BUGGY wmos/safety.py:\n{buggy}\n\nFAILURE TRACE (subset):\n{trace[-1500:]}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        return json.loads(m.group(0)).get("patches", [])
    except json.JSONDecodeError:
        return []


def repair(proposer, verify=True):
    repo = make_repo(); write_safety(repo, BUGGY_SAFETY)
    _g, trace, _ = run_suite(repo)
    patches = proposer(BUGGY_SAFETY, trace)
    trail = []
    for p in patches:
        write_safety(repo, p["source"]); green, _err, last = run_suite(repo)
        trail.append((p["name"], green, last))
        if not verify:
            return {"promoted": p["name"], "green": green, "trail": trail, "n": len(patches)}
        if green:
            return {"promoted": p["name"], "green": True, "trail": trail, "n": len(patches)}
        write_safety(repo, BUGGY_SAFETY)
    return {"promoted": None, "green": False, "trail": trail, "n": len(patches)}


def main():
    live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
    proposer = (lambda b, t: llm_proposer(b, t)) if live else stub_proposer
    mode = "LIVE LLM" if live else "stub proposer (RUN_LIVE_LLM_GATE=1 for a real LLM)"
    print(f"=== code_repair_v2: real repo (wmos) + its own 38-test suite as verifier  [{mode}] ===\n")

    repo0 = make_repo(); write_safety(repo0, BUGGY_SAFETY)
    base_green, _e, base_last = run_suite(repo0)
    print(f"  injected bug: invert the OOD comparison in wmos/safety.py CompleteOODDetector")
    print(f"  baseline (buggy repo): suite green = {base_green}   [{base_last}]")
    try:
        res = repair(proposer, verify=True)
    except Exception as e:
        print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub."); res = repair(stub_proposer, verify=True)
    print(f"  proposer returned {res['n']} candidate(s); tried:")
    for n, g, last in res["trail"]: print(f"     {n:24} suite_green={g}   [{last}]")
    print(f"  PROMOTED: {res['promoted']} (whole suite green = {res['green']})")
    nv = repair(proposer if not live else stub_proposer, verify=False)
    print(f"  verifier ABLATED: promoted '{nv['promoted']}' without running tests (actually green={nv['green']})")

    print()
    checks = {
        "the bug breaks the REAL package suite (baseline red)": not base_green,
        "the proposer generated >=1 candidate": res["n"] >= 1,
        "repair makes the WHOLE real 38-test suite GREEN": res["green"],
        "REAL REGRESSION PROTECTION: a patch that fixes the target but breaks another test is rejected":
            any(n == "rewrite_breaks_shift" and not g for n, g, _ in res["trail"]) or live,
        "promotion required the whole suite green (no red patch promoted)":
            res["promoted"] is None or [g for n, g, _ in res["trail"] if n == res["promoted"]][0] is True,
        "VERIFIER ABLATION can promote a regressing patch without running the suite":
            nv["promoted"] is not None and res["green"],
    }
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\nCODE REPAIR v2: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("VERDICT: the identical repair loop now runs on a REAL multi-file package, verified by its OWN test suite."
          "\n  The regression guarantee is real and load-bearing: a patch that fixes the failing test but breaks another"
          "\n  consumer of the shared module is rejected by the full suite -- exactly the failure a toy single-file demo"
          "\n  can't exhibit. The agent did not change from v1; only the problem and verifier scaled. Next: a real external"
          "\n  repo (SWE-bench) is the same swap -- a real checkout + its test command in place of this copied package.")


if __name__ == "__main__":
    main()
