#!/usr/bin/env python3
"""code_repair_harness: the Technique Discovery loop on CODE, with a REAL failing test.

problem  = a real failing unittest         technique = a source-transformation (patch strategy)
experiment = apply patch + RUN THE SUITE    verifier  = the actual test suite (subprocess; truth)
library  = reusable debugging technique     promote   = only when the WHOLE suite goes green

The verifier is not a mock: it writes the patched module to disk and runs `python -m unittest` as a
subprocess. A patch earns promotion only if every test passes -- including a held-out generalization
test that catches a GAMING patch (special-casing the failing input). Same invariant, real verifier:
the model PROPOSES patches; the test suite OWNS truth. No credit for a plausible-looking diff.
"""
import os, sys, tempfile, subprocess
from wmos.techniques import TechniqueLibrary, EvidenceBundle

BUGGY = (
    "def clamp(x, lo, hi):\n"
    "    if x < lo: return lo\n"
    "    if x > hi: return lo   # BUG: should return hi\n"
    "    return x\n"
)

TESTS = (
    "import unittest\n"
    "from mathutil import clamp\n"
    "class T(unittest.TestCase):\n"
    "    def test_below(self): self.assertEqual(clamp(-5, 0, 10), 0)\n"        # regression
    "    def test_inside(self): self.assertEqual(clamp(5, 0, 10), 5)\n"        # regression
    "    def test_above(self): self.assertEqual(clamp(99, 0, 10), 10)\n"        # the FAILING target test
    "    def test_above_general(self): self.assertEqual(clamp(50, 0, 10), 10)\n"  # held-out: gaming one input won't pass this
)

# techniques = source transformations (a reusable patching strategy), applied to the buggy source
TECHNIQUES = {
    "fix_wrong_branch_return": lambda s: s.replace("if x > hi: return lo   # BUG: should return hi",
                                                   "if x > hi: return hi"),                # the correct general fix
    "decoy_special_case":      lambda s: s.replace("def clamp(x, lo, hi):",
                                                   "def clamp(x, lo, hi):\n    if x == 99: return 10  # GAMING"),
    "wrong_unrelated":         lambda s: s.replace("if x < lo: return lo", "if x <= lo: return lo"),  # bug remains
    "breaks_regression":       lambda s: s.replace("if x < lo: return lo", "if x < lo: return hi"),   # breaks test_below
}


def setup():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "mathutil.py"), "w").write(BUGGY)
    open(os.path.join(d, "test_mathutil.py"), "w").write(TESTS)
    return d


def run_suite(workdir):                       # the REAL verifier: run the actual tests as a subprocess
    r = subprocess.run([sys.executable, "-m", "unittest", "test_mathutil"],
                       cwd=workdir, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.stderr else ""


def apply_and_verify(workdir, technique):
    open(os.path.join(workdir, "mathutil.py"), "w").write(TECHNIQUES[technique](BUGGY))
    return run_suite(workdir)


def discover(lib, verify=True, order=("decoy_special_case", "fix_wrong_branch_return", "wrong_unrelated")):
    """propose patches (decoy first -- it 'matches' the failing input, the attractive wrong move),
    RUN THE REAL SUITE per patch, promote only the one that makes the whole suite green."""
    d = setup(); trail = []
    for name in order:
        passed, last = apply_and_verify(d, name)
        trail.append((name, passed))
        if not verify:                         # ABLATION: promote the first proposal without running tests
            lib.promote(name, {"bug": "wrong_branch_return"}, EvidenceBundle({}, {}, "promote(NO-VERIFY)"))
            return {"promoted": name, "green": passed, "trail": trail}
        if passed:
            lib.promote(name, {"bug": "wrong_branch_return"},
                        EvidenceBundle({}, {"suite_green": True}, "promote"), effect="fix the wrong-branch return")
            return {"promoted": name, "green": True, "trail": trail}
    return {"promoted": None, "green": False, "trail": trail}


print("=== code_repair_harness: the technique loop on a REAL failing test ===\n")
d0 = setup()
base_green, base_last = run_suite(d0)
print(f"  baseline (buggy module): suite green = {base_green}   [{base_last}]")

# per-patch real verification
print("\n  per-patch verification (the actual test suite owns truth):")
for name in TECHNIQUES:
    dd = setup(); g, last = apply_and_verify(dd, name)
    note = {"fix_wrong_branch_return": "the correct general fix",
            "decoy_special_case": "GAMING: passes test_above but FAILS test_above_general",
            "wrong_unrelated": "bug remains -> target still fails",
            "breaks_regression": "breaks test_below"}[name]
    print(f"     {name:24} suite_green={g}   ({note})")

lib = TechniqueLibrary()
disc = discover(lib, verify=True)
print(f"\n  discovery (verify ON): tried {[t[0] for t in disc['trail']]} -> promoted '{disc['promoted']}' (green={disc['green']})")

lib_nv = TechniqueLibrary()
nv = discover(lib_nv, verify=False)
print(f"  discovery (verify ABLATED): promoted '{nv['promoted']}' WITHOUT running tests (actually green={nv['green']})")

# transfer: the promoted technique fixes a held-out module with the same bug pattern
d2 = tempfile.mkdtemp()
open(os.path.join(d2, "mathutil.py"), "w").write(TECHNIQUES[disc["promoted"]](BUGGY))   # reuse the card's transform
open(os.path.join(d2, "test_mathutil.py"), "w").write(TESTS)
transfer_green, _ = run_suite(d2)

print()
checks = {
    "the failing test is REAL (baseline suite is red)": not base_green,
    "discovery makes the REAL suite GREEN": disc["green"] and disc["promoted"] == "fix_wrong_branch_return",
    "the GAMING patch is REJECTED by the held-out test (not promoted)": disc["promoted"] != "decoy_special_case",
    "the regression-breaking patch fails the suite (caught)": not apply_and_verify(setup(), "breaks_regression")[0],
    "the wrong patch fails the target (caught)": not apply_and_verify(setup(), "wrong_unrelated")[0],
    "VERIFIER ABLATION: the attractive decoy is FALSELY promoted (and is NOT actually green)":
        nv["promoted"] == "decoy_special_case" and not nv["green"],
    "the promoted technique TRANSFERS (reused on the same bug -> green)": transfer_green,
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\n  library: {lib.inspect()}")
print(f"\nCODE REPAIR GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the technique loop runs on CODE with a REAL failing test -- the verifier is the actual unittest"
      "\n  suite run in a subprocess, not a mock. A patch is promoted ONLY when the whole suite goes green; the GAMING"
      "\n  patch that special-cases the failing input passes that one test but is REJECTED by the held-out generalization"
      "\n  test; the wrong and regression-breaking patches are caught by the suite. Ablate the verifier and the attractive"
      "\n  decoy is falsely promoted -- the test suite, not the diff's plausibility, owns truth. The fix transfers as a"
      "\n  reusable debugging technique. Same architecture, swap the verifier: tests for code, Lean for math, Δachievable for control.")
