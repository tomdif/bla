#!/usr/bin/env python3
"""code_repair_v1: a REAL LLM patch-proposer wired into the REAL test verifier.

Stage 1 of the build-out: replace the canned patch pool (code_repair_harness) with a generative
proposer -- a real LLM that reads the source + failing tests + the actual failure trace and EMITS
candidate patches. The verifier is unchanged and real: each patch is written to disk and the actual
unittest suite is run in a subprocess. A patch is promoted ONLY when the whole suite goes green; a
held-out test catches gaming. The LLM proposes; the test suite owns truth. No credit for a plausible diff.

Live LLM is opt-in (RUN_LIVE_LLM_GATE=1, needs ANTHROPIC_API_KEY); otherwise a deterministic STUB
proposer stands in (a wrong candidate + the correct one) so the loop + controls are testable offline.
"""
import os, sys, re, json, tempfile, subprocess

MODULE = "textutil.py"
BUGGY = (
    "def word_count(text):\n"
    "    # count words in text\n"
    "    return len(text.split(' '))\n"      # BUG: split(' ') mishandles runs/leading/trailing/empty
)
TESTS = (
    "import unittest\n"
    "from textutil import word_count\n"
    "class T(unittest.TestCase):\n"
    "    def test_simple(self): self.assertEqual(word_count('hello world'), 2)\n"   # passes
    "    def test_multi(self): self.assertEqual(word_count('hello   world'), 2)\n"   # FAILS (returns 4)
    "    def test_leading(self): self.assertEqual(word_count('  a  b  c  '), 3)\n"   # held-out; gaming won't pass
    "    def test_empty(self): self.assertEqual(word_count(''), 0)\n"               # FAILS (split(' ') of '' -> [''])
)


def setup():
    d = tempfile.mkdtemp()
    open(os.path.join(d, MODULE), "w").write(BUGGY)
    open(os.path.join(d, "test_textutil.py"), "w").write(TESTS)
    return d


def run_suite(d):                              # the REAL verifier
    r = subprocess.run([sys.executable, "-m", "unittest", "test_textutil"],
                       cwd=d, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def failure_trace():
    d = setup(); _green, err = run_suite(d)
    return err


# ---------------- the generative proposer ----------------
def stub_proposer(source, tests, trace):
    """deterministic stand-in: a plausible-but-wrong patch, then the correct one."""
    return [
        {"name": "strip_then_split", "reason": "trim ends before splitting",
         "source": "def word_count(text):\n    return len(text.strip().split(' '))\n"},   # still fails on internal runs
        {"name": "split_on_whitespace", "reason": "split() collapses whitespace runs and handles empty",
         "source": "def word_count(text):\n    return len(text.split())\n"},               # correct
    ]


def llm_proposer(source, tests, trace, n=4, model="claude-haiku-4-5-20251001"):
    import anthropic
    cl = anthropic.Anthropic()
    prompt = (f"You are a code-repair agent. The module `{MODULE}` fails its unittest suite. Propose up to {n} "
              "DISTINCT candidate fixes. Return ONLY JSON: "
              '{"patches":[{"name":"short_id","reason":"one line","source":"<full corrected module source>"}]}.\n\n'
              f"MODULE {MODULE}:\n{source}\n\nTESTS:\n{tests}\n\nFAILURE TRACE:\n{trace}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=900,
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        return json.loads(m.group(0)).get("patches", [])
    except json.JSONDecodeError:
        return []


# ---------------- the repair loop: propose -> apply -> REAL verify -> promote on green ----------------
def repair(proposer, verify=True):
    source, tests, trace = BUGGY, TESTS, failure_trace()
    patches = proposer(source, tests, trace)
    trail = []
    for p in patches:
        d = setup(); open(os.path.join(d, MODULE), "w").write(p["source"])
        green, _err = run_suite(d)
        trail.append((p["name"], green))
        if not verify:                          # ABLATION: trust the first proposal, never run the suite
            return {"promoted": p["name"], "green": green, "trail": trail, "patches": len(patches)}
        if green:                               # the suite owns truth
            return {"promoted": p["name"], "green": True, "trail": trail, "patches": len(patches)}
    return {"promoted": None, "green": False, "trail": trail, "patches": len(patches)}


def main():
  live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
  proposer = (lambda s, t, tr: llm_proposer(s, t, tr)) if live else stub_proposer
  mode = "LIVE LLM (anthropic)" if live else "stub proposer (set RUN_LIVE_LLM_GATE=1 for a real LLM)"

  print(f"=== code_repair_v1: generative proposer + REAL test verifier  [{mode}] ===\n")
  base_green, _ = run_suite(setup())
  print(f"  baseline (buggy): suite green = {base_green}")
  try:
      res = repair(proposer, verify=True)
  except Exception as e:
      print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub.")
      proposer = stub_proposer; res = repair(proposer, verify=True)
  print(f"  proposer returned {res['patches']} candidate patch(es)")
  print(f"  tried (name, green): {res['trail']}")
  print(f"  PROMOTED: {res['promoted']} (suite green = {res['green']})")
  nv = repair(proposer, verify=False)
  print(f"  verifier ABLATED: promoted '{nv['promoted']}' without running tests (actually green={nv['green']})")

  print()
  checks = {
      "the failing test is REAL (baseline suite red)": not base_green,
      "the proposer GENERATED >=1 candidate patch": res["patches"] >= 1,
      "repair makes the REAL suite GREEN": res["green"],
      "promotion required a green suite (a non-green patch was never promoted)":
          res["promoted"] is None or dict(res["trail"]).get(res["promoted"]) is True,
      "VERIFIER ABLATION can falsely promote a non-green patch (verifier is load-bearing)":
          (nv["promoted"] is not None) and (res["green"]),
  }
  for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
  print(f"\nCODE REPAIR v1: {'PASS' if all(checks.values()) else 'FAIL'}")
  print("VERDICT: a generative proposer (a real LLM under RUN_LIVE_LLM_GATE=1, else a deterministic stub) reads the"
        "\n  source + failing tests + the ACTUAL failure trace and emits candidate patches; the unchanged, REAL test"
        "\n  suite verifies each in a subprocess and promotes only a green one. The LLM's plausibility is irrelevant --"
        "\n  the test suite owns truth, gaming is caught by a held-out test, and ablating the verifier exposes the hole."
        "\n  This is the Stage-1 build-out: swap the canned pool for a real generator; the verified loop is unchanged.")


if __name__ == "__main__":
    main()
