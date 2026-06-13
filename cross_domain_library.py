#!/usr/bin/env python3
"""cross_domain_library: ONE technique store spanning code (pytest/unittest) and math (Lean 4).

The capstone of the arc: a single UnifiedStore keyed by an ABSTRACT bottleneck signature, holding
techniques with per-domain realizations (a code patch vs a Lean tactic). A technique verified in one
domain becomes PROPOSABLE for a same-signature bottleneck in another -- but transfer is PROPOSED, never
assumed. The target domain's REAL verifier must confirm it: a genuine shared meta-strategy transfers, a
spurious one is rejected. Both verifiers are real (real `python -m unittest`; real `lean`).

The invariant, lifted across domains: the store proposes across code and math; each domain's verifier
owns truth; nothing is marked verified-in-a-domain without that domain's verifier saying so.
"""
import os, shutil, tempfile, subprocess

LEAN = shutil.which("lean") or os.path.expanduser("~/.elan/bin/lean")

# an abstract technique "linear_identity" (a goal/test that reduces to a linear-arithmetic identity),
# with a CONCRETE realization in each domain. The cross-domain claim: the obstruction CLASS is shared.
TECH = {
    "linear_identity": {
        "lean": ("theorem t (n : Nat) : 0 + n = n :=", "by omega"),     # genuine: omega closes it
        "code": ("def zero_add(n):\n    return n\n",                     # genuine: the real fix
                 "from m import zero_add\nimport unittest\n"
                 "class T(unittest.TestCase):\n"
                 "    def test(self): self.assertEqual([zero_add(k) for k in range(5)], list(range(5)))\n"),
    },
}
# a SPURIOUS cross-domain transfer: a "technique" whose Lean realization is bogus (`rfl` does NOT close
# 0 + n = n -- not definitionally equal). It must be REJECTED by the real Lean verifier on re-verification.
SPURIOUS_LEAN = ("theorem t (n : Nat) : 0 + n = n :=", "by rfl")


def lean_verify(decl, tactic):
    d = tempfile.mkdtemp(); f = os.path.join(d, "T.lean")
    open(f, "w").write(f"{decl} {tactic}\n#print axioms t\n")
    r = subprocess.run([LEAN, f], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    return (r.returncode == 0) and ("error:" not in out) and ("sorryAx" not in out)


def code_verify(module_src, test_src):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "m.py"), "w").write(module_src)
    open(os.path.join(d, "test_m.py"), "w").write(test_src)
    import sys
    r = subprocess.run([sys.executable, "-m", "unittest", "test_m"], cwd=d, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


VERIFY = {"lean": lambda real: lean_verify(*real), "code": lambda real: code_verify(*real)}


class UnifiedStore:
    """techniques keyed by abstract signature; each card holds per-domain realizations + the set of
    domains whose REAL verifier has confirmed it."""
    def __init__(self): self.cards = {}
    def card(self, sig): return self.cards.setdefault(sig, {"real": {}, "verified": set()})
    def register(self, sig, domain, real): self.card(sig)["real"][domain] = real      # known, NOT yet verified here
    def promote(self, sig, domain, real): c = self.card(sig); c["real"][domain] = real; c["verified"].add(domain)
    def propose(self, sig, domain):
        """return (realization, already_verified_in_this_domain) or None -- a PROPOSAL, not a promotion."""
        c = self.cards.get(sig)
        if not c or domain not in c["real"]: return None
        return c["real"][domain], (domain in c["verified"])


def solve(store, sig, domain, real, reverify=True):
    """propose-or-explore, then VERIFY with the domain's REAL verifier; promote only on a real green."""
    prop = store.propose(sig, domain)
    realization = prop[0] if prop else real          # reuse the proposal if the store has one, else explore
    if not reverify:                                 # ABLATION: trust the proposal without the target verifier
        store.promote(sig, domain, realization); return True, (prop is not None), "ablated"
    ok = VERIFY[domain](realization)                 # the TARGET domain's real verifier owns truth
    if ok: store.promote(sig, domain, realization)
    return ok, (prop is not None), "verified" if ok else "rejected"


print("=== cross_domain_library: one store, code + Lean, transfer proposed & RE-VERIFIED ===\n")
store = UnifiedStore()

# (1) learn the technique in CODE (real unittest), and register (but do NOT verify) its Lean realization
code_ok, _, _ = solve(store, "linear_identity", "code", TECH["linear_identity"]["code"])
store.register("linear_identity", "lean", TECH["linear_identity"]["lean"])
verified_after_code = set(store.card("linear_identity")["verified"])
print(f"  (1) learned 'linear_identity' in CODE (real unittest green={code_ok}); verified_in={verified_after_code}")

# (2) SAME-DOMAIN reuse: a second code bottleneck of the same signature reuses the stored technique
reuse_ok, reuse_proposed, _ = solve(store, "linear_identity", "code", None)
print(f"  (2) same-domain reuse in CODE: proposed_from_store={reuse_proposed}, re-verified green={reuse_ok}")

# (3) CROSS-DOMAIN: a LEAN bottleneck of the same signature -- proposed from code-knowledge, then
#     RE-VERIFIED by the REAL Lean checker (genuine: omega closes 0 + n = n)
pre = set(store.card("linear_identity")["verified"])            # before re-verification: should NOT include lean
xdom_ok, xdom_proposed, xdom_status = solve(store, "linear_identity", "lean", None)
post = set(store.card("linear_identity")["verified"])
print(f"  (3) cross-domain to LEAN: proposed={xdom_proposed}, real-Lean re-verify -> {xdom_status}; "
      f"verified_in {pre} -> {post}")

# (4) SPURIOUS cross-domain transfer: a bogus Lean realization (`rfl`) -- the real Lean verifier must reject it
store.register("bad_transfer", "lean", SPURIOUS_LEAN)
spur_ok, spur_proposed, spur_status = solve(store, "bad_transfer", "lean", None)
print(f"  (4) spurious transfer (`by rfl` on 0+n=n): proposed={spur_proposed}, real-Lean -> {spur_status} "
      f"(promoted={'lean' in store.card('bad_transfer')['verified']})")

# (6) ABLATION: skip re-verification -> the spurious transfer is falsely 'promoted' on say-so
abl = UnifiedStore(); abl.register("bad_transfer", "lean", SPURIOUS_LEAN)
solve(abl, "bad_transfer", "lean", None, reverify=False)
abl_falsely = "lean" in abl.card("bad_transfer")["verified"]
print(f"  (6) verifier ABLATED: spurious transfer falsely verified_in_lean = {abl_falsely}\n")

checks = {
    "a technique is learned in CODE and verified by the real unittest verifier": code_ok,
    "SAME-DOMAIN REUSE: a second same-signature bottleneck reuses the stored technique (re-verified)":
        reuse_proposed and reuse_ok,
    "CROSS-DOMAIN PROPOSAL: a Lean bottleneck is offered the technique learned in code": xdom_proposed,
    "NO TRANSFER-BY-ASSUMPTION: it is NOT marked Lean-verified until the REAL Lean checker confirms":
        ("lean" not in pre) and ("lean" in post),
    "CROSS-DOMAIN RE-VERIFICATION (truth): a genuine transfer VERIFIES in real Lean; a SPURIOUS one is REJECTED":
        xdom_ok and (not spur_ok) and spur_status == "rejected",
    "CUMULATIVE: after re-verification the technique is verified in BOTH domains":
        store.card("linear_identity")["verified"] == {"code", "lean"},
    "VERIFIER ABLATION: skipping re-verification falsely promotes the spurious cross-domain transfer":
        abl_falsely,
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nCROSS-DOMAIN LIBRARY GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: one technique store spans code and math. A technique verified by the real unittest verifier in code"
      "\n  is PROPOSED for a same-signature Lean bottleneck and RE-VERIFIED by the real Lean checker -- a genuine shared"
      "\n  meta-strategy (omega closes the arithmetic identity) transfers and becomes verified in BOTH domains, while a"
      "\n  spurious transfer (`rfl`, which does not close 0+n=n) is REJECTED. Transfer is proposed, never assumed: the"
      "\n  store never marks a technique verified in a domain without that domain's real verifier. The single invariant"
      "\n  -- propose freely, the domain verifier owns truth -- now holds within AND across code and mathematics.")
