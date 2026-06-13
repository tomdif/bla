#!/usr/bin/env python3
"""cross_domain_library: ONE technique store spanning code (unittest), math (Lean 4), and SQL (sqlite).

The capstone: a single UnifiedStore keyed by an abstract bottleneck signature, holding techniques with
per-domain realizations (a code patch, a Lean tactic, a SQL query). A technique verified in one domain
becomes PROPOSABLE for a same-signature bottleneck in another -- but transfer is PROPOSED, never assumed.
The target domain's REAL verifier must confirm it: a genuine realization transfers, a spurious one is
rejected. All three verifiers are real (real `python -m unittest`; real `lean`; real `sqlite3`).

The invariant, lifted across three domains: the store proposes across code, math, and data; each domain's
verifier owns truth; nothing is marked verified-in-a-domain without that domain's verifier saying so.
"""
import os, sys, shutil, tempfile, subprocess, sqlite3

LEAN = shutil.which("lean") or os.path.expanduser("~/.elan/bin/lean")

# the abstract technique "comparison_correctness": a property that hinges on a numeric comparison being
# right. Each domain has its own concrete realization; the cross-domain claim is the obstruction CLASS.
TECH = {
    "code": ("def gt(v):\n    return v > 100000\n",
             "from m import gt\nimport unittest\n"
             "class T(unittest.TestCase):\n"
             "    def test(self): self.assertEqual([gt(x) for x in (50000,120000,100001)], [False,True,True])\n"),
    "lean": ("theorem t : (5 : Nat) < 9 :=", "by decide"),               # genuine: decide closes the comparison
    "sql":  ("SELECT v FROM t WHERE v > 100000 ORDER BY v",),            # genuine: the real predicate
}
# spurious realizations -- each must be REJECTED by its domain's real verifier:
SPURIOUS = {
    "lean": ("theorem t : (5 : Nat) < 9 :=", "by rfl"),                  # rfl does NOT close 5 < 9
    "sql":  ("SELECT v FROM t WHERE v IN (120000,150000) ORDER BY v",),  # hardcodes visible rows, fails held-out
}

SQL_SCHEMA, SQL_VIS, SQL_HELD = "CREATE TABLE t (v INTEGER)", [(50000,), (120000,), (99000,), (150000,)], \
    [(80000,), (200000,), (100001,), (60000,)]
SQL_CORRECT = "SELECT v FROM t WHERE v > 100000 ORDER BY v"


def lean_verify(real):
    decl, tactic = real; d = tempfile.mkdtemp(); f = os.path.join(d, "T.lean")
    open(f, "w").write(f"{decl} {tactic}\n#print axioms t\n")
    r = subprocess.run([LEAN, f], capture_output=True, text=True, timeout=120); out = r.stdout + r.stderr
    return (r.returncode == 0) and ("error:" not in out) and ("sorryAx" not in out)


def code_verify(real):
    module_src, test_src = real; d = tempfile.mkdtemp()
    open(os.path.join(d, "m.py"), "w").write(module_src)
    open(os.path.join(d, "test_m.py"), "w").write(test_src)
    return subprocess.run([sys.executable, "-m", "unittest", "test_m"], cwd=d, capture_output=True, timeout=120).returncode == 0


def _sql_run(q, rows):
    c = sqlite3.connect(":memory:"); c.execute(SQL_SCHEMA); c.executemany("INSERT INTO t VALUES (?)", rows)
    try: return c.execute(q).fetchall()
    except sqlite3.Error: return None
    finally: c.close()


def sql_verify(real):
    q = real[0]                                                          # correct on visible AND held-out data
    return _sql_run(q, SQL_VIS) == _sql_run(SQL_CORRECT, SQL_VIS) and _sql_run(q, SQL_HELD) == _sql_run(SQL_CORRECT, SQL_HELD)


VERIFY = {"lean": lean_verify, "code": code_verify, "sql": sql_verify}
DOMAINS = ("code", "lean", "sql")


class UnifiedStore:
    def __init__(self): self.cards = {}
    def card(self, sig): return self.cards.setdefault(sig, {"real": {}, "verified": set()})
    def register(self, sig, domain, real): self.card(sig)["real"][domain] = real      # known, NOT yet verified here
    def promote(self, sig, domain, real): c = self.card(sig); c["real"][domain] = real; c["verified"].add(domain)
    def propose(self, sig, domain):
        c = self.cards.get(sig)
        if not c or domain not in c["real"]: return None
        return c["real"][domain], (domain in c["verified"])


def solve(store, sig, domain, real, reverify=True):
    prop = store.propose(sig, domain)
    realization = prop[0] if prop else real
    if not reverify:                                                     # ABLATION: trust the proposal, no target verifier
        store.promote(sig, domain, realization); return True, (prop is not None), "ablated"
    ok = VERIFY[domain](realization)                                    # the TARGET domain's real verifier owns truth
    if ok: store.promote(sig, domain, realization)
    return ok, (prop is not None), "verified" if ok else "rejected"


SIG = "comparison_correctness"
print("=== cross_domain_library: one store, code + Lean + SQL, transfer proposed & RE-VERIFIED ===\n")
store = UnifiedStore()

# (1) learn the technique in CODE (real unittest), and register (NOT verify) the Lean + SQL realizations
code_ok, _, _ = solve(store, SIG, "code", TECH["code"])
store.register(SIG, "lean", TECH["lean"]); store.register(SIG, "sql", TECH["sql"])
print(f"  (1) learned '{SIG}' in CODE (real unittest green={code_ok}); verified_in={set(store.card(SIG)['verified'])}")

# (2) SAME-DOMAIN reuse
reuse_ok, reuse_proposed, _ = solve(store, SIG, "code", None)
print(f"  (2) same-domain reuse in CODE: proposed_from_store={reuse_proposed}, re-verified green={reuse_ok}")

# (3) CROSS-DOMAIN transfer to LEAN and SQL -- proposed from the store, RE-VERIFIED by each real verifier
pre = set(store.card(SIG)["verified"])
lean_ok, lean_prop, lean_st = solve(store, SIG, "lean", None)
sql_ok, sql_prop, sql_st = solve(store, SIG, "sql", None)
post = set(store.card(SIG)["verified"])
print(f"  (3) transfer -> LEAN (proposed={lean_prop}, real-lean {lean_st}) ; SQL (proposed={sql_prop}, real-sqlite {sql_st})")
print(f"      verified_in {pre} -> {post}")

# (4) SPURIOUS transfers -- each rejected by its domain's real verifier
store.register("bad_lean", "lean", SPURIOUS["lean"]); sp_lean, _, sp_lean_st = solve(store, "bad_lean", "lean", None)
store.register("bad_sql", "sql", SPURIOUS["sql"]); sp_sql, _, sp_sql_st = solve(store, "bad_sql", "sql", None)
print(f"  (4) spurious: lean `by rfl` -> {sp_lean_st} (promoted={'lean' in store.card('bad_lean')['verified']}); "
      f"sql hardcoded -> {sp_sql_st} (promoted={'sql' in store.card('bad_sql')['verified']})")

# (5) ABLATION: skip re-verification -> spurious falsely promoted
abl = UnifiedStore(); abl.register("bad_sql", "sql", SPURIOUS["sql"]); solve(abl, "bad_sql", "sql", None, reverify=False)
abl_false = "sql" in abl.card("bad_sql")["verified"]
print(f"  (5) verifier ABLATED: spurious SQL transfer falsely verified = {abl_false}\n")

checks = {
    "a technique is learned in CODE and verified by the real unittest verifier": code_ok,
    "SAME-DOMAIN REUSE: a second bottleneck reuses the stored technique (re-verified)": reuse_proposed and reuse_ok,
    "CROSS-DOMAIN PROPOSAL: the code-learned technique is offered to BOTH Lean and SQL bottlenecks":
        lean_prop and sql_prop,
    "NO TRANSFER-BY-ASSUMPTION: not marked verified in Lean/SQL until each REAL verifier confirms":
        ("lean" not in pre and "sql" not in pre) and ("lean" in post and "sql" in post),
    "CROSS-DOMAIN RE-VERIFICATION: genuine transfers VERIFY in real Lean AND real SQLite":
        lean_ok and sql_ok,
    "SPURIOUS transfers are REJECTED by each domain's real verifier (`rfl`; hardcoded query)":
        (not sp_lean) and (not sp_sql) and sp_lean_st == "rejected" and sp_sql_st == "rejected",
    "CUMULATIVE across THREE domains: the technique is verified in code, Lean, AND SQL":
        store.card(SIG)["verified"] == {"code", "lean", "sql"},
    "VERIFIER ABLATION: skipping re-verification falsely promotes a spurious cross-domain transfer": abl_false,
}
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nCROSS-DOMAIN LIBRARY (3 DOMAINS) GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: one technique store now spans code, mathematics, AND data. A technique verified by the real unittest"
      "\n  verifier in code is PROPOSED for same-signature Lean and SQL bottlenecks and RE-VERIFIED by the real Lean"
      "\n  checker and the real SQLite engine -- genuine realizations transfer and the technique becomes verified in all"
      "\n  three domains, while spurious transfers (`rfl` that can't close the goal; a hardcoded query that fails held-out"
      "\n  data) are rejected. Transfer is proposed, never assumed: no domain is marked verified without its own real"
      "\n  verifier. The single invariant -- propose freely, the domain verifier owns truth -- now holds across three.")
