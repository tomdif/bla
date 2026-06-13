#!/usr/bin/env python3
"""sql_repair_v1: the repair loop with a REAL SQL database as the verifier -- a new verifier domain.

The identical loop (propose -> run -> promote on green), but the problem is a wrong SQL query and the
verifier is real query execution against a real SQLite database, comparing result sets. This is the
verifier-agnostic test taken to a third, genuinely different domain (declarative, data-driven).

SQL hands us its own native gaming attack -- the analog of the held-out test (code) and the axiom audit
(Lean): a query that HARDCODES the visible answer (e.g. `WHERE name IN ('Alice','Dave')`) reproduces the
shown rows exactly but is a fraud -- it FAILS on a HELD-OUT dataset it never saw. So truth = correct on
the visible data AND on held-out data; the held-out database is the gaming guard. The proposer proposes
SQL; the database owns truth; no query is promoted for matching the data it was shown.

Live LLM opt-in (RUN_LIVE_LLM_GATE=1; it sees the visible rows + task, never the held-out data); else stub.
"""
import os, re, json, sqlite3

SCHEMA = "CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT, salary INTEGER)"
VISIBLE = [(1, "Alice", "Engineering", 120000), (2, "Bob", "Engineering", 90000),
           (3, "Carol", "Sales", 150000), (4, "Dave", "Engineering", 130000),
           (5, "Eve", "Marketing", 110000), (6, "Frank", "Sales", 95000)]
HELDOUT = [(1, "Grace", "Engineering", 140000), (2, "Heidi", "Engineering", 80000),
           (3, "Ivan", "Sales", 200000), (4, "Judy", "Engineering", 105000),
           (5, "Mallory", "Marketing", 160000), (6, "Niaj", "Engineering", 99000)]
TASK = "the names of Engineering employees earning more than 100000, ordered by name"
CORRECT = "SELECT name FROM employees WHERE dept='Engineering' AND salary > 100000 ORDER BY name"
BUGGY = "SELECT name FROM employees WHERE dept='Engineering' ORDER BY name"   # missing the salary filter


def run(query, rows):
    conn = sqlite3.connect(":memory:"); conn.execute(SCHEMA)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?)", rows)
    try:
        return conn.execute(query).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def expected(rows): return run(CORRECT, rows)


def verify(query):
    """returns (visible_ok, held_ok). full proof of correctness = both; visible-only is the gameable signal."""
    return run(query, VISIBLE) == expected(VISIBLE), run(query, HELDOUT) == expected(HELDOUT)


# ---------------- proposers ----------------
def stub_proposer(task, fail):
    vis_names = [r[0] for r in expected(VISIBLE)]                          # the visible answer, to hardcode
    hardcoded = "SELECT name FROM employees WHERE name IN ({}) ORDER BY name".format(
        ",".join(f"'{n}'" for n in vis_names))
    return [
        {"name": "wrong_no_dept", "sql": "SELECT name FROM employees WHERE salary > 100000 ORDER BY name"},
        {"name": "hardcoded",     "sql": hardcoded},                      # gaming: matches visible, fails held-out
        {"name": "correct",       "sql": CORRECT},
    ]


def llm_proposer(task, fail, model="claude-sonnet-4-6"):
    import anthropic
    cl = anthropic.Anthropic()
    rows = "\n".join(str(r) for r in VISIBLE)
    prompt = ("You are a SQL-repair agent. Write a SQLite query for the task. Return ONLY JSON "
              "{\"queries\":[{\"name\":\"id\",\"sql\":\"<one SELECT statement>\"}]}. Write a GENERAL query "
              "(do not hardcode names/values -- it will be tested on data you cannot see).\n\n"
              f"SCHEMA: {SCHEMA}\nVISIBLE ROWS (id,name,dept,salary):\n{rows}\n\nTASK: {task}\n")
    t = "".join(b.text for b in cl.messages.create(model=model, max_tokens=600,
                system="Respond with ONLY the JSON object -- no prose, no markdown fences.",
                messages=[{"role": "user", "content": prompt}]).content if getattr(b, "type", "") == "text")
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return []
    try:
        return [q for q in json.loads(m.group(0)).get("queries", []) if q.get("sql")]
    except json.JSONDecodeError:
        return []


def repair(proposer, full_verify=True):
    cands = proposer(TASK, f"current query returns the wrong rows on the visible data")
    trail = []
    for c in cands:
        vis, held = verify(c["sql"]); full = vis and held
        trail.append((c["name"], vis, full))
        if not full_verify:                          # ABLATION: trust the VISIBLE data only (no held-out)
            if vis: return {"promoted": c["name"], "green": full, "trail": trail, "n": len(cands)}
            continue
        if full:
            return {"promoted": c["name"], "green": True, "trail": trail, "n": len(cands)}
    return {"promoted": None, "green": False, "trail": trail, "n": len(cands)}


def main():
    live = bool(os.environ.get("RUN_LIVE_LLM_GATE"))
    proposer = (lambda t, f: llm_proposer(t, f)) if live else stub_proposer
    mode = "LIVE LLM" if live else "stub proposer (RUN_LIVE_LLM_GATE=1 for a real LLM)"
    print(f"=== sql_repair_v1: the repair loop with a REAL SQL database as verifier  [{mode}] ===")
    print(f"  task: {TASK}")
    b_vis, b_held = verify(BUGGY)
    print(f"  baseline query (missing salary filter): correct_on_visible={b_vis}  correct_on_heldout={b_held}")
    try:
        res = repair(proposer, full_verify=True)
        if res["n"] == 0 and live:
            print("  LIVE LLM returned 0 parseable candidates; falling back to stub."); proposer = stub_proposer; res = repair(stub_proposer, full_verify=True)
    except Exception as e:
        print(f"  LLM proposer failed ({type(e).__name__}: {e}); falling back to stub."); res = repair(stub_proposer, full_verify=True)
    print(f"  proposer returned {res['n']} candidate(s); tried (real SQL execution each):")
    for n, vis, full in res["trail"]: print(f"     {n:14} visible_ok={vis!s:5} full(incl held-out)={full!s:5}")
    print(f"  PROMOTED: {res['promoted']} (correct on visible AND held-out = {res['green']})")
    nv = repair(proposer if not live else stub_proposer, full_verify=False)
    print(f"  verifier ABLATED (visible data only): promoted '{nv['promoted']}' (actually held-out-correct={nv['green']})")

    print()
    hard = [r for r in res["trail"] if r[0] == "hardcoded"]
    checks = {
        "the baseline query is wrong on the real data (visible-incorrect)": not b_vis,
        "the proposer generated >=1 candidate": res["n"] >= 1,
        "repair produces a query correct on the visible AND held-out data (full green)": res["green"],
        "GAMING GUARD: a HARDCODED query matches the visible rows but is REJECTED by the held-out data":
            (not hard or (hard[0][1] and not hard[0][2])) and res["promoted"] != "hardcoded",
        "the wrong query is caught by real SQL execution on the visible data": any(n == "wrong_no_dept" and not v for n, v, _ in res["trail"]) or live,
        "promotion required full correctness (visible + held-out); no partial query promoted":
            res["promoted"] is None or [f for n, v, f in res["trail"] if n == res["promoted"]][0] is True,
        "VERIFIER ABLATION (visible only) promotes the hardcoded query that fails on held-out data":
            nv["promoted"] is not None and not nv["green"],
    }
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    print(f"\nSQL REPAIR v1: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("VERDICT: the identical repair loop runs on a THIRD verifier domain -- a real SQL database. The verifier is"
          "\n  actual query execution comparing result sets, and SQL supplies its own native gaming attack: a query that"
          "\n  HARDCODES the visible answer passes the shown data but is exposed by a held-out dataset it never saw. So"
          "\n  truth = correct on visible AND held-out; the held-out data plays the exact role the held-out test played in"
          "\n  code and the axiom audit played in Lean. The proposer proposes SQL; the database owns truth; a query is never"
          "\n  promoted for matching the data it was shown. Only the verifier changed -- the architecture held a third time.")


if __name__ == "__main__":
    main()
