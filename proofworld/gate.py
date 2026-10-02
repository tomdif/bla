#!/usr/bin/env python3
"""proofworld.gate -- THE trust boundary. One place decides whether a Lean claim is believed.

Every module used to roll its own `lean` subprocess + "rc==0 and no 'error:'" check. That pattern has failure
modes that turn into FALSE PROOFS (all reproduced against the old code, 2026-10-02):
  * `sorry` is only a WARNING -> rc 0, no 'error:' -> "VERIFIED"           (leankernel.verify_theorem, ramanujan)
  * `native_decide` trusts the compiler, not the kernel -> "VERIFIED"
  * an `axiom` declared in the candidate text -> "VERIFIED"
  * timeout / import failure / crash -> no error *inside the claim's lines* -> counted as a pass (decompose.try_direct)
  * `#print axioms` wraps long lists across lines -> first-line parsers miss `sorryAx`
  * fixed temp paths (/tmp/lean_batch.lean, ~/RamanujanTau/_pw_cand.lean) -> concurrent runs read each other's file

The rules here (lessons ported from a review of the BootLoops harness, whose gates failed the same ways):
  1. POSITIVE EVIDENCE ONLY. A claim is PROVED only if the kernel itself reports its axiom footprint (a one-line
     marker emitted by an in-file `collectAxioms` command) AND that footprint is within the standard three.
     Missing output, timeout, or an error we cannot attribute to a single claim -> INCONCLUSIVE, never a pass.
  2. CHEAP KILL BEFORE EXPENSIVE CHECK. Untrusted text (LLM proofs AND LLM-proposed statements) is linted for
     escape hatches (sorry/admit/axiom/native_decide/skipKernelTC/implemented_by/...) and REFUSED pre-kernel.
     The axiom footprint is the backstop for anything the lint misses.
  3. A GATE IS TRUSTED ONLY AFTER ITS CONTROLS FAIL CORRECTLY. Before the first real check in a given
     (toolchain, project) environment, planted claims run: a true one must be PROVED, a false one FAILED, and
     sorry / native_decide / injected-axiom ones REFUSED. If any control misbehaves, the gate refuses to run.
  4. RECEIPTS. Every verdict is appended (under a file lock) to a JSONL ledger with the exact source hash,
     toolchain, and project manifest hash, so a "proved" claim can be traced and re-checked.

Verdict statuses (closed vocabulary):  PROVED | FAILED | REFUSED | INCONCLUSIVE
    PROVED        kernel-checked, axioms ⊆ {propext, Classical.choice, Quot.sound}
    FAILED        the kernel rejected the proof (an error inside the claim)
    REFUSED       policy: escape hatch in the text, or sorry / non-standard axioms in the footprint
    INCONCLUSIVE  timeout, crash, missing marker, unattributable error -- no information, NOT a pass

API:
    from proofworld.gate import Claim, check, check_one
    v = check_one("theorem t (n : Nat) : n ≤ n + 1", "by omega")            # core Lean
    vs = check([Claim("t", "(n : ℕ) : n ≤ n^2 + n", "by nlinarith")],
               preamble="import Mathlib.Tactic\n", project="~/bla/proofworld/lean")
    vs["t"].proved, vs["t"].status, vs["t"].axioms

CLI:  python3 -m proofworld.gate --selftest [--project DIR]     (exit 0 pass, 1 fail, 77 lean unavailable)
"""
from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
STD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
STATUSES = ("PROVED", "FAILED", "REFUSED", "INCONCLUSIVE")
MARK = "@@PWTRUST@@"
FILE_BASENAME = "PwGate.lean"
ELAN_BIN = os.path.expanduser("~/.elan/bin")
DEFAULT_LEDGER = os.environ.get("PROOFWORLD_RECEIPTS", os.path.join(HERE, "receipts", "ledger.jsonl"))

# Escape hatches that must never appear in UNTRUSTED text (LLM proofs / LLM-proposed statements).
# Word-bounded so identifiers like `sorryFree` or `axiomatic` are not caught.
_BANNED = [
    (r"\bsorry\b", "sorry"), (r"\badmit\b", "admit"), (r"\baxiom\b", "axiom declaration"),
    (r"\bnative_decide\b", "native_decide (trusts the compiler, not the kernel)"),
    (r"\bdecide\s*\+\s*native\b|\bdecide\s*\(\s*config", "native/configured decide"),
    (r"ofReduceBool|ofReduceNat", "Lean.ofReduceBool/Nat (compiler-trusted)"),
    (r"skipKernelTC|\bdebug\.", "kernel-skipping debug option"),
    (r"implemented_by|\bextern\b|\bunsafe\b|\bpartial\s+def\b|\bopaque\b", "implemented_by/extern/unsafe/opaque"),
    (r"#eval|#exit|run_cmd|run_tac|run_elab|\belab\b|\bmacro\b|\bsyntax\b|\bnotation\b|\binfix[lr]?\b",
     "metaprogramming / command injection"),
    (r"^\s*import\b|\bset_option\b|@\[\s*(csimp|implemented_by|extern)", "import/set_option/attribute injection"),
    (r"\bprivate\s+(theorem|def|lemma)\b|^\s*(theorem|lemma|def|instance|abbrev|structure|class|inductive)\s",
     "extra top-level declaration inside a proof"),
]
_BANNED_RE = [(re.compile(p, re.MULTILINE), why) for p, why in _BANNED]

# A statement may legitimately start with nothing but binders/props; we lint it with the same list except the
# "extra declaration" rule (callers pass bare signatures, never `theorem ...` text, to Claim.sig).
_STMT_RE = [(r, why) for r, why in _BANNED_RE if "extra top-level" not in why]


def _elab_src() -> str:
    """the in-file axiom reporter: one line per target, on stdout, from the kernel's own collectAxioms.
    Names resolve like ordinary identifiers (current namespace / `open`s) when not found verbatim; the line
    is keyed by the REQUESTED identifier so the Python side can match it."""
    return (
        "open Lean Elab Command in\n"
        f'elab "#pw_trust " ids:ident* : command => do\n'
        "  for id in ids do\n"
        "    let req := id.getId\n"
        "    let n? : Option Name ← (do\n"
        "      if (← getEnv).contains req then pure (some req)\n"
        "      else try pure (some (← liftCoreM <| realizeGlobalConstNoOverload id)) catch _ => pure none)\n"
        "    match n? with\n"
        f'    | none => IO.println s!"{MARK}\\t{{req}}\\tMISSING\\t"\n'
        "    | some n =>\n"
        "      let axs ← collectAxioms n\n"
        f'      IO.println s!"{MARK}\\t{{req}}\\tFOUND\\t{{String.intercalate "," (axs.toList.map toString)}}"\n'
    )


class GateIntegrityError(RuntimeError):
    """raised when the gate's planted controls do not behave -- nothing it says can be trusted."""


@dataclass
class Claim:
    name: str                 # Lean identifier (unqualified; `namespace` arg qualifies it)
    sig: str                  # "(binders) : prop" -- everything between the name and `:=`
    proof: str                # "by ..." or a term
    trusted: bool = False     # True only for text the caller wrote itself (skips the lint, NOT the axiom check)
    keyword: str = "theorem"

    @classmethod
    def from_decl(cls, decl: str, proof: str, trusted: bool = False) -> "Claim":
        """from 'theorem NAME (binders) : P' (or `lemma`) + proof."""
        m = re.match(r"\s*(theorem|lemma)\s+([^\s(:{\[]+)\s*(.*)\Z", decl, re.DOTALL)
        if not m:
            raise ValueError(f"not a theorem declaration: {decl[:80]!r}")
        return cls(name=m.group(2), sig=m.group(3).strip(), proof=proof, trusted=trusted, keyword="theorem")

    def decl(self) -> str:
        return f"{self.keyword} {self.name} {self.sig} := {self.proof}"


@dataclass
class Verdict:
    name: str
    status: str               # one of STATUSES
    axioms: List[str] = field(default_factory=list)
    detail: str = ""
    register: str = "lean-kernel"
    env: str = ""             # toolchain / project the claim was checked in
    source_sha: str = ""
    elapsed_s: float = 0.0
    controls: str = ""        # "passed" | "skipped" | "n/a"

    @property
    def proved(self) -> bool:
        return self.status == "PROVED"

    def to_json(self) -> dict:
        d = asdict(self); d["proved"] = self.proved
        return d


# ------------------------------------------------------------------------------------------------ environment
def default_toolchain() -> str:
    """an already-installed toolchain (never trigger an elan download)."""
    pref = os.environ.get("PROOFWORLD_LEAN_TOOLCHAIN")
    if pref:
        return pref
    installed = sorted(glob.glob(os.path.expanduser("~/.elan/toolchains/leanprover--lean4---v*")))
    for p in installed:
        if p.endswith("v4.30.0"):
            return "leanprover/lean4:v4.30.0"
    if installed:
        return "leanprover/lean4:" + installed[-1].split("---")[-1]
    return "leanprover/lean4:v4.30.0"


def _env_key(project: Optional[str], toolchain: Optional[str]) -> str:
    if project:
        p = os.path.realpath(os.path.expanduser(project))
        man = os.path.join(p, "lake-manifest.json")
        tc = open(os.path.join(p, "lean-toolchain")).read().strip() if os.path.exists(os.path.join(p, "lean-toolchain")) else "?"
        msha = hashlib.sha256(open(man, "rb").read()).hexdigest()[:12] if os.path.exists(man) else "nomanifest"
        return f"project={p} toolchain={tc} manifest={msha}"
    return f"core toolchain={toolchain or default_toolchain()}"


def lean_available(project: Optional[str] = None) -> bool:
    env = dict(os.environ, PATH=ELAN_BIN + os.pathsep + os.environ.get("PATH", ""))
    exe = "lake" if project else "elan"
    try:
        return subprocess.run([exe, "--version"], capture_output=True, env=env, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ------------------------------------------------------------------------------------------------ lint
def lint(text: str, statement: bool = False) -> Optional[str]:
    """first escape hatch found in untrusted text, or None."""
    for rx, why in (_STMT_RE if statement else _BANNED_RE):
        if rx.search(text):
            return why
    return None


# ------------------------------------------------------------------------------------------------ assembly
def _split_imports(preamble: str) -> Tuple[List[str], str]:
    """Lean requires `import` lines first; pull them out (skipping blank lines, `--` comments and
    `/- ... -/` block comments that may precede them) so the trust elab can follow them."""
    lines = preamble.splitlines()
    imports, header, i, in_block = [], [], 0, False
    while i < len(lines):
        ln = lines[i].strip()
        if in_block:
            header.append(lines[i])
            if "-/" in ln:
                in_block = False
            i += 1
            continue
        if ln == "" or ln.startswith("--"):
            header.append(lines[i]); i += 1; continue
        if ln.startswith("/-"):
            header.append(lines[i])
            in_block = "-/" not in ln[2:]
            i += 1
            continue
        if ln.startswith("import "):
            imports.append(ln); i += 1; continue
        break
    # leading comments are kept (after the imports) so line numbers of the rest stay meaningful to readers
    return imports, "\n".join(header + lines[i:])


def assemble(claims: List[Claim], preamble: str = "", namespace: Optional[str] = None,
             opens: str = "") -> Tuple[str, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """build the single Lean file. Returns (source, {name: (first_line, last_line)}, {name: qualified_name})."""
    imports, rest = _split_imports(preamble)
    if "import Lean" not in imports:
        imports = ["import Lean"] + imports
    head = "\n".join(imports) + "\n\n" + _elab_src() + "\n" + rest.rstrip() + "\n\n"
    if namespace:
        head += f"namespace {namespace}\n"
    if opens:
        head += opens.rstrip() + "\n"
    head += "\n"
    src, spans, qual = head, {}, {}
    for c in claims:
        start = src.count("\n") + 1
        src += c.decl().rstrip() + "\n"
        spans[c.name] = (start, src.count("\n"))
        qual[c.name] = f"{namespace}.{c.name}" if namespace else c.name
        src += "\n"
    if namespace:
        src += f"end {namespace}\n"
    src += "#pw_trust " + " ".join(qual[c.name] for c in claims) + "\n"
    return src, spans, qual


# Lean identifiers may use Unicode letters (Ψ11, α_le, ...): any word char, not starting with a digit, dot-separated.
# Whitespace, ':', ':=', newlines etc. stay impossible, which is what blocks name-injection.
_IDENT = re.compile(r"^[^\W\d][\w'!?₀-₉]*(\.[^\W\d][\w'!?₀-₉]*)*$", re.UNICODE)
_POSITIONLESS = re.compile(r"^\s*(error\b|uncaught exception|INTERNAL PANIC|PANIC|Segmentation fault|Killed)",
                          re.IGNORECASE)
_DIAG = re.compile(re.escape(FILE_BASENAME) + r":(\d+):(\d+):\s*(error|warning)(?:\([^)]*\))?:\s*(.*)")


def parse(out: str, spans: Dict[str, Tuple[int, int]], qual: Dict[str, str]):
    """-> (per-claim errors {name:[msg]}, unattributed errors [msg], footprints {name: [axioms] | None})."""
    errs: Dict[str, List[str]] = {n: [] for n in spans}
    stray: List[str] = []
    for m in _DIAG.finditer(out):
        line, kind, msg = int(m.group(1)), m.group(3), m.group(4).strip()
        if kind != "error":
            continue
        owner = next((n for n, (s, e) in spans.items() if s <= line <= e), None)
        (errs[owner].append(msg) if owner else stray.append(f"line {line}: {msg}"))
    # errors that carry no file position at all (crashes, `unknown package`, OOM) are stray too
    for ln in out.splitlines():
        if _POSITIONLESS.match(ln) and FILE_BASENAME not in ln:
            stray.append(ln.strip()[:200])
    foot: Dict[str, Optional[List[str]]] = {n: None for n in spans}
    by_q = {q: n for n, q in qual.items()}
    for ln in out.splitlines():
        if not ln.startswith(MARK):
            continue
        parts = ln.split("\t")
        if len(parts) < 4 or parts[2] != "FOUND" or parts[1] not in by_q:
            continue
        foot[by_q[parts[1]]] = [a for a in parts[3].split(",") if a]
    return errs, stray, foot


def is_native_axiom(a: str) -> bool:
    """axioms minted by native_decide / Lean.ofReduceBool: the COMPILER is trusted, not just the kernel."""
    return "._native." in a or a in ("Lean.ofReduceBool", "Lean.ofReduceNat", "Lean.trustCompiler")


def _decide(c: Claim, errs, stray, foot, timed_out: bool, rc: Optional[int],
            allow_native: bool = False) -> Tuple[str, str, List[str]]:
    if timed_out:
        return "INCONCLUSIVE", "timeout", []
    if stray:
        return "INCONCLUSIVE", "error outside any claim (preamble/import/crash): " + stray[0][:200], []
    axs = foot.get(c.name)
    if errs.get(c.name):
        return "FAILED", errs[c.name][0][:300], axs or []
    if axs is None:
        return "INCONCLUSIVE", f"kernel reported no axiom footprint (rc={rc}); not a pass", []
    if "sorryAx" in axs:
        return "REFUSED", "footprint contains sorryAx", axs
    extra = sorted(set(axs) - STD_AXIOMS)
    native = [a for a in extra if is_native_axiom(a)]
    if allow_native:
        extra = [a for a in extra if not is_native_axiom(a)]
    if extra:
        return "REFUSED", "non-standard axioms: " + ", ".join(extra), axs
    # in a batch, lean exits 1 if ANY claim failed; a nonzero rc is only explained if some claim owns an error
    if rc not in (0, None) and not any(errs.values()):
        return "INCONCLUSIVE", f"lean rc={rc} with no attributable error", axs
    if native:          # only reachable with allow_native=True: say so on the verdict, never silently
        return "PROVED", f"relies on compiler-trusted native_decide ({len(native)}): {native[0]}", axs
    return "PROVED", "", axs


# ------------------------------------------------------------------------------------------------ receipts
def _append_receipts(rows: List[dict], ledger: str):
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------------------------ the gate
_CONTROLS_OK: Dict[str, bool] = {}


def check(claims: Iterable[Claim], preamble: str = "", project: Optional[str] = None,
          toolchain: Optional[str] = None, namespace: Optional[str] = None, opens: str = "",
          timeout: int = 300, controls: bool = True, receipts: bool = True,
          ledger: Optional[str] = None, tag: str = "", allow_native: bool = False) -> Dict[str, Verdict]:
    """check every claim in ONE Lean run; return {name: Verdict}. Never raises on a bad proof -- only on a
    malformed request or a gate whose controls failed (GateIntegrityError).

    allow_native: accept native_decide axioms ALREADY IN THE LIBRARY the claims build on (e.g. RamanujanTau's
    `tau_one := by native_decide`). Such verdicts carry register "lean-kernel+native" and say so in `detail`.
    Untrusted proof text still may not use native_decide itself (the lint refuses it)."""
    claims = list(claims)
    names = [c.name for c in claims]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate claim names: {sorted(n for n in set(names) if names.count(n) > 1)}")
    envk = _env_key(project, toolchain)
    ctl = "skipped"
    if controls:
        if envk not in _CONTROLS_OK:
            _CONTROLS_OK[envk] = run_controls(project=project, toolchain=toolchain)["ok"]
        if not _CONTROLS_OK[envk]:
            raise GateIntegrityError(f"gate controls FAILED in {envk}; refusing to certify anything there")
        ctl = "passed"

    out: Dict[str, Verdict] = {}
    to_run: List[Claim] = []
    for c in claims:
        if not _IDENT.match(c.name):                         # an LLM-supplied name is also injectable text
            out[c.name] = Verdict(c.name, "REFUSED", detail="invalid Lean identifier", env=envk, controls=ctl)
            continue
        why = None if c.trusted else (lint(c.proof) or (lint(c.sig, statement=True) and
                                                         "statement: " + lint(c.sig, statement=True)))
        if why:
            out[c.name] = Verdict(c.name, "REFUSED", detail=f"pre-kernel lint: {why}", env=envk, controls=ctl)
        else:
            to_run.append(c)

    if to_run:
        src, spans, qual = assemble(to_run, preamble, namespace, opens)
        sha = hashlib.sha256(src.encode()).hexdigest()
        env = dict(os.environ, PATH=ELAN_BIN + os.pathsep + os.environ.get("PATH", ""))
        t0 = time.time()
        timed_out, rc, text = False, None, ""
        with tempfile.TemporaryDirectory(prefix="pwgate_") as td:      # unique per call: no shared temp file
            f = os.path.join(td, FILE_BASENAME)
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(src)
            if project:
                cmd, cwd = ["lake", "env", "lean", f], os.path.expanduser(project)
            else:
                cmd, cwd = ["elan", "run", toolchain or default_toolchain(), "lean", f], td
            try:
                p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
                rc, text = p.returncode, p.stdout + p.stderr
            except subprocess.TimeoutExpired:
                timed_out = True
            except OSError as e:
                text = f"error: could not run lean: {e}"
        elapsed = round(time.time() - t0, 2)
        errs, stray, foot = parse(text, spans, qual)
        for c in to_run:
            st, det, axs = _decide(c, errs, stray, foot, timed_out, rc, allow_native)
            reg = "lean-kernel+native" if any(is_native_axiom(a) for a in axs) and st == "PROVED" else "lean-kernel"
            out[c.name] = Verdict(c.name, st, axs, det, register=reg, env=envk, source_sha=sha, elapsed_s=elapsed,
                                  controls=ctl)

    if receipts:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows = []
        for c in claims:
            v = out[c.name]
            rows.append({"ts": ts, "tag": tag, "name": c.name, "sig": c.sig, "proof": c.proof,
                         "trusted_text": c.trusted, **v.to_json()})
        try:
            _append_receipts(rows, ledger or DEFAULT_LEDGER)
        except OSError as e:                                   # receipts must never turn a FAIL into a crash
            print(f"[gate] warning: could not write receipts: {e}", file=sys.stderr)
    return {c.name: out[c.name] for c in claims}


def check_one(decl: str, proof: str, trusted: bool = False, **kw) -> Verdict:
    """convenience: check_one('theorem t (n : Nat) : n ≤ n + 1', 'by omega', project=..., preamble=...)."""
    c = Claim.from_decl(decl, proof, trusted=trusted)
    return check([c], **kw)[c.name]


# ------------------------------------------------------------------------------------------------ controls
CONTROLS = [
    # (claim, expected status) -- trusted=True so the KERNEL-side defences are what's being tested
    (Claim("pw_ctrl_true", "(n : Nat) : n ≤ n + 1", "by omega", trusted=True), "PROVED"),
    (Claim("pw_ctrl_classical", "(p : Prop) : p ∨ ¬p", "Classical.em p", trusted=True), "PROVED"),
    (Claim("pw_ctrl_false", "(n : Nat) : n = n + 1", "by omega", trusted=True), "FAILED"),
    (Claim("pw_ctrl_sorry", "(n : Nat) : n = n + 1", "by sorry", trusted=True), "REFUSED"),
    (Claim("pw_ctrl_native", ": 2 + 2 = 4", "by native_decide", trusted=True), "REFUSED"),
    (Claim("pw_ctrl_axiom", "(n : Nat) : n = n + 1", "pw_ctrl_bad_ax n", trusted=True), "REFUSED"),
    # untrusted text: must die in the lint before reaching the kernel
    (Claim("pw_ctrl_lint", "(n : Nat) : n = n + 1", "by sorry"), "REFUSED"),
    (Claim("pw_ctrl_inject", "(n : Nat) : n = n + 1", "by\n  exact pw_ctrl_bad_ax n"), "REFUSED"),
]
CONTROL_PREAMBLE = "axiom pw_ctrl_bad_ax : ∀ n : Nat, n = n + 1\n"


def run_controls(project: Optional[str] = None, toolchain: Optional[str] = None, log=None) -> dict:
    """planted controls -- the gate is trusted in an environment only if every one comes back as expected.
    The axiom-injection control proves the footprint check works (its proof text is clean; only the kernel's
    collectAxioms can see the injected axiom)."""
    claims = [c for c, _ in CONTROLS]
    vs = check(claims, preamble=CONTROL_PREAMBLE, project=project, toolchain=toolchain,
               controls=False, receipts=False, timeout=600)
    rows, ok = [], True
    for c, want in CONTROLS:
        got = vs[c.name].status
        # pw_ctrl_inject is untrusted but its text is clean -> reaches the kernel -> must be REFUSED by footprint
        good = got == want
        ok &= good
        rows.append({"control": c.name, "want": want, "got": got, "ok": good, "detail": vs[c.name].detail})
        if log:
            log(f"  {'ok ' if good else 'BAD'} {c.name:20} want {want:12} got {got:12} {vs[c.name].detail[:60]}")
    return {"ok": ok, "rows": rows, "env": _env_key(project, toolchain)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="proofworld trust gate")
    ap.add_argument("--selftest", action="store_true", help="run the planted controls (exit 0/1, 77 = no lean)")
    ap.add_argument("--project", default=None, help="lake project dir (else core Lean via elan)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.selftest:
        ap.print_help(); return 2
    if not lean_available(a.project):
        print("SKIP: lean/lake not available"); return 77
    res = run_controls(project=a.project, log=None if a.json else print)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"gate controls: {'PASS' if res['ok'] else 'FAIL'}  [{res['env']}]")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
