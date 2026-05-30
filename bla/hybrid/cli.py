"""Interactive REPL for bla.hybrid.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m bla.hybrid

Type a message; the harness routes it through the observe→predict→render
pipeline and prints the reply + the underlying packet for transparency.

Commands:
    /state                     show current object-file count + summary
    /show <id>                 dump one object file as JSON
    /trajectory                show the last step's full StepRecord
    /save <path>               save StateStore to JSON
    /load <path>               load StateStore from JSON
    /seed                      apply BLA-tracker seed objects
    /quit                      exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bla.hybrid.bla_tracker import (
    BLA_DOMAIN_SYSTEM_PROMPT,
    seed_bla_tracker,
)
from bla.hybrid.llm_client import AnthropicLLMClient
from bla.hybrid.loop import HybridLoop
from bla.hybrid.predictor import LLMPredictor
from bla.hybrid.state import StateStore


_HELP = """\
commands:
  /state                      show object count + ids by type
  /show <id>                  dump one object as JSON
  /trajectory                 print the last step's full StepRecord
  /save [path]                save state to path (default: state.json)
  /load <path>                load state from path
  /seed                       apply BLA-tracker seed objects
  /help                       this help
  /quit                       exit
"""


def _print_state_summary(state: StateStore) -> None:
    by_type: dict[str, list[str]] = {}
    for o in state.all():
        by_type.setdefault(o.type, []).append(o.id)
    print(f"# {len(state)} objects")
    for t in sorted(by_type):
        ids = by_type[t]
        print(f"  {t}: {len(ids)}  — {', '.join(sorted(ids))}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="bla.hybrid", description=__doc__)
    ap.add_argument(
        "--state-file",
        default="bla_hybrid_state.json",
        help="Persist state to this file between sessions (default: %(default)s)",
    )
    ap.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding with the BLA tracker baseline objects",
    )
    ap.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Anthropic model id (default: %(default)s)",
    )
    ap.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Effort level passed via output_config (default: %(default)s)",
    )
    args = ap.parse_args()

    state_path = Path(args.state_file)
    state = StateStore(path=state_path)
    if not args.no_seed and len(state) == 0:
        seed_bla_tracker(state)
        print(f"[seeded {len(state)} objects from bla_tracker]")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "and rerun.",
            file=sys.stderr,
        )
        return 2

    llm = AnthropicLLMClient(model=args.model, effort=args.effort, api_key=api_key)
    predictor = LLMPredictor(llm=llm)
    loop = HybridLoop(
        llm=llm,
        predictor=predictor,
        state=state,
        domain_preamble=BLA_DOMAIN_SYSTEM_PROMPT,
    )

    print(f"bla.hybrid REPL — model={args.model} effort={args.effort}")
    print(f"state file: {state_path.resolve()}")
    print(f"type /help for commands, /quit to exit")
    _print_state_summary(state)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd == "quit":
                break
            elif cmd == "help":
                print(_HELP)
            elif cmd == "state":
                _print_state_summary(state)
            elif cmd == "show":
                obj = state.find(rest)
                if obj is None:
                    print(f"no object with id {rest!r}")
                else:
                    print(json.dumps(obj.to_dict(), indent=2, sort_keys=True))
            elif cmd == "trajectory":
                if not loop.history:
                    print("no steps yet")
                else:
                    last = loop.history[-1]
                    print(json.dumps(
                        {
                            "user_input": last.user_input,
                            "observe": last.observe,
                            "predict": last.predict,
                            "critique": last.critique,
                            "render_text": last.render_text,
                            "state_changes": last.state_changes,
                        },
                        indent=2, sort_keys=True,
                    ))
            elif cmd == "save":
                target = Path(rest) if rest else state_path
                state.path = target
                p = state.save()
                print(f"saved → {p}")
            elif cmd == "load":
                if not rest:
                    print("usage: /load <path>")
                else:
                    state.path = Path(rest)
                    state.load()
                    print(f"loaded {len(state)} objects from {rest}")
            elif cmd == "seed":
                seed_bla_tracker(state)
                print(f"seed applied → {len(state)} objects")
            else:
                print(f"unknown command /{cmd}  (try /help)")
            continue

        # Otherwise treat as a user turn through the loop
        try:
            rec = loop.step(line)
        except Exception as e:
            print(f"[loop error: {type(e).__name__}: {e}]")
            continue

        print()
        print(rec.render_text)

        # Persist state after each turn
        try:
            state.save()
        except Exception as e:
            print(f"[save warning: {e}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
