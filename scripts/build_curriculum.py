"""Build the Phase 6 procedural-CPU curriculum and dump as JSONL.

Mixes Python execution traces, MetaMathQA, and FOLIO with configurable
weights. Each line of the output is a self-contained
{prompt, target, source, metadata} record.

Run:
    python3 scripts/build_curriculum.py \\
        --n 30000 --output runs/phase6/curriculum.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from system2_dca.curriculum import CurriculumMixer, dump_jsonl
from system2_dca.curriculum_logic import FormalLogicSource
from system2_dca.curriculum_math import MathQASource
from system2_dca.curriculum_python import PythonExecutionSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30_000)
    p.add_argument("--n-python", type=int, default=20_000)
    p.add_argument("--n-math", type=int, default=20_000)
    p.add_argument("--n-logic", type=int, default=10_000)
    p.add_argument("--w-python", type=float, default=0.4)
    p.add_argument("--w-math", type=float, default=0.4)
    p.add_argument("--w-logic", type=float, default=0.2)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({"event": "start", "n": args.n, "output": args.output,
                      "weights": {"python": args.w_python, "math": args.w_math,
                                  "logic": args.w_logic}}), flush=True)

    sources = [
        (PythonExecutionSource(n_examples=args.n_python, seed=args.seed), args.w_python),
        (MathQASource(n_examples=args.n_math, seed=args.seed + 1), args.w_math),
        (FormalLogicSource(n_examples=args.n_logic, seed=args.seed + 2), args.w_logic),
    ]
    mixer = CurriculumMixer(sources, seed=args.seed)
    t0 = time.time()
    stats = dump_jsonl(mixer, args.output, n=args.n)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps({"event": "done", **stats}, indent=2))


if __name__ == "__main__":
    main()
