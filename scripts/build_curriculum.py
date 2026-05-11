"""Build the Phase 6 procedural-CPU curriculum and dump as JSONL.

Mixes multiple curriculum sources at configurable weights:
  - PythonExecutionSource  : synthetic code -> stdout
  - MathQASource           : MetaMathQA from HF
  - FormalLogicSource      : propositional logic from FOLIO + synthetic
  - WordMathSource         : synthetic GSM8K-style word problems + CoT
  - WordToPythonSource     : synthetic word problems -> Python solutions
  - GSM8KTrainSource (v1)  : real GSM8K-train; v1 weak supervision (DEPRECATED — taught guessing not computation, see run10 result)
  - GSM8KTrainV2Source     : real GSM8K-train with <<expr=result>>-derived Python (PREFERRED for PAL training)

Run:
    python3 scripts/build_curriculum.py \\
        --n 300000 --output runs/phase6/curriculum.jsonl \\
        --w-python 0.20 --w-math 0.05 --w-logic 0.10 \\
        --w-word-math 0.15 --w-w2p 0.25 --w-gsm8k-v2 0.25
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
from system2_dca.curriculum_word_math import WordMathSource
from system2_dca.curriculum_word_to_python import WordToPythonSource
from system2_dca.curriculum_gsm8k_train import GSM8KTrainSource
from system2_dca.curriculum_gsm8k_v2 import GSM8KTrainV2Source
from system2_dca.curriculum_gsm8k_v3 import GSM8KTrainV3Source
from system2_dca.curriculum_cot_pal import CoTPALSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30_000)
    p.add_argument("--n-python", type=int, default=20_000)
    p.add_argument("--n-math", type=int, default=20_000)
    p.add_argument("--n-logic", type=int, default=10_000)
    p.add_argument("--n-word-math", type=int, default=0)
    p.add_argument("--n-w2p", type=int, default=0)
    p.add_argument("--n-gsm8k", type=int, default=0)
    p.add_argument("--n-gsm8k-v2", type=int, default=0)
    p.add_argument("--n-gsm8k-v3", type=int, default=0)
    p.add_argument("--n-cot-pal", type=int, default=0)
    p.add_argument("--w-python", type=float, default=0.4)
    p.add_argument("--w-math", type=float, default=0.4)
    p.add_argument("--w-logic", type=float, default=0.2)
    p.add_argument("--w-word-math", type=float, default=0.0)
    p.add_argument("--w-w2p", type=float, default=0.0)
    p.add_argument("--w-gsm8k", type=float, default=0.0)
    p.add_argument("--w-gsm8k-v2", type=float, default=0.0)
    p.add_argument("--w-gsm8k-v3", type=float, default=0.0)
    p.add_argument("--w-cot-pal", type=float, default=0.0)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = {
        "python": args.w_python, "math": args.w_math, "logic": args.w_logic,
        "word_math": args.w_word_math, "w2p": args.w_w2p,
        "gsm8k": args.w_gsm8k, "gsm8k_v2": args.w_gsm8k_v2,
        "gsm8k_v3": args.w_gsm8k_v3, "cot_pal": args.w_cot_pal,
    }
    print(json.dumps({"event": "start", "n": args.n, "output": args.output,
                      "weights": weights}), flush=True)

    sources = [
        (PythonExecutionSource(n_examples=args.n_python, seed=args.seed), args.w_python),
        (MathQASource(n_examples=args.n_math, seed=args.seed + 1), args.w_math),
        (FormalLogicSource(n_examples=args.n_logic, seed=args.seed + 2), args.w_logic),
        (WordMathSource(n_examples=args.n_word_math, seed=args.seed + 3), args.w_word_math),
        (WordToPythonSource(n_examples=args.n_w2p, seed=args.seed + 4), args.w_w2p),
        (GSM8KTrainSource(n_examples=args.n_gsm8k, seed=args.seed + 5, mode="mixed"), args.w_gsm8k),
        (GSM8KTrainV2Source(n_examples=args.n_gsm8k_v2, seed=args.seed + 6, mode="mixed"), args.w_gsm8k_v2),
        (GSM8KTrainV3Source(n_examples=args.n_gsm8k_v3, seed=args.seed + 7, mode="mixed"), args.w_gsm8k_v3),
        (CoTPALSource(n_examples=args.n_cot_pal, seed=args.seed + 8), args.w_cot_pal),
    ]
    mixer = CurriculumMixer(sources, seed=args.seed)
    t0 = time.time()
    stats = dump_jsonl(mixer, args.output, n=args.n)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps({"event": "done", **stats}, indent=2))


if __name__ == "__main__":
    main()
