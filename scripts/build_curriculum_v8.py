"""Build curriculum v8 from v7 + teacher-distilled pairs.

v7 base: 665620 examples (word_math, word_to_python, gsm8k_train, RFT, cot_pal,
python_exec, formal_logic, math_qa).

v8 adds the teacher-distilled (gsm8k_full + math_full) pool, repeated `--rep`
times so the high-quality teacher content is meaningfully represented in the
training mix instead of getting drowned. With rep=10 and ~10K teacher pairs,
that's 100K teacher examples mixed into ~765K total (~13% teacher share).

Output is a single JSONL keeping only the `prompt` and `target` fields used
by the trainer; teacher `_meta` is dropped at this stage but the source files
preserve it for provenance.
"""
from __future__ import annotations
import argparse, json, os, random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7", required=True,
                   help="Path to curriculum_v7_with_rft.jsonl on this machine")
    p.add_argument("--teacher", nargs="+", required=True,
                   help="One or more teacher-distill JSONL files to mix in")
    p.add_argument("--rep", type=int, default=10,
                   help="Repetition multiplier on the teacher pool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    teacher_examples = []
    for path in args.teacher:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                # Drop _meta to keep the trainer schema clean
                teacher_examples.append({"prompt": rec["prompt"],
                                          "target": rec["target"]})
    print(f"loaded {len(teacher_examples)} teacher examples")

    rng = random.Random(args.seed)
    rng.shuffle(teacher_examples)

    n_v7 = 0
    with open(args.output, "w") as out:
        with open(args.v7) as f:
            for line in f:
                out.write(line)
                n_v7 += 1
        for _ in range(args.rep):
            for rec in teacher_examples:
                out.write(json.dumps(rec) + "\n")

    n_teacher_written = len(teacher_examples) * args.rep
    print(f"wrote {n_v7} from v7 + {n_teacher_written} teacher (x{args.rep}) "
          f"= {n_v7 + n_teacher_written} total -> {args.output}")


if __name__ == "__main__":
    main()
