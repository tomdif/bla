"""Eval Pythia-410M (public, web-pretrained) on PAL GSM8K.

Compares an off-the-shelf web-pretrained LM to our procedural-trained
500M on the same PAL task. If Pythia-410M scores >4%, our procedural
curriculum + RFT recipe is not architecturally special. If <2%, our
specialized training is doing real work.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.phase6_eval_pal import exec_python, extract_python, parse_number


def load_pythia(model_name: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    @torch.no_grad()
    def generate(prompt: str, max_new: int = 256) -> str:
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda")
    print(json.dumps({"event": "loading", "model": args.model}), flush=True)
    generate, _ = load_pythia(args.model, device)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        items.append(ex)

    correct = 0
    code_ran = 0
    t0 = time.time()
    samples = []
    for ex in items:
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground)
        prompt = (
            "Write a Python program that prints the answer to this math problem.\n"
            "End with: print(answer)\n"
            f"Problem: {question}\nPython:\n"
        )
        gen = generate(prompt, 256)
        code = extract_python(gen) or ""
        output = exec_python(code) if code else "ERROR: no code"
        pred = parse_number(output) if not output.startswith("ERROR") else None
        if not output.startswith("ERROR"):
            code_ran += 1
        ok = pred is not None and pred.rstrip(".") == gold.rstrip(".")
        if ok:
            correct += 1
        samples.append({"q": question[:80], "gold": gold, "pred": pred, "ok": ok})

    summary = {
        "model": args.model,
        "n_tested": len(items),
        "accuracy": correct / len(items),
        "code_ran": code_ran,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump({"summary": summary, "samples": samples[:30]}, f, indent=2)
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
