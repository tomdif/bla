"""Multi-sample PAL eval with critic re-ranking and variable attractor iters.

For each GSM8K-test problem:
  - Generate N=16 candidates with sampling
  - Diversify by varying (temperature, attractor_iters)
  - Execute each, collect (code, pred) pairs
  - Score (problem, code) with the Phase 10 DistilBERT critic
  - Report 4 selection strategies: greedy / mode-vote / critic-pick / oracle@N

Tests whether the model's *true* PAL is hidden behind selection bias
(critic-pick approaches oracle), vs the model genuinely capping at 3.5%
(oracle itself stays low).

Usage:
    python scripts/multi_sample_critic_pal.py \
        --ckpt runs/run20/ckpt_step00011000.pt \
        --critic runs/phase10/critic.pt \
        --n-problems 200 \
        --n-samples 16 \
        --output runs/run20/multi_sample_pal
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn

from system2_dca.mt_ssm_core import MTSSMCore, MTSSMConfig
from scripts.phase6_eval_pal import exec_python, extract_python, parse_number


# Sampling/iter variants to apply across the 16 candidates.
# Designed to create diversity along two axes: sampling temperature
# (output randomness) and attractor iteration count (refinement depth).
SAMPLING_VARIANTS = [
    # (temperature, top_p, attractor_iters)
    (0.0, 1.00, 4),    # greedy baseline
    (0.0, 1.00, 8),    # greedy + deeper refinement
    (0.3, 0.95, 4),
    (0.3, 0.95, 8),
    (0.5, 0.95, 4),
    (0.5, 0.95, 8),
    (0.5, 0.95, 16),
    (0.7, 0.90, 4),
    (0.7, 0.90, 8),
    (0.7, 0.90, 16),
    (0.7, 0.90, 32),
    (0.9, 0.90, 4),
    (0.9, 0.90, 8),
    (0.9, 0.90, 16),
    (1.0, 0.90, 8),
    (1.0, 0.85, 16),
]


def load_mtssm(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = state["config"]
    cfg = MTSSMConfig(
        vocab_size=c["vocab"], d_model=c["d"], n_layers=c["n_layers"],
        state_fast=c.get("state_fast", 256),
        state_med=c.get("state_med", 512),
        state_slow=c.get("state_slow", 1024),
        max_seq_len=c["seq_len"], dropout=0.0, pred_loss_weight=0.0,
        use_memory=c.get("use_memory", False),
        n_slots=c.get("n_slots", 16),
        slot_chunk=c.get("slot_chunk", 64),
        use_attractor=c.get("use_attractor", False),
        attractor_layers=c.get("attractor_layers", 1),
        attractor_train_iters=c.get("attractor_train_iters", 3),
        attractor_infer_iters=c.get("attractor_infer_iters", 8),
        attractor_n_heads=c.get("attractor_n_heads", 8),
    )
    model = MTSSMCore(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, cfg


def load_critic(critic_path, device):
    """Load Phase 10 DistilBERT critic. state_dict format from train_critic.py."""
    from transformers import AutoTokenizer, AutoModel
    state = torch.load(critic_path, map_location=device, weights_only=False)
    model_name = state.get("config", {}).get("model_name", "distilbert-base-uncased")
    tok = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)

    class Critic(nn.Module):
        def __init__(self, backbone, hidden):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(hidden, 2)
        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(h[:, 0])

    critic = Critic(backbone, backbone.config.hidden_size).to(device)
    critic.load_state_dict(state["model_state"])
    critic.eval()
    return critic, tok


def make_generator(model, cfg, tok):
    """Returns generate(prompt, max_new, temperature, top_p, attractor_iters)."""

    @torch.no_grad()
    def generate(prompt: str, max_new: int, temperature: float,
                 top_p: float, attractor_iters: int) -> str:
        device = next(model.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        if ids.shape[1] >= cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len + max_new:]
        # Runtime override of attractor iter count
        if model.attractor is not None:
            old_infer = model.attractor.infer_iters
            model.attractor.infer_iters = attractor_iters
        try:
            out = ids
            for _ in range(max_new):
                logits = model.forward(out)[:, -1].float()
                if temperature <= 1e-5:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        sp, si = probs.sort(dim=-1, descending=True)
                        cs = sp.cumsum(dim=-1)
                        mask = cs - sp > top_p
                        sp[mask] = 0
                        sp = sp / sp.sum(dim=-1, keepdim=True)
                        picked = torch.multinomial(sp, 1)
                        next_id = si.gather(-1, picked)
                    else:
                        next_id = torch.multinomial(probs, 1)
                out = torch.cat([out, next_id], dim=1)
                if int(next_id.item()) == tok.eos_token_id or out.shape[1] >= cfg.max_seq_len:
                    break
            full = tok.decode(out[0], skip_special_tokens=True)
            return full[len(prompt):]
        finally:
            if model.attractor is not None:
                model.attractor.infer_iters = old_infer

    return generate


def critic_score(critic, tok, problem: str, code: str, device, max_len: int = 384):
    text = f"Problem: {problem}\nCode:\n{code}"
    enc = tok(text, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt").to(device)
    with torch.no_grad():
        logits = critic(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(prob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--critic", required=True)
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--start", type=int, default=None,
                   help="If set, evaluate problems [start, end). Overrides --n-problems.")
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=16,
                   help="Number of candidates per problem (uses first N of SAMPLING_VARIANTS)")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "loading_model", "ckpt": args.ckpt}), flush=True)
    model, cfg = load_mtssm(args.ckpt, device)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    generate = make_generator(model, cfg, tok)

    print(json.dumps({"event": "loading_critic", "critic": args.critic}), flush=True)
    critic, critic_tok = load_critic(args.critic, device)

    print(json.dumps({"event": "loading_dataset"}), flush=True)
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    if args.start is not None and args.end is not None:
        lo, hi = args.start, args.end
    else:
        lo, hi = 0, args.n_problems
    items = []
    for i, ex in enumerate(ds):
        if i >= hi:
            break
        if i >= lo:
            items.append(ex)

    variants = SAMPLING_VARIANTS[: args.n_samples]
    print(json.dumps({"event": "starting", "n_problems": len(items),
                      "n_samples_per": len(variants)}), flush=True)

    by_problem = []
    correct_greedy = 0
    correct_mode = 0
    correct_critic = 0
    correct_oracle = 0
    t0 = time.time()

    for prob_idx, ex in enumerate(items):
        question = ex["question"]
        ground = ex["answer"].split("####")[-1].strip()
        gold = re.sub(r"[\$,]", "", ground).rstrip(".")

        prompt = (
            "Write a Python program that prints the answer to this math problem.\n"
            "End with: print(answer)\n"
            f"Problem: {question}\nPython:\n"
        )

        candidates = []
        for v_idx, (temp, top_p, iters) in enumerate(variants):
            gen = generate(prompt, args.max_new, temp, top_p, iters)
            code = extract_python(gen) or ""
            output = exec_python(code) if code else "ERROR: no code"
            pred = parse_number(output) if not output.startswith("ERROR") else None
            ok = pred is not None and pred.rstrip(".") == gold
            critic_p = critic_score(critic, critic_tok, question, code, device) if code else 0.0
            candidates.append({
                "v_idx": v_idx, "temp": temp, "top_p": top_p, "iters": iters,
                "code": code, "pred": pred, "ok": int(ok),
                "critic_p": round(critic_p, 4),
            })

        # Strategy outcomes
        # Greedy = first variant (temp=0.0, iters=4)
        greedy_ok = candidates[0]["ok"]
        # Mode-vote
        valid_preds = [c["pred"] for c in candidates if c["pred"] is not None]
        if valid_preds:
            mode_pred, _ = collections.Counter(valid_preds).most_common(1)[0]
            mode_ok = int(mode_pred.rstrip(".") == gold)
        else:
            mode_ok = 0
        # Critic-pick: highest critic score wins
        best_c = max(candidates, key=lambda c: (c["critic_p"], c["pred"] is not None))
        critic_ok = best_c["ok"]
        # Oracle: any candidate correct
        oracle_ok = int(any(c["ok"] for c in candidates))

        correct_greedy += greedy_ok
        correct_mode += mode_ok
        correct_critic += critic_ok
        correct_oracle += oracle_ok

        by_problem.append({
            "q": question[:80], "gold": gold,
            "greedy": greedy_ok, "mode": mode_ok,
            "critic": critic_ok, "oracle": oracle_ok,
            "n_valid": len(valid_preds),
            "best_critic_p": best_c["critic_p"],
            "best_critic_ok": best_c["ok"],
        })

        if (prob_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (prob_idx + 1) / elapsed
            eta = (len(items) - prob_idx - 1) / rate
            print(json.dumps({
                "done": prob_idx + 1,
                "greedy": correct_greedy / (prob_idx + 1),
                "mode": correct_mode / (prob_idx + 1),
                "critic": correct_critic / (prob_idx + 1),
                "oracle": correct_oracle / (prob_idx + 1),
                "rate_problems_per_s": round(rate, 2),
                "eta_s": round(eta),
            }), flush=True)

    summary = {
        "n_problems": len(items),
        "n_samples": len(variants),
        "ckpt": args.ckpt,
        "critic": args.critic,
        "greedy_acc": correct_greedy / len(items),
        "mode_vote_acc": correct_mode / len(items),
        "critic_pick_acc": correct_critic / len(items),
        "oracle_acc": correct_oracle / len(items),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump({"summary": summary,
                   "samples": by_problem[:40],
                   "variants": variants}, f, indent=2)
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
