"""PAL eval for MT-SSM checkpoints.

Mirrors scripts/phase6_eval_pal.py but loads an MTSSMCore from a
phase6_train_mtssm.py checkpoint. Greedy decode → extract Python →
sandbox-exec → compare to GSM8K-test gold.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from system2_dca.mt_ssm_core import MTSSMCore, MTSSMConfig
from scripts.phase6_eval_pal import run_math_pal


def load_mtssm(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = state["config"]
    cfg = MTSSMConfig(
        vocab_size=cfg_d["vocab"],
        d_model=cfg_d["d"],
        n_layers=cfg_d["n_layers"],
        state_fast=cfg_d.get("state_fast", 256),
        state_med=cfg_d.get("state_med", 512),
        state_slow=cfg_d.get("state_slow", 1024),
        max_seq_len=cfg_d["seq_len"],
        dropout=0.0,
        pred_loss_weight=0.0,
        use_memory=cfg_d.get("use_memory", False),
        n_slots=cfg_d.get("n_slots", 16),
        slot_chunk=cfg_d.get("slot_chunk", 64),
        use_attractor=cfg_d.get("use_attractor", False),
        attractor_layers=cfg_d.get("attractor_layers", 1),
        attractor_train_iters=cfg_d.get("attractor_train_iters", 3),
        attractor_infer_iters=cfg_d.get("attractor_infer_iters", 8),
        attractor_n_heads=cfg_d.get("attractor_n_heads", 8),
    )
    model = MTSSMCore(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(state["state_dict"])
    model.eval()

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int) -> str:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        if ids.shape[1] >= cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len + max_new_tokens:]
        out = ids
        for _ in range(max_new_tokens):
            logits = model.forward(out)[:, -1]
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if int(next_id.item()) == tok.eos_token_id or out.shape[1] >= cfg.max_seq_len:
                break
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):]

    return generate, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    print(json.dumps({"event": "loading", "ckpt": args.ckpt}), flush=True)
    generate, _ = load_mtssm(args.ckpt, device)

    t0 = time.time()
    result = run_math_pal(generate, args.n, args.max_new)
    result["elapsed_s"] = round(time.time() - t0, 1)
    summary = {
        "ckpt": args.ckpt,
        "accuracy": result["accuracy"],
        "code_extracted": result["code_extracted"],
        "code_ran": result["code_ran"],
        "n_tested": result["n_tested"],
        "elapsed_s": result["elapsed_s"],
    }
    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump({"summary": summary, "samples": result["results"][:30]}, f, indent=2)
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
