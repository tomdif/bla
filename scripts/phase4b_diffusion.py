"""Phase 4b — discrete diffusion text + retrieval-augmented generation.

A small D3PM-style masked-language-modeling diffusion model. Trains on
factual sentences (from `factual_corpus.py`) plus a small WikiText slice
for general English fluency. Compares parametric vs retrieval-augmented
hallucination on the Phase 4 QA benchmark.

The "non-AR text generation" property — predict every masked token in
parallel from full context — is the SEDD-class property the Phase 4b
gate cares about. We don't reproduce full SEDD here; we ship the
load-bearing comparison: does retrieval reduce hallucination on a
non-AR text model the same way it does on GPT-2 (Phase 4a)?

Tokenizer: GPT-2 BPE.
Architecture: 6-layer encoder transformer, d=256.
Diffusion: masked-LM with random mask ratio per example (D3PM
absorbing-state).

Run:
    python3 scripts/phase4b_diffusion.py \\
        --memory-db runs/phase3_facts/symbolic.db \\
        --output runs/phase4b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_jepa.factual_corpus import (
    COUNTRY_CAPITALS,
    PLANET_ORBITS,
    ELEMENT_ATOMIC_NUMBERS,
)
from system2_dca.symbolic_memory import SymbolicMemory
from verification import CommitmentObject


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-db", required=True)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wt-decay", type=float, default=0.01)
    p.add_argument("--n-denoise-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sentence templates for training (declarative form, the same template family
# as Phase 4a's natural-language context format)
# ---------------------------------------------------------------------------

def render_facts() -> list[str]:
    sents = []
    for country, capital, _ in COUNTRY_CAPITALS:
        sents.append(f"The capital of {country} is {capital}.")
    for body, parent, _ in PLANET_ORBITS:
        sents.append(f"{body} orbits {parent}.")
    for elem, z in ELEMENT_ATOMIC_NUMBERS:
        sents.append(f"{elem} has atomic number {z}.")
    return sents


def render_question_template(case: dict) -> tuple[str, str]:
    """Returns (template_with_mask, ground_truth_answer)."""
    pred = case["predicate"]
    if pred == "country_capital_is":
        return f"The capital of {case['subject']} is {{ANSWER}}.", case["ground_truth"]
    if pred == "orbits":
        return f"{case['subject']} orbits {{ANSWER}}.", case["ground_truth"]
    if pred == "atomic_number":
        return f"{case['subject']} has atomic number {{ANSWER}}.", case["ground_truth"]
    raise ValueError(f"unknown predicate {pred}")


# ---------------------------------------------------------------------------
# Model: encoder transformer + MLM head
# ---------------------------------------------------------------------------

class DiffusionTextModel(nn.Module):
    def __init__(self, vocab_size: int, d: int, n_layers: int, n_heads: int, max_len: int):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Embedding(max_len, d)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  # tied

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l = tokens.shape
        positions = torch.arange(l, device=tokens.device).unsqueeze(0).expand(b, l)
        x = self.tok_embed(tokens) + self.pos_embed(positions)
        time_feat = self.time_embed(t.float().unsqueeze(-1))
        x = x + time_feat.unsqueeze(1)
        if attention_mask is not None:
            key_padding = ~attention_mask.bool()
        else:
            key_padding = None
        x = self.blocks(x, src_key_padding_mask=key_padding)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training data: factual corpus + small WikiText slice
# ---------------------------------------------------------------------------

def build_training_corpus(extra_lines: int = 1000) -> list[str]:
    """Factual corpus (227 sentences) + a small WikiText-2 slice."""
    sents = render_facts()
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        added = 0
        for example in wiki:
            line = example["text"].strip()
            if not line or line.startswith("="):
                continue
            line = line.split(".")[0].strip() + "."
            if 8 <= len(line.split()) <= 30:
                sents.append(line)
                added += 1
                if added >= extra_lines:
                    break
    except Exception as exc:
        print(f"[warn] WikiText unavailable: {exc}; falling back to facts only", flush=True)
    return sents


# ---------------------------------------------------------------------------
# Tokenization + corruption
# ---------------------------------------------------------------------------

def tokenize_corpus(tokenizer, sents: list[str], max_len: int) -> list[torch.Tensor]:
    out = []
    for s in sents:
        ids = tokenizer.encode(s)
        if len(ids) > max_len - 1:
            continue
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


def make_batch(token_lists: list[torch.Tensor], batch_size: int, max_len: int,
               pad_id: int, mask_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch, corrupt with random mask ratio per example.
    Returns (corrupted, original, mask_indicator, t)."""
    chosen = random.choices(token_lists, k=batch_size)
    lens = [min(t.numel(), max_len) for t in chosen]
    L = max(lens)
    corrupted = torch.full((batch_size, L), pad_id, dtype=torch.long)
    original = torch.full((batch_size, L), pad_id, dtype=torch.long)
    attn = torch.zeros(batch_size, L, dtype=torch.bool)
    t_vec = torch.zeros(batch_size, dtype=torch.float32)
    is_masked = torch.zeros(batch_size, L, dtype=torch.bool)
    for i, (toks, ln) in enumerate(zip(chosen, lens)):
        original[i, :ln] = toks[:ln]
        corrupted[i, :ln] = toks[:ln]
        attn[i, :ln] = True
        t = float(np.random.uniform(0.05, 0.95))
        t_vec[i] = t
        # mask each non-pad position with prob t
        mask_probs = torch.rand(ln) < t
        is_masked[i, :ln] = mask_probs
        corrupted[i, :ln] = torch.where(mask_probs, torch.full_like(toks[:ln], mask_id), toks[:ln])
    return corrupted.to(device), original.to(device), is_masked.to(device), t_vec.to(device)


# ---------------------------------------------------------------------------
# Inference: iterative denoising
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_query(
    model: DiffusionTextModel,
    tokenizer,
    template: str,
    n_mask_tokens: int,
    n_steps: int,
    mask_id: int,
    device: torch.device,
) -> str:
    """Denoise a single query template with `n_mask_tokens` MASK placeholders.

    template uses '{ANSWER}' as the slot. We tokenize the prefix and suffix
    then insert n_mask_tokens MASK tokens between them, run iterative
    denoising over only the masked positions.
    """
    prefix, suffix = template.split("{ANSWER}")
    prefix_ids = tokenizer.encode(prefix.rstrip(), add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    seq = prefix_ids + [mask_id] * n_mask_tokens + suffix_ids
    if len(seq) > model.max_len:
        seq = seq[: model.max_len]
    tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    masked_positions = list(range(len(prefix_ids), len(prefix_ids) + n_mask_tokens))
    masked_set = set(masked_positions)

    for step in range(n_steps):
        t = torch.tensor([(n_steps - step) / max(n_steps, 1)], device=device, dtype=torch.float32)
        logits = model(tokens, t, attention_mask=attn)
        # softmax-argmax over masked positions; commit the most-confident first
        probs = F.softmax(logits[0, list(masked_set)], dim=-1)
        confs, picks = probs.max(dim=-1)
        # pick the position with highest confidence to commit this step
        commit_idx = int(confs.argmax().item())
        pos = list(masked_set)[commit_idx]
        tokens[0, pos] = picks[commit_idx]
        masked_set.remove(pos)
        if not masked_set:
            break

    # also fill any remaining masked positions
    for pos in masked_set:
        with torch.no_grad():
            t = torch.tensor([0.0], device=device, dtype=torch.float32)
            logits = model(tokens, t, attention_mask=attn)
            tokens[0, pos] = logits[0, pos].argmax()

    answer_ids = tokens[0, len(prefix_ids) : len(prefix_ids) + n_mask_tokens].tolist()
    return tokenizer.decode(answer_ids).strip()


def build_eval_set() -> list[dict]:
    cases = []
    for country, capital, src in COUNTRY_CAPITALS:
        cases.append({"predicate": "country_capital_is", "subject": country,
                      "ground_truth": capital, "source": src})
    for body, parent, src in PLANET_ORBITS:
        cases.append({"predicate": "orbits", "subject": body,
                      "ground_truth": parent, "source": src})
    for elem, z in ELEMENT_ATOMIC_NUMBERS:
        cases.append({"predicate": "atomic_number", "subject": elem,
                      "ground_truth": str(z), "source": f"periodic_table#{elem}"})
    return cases


def retrieve_context_sentence(sm: SymbolicMemory, case: dict) -> tuple[str, list]:
    """Same logic as Phase 4a, returning a single declarative sentence."""
    triples = sm.query_subject_predicate(case["subject"], predicate=case["predicate"])
    if not triples:
        resp = sm.query(f"{case['subject']} {case['predicate']}", top_k=1)
        triples = resp["triples"]
    if not triples:
        return "", []
    tr = triples[0]
    obj_text = tr.get("object_value")
    if obj_text is None and tr.get("object_id"):
        obj_ent = sm._memoria.kg.get_entity(tr["object_id"])
        obj_text = obj_ent.name if obj_ent else "?"
    pred = case["predicate"]
    if pred == "country_capital_is":
        sentence = f"The capital of {case['subject']} is {obj_text}."
    elif pred == "orbits":
        sentence = f"{case['subject']} orbits {obj_text}."
    elif pred == "atomic_number":
        sentence = f"{case['subject']} has atomic number {obj_text}."
    else:
        sentence = f"{case['subject']} {pred} {obj_text}."
    return sentence, [{"subject": case['subject'], "predicate": pred, "object": obj_text,
                       "source_ref": tr.get("source_ref")}]


def score(answer: str, ground_truth: str) -> bool:
    return ground_truth.lower() in answer.lower()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    pad_id = tokenizer.eos_token_id  # use EOS as pad
    # Add a [MASK] token to the vocab (SEDD-style absorbing state)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    vocab_size = len(tokenizer)
    print(json.dumps({"event": "tokenizer", "vocab_size": vocab_size,
                      "pad_id": pad_id, "mask_id": mask_id, "added": n_added}), flush=True)

    print(json.dumps({"event": "loading_corpus"}), flush=True)
    corpus = build_training_corpus(extra_lines=2000)
    token_lists = tokenize_corpus(tokenizer, corpus, args.max_len)
    print(json.dumps({"event": "corpus", "n_sentences": len(corpus),
                      "n_tokenized": len(token_lists),
                      "median_len": int(np.median([t.numel() for t in token_lists]))}), flush=True)

    sm = SymbolicMemory(db_path=args.memory_db)

    model = DiffusionTextModel(
        vocab_size=vocab_size, d=args.d, n_layers=args.n_layers,
        n_heads=args.n_heads, max_len=args.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "model", "params": n_params}), flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)

    t0 = time.time()
    losses = []
    log_every = max(args.steps // 20, 50)
    for step in range(args.steps):
        corrupted, original, is_masked, t_vec = make_batch(
            token_lists, args.batch_size, args.max_len, pad_id, mask_id, device,
        )
        attn = (corrupted != pad_id) | is_masked
        logits = model(corrupted, t_vec, attention_mask=attn)
        # CE loss only on masked positions
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            original.masked_fill(~is_masked, -100),
            ignore_index=-100,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.detach()))
        if (step + 1) % log_every == 0:
            print(json.dumps({"event": "train", "step": step + 1,
                              "loss": float(np.mean(losses[-log_every:])),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    train_elapsed = time.time() - t0

    # ------------- Evaluation -------------
    cases = build_eval_set()
    n_per_relation = {p: 0 for p in {"country_capital_is", "orbits", "atomic_number"}}
    para_results = []
    rag_results = []
    para_correct = 0
    rag_correct = 0

    for i, case in enumerate(cases):
        template, gt = render_question_template(case)
        gt_token_ids = tokenizer.encode(" " + gt, add_special_tokens=False)
        n_mask = max(1, len(gt_token_ids))

        # Parametric: just the template with masks, no context
        ans_para = denoise_query(model, tokenizer, template, n_mask,
                                 args.n_denoise_steps, mask_id, device)
        ok_para = score(ans_para, gt)

        # RAG: prepend retrieved context sentence
        context_sentence, evidence = retrieve_context_sentence(sm, case)
        rag_template = f"{context_sentence} {template}" if context_sentence else template
        ans_rag = denoise_query(model, tokenizer, rag_template, n_mask,
                                args.n_denoise_steps, mask_id, device)
        ok_rag = score(ans_rag, gt)

        para_results.append({**case, "answer": ans_para, "correct": ok_para})
        rag_results.append({**case, "answer": ans_rag, "correct": ok_rag,
                            "evidence_len": len(evidence)})
        if ok_para:
            para_correct += 1
        if ok_rag:
            rag_correct += 1
        n_per_relation[case["predicate"]] += 1

        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "eval", "i": i + 1, "n": len(cases),
                              "para": para_correct, "rag": rag_correct}), flush=True)

    n = len(cases)
    para_acc = para_correct / max(n, 1)
    rag_acc = rag_correct / max(n, 1)
    para_hall = 1.0 - para_acc
    rag_hall = 1.0 - rag_acc
    ratio = rag_hall / max(para_hall, 1e-6)

    by_pred = {}
    for p in n_per_relation:
        ps = [r for r in para_results if r["predicate"] == p]
        rs = [r for r in rag_results if r["predicate"] == p]
        by_pred[p] = {
            "n": len(ps),
            "parametric_acc": sum(r["correct"] for r in ps) / max(len(ps), 1),
            "rag_acc": sum(r["correct"] for r in rs) / max(len(rs), 1),
        }

    summary = {
        "model_params": n_params,
        "train_steps": args.steps,
        "train_elapsed_s": round(train_elapsed, 1),
        "n_questions": n,
        "parametric_accuracy": para_acc,
        "rag_accuracy": rag_acc,
        "parametric_hallucination": para_hall,
        "rag_hallucination": rag_hall,
        "rag_vs_parametric_ratio": ratio,
        "by_predicate": by_pred,
        "gate_ratio": 0.5,
        "gate_passed": rag_hall <= 0.5 * para_hall,
    }
    print(json.dumps({"event": "summary", **summary}, indent=2), flush=True)

    with open(os.path.join(args.output, "phase4b.json"), "w") as f:
        json.dump({"summary": summary,
                   "parametric_results": para_results,
                   "rag_results": rag_results}, f, indent=2)

    # Save trained model + sample commitment objects
    torch.save({
        "state_dict": model.state_dict(),
        "config": vars(args),
        "summary": summary,
    }, os.path.join(args.output, "model.pt"))


if __name__ == "__main__":
    main()
