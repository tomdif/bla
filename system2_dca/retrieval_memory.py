"""Retrieval memory backend for the BLA RETRIEVE action.

Indexes a corpus of (problem, solution) pairs and exposes a nearest-
neighbor lookup. Used at inference to populate few-shot demonstrations
in the SIMULATE prompt — the practical embodiment of:

    router.dispatch(RETRIEVE) → memory.lookup(query) → demos
    router.dispatch(SIMULATE) → procedural_core.generate(demos + query)

Two backends:

  TFIDFRetriever  — local, fast, no GPU. Uses sklearn TfidfVectorizer
                    + cosine similarity. ~3MB index for GSM8K-train.

  EmbeddingRetriever — placeholder for sentence-transformer or learned
                       embeddings. Same .lookup() signature so callers
                       are agnostic.

For GSM8K specifically the corpus is gsm8k-train (7473 problems with
chain-of-thought answers including <<expr=result>> markers).
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class RetrievedExample:
    question: str
    answer_cot: str  # natural-language reasoning
    answer_python: str  # extracted/built Python solution
    final_answer: str  # the #### N
    score: float  # similarity to query (higher = more similar)


def _extract_answer(answer_field: str) -> str:
    m = re.search(r"####\s*(-?\d[\d,\.]*)", answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def _strip_markers(s: str) -> str:
    return re.sub(r"<<[^>]*>>", "", s)


def _build_python(answer_field: str) -> str:
    """Build chained-variable Python from <<expr=result>> markers.

    Mirrors curriculum_gsm8k_v3 logic so retrieved demos use the same
    Python format the model was trained to emit.
    """
    NUM_TOKEN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
    CALC = re.compile(r"<<([^>=]+?)=([^>]+?)>>")
    matches = list(CALC.finditer(answer_field))
    if not matches:
        return ""
    step_results: list[tuple[str, str]] = []
    lines: list[str] = []
    for i, m in enumerate(matches, start=1):
        expr = re.sub(r"[\$,]", "", m.group(1)).strip()
        res = re.sub(r"[\$,]", "", m.group(2)).strip()
        out_expr = ""
        last_end = 0
        for tm in NUM_TOKEN.finditer(expr):
            out_expr += expr[last_end:tm.start()]
            tok = tm.group()
            replacement = None
            for s_name, s_res in reversed(step_results):
                try:
                    if float(s_res) == float(tok):
                        replacement = s_name
                        break
                except ValueError:
                    pass
            out_expr += replacement if replacement is not None else tok
            last_end = tm.end()
        out_expr += expr[last_end:]
        var = f"step{i}"
        lines.append(f"{var} = {out_expr}")
        step_results.append((var, res))
    lines.append(f"answer = step{len(matches)}")
    lines.append("print(answer)")
    return "\n".join(lines)


class TFIDFRetriever:
    """Local TF-IDF retriever. Cheap to build, no GPU, ~ms lookups for
    7K-problem corpus.
    """

    def __init__(self, problems: list[dict],
                 ngram_range: tuple[int, int] = (1, 2),
                 max_features: int = 20000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity  # noqa

        self.problems = problems
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform([p["question"] for p in problems])

    def lookup(self, query: str, k: int = 5,
               exclude_exact: bool = True) -> list[RetrievedExample]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        # Get top-k indices
        idx_sorted = sims.argsort()[::-1]
        results = []
        for i in idx_sorted:
            if len(results) >= k:
                break
            if exclude_exact and self.problems[i]["question"].strip() == query.strip():
                continue
            p = self.problems[i]
            results.append(RetrievedExample(
                question=p["question"],
                answer_cot=p.get("answer_cot", ""),
                answer_python=p.get("answer_python", ""),
                final_answer=p.get("final_answer", ""),
                score=float(sims[i]),
            ))
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"problems": self.problems,
                         "vectorizer": self.vectorizer,
                         "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        inst = cls.__new__(cls)
        inst.problems = d["problems"]
        inst.vectorizer = d["vectorizer"]
        inst.matrix = d["matrix"]
        return inst


def build_gsm8k_train_index() -> TFIDFRetriever:
    """Build a TF-IDF index over the full GSM8K-train split."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    problems = []
    for ex in ds:
        cot = _strip_markers(ex["answer"]).split("####")[0].strip()
        py = _build_python(ex["answer"])
        final = _extract_answer(ex["answer"])
        problems.append({
            "question": ex["question"],
            "answer_cot": cot,
            "answer_python": py,
            "final_answer": final,
        })
    return TFIDFRetriever(problems)


def format_few_shot_prompt(query: str, demos: list[RetrievedExample],
                            include_python: bool = True) -> str:
    """Render demos + query into a PAL prompt the model can use."""
    lines = [
        "Write a Python program that prints the answer to this math problem.",
        "End with: print(answer)",
        "",
        "Here are similar problems for reference:",
    ]
    for i, d in enumerate(demos, start=1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Problem: {d.question}")
        if include_python and d.answer_python:
            lines.append("Python:")
            lines.append(d.answer_python)
            lines.append(f"Answer: {d.final_answer}")
        else:
            lines.append(f"Answer: {d.final_answer}")
    lines.append("")
    lines.append("--- Your turn ---")
    lines.append(f"Problem: {query}")
    lines.append("Python:")
    return "\n".join(lines)
