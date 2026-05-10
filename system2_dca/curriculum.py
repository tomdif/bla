"""Curriculum data loaders for the Phase 6 procedural CPU.

Each `CurriculumSource` produces (prompt, target) pairs of pure
procedural content — no factual trivia. The mixer combines sources at
configurable weights and yields a unified stream the trainer consumes.

The contract:
  prompt   — text the model conditions on (problem statement)
  target   — text the model should produce (solution / output / proof)
  source   — string identifying which curriculum branch
  metadata — optional dict with provenance / difficulty / tokens

Sources implemented in this commit:
  PythonExecutionSource     — synthetic Python programs + their stdout
  MathQASource              — MetaMathQA from HuggingFace
  FormalLogicSource         — FOLIO from HuggingFace

Stays factual-free: we never include Wikipedia-style facts. Code
strings reference variables and procedures only; math problems are
abstract; logic premises use synthetic entities.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass
class CurriculumExample:
    prompt: str
    target: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "target": self.target,
                "source": self.source, "metadata": self.metadata}


class CurriculumSource(ABC):
    name: str = "abstract"

    @abstractmethod
    def __iter__(self) -> Iterator[CurriculumExample]: ...


class CurriculumMixer:
    """Round-robin over multiple sources at configurable weights."""

    def __init__(self, sources_with_weights: list[tuple[CurriculumSource, float]],
                 seed: int = 0):
        self.sources = [s for s, _ in sources_with_weights]
        weights = [w for _, w in sources_with_weights]
        total = sum(weights)
        self.probs = [w / total for w in weights]
        self.iters = [iter(s) for s in self.sources]
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[CurriculumExample]:
        return self

    def __next__(self) -> CurriculumExample:
        for _ in range(len(self.sources) * 2):  # bounded fallthrough
            idx = self.rng.choices(range(len(self.sources)), weights=self.probs, k=1)[0]
            try:
                return next(self.iters[idx])
            except StopIteration:
                # restart that source's iterator
                self.iters[idx] = iter(self.sources[idx])
        raise StopIteration("all curriculum sources exhausted")


def dump_jsonl(stream: Iterable[CurriculumExample], path: str, n: int) -> dict:
    """Write up to n examples from `stream` to `path` as JSONL.
    Returns stats dict."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    by_source: dict[str, int] = {}
    written = 0
    with open(path, "w") as f:
        for ex in stream:
            f.write(json.dumps(ex.to_dict()) + "\n")
            by_source[ex.source] = by_source.get(ex.source, 0) + 1
            written += 1
            if written >= n:
                break
    return {"written": written, "by_source": by_source, "path": path}
