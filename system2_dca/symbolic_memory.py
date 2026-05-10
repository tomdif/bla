"""SymbolicMemory — typed entity / typed triple store with provenance,
freshness, contradiction detection, and graph-walk retrieval.

Wraps the existing `memoria` repo (https://github.com/tomdif/memoria),
which already ships:
  * SQLite-backed knowledge graph (entities + triples)
  * Three-pass retrieval (graph walk + bi-encoder + cross-encoder rerank,
    95% R@5 on LongMemEval)
  * Contradiction detection on write
  * Temporal validity (valid_from / valid_until)
  * Confidence + access tracking

What this wrapper adds:
  * Bypasses memoria's LLM-based extractor for *structured* ingestion
    (Wikidata RDF, code AST, math objects) — call `add_typed_entity`
    and `add_typed_triple` directly instead of `Memoria.remember(text)`.
  * Torch-side similarity for differentiable read paths — entity
    embeddings are loaded as a frozen tensor buffer, query gradients
    flow through inner-product scores.
  * `query_subject_predicate` for direct typed lookups with provenance.

Memoria itself stays the source of truth for storage and retrieval; this
wrapper is a thin contract layer for B.L.A.'s symbolic memory layer.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

# Ensure memoria package is importable
_MEMORIA_PATH = Path("/Users/thomasdifiore/memoria")
if str(_MEMORIA_PATH) not in sys.path:
    sys.path.insert(0, str(_MEMORIA_PATH))


def _import_memoria():
    from memoria.core import Memoria  # type: ignore
    return Memoria


class SymbolicMemory:
    """Typed knowledge graph with provenance, backed by memoria."""

    def __init__(
        self,
        db_path: str | Path = "~/.bla/symbolic_memory.db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        Memoria = _import_memoria()
        self._memoria = Memoria(db_path=db_path, model_name=embedding_model)

    # --- structured ingestion (bypass LLM extractor) -------------------

    def add_typed_entity(
        self,
        name: str,
        entity_type: Optional[str] = None,
        confidence: float = 1.0,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Add or return an existing entity. Returns its ID.

        Pre-computed embedding is optional; if None, no embedding is stored
        (fine for entities that won't be retrieved by similarity, only by name).
        """
        return self._memoria.kg.add_entity(
            name=name, entity_type=entity_type, confidence=confidence, embedding=embedding,
        )

    def add_typed_triple(
        self,
        subject_id: str,
        predicate: str,
        object_id: Optional[str] = None,
        object_value: Optional[str] = None,
        relation_type: str = "fact",
        confidence: float = 1.0,
        valid_from: Optional[float] = None,
        source_ref: Optional[str] = None,
    ) -> tuple[str, list]:
        """Insert a triple. Returns (triple_id, contradictions). The
        memoria graph layer auto-detects contradictions and supersedes
        old facts via valid_until — provenance and freshness are
        preserved without manual bookkeeping."""
        return self._memoria.kg.add_triple(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            object_value=object_value,
            relation_type=relation_type,
            confidence=confidence,
            valid_from=valid_from,
            source_ref=source_ref,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """Convenience: compute an embedding for a string using the
        configured embedder. Useful for adding entity embeddings during
        structured ingestion."""
        return self._memoria.embedder.embed_single(text)

    # --- retrieval ------------------------------------------------------

    def query(self, text: str, top_k: int = 8, mode: str = "balanced") -> dict:
        """Three-pass retrieval over the full knowledge graph. Returns the
        memoria response (entities + triples with provenance + scores)."""
        response = self._memoria.recall(text, top_k=top_k, mode=mode)
        return self._format_response(response)

    def query_subject_predicate(
        self,
        subject_name: str,
        predicate: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """Direct graph lookup for "what does the model know about
        (subject, predicate)?" Returns triples with provenance and
        validity windows."""
        ents = self._memoria.kg.find_entities(subject_name)
        if not ents:
            return []
        results = []
        for ent in ents:
            triples = self._memoria.kg.get_triples(
                subject_id=ent.id, predicate=predicate, active_only=active_only
            )
            for tr in triples:
                results.append(self._format_triple(tr, subject_entity=ent))
        return results

    def find_entity(self, name: str) -> Optional[dict]:
        """Return the first entity matching a name, with metadata. None if not found."""
        ents = self._memoria.kg.find_entities(name)
        if not ents:
            return None
        ent = ents[0]
        return {
            "id": ent.id,
            "name": ent.name,
            "type": ent.entity_type,
            "confidence": ent.confidence,
            "access_count": ent.access_count,
        }

    # --- torch-side differentiable similarity --------------------------

    def export_entity_embedding_table(
        self, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, list[str]]:
        """Materialize all stored entity embeddings into a single tensor
        and a parallel list of entity IDs. The tensor is a frozen buffer
        — freeze it on the model side and use `differentiable_similarity`
        for a query-side gradient path."""
        rows = self._memoria.db.execute(
            "SELECT id, embedding FROM entities WHERE embedding IS NOT NULL"
        ).fetchall()
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            entity_id, blob = row
            if blob is None:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            ids.append(entity_id)
            vectors.append(vec)
        if not vectors:
            return torch.zeros(0, dtype=dtype), []
        stacked = np.stack(vectors, axis=0)
        return torch.from_numpy(stacked).to(dtype=dtype), ids

    @staticmethod
    def differentiable_similarity(
        query: torch.Tensor,
        keys: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Inner-product similarity with optional cosine normalization.
        Gradient flows through query; keys are treated as a frozen buffer."""
        if normalize:
            query = torch.nn.functional.normalize(query, dim=-1)
            keys = torch.nn.functional.normalize(keys, dim=-1)
        return torch.einsum("...d,nd->...n", query, keys)

    # --- introspection --------------------------------------------------

    def stats(self) -> dict:
        return self._memoria.graph_stats()

    def close(self) -> None:
        self._memoria.close()

    # --- internal -------------------------------------------------------

    def _format_response(self, response) -> dict:
        """Adapt memoria's RetrievalResponse(results=[RetrievalResult(triple, score, source, depth, entity_name)])
        into a flat list of typed triple dicts annotated with retrieval metadata."""
        out: dict = {
            "query_entities": list(getattr(response, "query_entities", []) or []),
            "screening_depth": getattr(response, "screening_depth", None),
            "spectral_gap": getattr(response, "local_spectral_gap", None),
            "passes_used": list(getattr(response, "passes_used", []) or []),
            "mode": getattr(response, "mode", None),
            "triples": [],
        }
        for r in getattr(response, "results", []) or []:
            tr = self._format_triple(r.triple)
            tr["score"] = r.score
            tr["source"] = r.source
            tr["depth"] = r.depth
            tr["entity_name"] = r.entity_name
            out["triples"].append(tr)
        return out

    def _format_triple(self, tr: Any, subject_entity: Any = None) -> dict:
        get = lambda k, default=None: tr.get(k, default) if isinstance(tr, dict) else getattr(tr, k, default)
        return {
            "id": get("id"),
            "subject_id": get("subject_id"),
            "subject_name": getattr(subject_entity, "name", None),
            "predicate": get("predicate"),
            "object_id": get("object_id"),
            "object_value": get("object_value"),
            "relation_type": get("relation_type"),
            "confidence": get("confidence"),
            "valid_from": get("valid_from"),
            "valid_until": get("valid_until"),
            "source_ref": get("source_ref"),
        }
