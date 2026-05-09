from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - exercised only when faiss is absent
    faiss = None


@dataclass
class RAMRetrieval:
    scores: np.ndarray
    indices: np.ndarray
    vectors: np.ndarray
    weights: np.ndarray
    weighted: np.ndarray


class _NumpyMIPS:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.concatenate([self.vectors, vectors.astype(np.float32)], axis=0)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query.astype(np.float32) @ self.vectors.T
        k = min(k, self.vectors.shape[0])
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(scores.shape[0])[:, None]
        partial = scores[row, idx]
        order = np.argsort(-partial, axis=1)
        idx = idx[row, order]
        return scores[row, idx], idx.astype(np.int64)

    def reconstruct_batch(self, indices: np.ndarray) -> np.ndarray:
        return self.vectors[indices]


class FaissTensorRAM:
    """External frozen tensor RAM with MIPS retrieval and softmax weighting."""

    def __init__(self, d_ram: int = 4096, use_faiss: bool = False):
        self.d_ram = d_ram
        self.use_faiss = bool(use_faiss and faiss is not None)
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(d_ram)
        else:
            self.index = _NumpyMIPS(d_ram)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.d_ram:
            raise ValueError(f"expected [n, {self.d_ram}], got {vectors.shape}")
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            raise RuntimeError("RAM index is empty")
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        if query.shape[1] != self.d_ram:
            raise ValueError(f"expected query dim {self.d_ram}, got {query.shape[1]}")
        k = min(k, self.ntotal)
        scores, indices = self.index.search(query, k)
        return scores.astype(np.float32), indices.astype(np.int64)

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self.use_faiss:
            flat = indices.reshape(-1)
            vectors = np.stack([self.index.reconstruct(int(i)) for i in flat], axis=0)
            return vectors.reshape(*indices.shape, self.d_ram).astype(np.float32)
        return self.index.reconstruct_batch(indices).astype(np.float32)

    def weighted_retrieve(
        self,
        query: np.ndarray,
        k: int = 8,
        temperature: float = 1.0,
    ) -> RAMRetrieval:
        scores, indices = self.search(query, k=k)
        vectors = self.reconstruct(indices)
        scaled = scores / max(temperature, 1e-6)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights = weights / weights.sum(axis=1, keepdims=True)
        weighted = (weights[:, :, None] * vectors).sum(axis=1)
        return RAMRetrieval(
            scores=scores,
            indices=indices,
            vectors=vectors,
            weights=weights.astype(np.float32),
            weighted=weighted.astype(np.float32),
        )

    def save(self, path: str) -> None:
        if self.use_faiss:
            if not path.endswith(".faiss"):
                raise ValueError("FAISS-backed RAM must be saved with a .faiss extension")
            faiss.write_index(self.index, path)
            return
        if not path.endswith(".npy"):
            raise ValueError("NumPy-backed RAM must be saved with a .npy extension")
        np.save(path, self.index.vectors)

    @classmethod
    def load(cls, path: str, d_ram: int = 4096) -> "FaissTensorRAM":
        ram = cls(d_ram=d_ram, use_faiss=path.endswith(".faiss"))
        if ram.use_faiss:
            ram.index = faiss.read_index(path)
        else:
            ram.index.add(np.load(path))
        return ram
