"""ANN index wrapper: FAISS (HNSW) if available, otherwise exact numpy search."""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = True, hnsw_m: int = 32):
        self.dim = dim
        self.ids: list[str | None] = []
        self._pos: dict[str, int] = {}
        self._vecs: list[np.ndarray | None] = []
        self._faiss = None
        self._dirty = False
        if use_faiss:
            try:
                import faiss
                self._faiss = faiss.IndexHNSWFlat(dim, hnsw_m)
            except ImportError:
                self._faiss = None

    def add(self, rid: str, vec: np.ndarray) -> None:
        if rid in self._pos:
            self.remove(rid)
        self._pos[rid] = len(self.ids)
        self.ids.append(rid)
        self._vecs.append(vec.astype(np.float32))
        if self._faiss is not None:
            self._faiss.add(vec.astype(np.float32)[None, :])

    def remove(self, rid: str) -> None:
        # HNSW does not support deletion; tombstone and fall back to exact search.
        pos = self._pos.pop(rid)
        self.ids[pos] = None
        self._vecs[pos] = None
        self._dirty = True

    def has(self, rid: str) -> bool:
        return rid in self._pos

    def get(self, rid: str) -> np.ndarray:
        return self._vecs[self._pos[rid]]

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        live = [(i, v) for i, v in zip(self.ids, self._vecs) if i is not None]
        if not live:
            return []
        if self._faiss is not None and not self._dirty:
            d, idx = self._faiss.search(q.astype(np.float32)[None, :], min(k, len(self.ids)))
            return [(self.ids[j], float(dd)) for dd, j in zip(d[0], idx[0]) if j >= 0]
        ids, vecs = zip(*live)
        m = np.stack(vecs)
        d = np.linalg.norm(m - q[None, :], axis=1)
        order = np.argsort(d)[:k]
        return [(ids[i], float(d[i])) for i in order]

    def __len__(self) -> int:
        return len(self._pos)
