"""Tier B — embedding-based ranking of surviving candidate pairs.

Given an embedder e(·) and an ANN index, rank partners of rule r by an
*overlap score* s(r, r') = -dist(e(r), e(r')). The ranking NEVER decides
conflict; it only orders (approximate mode: truncates) what verification sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecd.core import Rule
from ecd.embedding.base import Embedder
from ecd.embedding.index import VectorIndex


@dataclass
class RankedCandidate:
    rule: Rule
    score: float   # higher = more likely to overlap


class EmbeddingRanker:
    def __init__(self, embedder: Embedder, index: VectorIndex | None = None):
        self.embedder = embedder
        self.index = index

    def rank(self, target: Rule, candidates: list[Rule]) -> list[RankedCandidate]:
        if not candidates:
            return []
        q = self.embedder.embed([target])[0]
        if self.index is not None and all(self.index.has(c.id) for c in candidates):
            vecs = np.stack([self.index.get(c.id) for c in candidates])
        else:
            vecs = self.embedder.embed(candidates)
        d = np.linalg.norm(vecs - q[None, :], axis=1)
        order = np.argsort(d)
        return [RankedCandidate(candidates[i], float(-d[i])) for i in order]
