"""Dependency-free embedders used as baselines / fallbacks.

HashedAtomEmbedder — bag of hashed (attr, value) features (no learning).
                     `use_ontology=True` hashes *canonical* tokens instead
                     (ablation: "hand-coded ontology, no learning").
SBertEmbedder      — pure-text Sentence-BERT baseline (needs sentence-transformers).
"""
from __future__ import annotations

import numpy as np

from ecd.core import AttributeSpace, Policy, Rule
from .base import Embedder
from ecd.xfg.builder import tok_id


class HashedAtomEmbedder(Embedder):
    def __init__(self, dim: int = 256, use_ontology: bool = False, space: AttributeSpace | None = None):
        self.dim = dim
        self.use_ontology = use_ontology
        self.space = space
        self._proj: dict[int, np.ndarray] = {}

    def bind(self, policy: Policy) -> None:
        self.space = policy.space

    def _vec(self, tok: int) -> np.ndarray:
        if tok not in self._proj:
            r = np.random.default_rng(tok)
            self._proj[tok] = r.standard_normal(self.dim).astype(np.float32)
        return self._proj[tok]

    def _pred(self, r: Rule):
        p = r.applicability
        if self.use_ontology and self.space is not None:
            p = self.space.canonical(p)
        return p

    def embed(self, rules: list[Rule]) -> np.ndarray:
        out = np.zeros((len(rules), self.dim), dtype=np.float32)
        for i, r in enumerate(rules):
            pred = self._pred(r)
            toks = [tok_id(f"attr:{a.attr}") for a in pred.atoms]
            for a in pred.atoms:
                vals = a.value if isinstance(a.value, tuple) else (a.value,)
                toks += [tok_id(f"val:{a.attr}={v}") for v in vals]
            for t in toks:
                out[i] += self._vec(t)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def rule_to_text(r: Rule) -> str:
    parts = [f"{a.attr} {a.op} {a.value}" for a in r.applicability.atoms]
    if r.applicability.raw_expr:
        parts.append(r.applicability.raw_expr)
    return f"{r.effect.value} if " + " and ".join(parts)


class SBertEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, rules: list[Rule]) -> np.ndarray:
        return self.model.encode([rule_to_text(r) for r in rules], batch_size=256,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
