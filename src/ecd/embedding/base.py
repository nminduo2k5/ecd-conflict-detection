from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ecd.core import Policy, Rule


class Embedder(ABC):
    """e : Rule → R^d. Implementations must be deterministic for a given rule."""
    dim: int

    @abstractmethod
    def embed(self, rules: list[Rule]) -> np.ndarray:  # (n, dim)
        ...

    def bind(self, policy: Policy) -> None:
        """Give the embedder the policy context (attribute space / ontology). Optional."""
        return None
