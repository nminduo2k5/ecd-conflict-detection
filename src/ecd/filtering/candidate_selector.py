"""Step [3]: two-tier candidate selection for a new/modified rule r.

Modes (paper §3):
  exact       — Tier A, then verify ALL survivors in embedding order
                (recall 100%; embedding only accelerates time-to-first-conflict).
  approximate — Tier A, then verify only the top-k survivors
                (fast; report empirical recall; intended for CI/CD).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ecd.core import Ontology, Rule
from .sound_pruning import PruneDecision, PruneReason, Pruner
from .ranking import EmbeddingRanker


class Mode(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


@dataclass
class SelectionStats:
    total: int = 0
    pruned_same_effect: int = 0
    pruned_disjoint: int = 0
    survivors: int = 0
    verified: int = 0

    @property
    def pruning_ratio(self) -> float:
        return 0.0 if self.total == 0 else 1 - self.survivors / self.total


@dataclass
class Selection:
    ordered: list[Rule]              # candidates to verify, in this order
    stats: SelectionStats
    decisions: dict[str, PruneDecision] = field(default_factory=dict)


class CandidateSelector:
    def __init__(self, ranker: EmbeddingRanker | None = None,
                 mode: Mode = Mode.EXACT, top_k: int = 50,
                 ontology: Ontology | None = None, use_tier_a: bool = True,
                 keep_decisions: bool = False):
        self.ranker = ranker
        self.mode = mode
        self.top_k = top_k
        self.pruner = Pruner(ontology)
        self.use_tier_a = use_tier_a          # False = ablation "without Tier A"
        self.keep_decisions = keep_decisions

    def select(self, target: Rule, others: list[Rule]) -> Selection:
        st = SelectionStats()
        survivors: list[Rule] = []
        decisions: dict[str, PruneDecision] = {}
        if self.use_tier_a:
            for r, dec in self.pruner.prune_candidates(target, others):
                st.total += 1
                if self.keep_decisions:
                    decisions[r.id] = dec
                if dec.pruned:
                    if dec.reason is PruneReason.SAME_EFFECT:
                        st.pruned_same_effect += 1
                    else:
                        st.pruned_disjoint += 1
                else:
                    survivors.append(r)
        else:
            # ablation: only the definitional (P0) filter, which is not a heuristic
            for r in others:
                if r.id == target.id:
                    continue
                st.total += 1
                if r.effect == target.effect:
                    st.pruned_same_effect += 1
                else:
                    survivors.append(r)
        st.survivors = len(survivors)

        if self.ranker is not None:
            ordered = [rc.rule for rc in self.ranker.rank(target, survivors)]
        else:
            ordered = survivors  # no embedding: arbitrary order (baseline)

        if self.mode is Mode.APPROXIMATE:
            ordered = ordered[: self.top_k]
        st.verified = len(ordered)
        return Selection(ordered, st, decisions)
