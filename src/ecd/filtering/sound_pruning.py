"""Tier A — sound symbolic pruning.

Theorem 1 (soundness of Tier A): every pair discarded by Tier A is NOT a conflict.

Proof sketch. Let ⟦r⟧ be the canonical applicability of r (core.ontology:
aliases renamed, hierarchies expanded — the same predicate the SMT step and
Event-B reason about).  A pair (r_i, r_j) is discarded iff one of:
  (P0) eff_i = eff_j                                   — violates the definition of conflict.
  (P1) ∃ attribute x referenced by BOTH ⟦r_i⟧ and ⟦r_j⟧ such that
       α_i(x) ∩ α_j(x) = ∅, where α_k(x) is an over-approximation of
       {q(x) | ⟦r_k⟧(q)} computed from the atoms on x alone.
Since applicability is a conjunction (atoms ∧ raw_expr), the concrete set of
q(x) values is a subset of α_k(x); disjoint over-approximations therefore
imply disjoint concrete sets, so no q satisfies both rules. ∎

Pairs that survive are "maybe overlapping" and go to Tier B / verification.

Performance note: abstractions are memoised per Rule object (rules are
immutable), so a Δ against a 10⁵-rule policy costs one dict lookup and a few
set intersections per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ecd.core import Rule, Ontology
from ecd.core.intervals import Interval, ValueSet, abstract_attr


class PruneReason(str, Enum):
    SAME_EFFECT = "same-effect"
    DISJOINT_ATTR = "disjoint-attribute"
    KEEP = "keep"


@dataclass(frozen=True)
class PruneDecision:
    pruned: bool
    reason: PruneReason
    attr: str | None = None

    @staticmethod
    def keep() -> "PruneDecision":
        return PruneDecision(False, PruneReason.KEEP)


_KEEP = PruneDecision.keep()
_SAME = PruneDecision(True, PruneReason.SAME_EFFECT)

Abstraction = dict[str, tuple[Interval | None, ValueSet | None]]


class Pruner:
    """Tier A with an ontology and a per-rule abstraction cache."""

    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or Ontology()
        self._abs: dict[int, tuple[Rule, Abstraction]] = {}   # id(rule) -> (rule, abstraction)

    def abstraction(self, r: Rule) -> Abstraction:
        key = id(r)
        hit = self._abs.get(key)
        if hit is not None and hit[0] is r:
            return hit[1]
        pred = self.ontology.canonical_predicate(r.applicability)
        by_attr: dict[str, list] = {}
        for a in pred.atoms:
            by_attr.setdefault(a.attr, []).append(a)
        abs_ = {x: abstract_attr(atoms) for x, atoms in by_attr.items()}
        self._abs[key] = (r, abs_)
        return abs_

    def decide(self, a: Rule, b: Rule, *, ignore_effect: bool = False) -> PruneDecision:
        if not ignore_effect and a.effect == b.effect:
            return _SAME
        aa, ab = self.abstraction(a), self.abstraction(b)
        if len(aa) > len(ab):
            aa, ab = ab, aa
        for x, (ia, va) in aa.items():
            other = ab.get(x)
            if other is None:
                continue
            ib, vb = other
            if ia is not None and ib is not None and ia.intersect(ib).is_empty():
                return PruneDecision(True, PruneReason.DISJOINT_ATTR, x)
            if va is not None and vb is not None and va.intersect(vb).is_empty():
                return PruneDecision(True, PruneReason.DISJOINT_ATTR, x)
            # mixed numeric/categorical on same attr: cannot conclude → keep
        return _KEEP

    def prune_candidates(self, target: Rule, others, *, ignore_effect: bool = False):
        """Yield (rule, decision) for every candidate partner of `target`."""
        for r in others:
            if r.id == target.id:
                continue
            yield r, self.decide(target, r, ignore_effect=ignore_effect)

    def clear(self) -> None:
        self._abs.clear()


# ----- module-level convenience (no ontology, shared cache) ---------------
_default = Pruner()


def decide(a: Rule, b: Rule, *, ignore_effect: bool = False,
           ontology: Ontology | None = None) -> PruneDecision:
    """Return a pruning decision for the pair (a, b).

    ignore_effect=True is used when labelling pairs for the embedding
    (we care about *overlap*, not conflict), so (P0) is skipped.
    """
    if ontology is None or ontology.is_empty():
        return _default.decide(a, b, ignore_effect=ignore_effect)
    return Pruner(ontology).decide(a, b, ignore_effect=ignore_effect)


def prune_candidates(target: Rule, others: list[Rule], *, ignore_effect: bool = False,
                     ontology: Ontology | None = None):
    p = _default if (ontology is None or ontology.is_empty()) else Pruner(ontology)
    yield from p.prune_candidates(target, others, ignore_effect=ignore_effect)
